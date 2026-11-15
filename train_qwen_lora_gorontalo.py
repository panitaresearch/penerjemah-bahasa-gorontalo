#!/usr/bin/env python3
"""
train_qwen_lora_gorontalo.py
============================
Fine-Tuning LLM Instruksi & Nalar Morfologi Bahasa Indonesia <-> Bahasa Gorontalo
menggunakan Qwen/Qwen2.5-7B-Instruct + LoRA 16-bit (BF16) pada NVIDIA A100-SXM4-40GB GPU.
Konfigurasi sesuai AGENTS.md & SKILL.md:
  - Base Model: Qwen/Qwen2.5-7B-Instruct (penguasaan KBBI + penalaran morfologi)
  - LoRA: r=32, lora_alpha=64, target seluruh linear layers (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
  - Dataset: sft_augmented_train.parquet (4 Siasat Linguistik Prof. Dr. H. Mansoer Pateda)
  - Evaluasi: SacreBLEU, chrF++, dan akurasi Morfologi/Kala/Serapan pada Test Set
"""

import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)

from training_resilience import (
    PartialEvalStore,
    ProgressAndGracefulStopCallback,
    find_latest_valid_checkpoint,
    is_training_already_completed,
    mark_training_completed,
)

SEED = 42
set_seed(SEED)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

BASE_DIR = Path("/content/drive/MyDrive/REPO/richie/modelgorontalo")
DATA_DIR = BASE_DIR / "dataset_terpadu_indonesia_gorontalo"
FINAL_LORA_DIR = BASE_DIR / "models" / "qwen2.5-7b-gorontalo-lora"
LOCAL_CKPT_DIR = BASE_DIR / "ckpts_qwen"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN = 256


SYSTEM_PROMPT = (
    "Anda adalah pakar bahasa dan penerjemah dwibahasa Bahasa Indonesia <-> Bahasa Gorontalo "
    "yang menguasai tata bahasa, 5 sistem kala, morfofonemik, morfem penunjuk arah (mayi, mao, mola, mota), "
    "dan kaidah leksikografi Prof. Dr. H. Mansoer Pateda."
)


def format_and_tokenize_sft(df: pd.DataFrame, tokenizer) -> Dataset:
    """
    Memformat sampel instruksi menjadi format ChatML Qwen2.5 dan melakukan masking
    (-100) pada bagian prompt agar loss hanya dihitung pada respons asisten.
    """
    t0 = time.time()
    prompts = []
    full_texts = []

    for _, row in df.iterrows():
        user_msg = f"{row['instruction']}\n\nInput: {row['input']}"
        prompt_str = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        full_str = f"{prompt_str}{row['output']}<|im_end|>"
        prompts.append(prompt_str)
        full_texts.append(full_str)

    prompt_enc = tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)
    full_enc = tokenizer(full_texts, add_special_tokens=False, truncation=True, max_length=MAX_SEQ_LEN)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for p_ids, f_ids in zip(prompt_enc["input_ids"], full_enc["input_ids"]):
        p_len = min(len(p_ids), len(f_ids))
        labels = [-100] * p_len + f_ids[p_len:]
        input_ids_list.append(f_ids)
        attention_mask_list.append([1] * len(f_ids))
        labels_list.append(labels)

    ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })
    print(f"Tokenisasi SFT {len(df)} baris selesai dalam {time.time() - t0:.2f} detik.")
    return ds


@torch.inference_mode()
def generate_responses_batch(df_sub: pd.DataFrame, model, tokenizer, batch_size: int = 32) -> list[str]:
    model.eval()
    tokenizer.padding_side = "left"
    outputs = []

    prompts = [
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{r['instruction']}\n\nInput: {r['input']}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        for _, r in df_sub.iterrows()
    ]

    for i in range(0, len(prompts), batch_size):
        batch_p = prompts[i : i + batch_size]
        enc = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(
            model.device
        )
        input_len = enc["input_ids"].shape[1]
        gen_ids = model.generate(
            **enc,
            max_new_tokens=96,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        new_tokens = gen_ids[:, input_len:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])

    tokenizer.padding_side = "right"
    return outputs


def main():
    t_start = time.time()
    num_epochs = int(os.environ.get("QWEN_EPOCHS", "1"))
    train_bs = int(os.environ.get("QWEN_BATCH_SIZE", "32"))
    grad_accum = int(os.environ.get("QWEN_GRAD_ACCUM", "2"))
    save_steps = int(os.environ.get("QWEN_SAVE_STEPS", "150"))
    force_retrain = os.environ.get("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    resume_eval_only = (not force_retrain) and is_training_already_completed(FINAL_LORA_DIR, is_lora=True)

    print(f"Memuat tokenizer & base model LLM: {MODEL_NAME}")
    tok_src = FINAL_LORA_DIR if resume_eval_only else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.config.use_cache = False

    df_test = pd.read_parquet(DATA_DIR / "sft_augmented_test.parquet")

    if resume_eval_only:
        print(f"\n=== Adapter Qwen2.5-7B sudah selesai dilatih di {FINAL_LORA_DIR}; melanjutkan fase evaluasi test set ===")
        model = PeftModel.from_pretrained(model, FINAL_LORA_DIR)
        completed_meta = json.loads((FINAL_LORA_DIR / "training_completed.json").read_text(encoding="utf-8"))
        train_samples_count = int(completed_meta.get("train_samples", 0))
        final_train_loss = float(completed_meta.get("train_loss", 0.0))
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        grad_ckpt_env = os.environ.get("QWEN_GRAD_CKPT", "1").lower()
        use_grad_ckpt = grad_ckpt_env not in {"0", "false", "no"}
        if use_grad_ckpt:
            model.enable_input_require_grads()
        print(f"Gradient checkpointing: {use_grad_ckpt} (VRAM GPU: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)")
        model.print_trainable_parameters()

        print("Memuat dataset SFT Augmented dari:", DATA_DIR)
        df_train_full = pd.read_parquet(DATA_DIR / "sft_augmented_train.parquet")
        df_val = pd.read_parquet(DATA_DIR / "sft_augmented_val.parquet")
        train_samples_count = len(df_train_full)

        train_ds = format_and_tokenize_sft(df_train_full, tokenizer)
        val_ds = format_and_tokenize_sft(df_val, tokenizer)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        )

        training_args = TrainingArguments(
            output_dir=str(LOCAL_CKPT_DIR),
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=train_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=2e-4,
            weight_decay=0.01,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            num_train_epochs=num_epochs,
            bf16=True,
            tf32=True,
            gradient_checkpointing=use_grad_ckpt,
            logging_steps=50,
            eval_strategy="steps",
            eval_steps=save_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=4,
            report_to="none",
            seed=SEED,
        )

        progress_cb = ProgressAndGracefulStopCallback(BASE_DIR / "logs" / "qwen_progress.json", t_start)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[progress_cb],
        )

        latest_ckpt = find_latest_valid_checkpoint(LOCAL_CKPT_DIR, is_lora=True)
        if latest_ckpt:
            print(f"\n=== Resume LoRA Qwen2.5-7B dari checkpoint valid: {latest_ckpt} ===")
        else:
            print(f"\n=== Memulai Fine-Tuning Qwen2.5-7B-Instruct LoRA (BF16 di A100-80GB, bs={train_bs}x{grad_accum}, save_steps={save_steps}) ===")

        try:
            train_result = trainer.train(resume_from_checkpoint=latest_ckpt)
        finally:
            progress_cb.restore_signal_handlers()

        if progress_cb.stop_requested:
            print("Pelatihan Qwen2.5-7B dihentikan aman karena sinyal sistem; checkpoint terakhir dipertahankan di:", LOCAL_CKPT_DIR)
            return

        final_train_loss = round(train_result.training_loss, 4)
        print(f"Pelatihan LoRA selesai dalam {(time.time() - t_start)/60:.2f} menit. Train Loss: {final_train_loss:.4f}")

        FINAL_LORA_DIR.mkdir(parents=True, exist_ok=True)
        print("Menyimpan adapter LoRA & tokenizer ke:", FINAL_LORA_DIR)
        model.save_pretrained(FINAL_LORA_DIR)
        tokenizer.save_pretrained(FINAL_LORA_DIR)
        mark_training_completed(
            FINAL_LORA_DIR,
            {
                "base_model": MODEL_NAME,
                "train_samples": train_samples_count,
                "epochs": num_epochs,
                "train_loss": final_train_loss,
            },
        )

    # Evaluasi pada Test Set (Keseluruhan & Subset Kalimat Bersih)
    import re
    def is_clean_sft(r):
        s, t = str(r["input"]).strip(), str(r["output"]).strip()
        if "=" in s or "=" in t or ";" in s or ";" in t or "—" in s or "—" in t:
            return False
        if re.search(r"\(\d+\)|\b\d+\.", s) or re.search(r"\(\d+\)|\b\d+\.", t):
            return False
        ws, wt = len(s.split()), len(t.split())
        if ws < 2 or wt < 2 or ws > 22 or wt > 22 or ws / wt > 2.2 or wt / ws > 2.2:
            return False
        return True

    print("\n=== Evaluasi Qwen2.5-7B-Instruct LoRA pada Test Set ===")
    eval_store = PartialEvalStore(BASE_DIR / "logs" / "qwen_eval_partial.json")
    eval_results = {}
    for task_name in ["translate_id_to_gor", "translate_gor_to_id", "translate_gor1987_to_id"]:
        sub = df_test[df_test["task"] == task_name].reset_index(drop=True)
        if len(sub) == 0:
            continue
        cached = eval_store.get(task_name)
        if isinstance(cached, dict) and cached.get("num_samples") == len(sub):
            eval_results[task_name] = cached
            print(f"[{task_name}] Memuat hasil evaluasi parsial tersimpan (n={len(sub)}): BLEU={cached['sacrebleu']}, chrF++={cached['chrf_plusplus']}")
            continue

        preds = generate_responses_batch(sub, model, tokenizer, batch_size=32)
        refs = sub["output"].tolist()
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)

        clean_mask = sub.apply(is_clean_sft, axis=1)
        clean_preds = [p for p, m in zip(preds, clean_mask) if m]
        clean_refs = [r for r, m in zip(refs, clean_mask) if m]
        c_bleu = sacrebleu.corpus_bleu(clean_preds, [clean_refs]).score if clean_preds else 0.0
        c_chrf = sacrebleu.corpus_chrf(clean_preds, [clean_refs], word_order=2).score if clean_preds else 0.0

        print(
            f"[{task_name}] All (n={len(sub)}): SacreBLEU={bleu.score:.2f}, chrF++={chrf.score:.2f} | "
            f"Clean (n={len(clean_preds)}): SacreBLEU={c_bleu:.2f}, chrF++={c_chrf:.2f}"
        )
        task_payload = {
            "num_samples": len(sub),
            "sacrebleu": round(bleu.score, 2),
            "chrf_plusplus": round(chrf.score, 2),
            "clean_num_samples": len(clean_preds),
            "clean_sacrebleu": round(c_bleu, 2),
            "clean_chrf_plusplus": round(c_chrf, 2),
            "samples": [
                {"input": sub["input"].iloc[i], "reference": refs[i], "prediction": preds[i]}
                for i in range(min(10, len(sub)))
            ],
        }
        eval_results[task_name] = task_payload
        eval_store.put(task_name, task_payload)

    # Uji Penalaran Morfologi, Kata Serapan, dan Kaidah Tata Bahasa Pateda
    siasat_test_df = pd.DataFrame([
        {
            "instruction": "Bentuklah kata turunan bahasa Gorontalo dari akar kata 'tuladu' beserta artinya dalam bahasa Indonesia.",
            "input": "Akar kata: tuladu -> Makna sasaran: menulis",
        },
        {
            "instruction": "Adaptasikan kata bahasa Indonesia berikut ke dalam bahasa Gorontalo sesuai kaidah fonotaktik vokalis Prof. Mansoer Pateda beserta contoh imbuhannya.",
            "input": "adil",
        },
        {
            "instruction": "Jelaskan 5 sistem kala (tense/aspect) dalam Bahasa Gorontalo menurut Prof. Dr. H. Mansoer Pateda beserta penanda morfologis dan contohnya.",
            "input": "Sistem Kala Bahasa Gorontalo",
        },
        {
            "instruction": "Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Gorontalo yang baik dan benar sesuai kaidah Prof. Mansoer Pateda.",
            "input": "Saya pergi ke pasar kemarin.",
        },
    ])
    cached_siasat = eval_store.get("siasat_reasoning_samples")
    if isinstance(cached_siasat, list) and len(cached_siasat) == len(siasat_test_df):
        eval_results["siasat_reasoning_samples"] = cached_siasat
    else:
        siasat_preds = generate_responses_batch(siasat_test_df, model, tokenizer, batch_size=8)
        eval_results["siasat_reasoning_samples"] = [
            {"instruction": r["instruction"], "input": r["input"], "prediction": p}
            for (_, r), p in zip(siasat_test_df.iterrows(), siasat_preds)
        ]
        eval_store.put("siasat_reasoning_samples", eval_results["siasat_reasoning_samples"])

    eval_results["training_summary"] = {
        "base_model": MODEL_NAME,
        "saved_lora_path": str(FINAL_LORA_DIR),
        "lora_r": 32,
        "lora_alpha": 64,
        "train_samples": train_samples_count,
        "epochs": num_epochs,
        "train_loss": final_train_loss,
        "elapsed_minutes": round((time.time() - t_start) / 60, 2),
    }

    eval_out_path = DATA_DIR / "evaluasi_qwen_lora_gorontalo.json"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print("Laporan evaluasi Qwen LoRA disimpan ke:", eval_out_path)
    shutil.copy2(eval_out_path, FINAL_LORA_DIR / "evaluation_results.json")
    print("Salinan evaluasi disimpan ke:", FINAL_LORA_DIR / "evaluation_results.json")
    eval_store.clear()

    if LOCAL_CKPT_DIR.exists():
        shutil.rmtree(LOCAL_CKPT_DIR, ignore_errors=True)
        print("Checkpoint sementara Qwen dibersihkan.")


if __name__ == "__main__":
    main()
