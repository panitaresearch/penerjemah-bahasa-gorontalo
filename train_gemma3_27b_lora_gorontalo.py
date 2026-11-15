#!/usr/bin/env python3
"""
train_gemma3_27b_lora_gorontalo.py
==================================
Fine-Tuning LLM Instruksi & Nalar Morfologi Bahasa Indonesia <-> Bahasa Gorontalo
menggunakan GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct + LoRA (BF16 default di A100-80GB,
dengan fallback opsional QLoRA 4-bit via GEMMA_LOAD_IN_4BIT=1).

Dirancang khusus untuk arsitektur Gemma 3 Multimodal (`Gemma3ForConditionalGeneration`):
  - Base Model: GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct (CPT Nusantara & bahasa daerah)
  - Chat Template: format resmi `<bos><start_of_turn>user ... <end_of_turn>\\n<start_of_turn>model`
  - Isolasi Teks: membekukan `vision_tower` & `multi_modal_projector`; LoRA hanya pada `language_model`
  - Dataset: sft_augmented_{train,val,test}.parquet (korpus v6, 4 Siasat Prof. Dr. H. Mansoer Pateda)
  - Evaluasi: SacreBLEU, chrF++, dan penalaran Morfologi/Kala/Serapan pada Test Set
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path

import pandas as pd
import sacrebleu
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
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

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

BASE_DIR = Path("/content/drive/MyDrive/REPO/richie/modelgorontalo")
DATA_DIR = BASE_DIR / "dataset_terpadu_indonesia_gorontalo"
FINAL_LORA_DIR = BASE_DIR / "models" / "gemma3-27b-gotoai-gorontalo-lora"
LOCAL_CKPT_DIR = BASE_DIR / "ckpts_gemma3"

MODEL_NAME = os.environ.get("GEMMA_BASE_MODEL", "GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct")
MAX_SEQ_LEN = int(os.environ.get("GEMMA_MAX_SEQ_LEN", "256"))

# Target LoRA dibatasi eksplisit ke layer bahasa agar tidak menempel ke vision_tower SigLIP.
TEXT_LORA_TARGET_REGEX = (
    r"model\.language_model\.layers\.\d+\."
    r"(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)"
)

SYSTEM_PROMPT = (
    "Anda adalah pakar bahasa dan penerjemah dwibahasa Bahasa Indonesia <-> Bahasa Gorontalo "
    "yang menguasai tata bahasa, 5 sistem kala, morfofonemik, morfem penunjuk arah (mayi, mao, mola, mota), "
    "dan kaidah leksikografi Prof. Dr. H. Mansoer Pateda."
)


def preflight_dependency_check(use_4bit: bool) -> None:
    """Validasi paket kritis sebelum mengunduh bobot ~51 GiB."""
    try:
        from peft.import_utils import is_torchao_available

        is_torchao_available()
    except ImportError as exc:
        raise RuntimeError(
            "Konflik versi torchao terdeteksi oleh PEFT sebelum get_peft_model(). "
            "Jalankan `pip install -U torchao` atau `pip uninstall -y torchao` lalu ulangi."
        ) from exc

    if use_4bit:
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "GEMMA_LOAD_IN_4BIT=1 memerlukan `bitsandbytes`. Pasang dengan `pip install -U bitsandbytes`."
            ) from exc


def build_gemma3_prompt(instruction: str, user_input: str, tokenizer) -> str:
    """Format prompt sesuai chat_template.jinja resmi Gemma 3 GoTo."""
    user_msg = f"{instruction}\n\nInput: {user_input}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def get_gemma3_eos_ids(tokenizer) -> list[int]:
    """Kumpulkan token penghenti resmi Gemma 3 (<eos> dan <end_of_turn>)."""
    eos_ids = []
    if tokenizer.eos_token_id is not None:
        eos_ids.append(int(tokenizer.eos_token_id))
    eot_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot_id, int) and eot_id >= 0 and eot_id not in eos_ids and eot_id != tokenizer.unk_token_id:
        eos_ids.append(eot_id)
    return eos_ids or [1]


def format_and_tokenize_sft(df: pd.DataFrame, tokenizer, max_seq_len: int = MAX_SEQ_LEN) -> Dataset:
    """Tokenisasi SFT format Gemma 3 dengan masking -100 pada seluruh prefix prompt."""
    t0 = time.time()
    prompts = []
    full_texts = []

    for _, row in df.iterrows():
        prompt_str = build_gemma3_prompt(str(row["instruction"]), str(row["input"]), tokenizer)
        full_str = f"{prompt_str}{row['output']}<end_of_turn>\n"
        prompts.append(prompt_str)
        full_texts.append(full_str)

    prompt_enc = tokenizer(prompts, add_special_tokens=False, truncation=True, max_length=max_seq_len)
    full_enc = tokenizer(full_texts, add_special_tokens=False, truncation=True, max_length=max_seq_len)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for p_ids, f_ids in zip(prompt_enc["input_ids"], full_enc["input_ids"]):
        # Gunakan panjang prefiks identik agar batas SentencePiece tidak pernah salah geser.
        p_len = 0
        limit = min(len(p_ids), len(f_ids))
        while p_len < limit and p_ids[p_len] == f_ids[p_len]:
            p_len += 1
        labels = [-100] * p_len + f_ids[p_len:]
        input_ids_list.append(f_ids)
        attention_mask_list.append([1] * len(f_ids))
        labels_list.append(labels)

    ds = Dataset.from_dict({
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    })
    print(f"Tokenisasi Gemma3 SFT {len(df)} baris selesai dalam {time.time() - t0:.2f} detik.")
    return ds


@torch.inference_mode()
def generate_responses_batch(df_sub: pd.DataFrame, model, tokenizer, batch_size: int = 32) -> list[str]:
    model.eval()
    old_use_cache = getattr(model.config, "use_cache", False)
    model.config.use_cache = True
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    eos_ids = get_gemma3_eos_ids(tokenizer)
    outputs = []

    prompts = [
        build_gemma3_prompt(str(r["instruction"]), str(r["input"]), tokenizer)
        for _, r in df_sub.iterrows()
    ]

    try:
        for i in range(0, len(prompts), batch_size):
            batch_p = prompts[i : i + batch_size]
            enc = tokenizer(
                batch_p,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=MAX_SEQ_LEN,
            ).to(model.device)
            input_len = enc["input_ids"].shape[1]
            gen_ids = model.generate(
                **enc,
                max_new_tokens=96,
                do_sample=False,
                top_k=None,
                top_p=None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids,
            )
            new_tokens = gen_ids[:, input_len:]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            outputs.extend([d.strip() for d in decoded])
    finally:
        tokenizer.padding_side = orig_padding_side
        model.config.use_cache = old_use_cache

    return outputs


def is_clean_sft(r) -> bool:
    s, t = str(r["input"]).strip(), str(r["output"]).strip()
    if "=" in s or "=" in t or ";" in s or ";" in t or "—" in s or "—" in t:
        return False
    if re.search(r"\(\d+\)|\b\d+\.", s) or re.search(r"\(\d+\)|\b\d+\.", t):
        return False
    ws, wt = len(s.split()), len(t.split())
    if ws < 2 or wt < 2 or ws > 22 or wt > 22 or ws / wt > 2.2 or wt / ws > 2.2:
        return False
    return True


def main():
    t_start = time.time()
    use_4bit = os.environ.get("GEMMA_LOAD_IN_4BIT", "0").lower() in {"1", "true", "yes"}
    preflight_dependency_check(use_4bit=use_4bit)

    attn_impl = os.environ.get("GEMMA_ATTN_IMPL", "sdpa")
    lora_r = int(os.environ.get("GEMMA_LORA_R", "16"))
    lora_alpha = int(os.environ.get("GEMMA_LORA_ALPHA", str(lora_r * 2)))
    num_epochs = int(os.environ.get("GEMMA_EPOCHS", "1"))
    train_bs = int(os.environ.get("GEMMA_BATCH_SIZE", "8"))
    grad_accum = int(os.environ.get("GEMMA_GRAD_ACCUM", "8"))
    eval_bs = int(os.environ.get("GEMMA_EVAL_BATCH_SIZE", str(train_bs)))
    save_steps = int(os.environ.get("GEMMA_SAVE_STEPS", "100"))
    lr = float(os.environ.get("GEMMA_LR", "1e-4"))
    use_grad_ckpt = os.environ.get("GEMMA_GRAD_CKPT", "1").lower() not in {"0", "false", "no"}
    force_retrain = os.environ.get("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    resume_eval_only = (not force_retrain) and is_training_already_completed(FINAL_LORA_DIR, is_lora=True)

    print(f"Memuat tokenizer & base model GoTo Gemma 3 27B: {MODEL_NAME}")
    print(f"Mode: {'QLoRA 4-bit (NF4)' if use_4bit else 'BF16 LoRA (A100-80GB)'} | attn_implementation={attn_impl}")
    tok_src = FINAL_LORA_DIR if resume_eval_only else MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(tok_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda" if torch.cuda.is_available() else "cpu",
        "attn_implementation": attn_impl,
    }
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **model_kwargs)
    model.config.use_cache = False

    df_test = pd.read_parquet(DATA_DIR / "sft_augmented_test.parquet")

    if resume_eval_only:
        print(f"\n=== Adapter GoTo Gemma 3 sudah selesai dilatih di {FINAL_LORA_DIR}; melanjutkan fase evaluasi test set ===")
        model = PeftModel.from_pretrained(model, FINAL_LORA_DIR)
        completed_meta = json.loads((FINAL_LORA_DIR / "training_completed.json").read_text(encoding="utf-8"))
        train_samples_count = int(completed_meta.get("train_samples", 0))
        final_train_loss = float(completed_meta.get("train_loss", 0.0))
    else:
        if use_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=use_grad_ckpt)

        # Bekukan semua bobot dasar (termasuk 439 tensor vision_tower & projector).
        for _, param in model.named_parameters():
            param.requires_grad = False

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            target_modules=TEXT_LORA_TARGET_REGEX,
            bias="none",
        )
        model = get_peft_model(model, lora_config)
        if use_grad_ckpt and not use_4bit:
            model.enable_input_require_grads()

        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"GPU: {torch.cuda.get_device_name(0)} ({vram_gb:.1f} GB) | Gradient checkpointing: {use_grad_ckpt}")
        model.print_trainable_parameters()

        print("Memuat dataset SFT Augmented v6 dari:", DATA_DIR)
        df_train_full = pd.read_parquet(DATA_DIR / "sft_augmented_train.parquet")
        df_val = pd.read_parquet(DATA_DIR / "sft_augmented_val.parquet")
        train_samples_count = len(df_train_full)

        train_ds = format_and_tokenize_sft(df_train_full, tokenizer, max_seq_len=MAX_SEQ_LEN)
        val_ds = format_and_tokenize_sft(df_val, tokenizer, max_seq_len=MAX_SEQ_LEN)

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        )

        training_args = TrainingArguments(
            output_dir=str(LOCAL_CKPT_DIR),
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=100,
            lr_scheduler_type="cosine",
            num_train_epochs=num_epochs,
            bf16=True,
            tf32=torch.cuda.is_available(),
            gradient_checkpointing=use_grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else None,
            prediction_loss_only=True,
            logging_steps=25,
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

        progress_cb = ProgressAndGracefulStopCallback(BASE_DIR / "logs" / "gemma3_27b_progress.json", t_start)
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
            print(f"\n=== Resume LoRA GoTo Gemma 3 27B dari checkpoint valid: {latest_ckpt} ===")
        else:
            print(
                f"\n=== Memulai Fine-Tuning GoTo Gemma 3 27B LoRA "
                f"(r={lora_r}, alpha={lora_alpha}, bs={train_bs}x{grad_accum}, lr={lr}, save_steps={save_steps}) ==="
            )

        try:
            train_result = trainer.train(resume_from_checkpoint=latest_ckpt)
        finally:
            progress_cb.restore_signal_handlers()

        if progress_cb.stop_requested:
            print("Pelatihan dihentikan aman karena sinyal sistem; checkpoint terakhir dipertahankan di:", LOCAL_CKPT_DIR)
            return

        final_train_loss = round(train_result.training_loss, 4)
        print(f"Pelatihan GoTo Gemma 3 27B selesai dalam {(time.time() - t_start)/60:.2f} menit. Train Loss: {final_train_loss:.4f}")

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

    print("\n=== Evaluasi GoTo Gemma 3 27B LoRA pada Test Set ===")
    eval_gen_bs = int(os.environ.get("GEMMA_TEST_GEN_BS", "32"))
    eval_store = PartialEvalStore(BASE_DIR / "logs" / "gemma3_eval_partial.json")
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

        preds = generate_responses_batch(sub, model, tokenizer, batch_size=eval_gen_bs)
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
        siasat_preds = generate_responses_batch(siasat_test_df, model, tokenizer, batch_size=4)
        eval_results["siasat_reasoning_samples"] = [
            {"instruction": r["instruction"], "input": r["input"], "prediction": p}
            for (_, r), p in zip(siasat_test_df.iterrows(), siasat_preds)
        ]
        eval_store.put("siasat_reasoning_samples", eval_results["siasat_reasoning_samples"])

    eval_results["training_summary"] = {
        "base_model": MODEL_NAME,
        "saved_lora_path": str(FINAL_LORA_DIR),
        "quantization": "4bit_nf4" if use_4bit else "bf16",
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "target_modules": TEXT_LORA_TARGET_REGEX,
        "learning_rate": lr,
        "batch_size": train_bs,
        "grad_accum": grad_accum,
        "train_samples": train_samples_count,
        "epochs": num_epochs,
        "train_loss": final_train_loss,
        "elapsed_minutes": round((time.time() - t_start) / 60, 2),
    }

    eval_out_path = DATA_DIR / "evaluasi_gemma3_27b_lora_gorontalo.json"
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print("Laporan evaluasi GoTo Gemma 3 27B LoRA disimpan ke:", eval_out_path)
    shutil.copy2(eval_out_path, FINAL_LORA_DIR / "evaluation_results.json")
    eval_store.clear()

    if LOCAL_CKPT_DIR.exists():
        shutil.rmtree(LOCAL_CKPT_DIR, ignore_errors=True)
        print("Checkpoint sementara Gemma 3 dibersihkan.")


if __name__ == "__main__":
    main()
