#!/usr/bin/env python3
"""
train_nllb_gorontalo.py
=======================
Full Fine-Tuning (BF16) model Seq2Seq Dwibahasa (Indonesia <-> Gorontalo)
menggunakan facebook/nllb-200-distilled-600M pada NVIDIA A100-SXM4-40GB GPU.
Mendukung:
  - Token bahasa khusus `gor_Latn` (diinisialisasi dari rata-rata rumpun Austronesia/Sulawesi-Filipina: ind_Latn, bug_Latn, tgl_Latn, ceb_Latn)
  - Pelatihan dua arah: ind_Latn -> gor_Latn dan gor_Latn -> ind_Latn (96.004 sampel)
  - Evaluasi otomatis SacreBLEU & chrF++ pada Test Set terstratifikasi + Uji 4 Siasat Linguistik Pateda
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
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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
FINAL_MODEL_DIR = Path(os.environ.get("NLLB_OUT_DIR", BASE_DIR / "models" / "nllb-200-gorontalo-bi-600M"))
EVAL_JSON_NAME = os.environ.get("NLLB_EVAL_JSON", "evaluasi_nllb_gorontalo.json")
LOCAL_CKPT_DIR = BASE_DIR / "ckpts_nllb"

MODEL_NAME = "facebook/nllb-200-distilled-600M"
MAX_LENGTH = 128

# v3: lebih banyak epoch + bobot ekstra (oversampling) untuk leksikon/morfologi frasa pendek
NUM_EPOCHS = int(os.environ.get("NLLB_EPOCHS", "4"))
SHORT_LEX_WEIGHT = int(os.environ.get("NLLB_LEX_WEIGHT", "3"))  # 1 = tanpa oversampling
SHORT_PHRASE_MAX_WORDS = 3
LEXICON_TYPES = {"lexicon", "lexicon_orth1987", "morphology"}


def upweight_short_lexicon(df: pd.DataFrame, weight: int) -> pd.DataFrame:
    """Gandakan baris leksikon/morfologi berfrasa pendek (<=3 kata di sisi sumber) sebanyak `weight` kali."""
    if weight <= 1:
        return df
    is_short = df["type"].isin(LEXICON_TYPES) & (
        df["src_text"].astype(str).str.split().str.len() <= SHORT_PHRASE_MAX_WORDS
    )
    short = df[is_short]
    out = pd.concat([df] + [short] * (weight - 1), ignore_index=True)
    out = out.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"Oversampling leksikon frasa pendek x{weight}: {len(short)} baris dasar -> total latih {len(df)} -> {len(out)}")
    return out


def register_gorontalo_token(tokenizer, model):
    """
    Mendaftarkan token bahasa `gor_Latn` ke dalam NllbTokenizer dan menginisialisasi
    embedding-nya dari rata-rata bahasa serumpun (Filipina-Gorontalo-Mongondow & Sulawesi/Indonesia):
    `ind_Latn`, `bug_Latn`, `tgl_Latn`, `ceb_Latn`.
    """
    if "gor_Latn" not in tokenizer.get_vocab():
        tokenizer.add_tokens(["gor_Latn"], special_tokens=True)

    model.resize_token_embeddings(len(tokenizer))
    gor_id = tokenizer.convert_tokens_to_ids("gor_Latn")
    donor_langs = ["ind_Latn", "bug_Latn", "tgl_Latn", "ceb_Latn"]
    donor_ids = [tokenizer.convert_tokens_to_ids(l) for l in donor_langs if l in tokenizer.get_vocab()]

    with torch.no_grad():
        emb_weight = model.get_input_embeddings().weight
        mean_emb = emb_weight[donor_ids].mean(dim=0)
        emb_weight[gor_id] = mean_emb
        if model.get_output_embeddings() is not None:
            out_weight = model.get_output_embeddings().weight
            if out_weight.shape[0] > gor_id:
                out_weight[gor_id] = out_weight[donor_ids].mean(dim=0)

    print(f"Token 'gor_Latn' berhasil didaftarkan dengan ID={gor_id} (diinisialisasi dari {donor_langs})")
    return gor_id


def fast_tokenize_dataframe(df: pd.DataFrame, tokenizer) -> Dataset:
    """
    Tokenisasi cepat secara vektor/batch mengikuti struktur NLLB-200:
    input_ids  = [src_lang_id] + subword_ids(src_text) + [eos_token_id]
    labels     = [tgt_lang_id] + subword_ids(tgt_text) + [eos_token_id]
    """
    t0 = time.time()
    eos_id = tokenizer.eos_token_id
    lang_to_id = {
        "ind_Latn": tokenizer.convert_tokens_to_ids("ind_Latn"),
        "gor_Latn": tokenizer.convert_tokens_to_ids("gor_Latn"),
    }

    src_texts = df["src_text"].astype(str).tolist()
    tgt_texts = df["tgt_text"].astype(str).tolist()
    src_langs = df["src_lang"].tolist()
    tgt_langs = df["tgt_lang"].tolist()

    src_enc = tokenizer(src_texts, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH - 2)
    tgt_enc = tokenizer(tgt_texts, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH - 2)

    input_ids = []
    attention_mask = []
    labels = []

    for s_ids, t_ids, s_lang, t_lang in zip(src_enc["input_ids"], tgt_enc["input_ids"], src_langs, tgt_langs):
        inp = [lang_to_id[s_lang]] + s_ids + [eos_id]
        lbl = [lang_to_id[t_lang]] + t_ids + [eos_id]
        input_ids.append(inp)
        attention_mask.append([1] * len(inp))
        labels.append(lbl)

    ds = Dataset.from_dict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    })
    print(f"Tokenisasi {len(df)} baris selesai dalam {time.time() - t0:.2f} detik.")
    return ds


@torch.inference_mode()
def translate_batch(texts: list[str], src_lang: str, tgt_lang: str, model, tokenizer, batch_size: int = 64) -> list[str]:
    model.eval()
    eos_id = tokenizer.eos_token_id
    src_lang_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    pad_id = tokenizer.pad_token_id

    outputs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(batch_texts, add_special_tokens=False, truncation=True, max_length=MAX_LENGTH - 2)
        batch_ids = [[src_lang_id] + ids + [eos_id] for ids in enc["input_ids"]]
        max_len = max(len(x) for x in batch_ids)

        padded_ids = []
        attn_masks = []
        for x in batch_ids:
            pad_len = max_len - len(x)
            padded_ids.append(x + [pad_id] * pad_len)
            attn_masks.append([1] * len(x) + [0] * pad_len)

        inp_tensor = torch.tensor(padded_ids, dtype=torch.long, device=model.device)
        mask_tensor = torch.tensor(attn_masks, dtype=torch.long, device=model.device)

        gen_ids = model.generate(
            input_ids=inp_tensor,
            attention_mask=mask_tensor,
            forced_bos_token_id=tgt_lang_id,
            max_new_tokens=96,
            num_beams=4,
            early_stopping=True,
        )
        decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        outputs.extend([d.strip() for d in decoded])
    return outputs


def evaluate_on_test(model, tokenizer, df_test: pd.DataFrame, eval_store: PartialEvalStore | None = None) -> dict:
    print("\n=== Evaluasi Komprehensif pada Test Set ===")
    results = {}

    for direction, s_lang, t_lang, filter_type in [
        ("id_to_gor", "ind_Latn", "gor_Latn", "parallel_sentence"),
        ("gor_to_id", "gor_Latn", "ind_Latn", "parallel_sentence"),
        ("gor1987_to_id", "gor_Latn", "ind_Latn", "parallel_sentence_orth1987"),
    ]:
        sub = df_test[
            (df_test["src_lang"] == s_lang)
            & (df_test["tgt_lang"] == t_lang)
            & (df_test["type"] == filter_type)
        ].reset_index(drop=True)
        if len(sub) == 0:
            continue

        cached = eval_store.get(direction) if eval_store else None
        if isinstance(cached, dict) and cached.get("num_samples") == len(sub):
            results[direction] = cached
            print(f"[{direction}] Memuat hasil evaluasi parsial tersimpan (n={len(sub)}): BLEU={cached['sacrebleu']}, chrF++={cached['chrf_plusplus']}")
            continue

        preds = translate_batch(sub["src_text"].tolist(), s_lang, t_lang, model, tokenizer, batch_size=64)
        refs = sub["tgt_text"].tolist()

        bleu = sacrebleu.corpus_bleu(preds, [refs])
        chrf = sacrebleu.corpus_chrf(preds, [refs], word_order=2)  # chrF++

        print(f"[{direction}] ({len(sub)} kalimat) -> SacreBLEU: {bleu.score:.2f} | chrF++: {chrf.score:.2f}")
        sample_pairs = []
        for idx in range(min(10, len(sub))):
            sample_pairs.append({
                "source": sub["src_text"].iloc[idx],
                "reference": refs[idx],
                "prediction": preds[idx],
            })
        dir_payload = {
            "num_samples": len(sub),
            "sacrebleu": round(bleu.score, 2),
            "chrf_plusplus": round(chrf.score, 2),
            "samples": sample_pairs,
        }
        results[direction] = dir_payload
        if eval_store:
            eval_store.put(direction, dir_payload)

    # Uji Kualitatif 4 Siasat Linguistik Prof. Mansoer Pateda (5 Kala, MPA, Serapan, Register)
    diagnostic_prompts_id_to_gor = [
        "Saya pergi ke pasar kemarin.",
        "Aku akan pergi membeli beras di pasar.",
        "Dia sedang duduk di rumah.",
        "Mereka sudah membawa buku itu ke sini.",
        "Tiap-tiap pagi halaman rumah mesti disapu.",
        "Kami mengabadikan namanya di prasasti.",
        "Siapa yang mengambil bukumu?",
        "Jangan lupa berobat ke dokter.",
        "Rumah Pak Dula sangat bersih.",
        "Ambillah ke sini buku itu.",
        # Probe leksikon frasa pendek (kasus yang pernah salah: matahari, air mata)
        "matahari",
        "air mata",
        "mata air",
        "pengantin",
        "siang",
        "hujan",
        "Matahari sudah terbit",
    ]
    diagnostic_prompts_gor_to_id = [
        "Wa:tiya loona:o ode pa:tali ola:ngo.",
        "Timi-timiidu dumodupo pa:ngo musi aatalo.",
        "Yilongola yio di:la losikola?",
        "Ngoolo haraga lo buku botiya?",
        "Tiyo hulo-huloo to bele.",
        "Lapatao mongaato yio mao polihulo.",
    ]
    cached_q1 = eval_store.get("qualitative_id_to_gor") if eval_store else None
    if isinstance(cached_q1, list) and len(cached_q1) == len(diagnostic_prompts_id_to_gor):
        results["qualitative_id_to_gor"] = cached_q1
    else:
        diag_preds_gor = translate_batch(diagnostic_prompts_id_to_gor, "ind_Latn", "gor_Latn", model, tokenizer)
        results["qualitative_id_to_gor"] = [
            {"indonesia": s, "gorontalo_pred": p}
            for s, p in zip(diagnostic_prompts_id_to_gor, diag_preds_gor)
        ]
        if eval_store:
            eval_store.put("qualitative_id_to_gor", results["qualitative_id_to_gor"])

    cached_q2 = eval_store.get("qualitative_gor_to_id") if eval_store else None
    if isinstance(cached_q2, list) and len(cached_q2) == len(diagnostic_prompts_gor_to_id):
        results["qualitative_gor_to_id"] = cached_q2
    else:
        diag_preds_id = translate_batch(diagnostic_prompts_gor_to_id, "gor_Latn", "ind_Latn", model, tokenizer)
        results["qualitative_gor_to_id"] = [
            {"gorontalo": s, "indonesia_pred": p}
            for s, p in zip(diagnostic_prompts_gor_to_id, diag_preds_id)
        ]
        if eval_store:
            eval_store.put("qualitative_gor_to_id", results["qualitative_gor_to_id"])
    return results


def main():
    t_start = time.time()
    force_retrain = os.environ.get("FORCE_RETRAIN", "0").lower() in {"1", "true", "yes"}
    resume_eval_only = (not force_retrain) and is_training_already_completed(FINAL_MODEL_DIR, is_lora=False)
    save_steps = int(os.environ.get("NLLB_SAVE_STEPS", "500"))

    df_test = pd.read_parquet(DATA_DIR / "seq2seq_test.parquet")

    if resume_eval_only:
        print(f"Model NLLB-200 sudah selesai dilatih di {FINAL_MODEL_DIR}; melanjutkan fase evaluasi test set...")
        tokenizer = AutoTokenizer.from_pretrained(FINAL_MODEL_DIR)
        model = AutoModelForSeq2SeqLM.from_pretrained(FINAL_MODEL_DIR, dtype=torch.bfloat16).to("cuda")
        completed_meta = json.loads((FINAL_MODEL_DIR / "training_completed.json").read_text(encoding="utf-8"))
        n_train_after = int(completed_meta.get("train_samples", 0))
        n_train_original = int(completed_meta.get("train_samples_before_oversampling", 0))
        n_val_samples = int(completed_meta.get("val_samples", 0))
        final_train_loss = float(completed_meta.get("train_loss", 0.0))
        base_bleu_id_gor = float(completed_meta.get("baseline_id_to_gor_bleu", 0.0))
        base_chrf_id_gor = float(completed_meta.get("baseline_id_to_gor_chrf", 0.0))
        base_bleu_gor_id = float(completed_meta.get("baseline_gor_to_id_bleu", 0.0))
        base_chrf_gor_id = float(completed_meta.get("baseline_gor_to_id_chrf", 0.0))
    else:
        print(f"Memuat tokenizer & base model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

        register_gorontalo_token(tokenizer, model)
        model.to("cuda")

        print("Memuat dataset Seq2Seq dari:", DATA_DIR)
        df_train = pd.read_parquet(DATA_DIR / "seq2seq_train.parquet")
        df_val = pd.read_parquet(DATA_DIR / "seq2seq_val.parquet")
        n_train_original = len(df_train)
        n_val_samples = len(df_val)
        df_train = upweight_short_lexicon(df_train, SHORT_LEX_WEIGHT)
        n_train_after = len(df_train)

        train_ds = fast_tokenize_dataframe(df_train, tokenizer)
        val_ds = fast_tokenize_dataframe(df_val, tokenizer)

        latest_ckpt = find_latest_valid_checkpoint(LOCAL_CKPT_DIR, is_lora=False)

        # Evaluasi Zero-Shot sebelum fine-tuning sebagai baseline (skip jika sedang resume dari checkpoint)
        base_bleu_id_gor = base_chrf_id_gor = base_bleu_gor_id = base_chrf_gor_id = 0.0
        if latest_ckpt is None:
            print("\n--- Evaluasi Baseline (Zero-Shot NLLB-200 Sebelum Fine-Tuning) ---")
            sub_id_gor = df_test[(df_test["src_lang"] == "ind_Latn") & (df_test["type"] == "parallel_sentence")].iloc[:200]
            sub_gor_id = df_test[(df_test["src_lang"] == "gor_Latn") & (df_test["type"] == "parallel_sentence")].iloc[:200]
            base_pred_gor = translate_batch(sub_id_gor["src_text"].tolist(), "ind_Latn", "gor_Latn", model, tokenizer)
            base_pred_id = translate_batch(sub_gor_id["src_text"].tolist(), "gor_Latn", "ind_Latn", model, tokenizer)
            base_bleu_id_gor = sacrebleu.corpus_bleu(base_pred_gor, [sub_id_gor["tgt_text"].tolist()]).score
            base_chrf_id_gor = sacrebleu.corpus_chrf(base_pred_gor, [sub_id_gor["tgt_text"].tolist()], word_order=2).score
            base_bleu_gor_id = sacrebleu.corpus_bleu(base_pred_id, [sub_gor_id["tgt_text"].tolist()]).score
            base_chrf_gor_id = sacrebleu.corpus_chrf(base_pred_id, [sub_gor_id["tgt_text"].tolist()], word_order=2).score
            print(f"Baseline ID->GOR: SacreBLEU={base_bleu_id_gor:.2f}, chrF++={base_chrf_id_gor:.2f}")
            print(f"Baseline GOR->ID: SacreBLEU={base_bleu_gor_id:.2f}, chrF++={base_chrf_gor_id:.2f}")
        else:
            print(f"\n--- Resume terdeteksi dari checkpoint valid: {latest_ckpt} (lewati zero-shot baseline) ---")

        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            padding=True,
            pad_to_multiple_of=8,
        )

        train_bs = int(os.environ.get("NLLB_BATCH_SIZE", "64"))
        grad_accum = int(os.environ.get("NLLB_GRAD_ACCUM", "1"))

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(LOCAL_CKPT_DIR),
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=train_bs,
            gradient_accumulation_steps=grad_accum,
            learning_rate=1.5e-4,
            weight_decay=0.01,
            warmup_steps=400,
            lr_scheduler_type="cosine",
            num_train_epochs=NUM_EPOCHS,
            bf16=True,
            tf32=True,
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=save_steps,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,  # cegah overfitting pada epoch tambahan
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=2,
            report_to="none",
            seed=SEED,
        )

        progress_cb = ProgressAndGracefulStopCallback(BASE_DIR / "logs" / "nllb_progress.json", t_start)
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[progress_cb],
        )

        print(f"\n=== Memulai Full Fine-Tuning NLLB-200-600M (BF16 di A100-80GB, bs={train_bs}x{grad_accum}, save_steps={save_steps}) ===")
        try:
            train_result = trainer.train(resume_from_checkpoint=latest_ckpt)
        finally:
            progress_cb.restore_signal_handlers()

        if progress_cb.stop_requested:
            print("Pelatihan NLLB-200 dihentikan aman karena sinyal sistem; checkpoint terakhir dipertahankan di:", LOCAL_CKPT_DIR)
            return

        final_train_loss = round(train_result.training_loss, 4)
        print(f"Pelatihan selesai dalam {(time.time() - t_start)/60:.2f} menit. Train Loss: {final_train_loss:.4f}")

        # Simpan model final ke Google Drive
        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print("Menyimpan model dan tokenizer ke:", FINAL_MODEL_DIR)
        model.save_pretrained(FINAL_MODEL_DIR)
        tokenizer.save_pretrained(FINAL_MODEL_DIR)
        mark_training_completed(
            FINAL_MODEL_DIR,
            {
                "model_name": MODEL_NAME,
                "train_samples": n_train_after,
                "train_samples_before_oversampling": n_train_original,
                "val_samples": n_val_samples,
                "epochs": NUM_EPOCHS,
                "train_loss": final_train_loss,
                "baseline_id_to_gor_bleu": round(base_bleu_id_gor, 2),
                "baseline_id_to_gor_chrf": round(base_chrf_id_gor, 2),
                "baseline_gor_to_id_bleu": round(base_bleu_gor_id, 2),
                "baseline_gor_to_id_chrf": round(base_chrf_gor_id, 2),
            },
        )

    # Evaluasi pada Test Set dengan cache parsial per arah
    eval_store = PartialEvalStore(BASE_DIR / "logs" / "nllb_eval_partial.json")
    eval_results = evaluate_on_test(model, tokenizer, df_test, eval_store=eval_store)
    eval_results["baseline_zero_shot"] = {
        "id_to_gor": {"sacrebleu": round(base_bleu_id_gor, 2), "chrf_plusplus": round(base_chrf_id_gor, 2)},
        "gor_to_id": {"sacrebleu": round(base_bleu_gor_id, 2), "chrf_plusplus": round(base_chrf_gor_id, 2)},
    }
    eval_results["training_summary"] = {
        "model_name": MODEL_NAME,
        "saved_path": str(FINAL_MODEL_DIR),
        "train_samples": n_train_after,
        "train_samples_before_oversampling": n_train_original,
        "short_lexicon_weight": SHORT_LEX_WEIGHT,
        "val_samples": n_val_samples,
        "test_samples": len(df_test),
        "epochs": NUM_EPOCHS,
        "train_loss": final_train_loss,
        "elapsed_minutes": round((time.time() - t_start) / 60, 2),
    }

    eval_out_path = DATA_DIR / EVAL_JSON_NAME
    with open(eval_out_path, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    print("\nLaporan evaluasi lengkap disimpan ke:", eval_out_path)

    shutil.copy2(eval_out_path, FINAL_MODEL_DIR / "evaluation_results.json")
    print("Salinan evaluasi disimpan ke:", FINAL_MODEL_DIR / "evaluation_results.json")
    eval_store.clear()

    # Bersihkan ckpt sementara di Drive agar hemat ruang
    if LOCAL_CKPT_DIR.exists():
        shutil.rmtree(LOCAL_CKPT_DIR, ignore_errors=True)
        print("Checkpoint sementara dibersihkan.")


if __name__ == "__main__":
    main()
