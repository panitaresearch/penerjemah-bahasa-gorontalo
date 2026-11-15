#!/usr/bin/env python3
"""
push_all_to_hf.py
=================
Sinkronisasi Data Korpus v6, Notebook Colab, 3 Model (`nllb`, `qwen`, `gemma3`),
dan Hugging Face Space (`panitagorontalo`).

Pemakaian:
  python3 push_all_to_hf.py --sync-only   # Hanya siapkan aset lokal v6 tanpa upload ke HF
  HF_TOKEN=hf_... python3 push_all_to_hf.py  # Sinkronisasi lokal + upload ke HF Hub & Space
"""

import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

BASE_DIR = Path("/content/drive/MyDrive/REPO/richie/modelgorontalo")
DATA_DIR = BASE_DIR / "dataset_terpadu_indonesia_gorontalo"
NLLB_DIR = BASE_DIR / "models" / "nllb-200-gorontalo-bi-600M"
QWEN_DIR = BASE_DIR / "models" / "qwen2.5-7b-gorontalo-lora"
GEMMA3_DIR = BASE_DIR / "models" / "gemma3-27b-gotoai-gorontalo-lora"
SPACE_DIR = BASE_DIR / "hf_space_gorontalo"
NOTEBOOK_PATH = BASE_DIR / "Tutorial_Penerjemah_AI_Bahasa_Gorontalo_Panita_Research.ipynb"


def sync_local_v6_assets() -> None:
    """Sinkronkan data korpus v6 dan notebook tutorial ke Space serta folder model."""
    print("=== 1. Memperbarui Data Korpus v6 & Notebook pada Hugging Face Space & Model ===")
    space_data = SPACE_DIR / "data"
    space_data.mkdir(parents=True, exist_ok=True)
    if (DATA_DIR / "combined_lexicon_id_to_gor.parquet").exists():
        shutil.copy2(DATA_DIR / "combined_lexicon_id_to_gor.parquet", space_data / "lexicon.parquet")
    if (DATA_DIR / "combined_parallel_sentences.parquet").exists():
        shutil.copy2(DATA_DIR / "combined_parallel_sentences.parquet", space_data / "parallel_sentences.parquet")

    morph_path = BASE_DIR / "gorontaloindonesia/dataset/training_ready/train_morphology_grammar.parquet"
    morph_ig_path = BASE_DIR / "indonesiagorontalo/dataset/training_ready/train_morphology_grammar.parquet"
    if morph_path.exists():
        shutil.copy2(morph_path, space_data / "morphology.parquet")

    df_lex = pd.read_parquet(DATA_DIR / "combined_lexicon_id_to_gor.parquet")
    df_par = pd.read_parquet(DATA_DIR / "combined_parallel_sentences.parquet")
    lex_rows = [
        [
            str(r["indonesia"] or ""),
            str(r["gorontalo"] or ""),
            str(r["gorontalo_asli"] or ""),
            str(r["tipe"] or ""),
            str(r["sumber_kamus"] or ""),
        ]
        for _, r in df_lex.iterrows()
    ]
    par_rows = [
        [
            str(r["src_indonesia"] or ""),
            str(r["tgt_gorontalo"] or ""),
            str(r["sumber_kamus"] or ""),
        ]
        for _, r in df_par.iterrows()
    ]
    mor_rows = []
    if morph_path.exists():
        df_gi_m = pd.read_parquet(morph_path)
        for _, r in df_gi_m.iterrows():
            ex = f"{r['contoh_kalimat']} = {r['arti_kalimat']}" if str(r.get("contoh_kalimat") or "").strip() else ""
            mor_rows.append([str(r["akar_kata"] or ""), str(r["kata_turunan"] or ""), str(r["arti_turunan"] or ""), ex, "2001"])
    if morph_ig_path.exists():
        df_ig_m = pd.read_parquet(morph_ig_path)
        for _, r in df_ig_m.iterrows():
            ex = f"{r['contoh_gorontalo']} = {r['contoh_indonesia']}" if str(r.get("contoh_gorontalo") or "").strip() else ""
            mor_rows.append([
                str(r["padanan_akar_gorontalo"] or r["akar_indonesia"] or ""),
                str(r["padanan_turunan_gorontalo"] or ""),
                str(r["turunan_indonesia"] or ""),
                ex,
                "1987",
            ])

    corpus_json = {
        "meta": {
            "organization": "Panita Research Gorontalo",
            "website": "https://panita.web.id",
            "emails": ["ai@panita.web.id", "richie@panita.web.id", "funco@panita.web.id"],
            "team": ["Funco Tanipu", "Richic Oct Sumual", "Budiyanto", "Thomas Jr"],
            "version": "v6",
            "lexicon_count": len(lex_rows),
            "parallel_count": len(par_rows),
            "morphology_count": len(mor_rows),
            "models": [
                "panitagorontalo/nllb-200-gorontalo-bi-600M",
                "panitagorontalo/qwen2.5-7b-gorontalo-lora",
                "panitagorontalo/gemma3-27b-gotoai-gorontalo-lora",
            ],
        },
        "lexicon": lex_rows,
        "morphology": mor_rows,
        "parallel": par_rows,
    }
    with open(SPACE_DIR / "corpus_gorontalo.json", "w", encoding="utf-8") as f:
        json.dump(corpus_json, f, ensure_ascii=False)

    if NOTEBOOK_PATH.exists():
        for target_dir in [SPACE_DIR, NLLB_DIR, QWEN_DIR, GEMMA3_DIR]:
            if target_dir.exists():
                shutil.copy2(NOTEBOOK_PATH, target_dir / NOTEBOOK_PATH.name)

    paper_dir = BASE_DIR / "paper"
    paper_mapping = (
        ("paper_ieee_en.pdf", "paper_en.pdf"),
        ("paper_ieee_id.pdf", "paper_id.pdf"),
    )
    for target_dir in [SPACE_DIR, NLLB_DIR, QWEN_DIR, GEMMA3_DIR]:
        if not target_dir.exists():
            continue
        target_paper_dir = target_dir / "paper"
        target_paper_dir.mkdir(parents=True, exist_ok=True)
        for old_pdf in target_paper_dir.glob("paper_ieee_*.pdf"):
            old_pdf.unlink()
        for src_name, dst_name in paper_mapping:
            src_pdf = paper_dir / src_name
            if src_pdf.exists():
                shutil.copy2(src_pdf, target_paper_dir / dst_name)

    for p in SPACE_DIR.glob("**/__pycache__"):
        shutil.rmtree(p, ignore_errors=True)

    print(
        f"Data Space v6 berhasil disinkronkan: {len(lex_rows)} leksikon, "
        f"{len(par_rows)} paralel, {len(mor_rows)} morfologi."
    )


def upload_all_to_hf(api: HfApi, only: str = "all") -> None:
    """Unggah ketiga model v6 dan Hugging Face Space ke Hub."""
    ignore_list = ["__pycache__", "*.pyc", "*.tmp"]

    uploads = [
        (
            "2. NLLB-200 v6",
            NLLB_DIR,
            "panitagorontalo/nllb-200-gorontalo-bi-600M",
            "model",
            "Update NLLB-200 v6: korpus rekonsiliasi v6 (SacreBLEU 40.59 / chrF++ 57.98 ID->GOR)",
        ),
        (
            "3. Qwen2.5-7B LoRA v6",
            QWEN_DIR,
            "panitagorontalo/qwen2.5-7b-gorontalo-lora",
            "model",
            "Update Qwen2.5-7B LoRA v6: 130.499 SFT korpus v6 (SacreBLEU 32.86 clean / 32.28 all, chrF++ 51.60)",
        ),
        (
            "4. GoTo Gemma 3 27B LoRA v6",
            GEMMA3_DIR,
            "panitagorontalo/gemma3-27b-gotoai-gorontalo-lora",
            "model",
            "Release GoTo Gemma 3 27B LoRA v6: 130.499 SFT korpus v6 (SacreBLEU 34.45 clean ID->GOR, 29.20 clean GOR->ID)",
        ),
        (
            "5. Hugging Face Space v6",
            SPACE_DIR,
            "panitagorontalo/penerjemah-bahasa-gorontalo",
            "space",
            "Sync Space v6: korpus 32.039 leksikon / 29.546 paralel, dukungan paragraf, model Gemma 3 27B, dan paper teknis v6 (EN/ID)",
        ),
    ]

    if only == "space":
        uploads = [u for u in uploads if u[3] == "space"]
    elif only == "models":
        uploads = [u for u in uploads if u[3] == "model"]

    failures = []
    for label, folder, repo_id, repo_type, msg in uploads:
        print(f"\n=== {label} -> {repo_id} ===")
        try:
            if repo_type == "model":
                api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
            api.upload_folder(
                folder_path=str(folder),
                repo_id=repo_id,
                repo_type=repo_type,
                commit_message=msg,
                ignore_patterns=ignore_list,
                delete_patterns=["paper/paper_ieee_*.pdf"],
            )
            print(f"✓ Berhasil mengunggah {repo_id}!")
        except Exception as e:
            print(f"Error saat mengunggah {repo_id}: {e}")
            failures.append((repo_id, str(e)))

    if failures:
        summary = "; ".join(f"{repo}: {err}" for repo, err in failures)
        raise RuntimeError(f"Sebagian unggahan HF gagal -> {summary}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sinkronisasi aset v6 dan unggah ke Hugging Face Hub")
    parser.add_argument(
        "--sync-only",
        action="store_true",
        help="Hanya perbarui data Space, corpus_gorontalo.json, dan notebook di lokal tanpa push ke HF.",
    )
    parser.add_argument(
        "--only",
        choices=["all", "models", "space"],
        default="all",
        help="Pilih target unggahan (default: all).",
    )
    args = parser.parse_args()

    sync_local_v6_assets()
    if args.sync_only:
        print("\nMode `--sync-only` aktif: tidak ada unggahan yang dikirim ke Hugging Face Hub.")
        return

    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    try:
        api.whoami()
    except Exception as exc:
        raise RuntimeError(
            "Autentikasi Hugging Face tidak ditemukan. Set environment variable `HF_TOKEN` "
            "atau jalankan `hf auth login` terlebih dahulu."
        ) from exc

    upload_all_to_hf(api, only=args.only)
    print("\n=== Selesai 100%! Seluruh target repositori Hugging Face v6 telah diperbarui. ===")


if __name__ == "__main__":
    main()
