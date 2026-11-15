---
language:
- id
- gor
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: Qwen/Qwen2.5-7B-Instruct
tags:
- base_model:adapter:Qwen/Qwen2.5-7B-Instruct
- lora
- transformers
- gorontalo
- bahasa-gorontalo
- hulontalo
- indonesian
- panita-research-gorontalo
- mansoer-pateda
metrics:
- sacrebleu
- chrf
model-index:
- name: qwen2.5-7b-gorontalo-lora
  results:
  - task:
      type: translation
      name: Translation (Indonesia -> Gorontalo)
    dataset:
      name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
      type: panitagorontalo/gorontalo-indonesia-parallel
    metrics:
    - type: sacrebleu
      value: 32.86
      name: SacreBLEU (ID -> GOR Clean Test v6)
    - type: chrf
      value: 51.60
      name: chrF++ (ID -> GOR Clean Test v6)
  - task:
      type: translation
      name: Translation (Gorontalo -> Indonesia)
    dataset:
      name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
      type: panitagorontalo/gorontalo-indonesia-parallel
    metrics:
    - type: sacrebleu
      value: 25.27
      name: SacreBLEU (GOR 2001 -> ID Clean Test v6)
    - type: chrf
      value: 50.50
      name: chrF++ (GOR 2001 -> ID Clean Test v6)
  - task:
      type: translation
      name: Translation (Gorontalo 1987 -> Indonesia)
    dataset:
      name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
      type: panitagorontalo/gorontalo-indonesia-parallel
    metrics:
    - type: sacrebleu
      value: 27.68
      name: SacreBLEU (GOR 1987 -> ID Clean Test v6)
    - type: chrf
      value: 52.77
      name: chrF++ (GOR 1987 -> ID Clean Test v6)
---

# `panitagorontalo/qwen2.5-7b-gorontalo-lora` (v6)
**Model Bahasa Besar (LLM) Instruksi, Penerjemah Dwibahasa, & Penalaran Morfologi Bahasa Indonesia $\leftrightarrow$ Bahasa Gorontalo**

[![Hugging Face Space](https://img.shields.io/badge/🦅_Hugging_Face_Space-Penerjemah_&_Kamus_Gorontalo-indigo)](https://huggingface.co/spaces/panitagorontalo/penerjemah-bahasa-gorontalo)
[![Notebook Tutorial Colab](https://img.shields.io/badge/📓_Google_Colab-Unduh_&_Jalankan_Notebook_(.ipynb)-orange)](https://huggingface.co/panitagorontalo/qwen2.5-7b-gorontalo-lora/blob/main/Tutorial_Penerjemah_AI_Bahasa_Gorontalo_Panita_Research.ipynb)
[![Paper Teknis v6 ID](https://img.shields.io/badge/📄_Paper_Teknis_v6-Bahasa_Indonesia_(PDF)-blue)](https://huggingface.co/panitagorontalo/qwen2.5-7b-gorontalo-lora/blob/main/paper/paper_id.pdf)
[![Technical Paper v6 EN](https://img.shields.io/badge/📄_Technical_Paper_v6-English_(PDF)-green)](https://huggingface.co/panitagorontalo/qwen2.5-7b-gorontalo-lora/blob/main/paper/paper_en.pdf)

Dikembangkan oleh **Panita Research Gorontalo** sebagai bagian dari riset pelestarian digital dan pemrosesan bahasa alami (*Natural Language Processing*) untuk bahasa daerah **Bahasa Gorontalo (*Bahasa Hulontalo*)**. Model **v6** ini merupakan adapter **LoRA 16-bit (`bfloat16`, `r=32`, `lora_alpha=64`)** di atas **`Qwen/Qwen2.5-7B-Instruct`** yang dilatih pada **130.499 sampel instruksi multi-tugas (*Multi-Task SFT v6*)** bersumber dari dua kamus utama karya **Prof. Dr. H. Mansoer Pateda** yang telah diaudit & direkonsiliasi:
1. ***Kamus Bahasa Gorontalo–Indonesia*** (Pusat Bahasa Departemen Pendidikan Nasional & Balai Pustaka, 2001 — 302 halaman leksikon + pengantar).
2. ***Kamus Bahasa Indonesia–Gorontalo*** (Pusat Pembinaan dan Pengembangan Bahasa, 1987 — 489 halaman leksikon + pengantar).

---

## 1. Keunggulan & Empat Siasat Linguistik Prof. Dr. H. Mansoer Pateda (v6)

Karena jumlah akar kata (*morfem asal*) Bahasa Gorontalo lebih sedikit dibandingkan kosakata Bahasa Indonesia modern (KBBI), **Panita Research Gorontalo** menggabungkan penguasaan Bahasa Indonesia pada *base model* `Qwen/Qwen2.5-7B-Instruct` dengan **4 Siasat Linguistik & Augmentasi Morfologi (Korpus v6)**:

1. **Siasat A — Mesin Perakitan Imbuhan & Morfofonemik Produktif (*Morphological Engine*)**:
   - Dilatih pada **9.302 sampel Siasat A** (`5.461` `morphology_id_to_gor` + `3.808` `morphology_derivation` + `33` `grammar_rule_gorontalo`) mencakup 31 prefiks berpasangan/tak berpasangan, infiks (`-um-`, `-il-`, `-im-`), sufiks (`-lo`, `-a`, `-i`), peluluhan konsonan awal (`/t/->/l/, /s/->/n/, /p/->/m/, /k/->/ng/`), dan pergeseran vokal akhir `-o -> -a`.
2. **Siasat B — Kaidah Fonologisasi Kata Serapan Vokalis (*Loanword Phonologization Rule* `BI -> BG`)**:
   - Dilatih pada **1.556 tugas adaptasi kata serapan** agar kosakata Indonesia modern berakhiran konsonan diadaptasi menjadi berakhiran vokal terbuka (`-i, -u, -o, -a`) serta mampu menerima imbuhan produktif (mis. *adil* $\rightarrow$ `a:dili` $\rightarrow$ `moa:dili / a:diliyolo`).
3. **Siasat C — Dekomposisi Semantik & Parafrase Analitis dengan Morfem Penunjuk Arah (`MPA`)**:
   - Dilatih pada **699 sampel analitis MPA** untuk memetakan verba berarah Bahasa Indonesia ke kata dasar + **`mayi`**, **`mao`**, **`mola`**, dan **`mota`**.
4. **Siasat D — Jembatan Sinonim KBBI & Tingkat Tutur Biasa vs Hormat**:
   - Dilatih pada **9.135 perluasan variasi sinonim KBBI** pada sisi input Bahasa Indonesia serta pembedaan register **Biasa (`wau / yio`)** vs **Hormat (`wa:tiya / ito`)**.

---

## 2. Distribusi Tugas Pelatihan (*Multi-Task SFT Dataset v6* — 130.499 Sampel `Train`)

| Nama Tugas (*Task*) | Jumlah Sampel (`Train`) | Deskripsi |
| :--- | :---: | :--- |
| `lexicon_id_to_gor` | `28.727` | Pencarian kosakata/istilah Bahasa Indonesia $\rightarrow$ Bahasa Gorontalo |
| `lexicon_gor_to_id` | `28.727` | Pencarian arti kosakata/istilah Bahasa Gorontalo $\rightarrow$ Bahasa Indonesia |
| `translate_id_to_gor` | `20.479` | Penerjemahan kalimat Bahasa Indonesia $\rightarrow$ Bahasa Gorontalo |
| `translate_gor_to_id` | `20.479` | Penerjemahan kalimat Bahasa Gorontalo (Ejaan 2001) $\rightarrow$ Bahasa Indonesia |
| `translate_gor1987_to_id` | `11.395` | Penerjemahan kalimat Bahasa Gorontalo (Ejaan 1987) $\rightarrow$ Bahasa Indonesia |
| `translate_id_to_gor_synonym` | `9.135` | Jembatan sinonim KBBI & register tutur Biasa vs Hormat (`Siasat D`) |
| `morphology_id_to_gor` | `5.461` | Penerjemahan kata berimbuhan Indonesia ke derivasi Gorontalo (`Siasat A`) |
| `morphology_derivation` | `3.808` | Pembentukan kata turunan Bahasa Gorontalo dari akar kata (`Siasat A`) |
| `loanword_phonologization` | `1.556` | Adaptasi fonotaktik vokalis kata serapan Indonesia $\rightarrow$ Gorontalo (`Siasat B`) |
| `analytical_paraphrase_mpa` | `699` | Penerjemahan analitis dengan Morfem Penunjuk Arah `mayi/mao/mola/mota` (`Siasat C`) |
| `grammar_rule_gorontalo` | `33` | Kaidah tata bahasa, 5 kala, & kata bantu bilangan Pateda |
| **Total `Train` / `Val` / `Test`** | **`130.499`** / **`2.898`** / **`2.890`** | Pembagian terstratifikasi dengan **0% kebocoran data (*zero leakage*)** |

---

## 3. Hasil Evaluasi Kuantitatif & Kualitatif (Korpus v6)

### A. Skor `SacreBLEU` & `chrF++` pada `Test Set v6`

| Tugas Penerjemahan (`Test Set v6`) | Clean (`SacreBLEU` / `chrF++`) | All (`SacreBLEU` / `chrF++`) | Baseline v5 Clean | Baseline v4 Clean |
| :--- | :---: | :---: | :---: | :---: |
| **Indonesia $\rightarrow$ Gorontalo (`translate_id_to_gor`)** | **`32.86` / `51.60`** ($n=1.083$) | **`32.28` / `50.99`** ($n=1.136$) | `32.07 / 50.21` | `26.43 / 44.23` |
| **Gorontalo (Ejaan 2001) $\rightarrow$ Indonesia (`translate_gor_to_id`)** | **`25.27` / `50.50`** ($n=1.083$) | **`24.36` / `49.14`** ($n=1.136$) | `22.94 / 47.08` | `18.95 / 42.67` |
| **Gorontalo (Ejaan 1987) $\rightarrow$ Indonesia (`translate_gor1987_to_id`)** | **`27.68` / `52.77`** ($n=600$) | **`27.21` / `52.17`** ($n=618$) | `24.19 / 48.85` | `22.48 / 47.00` |

### B. Contoh Kemampuan Multi-Tugas Model (Terverifikasi `evaluasi_qwen_lora_gorontalo.json` v6)

1. **Penerjemahan Kalimat Terikat Kala & Tingkat Tutur**:
   * **Input (`ID -> GOR`)**: *"Saya pergi ke pasar kemarin."*
   * **Output**: `Wa:tiya lo:nao ode pa:tali ola:ngo` *(pronomina hormat `Wa:tiya` + kala lampau `lo:nao`)*.
   * **Input (`GOR -> ID`)**: *"Ti Pa:pa lopopandango buatulo alihu batade di:la mowali moliba:ya."*
   * **Output**: `Ayah memanjangkan tali ijuk agar kambing tidak bisa bersembunyi`
2. **Perakitan Morfologi & Peluluhan Konsonan (`Siasat A`)**:
   * **Instruksi**: *"Bentuklah kata turunan bahasa Gorontalo dari akar kata 'tuladu' beserta artinya dalam bahasa Indonesia."*
   * **Input**: `Akar kata: tuladu -> Makna sasaran: menulis`
   * **Output**: `Kata turunan: moluladu (Artinya: menulis) | Contoh: wau di:la mowali moluladu = aku tidak bisa menulis`
3. **Adaptasi Fonologisasi Kata Serapan Vokalis (`Siasat B`)**:
   * **Instruksi**: *"Adaptasikan kata bahasa Indonesia berikut ke dalam bahasa Gorontalo sesuai kaidah fonotaktik vokalis Prof. Mansoer Pateda beserta contoh imbuhannya."*
   * **Input**: `adil`
   * **Output**: `Bentuk serapan Gorontalo: a:dili | Contoh turunan: moa:dili / a:diliyolo | Kaidah: Adaptasi fonotaktik vokalis Bahasa Gorontalo dari kata 'adil' menjadi 'a:dili' (berakhiran vokal terbuka '-i').`

---

## 4. Kompatibilitas Hugging Face Space & Tutorial Menjalankan Model

> **Kompatibilitas:**
> - **HF Space Free Tier (`CPU Basic`):** Tidak memadai (7,6B butuh ~15,2 GB `bf16` / GPU CUDA untuk 4-bit).
> - **HF Space GPU / Google Colab T4 (15 GB VRAM):** Berjalan lancar dalam **4-bit NF4 (`~5,5 GB` VRAM)** maupun `bfloat16` di L4 / A10G / A100 ($\ge 24$ GB VRAM).
> - **Catatan Paragraf:** Karena dilatih pada pasangan kalimat (`MAX_SEQ_LEN=256`), pecah teks paragraf per kalimat terlebih dahulu (`re.split(r"(?<=[.!?])\s+", ...)`) dan aktifkan `no_repeat_ngram_size=4`.

```python
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
LORA_ADAPTER = "panitagorontalo/qwen2.5-7b-gorontalo-lora"

SYSTEM_PROMPT = (
    "Anda adalah pakar bahasa dan penerjemah dwibahasa Bahasa Indonesia <-> Bahasa Gorontalo "
    "yang menguasai tata bahasa, 5 sistem kala, morfofonemik, morfem penunjuk arah (mayi, mao, mola, mota), "
    "dan kaidah leksikografi Prof. Dr. H. Mansoer Pateda."
)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER)
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

def ask_gorontalo_llm(instruction: str, input_text: str, split_sentences: bool = True) -> str:
    def infer_one(seg: str) -> str:
        prompt = (
            f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n\nInput: {seg}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            gen_ids = model.generate(
                **enc,
                max_new_tokens=192,
                do_sample=False,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        out_ids = gen_ids[0, enc["input_ids"].shape[1]:]
        return tokenizer.decode(out_ids, skip_special_tokens=True).strip()

    if not split_sentences:
        return infer_one(input_text)
    paras = [p.strip() for p in re.split(r"\n+", input_text.strip()) if p.strip()]
    out_paras = []
    for para in paras:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        parts = []
        for s in sents:
            pred = infer_one(s)
            if pred and not pred.endswith((".", "!", "?")):
                end_ch = s[-1] if s.endswith((".", "!", "?")) else "."
                if len(sents) > 1 or end_ch in ("?", "!"):
                    pred += end_ch
            parts.append(pred)
        out_paras.append(" ".join(parts))
    return "\n\n".join(out_paras)
```

---

## 5. Detail Konfigurasi Pelatihan (*Training Hyperparameters v6*)

* **Perangkat Keras**: 1× NVIDIA A100-SXM4-80GB GPU (`elapsed_minutes = 168.57`)
* **Base Model**: `Qwen/Qwen2.5-7B-Instruct`
* **Konfigurasi LoRA**: `r=32`, `lora_alpha=64`, `lora_dropout=0.05`, `bias="none"`, target modul: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` (**80.740.352 parameter terlatih / `1,0491%` dari `7.696.356.864` total parameter**)
* **Presisi & Batch**: `bfloat16` (`bf16=True`, `tf32=True`), `MAX_SEQ_LEN=256`, batch efektif `64` (`per_device_train_batch_size=32`, `gradient_accumulation_steps=2`), `1 epoch` (`2.040` langkah)
* **Learning Rate & Konvergensi**: `2e-4` (jadwal `cosine`, `warmup_steps=100`); `train_loss = 1.6235`, `eval_loss` turun konsisten dari `2.669` $\rightarrow$ **`1.433`** di akhir epoch.

---

## 6. Tim Pengembang & Keterbatasan Diketahui

Dikembangkan oleh **Panita Research Gorontalo** (Funco Tanipu, Richic Oct Sumual, Budiyanto, Thomas Jr — [https://panita.web.id](https://panita.web.id), [ai@panita.web.id](mailto:ai@panita.web.id)).

* **Keterbatasan Utama:** Dilatih pada tingkat kalimat tunggal (`MAX_SEQ_LEN=256`) dari dua kamus Pateda (1987 & 2001) + rekonsiliasi v6; pada pertanyaan definisi teori tata bahasa terbuka (mis. menjelaskan *"5 sistem kala"*), LLM masih dapat berhalusinasi sehingga fitur teori/kaidah sebaiknya merujuk ke kamus/JSON secara deterministik. Hasil terjemahan tetap perlu tinjauan penutur asli untuk penggunaan resmi.

---

## 7. Sitasi & Penghargaan (*Citation & Credits*)

```bibtex
@misc{panitagorontalo2026qwen,
  title        = {Qwen2.5-7B-Gorontalo-LoRA v6: Instruction-Tuned LLM for Indonesian-Gorontalo Translation and Morphological Reasoning},
  author       = {{Panita Research Gorontalo}},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/panitagorontalo/qwen2.5-7b-gorontalo-lora}}
}
```
### Framework versions

- PEFT 0.21.2
