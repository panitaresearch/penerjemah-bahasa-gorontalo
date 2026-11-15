---
language:
- id
- gor
license: apache-2.0
library_name: peft
pipeline_tag: text-generation
base_model: GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct
tags:
- base_model:adapter:GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct
- lora
- gemma3
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
- name: gemma3-27b-gotoai-gorontalo-lora
  results:
  - task:
      type: translation
      name: Translation (Indonesia -> Gorontalo)
    dataset:
      name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
      type: panitagorontalo/gorontalo-indonesia-parallel
    metrics:
    - type: sacrebleu
      value: 34.45
      name: SacreBLEU (ID -> GOR Clean Test v6)
    - type: chrf
      value: 53.98
      name: chrF++ (ID -> GOR Clean Test v6)
  - task:
      type: translation
      name: Translation (Gorontalo -> Indonesia)
    dataset:
      name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
      type: panitagorontalo/gorontalo-indonesia-parallel
    metrics:
    - type: sacrebleu
      value: 29.20
      name: SacreBLEU (GOR 2001 -> ID Clean Test v6)
    - type: chrf
      value: 53.19
      name: chrF++ (GOR 2001 -> ID Clean Test v6)
  - task:
      type: translation
      name: Translation (Gorontalo 1987 -> Indonesia)
    dataset:
      name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
      type: panitagorontalo/gorontalo-indonesia-parallel
    metrics:
    - type: sacrebleu
      value: 32.06
      name: SacreBLEU (GOR 1987 -> ID Clean Test v6)
    - type: chrf
      value: 56.42
      name: chrF++ (GOR 1987 -> ID Clean Test v6)
---

# `panitagorontalo/gemma3-27b-gotoai-gorontalo-lora` (v6)
**Model Bahasa Besar 27B (`GoTo-AI v3 Gemma 3`) untuk Penerjemahan Dwibahasa, Konteks Antar-Kalimat, & Penalaran Morfologi Bahasa Indonesia $\leftrightarrow$ Bahasa Gorontalo**

[![Hugging Face Space](https://img.shields.io/badge/🦅_Hugging_Face_Space-Penerjemah_&_Kamus_Gorontalo-indigo)](https://huggingface.co/spaces/panitagorontalo/penerjemah-bahasa-gorontalo)
[![Notebook Tutorial Colab](https://img.shields.io/badge/📓_Google_Colab-Unduh_&_Jalankan_Notebook_(.ipynb)-orange)](https://huggingface.co/panitagorontalo/gemma3-27b-gotoai-gorontalo-lora/blob/main/Tutorial_Penerjemah_AI_Bahasa_Gorontalo_Panita_Research.ipynb)
[![Paper Teknis v6 ID](https://img.shields.io/badge/📄_Paper_Teknis_v6-Bahasa_Indonesia_(PDF)-blue)](https://huggingface.co/panitagorontalo/gemma3-27b-gotoai-gorontalo-lora/blob/main/paper/paper_id.pdf)
[![Technical Paper v6 EN](https://img.shields.io/badge/📄_Technical_Paper_v6-English_(PDF)-green)](https://huggingface.co/panitagorontalo/gemma3-27b-gotoai-gorontalo-lora/blob/main/paper/paper_en.pdf)

Dikembangkan oleh **Panita Research Gorontalo** di atas *base model* **`GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct`** (model Gemma 3 27B hasil *Continued Pretraining* berbahasa Indonesia & bahasa daerah Nusantara) menggunakan **LoRA `bfloat16` (`r=16`, `lora_alpha=32`) khusus pada layer teks** (`model.language_model.layers.*`, membekukan seluruh 439 tensor `vision_tower` & proyektor multimodal) dengan **130.499 sampel instruksi multi-tugas (Korpus Terpadu v6)** dari dua kamus karya **Prof. Dr. H. Mansoer Pateda** (1987 & 2001).

---

## 1. Performa Terbaik di Kelas LLM & Keunggulan Arsitektur v6

Pada pengujian benchmark korpus **v6**, **GoTo Gemma 3 27B LoRA** menjadi **model LLM terkuat** di seluruh arah penerjemahan (melampaui `Qwen2.5-7B LoRA` v6 di seluruh metrik `SacreBLEU` dan `chrF++`):

| Tugas Penerjemahan (`Test Set v6`) | **Gemma 3 27B LoRA Clean** | Gemma 3 27B LoRA All | Qwen2.5-7B LoRA Clean | NLLB-200-600M v6 (All) |
| :--- | :---: | :---: | :---: | :---: |
| **Indonesia $\rightarrow$ Gorontalo (`ID -> GOR`)** | **`34.45` / `53.98`** ($n=1.083$) | `33.75` / `53.27` ($n=1.136$) | `32.86` / `51.60` | `40.59` / `57.98` |
| **Gorontalo (2001) $\rightarrow$ Indonesia** | **`29.20` / `53.19`** ($n=1.083$) | `28.15` / `51.82` ($n=1.136$) | `25.27` / `50.50` | `32.57` / `56.80` |
| **Gorontalo (1987) $\rightarrow$ Indonesia** | **`32.06` / `56.42`** ($n=600$) | `31.37` / `55.72` ($n=618$) | `27.68` / `52.77` | `35.60` / `59.29` |

**Keunggulan Teramati pada Uji Kualitatif & Paragraf:**
1. **Penjelasan Morfologi & Fonotaktik Vokalis Presisi (`Siasat A & B`)**: Mampu menurunkan verba berpeluluhan konsonan `/t/->/l/` (`tuladu` $\rightarrow$ `moluladu`, contoh: `wau di:la moluladu`) serta menjelaskan adaptasi kata serapan konsonan akhir (`adil` $\rightarrow$ `a:dili`, `moa:dili / a:diliyolo`).
2. **Pemeliharaan Konteks Paragraf Pendek**: Di antara ketiga model v6, Gemma 3 27B paling mampu menerjemahkan paragraf 4 kalimat secara utuh pada kedua arah (`ID->GOR` dan `GOR->ID`) serta menjaga koherensi makna antar-kalimat (`GOR -> ID`).

---

## 2. Contoh Keluaran Multi-Tugas (Terverifikasi `evaluasi_gemma3_27b_lora_gorontalo.json` v6)

* **Terjemahan Hormat & Kala Lampau (`ID -> GOR`)**:
  * `"Saya pergi ke pasar kemarin."` $\rightarrow$ `Wa:tiya lo:nao ode pa:tali ola:ngo`
  * `"Tiap pagi halaman rumah harus disapu."` $\rightarrow$ `Timiidu dumodupo pa:ngo bele musi aatolo`
* **Terjemahan `GOR -> ID`**:
  * `"Wa:tiya lo:nao ode pa:tali ola:ngo."` $\rightarrow$ `Saya pergi ke pasar kemarin`
* **Siasat A (`tuladu` $\rightarrow$ menulis)**:
  * `Kata turunan: moluladu (Artinya: menulis) | Contoh: wau di:la moluladu = aku tidak menulis`
* **Siasat B (`adil`)**:
  * `Bentuk serapan Gorontalo: a:dili | Contoh turunan: moa:dili / a:diliyolo | Kaidah: Kata Indonesia 'adil' berakhir dengan konsonan 'l', sehingga diadaptasi ke sifat vokalis Bahasa Gorontalo dengan akhiran vokal '-i' menjadi 'a:dili'.`

---

## 3. Cara Menjalankan Inferensi (A100 `bf16` atau L4 24 GB 4-bit QLoRA)

> **Persyaratan Perangkat Keras:** *Base model* 27,5B (`51,10 GiB` dalam `bfloat16`) membutuhkan **A100-80GB / H100** untuk presisi `bfloat16`, atau GPU $\ge 24$ GB VRAM (mis. **NVIDIA L4 / RTX 3090 / 4090 / A10G**) menggunakan kuantisasi **4-bit NF4 (`bitsandbytes`)**. Tidak dapat dijalankan di T4 15 GB atau HF Space CPU Basic.

```python
import re
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE_MODEL = "GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct"
LORA_ADAPTER = "panitagorontalo/gemma3-27b-gotoai-gorontalo-lora"

SYSTEM_PROMPT = (
    "Anda adalah pakar bahasa dan penerjemah dwibahasa Bahasa Indonesia <-> Bahasa Gorontalo "
    "yang menguasai tata bahasa, 5 sistem kala, morfofonemik, morfem penunjuk arah (mayi, mao, mola, mota), "
    "dan kaidah leksikografi Prof. Dr. H. Mansoer Pateda."
)

# Gunakan BitsAndBytesConfig untuk GPU 24 GB (L4/A10G), atau dtype=torch.bfloat16 di A100-80GB
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(LORA_ADAPTER)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,  # Ganti dengan dtype=torch.bfloat16 di A100-80GB
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, LORA_ADAPTER)
model.eval()

eos_ids = [
    i for i in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    if isinstance(i, int) and i >= 0
]

def ask_gemma3_gorontalo(instruction: str, input_text: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"{instruction}\n\nInput: {input_text}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        gen_ids = model.generate(
            **enc,
            max_new_tokens=192,
            do_sample=False,
            top_k=None,
            top_p=None,
            no_repeat_ngram_size=4,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_ids or tokenizer.eos_token_id,
        )
    out_ids = gen_ids[0, enc["input_ids"].shape[1]:]
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()
```

---

## 4. Detail Konfigurasi Pelatihan (*Training Hyperparameters v6*)

* **Perangkat Keras**: 1× NVIDIA A100-SXM4-80GB GPU (`~64,26 GiB` VRAM puncak, `elapsed_minutes = 545.02`)
* **Base Model**: `GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct`
* **Konfigurasi LoRA (Text-Only)**: `r=16`, `lora_alpha=32`, `lora_dropout=0.05`, `bias="none"`, target regex: `model\.language_model\.layers\.\d+\.(?:self_attn\.(?:q|k|v|o)_proj|mlp\.(?:gate|up|down)_proj)` (**113.516.544 parameter terlatih / `0,4121%` dari `27.545.923.184` total parameter**)
* **Presisi & Batch**: `bfloat16` (`bf16=True`, `tf32=True`, `attn_implementation="sdpa"`, `gradient_checkpointing=True`), `MAX_SEQ_LEN=256`, batch efektif `64` (`per_device_train_batch_size=8`, `gradient_accumulation_steps=8`), `1 epoch` (`2.040` langkah, `130.499` sampel latih / `2.898` validasi)
* **Learning Rate & Konvergensi**: `1e-4` (jadwal `cosine`, `warmup_steps=100`); `train_loss = 1.6540`, `eval_loss` turun konsisten dari `2.849` $\rightarrow$ **`1.393`** di akhir epoch.

---

## 5. Tim Pengembang & Keterbatasan Diketahui

Dikembangkan oleh **Panita Research Gorontalo** (Funco Tanipu, Richic Oct Sumual, Budiyanto, Thomas Jr — [https://panita.web.id](https://panita.web.id), [ai@panita.web.id](mailto:ai@panita.web.id)).

* **Keterbatasan Utama:** Dilatih pada pasangan tingkat kalimat tunggal (`MAX_SEQ_LEN=256`) dari korpus v6 (Kamus Pateda 1987 & 2001); untuk teks panjang (>3–4 kalimat), disarankan memecah per kalimat atau blok 2 kalimat serta memakai `no_repeat_ngram_size=4`. Pada pertanyaan definisi teori tata bahasa terbuka (*"5 sistem kala"*), keluaran masih bisa kurang akurat dan sebaiknya merujuk ke korpus/JSON secara langsung. Hasil terjemahan perlu ditinjau penutur asli untuk dokumen resmi.

---

## 6. Sitasi

```bibtex
@misc{panitagorontalo2026gemma3,
  title        = {Gemma3-27B-GoToAI-Gorontalo-LoRA v6: Indonesian-Gorontalo Bilingual Translation and Morphological Reasoning},
  author       = {{Panita Research Gorontalo}},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/panitagorontalo/gemma3-27b-gotoai-gorontalo-lora}}
}
```
### Framework versions

- PEFT 0.21.2
