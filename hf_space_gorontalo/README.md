---
title: Penerjemah & Kamus AI Bahasa Gorontalo
emoji: 🦅
colorFrom: indigo
colorTo: blue
sdk: static
pinned: true
license: apache-2.0
models:
  - panitagorontalo/nllb-200-gorontalo-bi-600M
  - panitagorontalo/qwen2.5-7b-gorontalo-lora
  - panitagorontalo/gemma3-27b-gotoai-gorontalo-lora
short_description: Penerjemah AI & Kamus Indonesia-Gorontalo v6 (Panita)
---

# 🦅 Penerjemah AI & Kamus Digital Dwibahasa Bahasa Indonesia ↔ Bahasa Gorontalo (v6)
**Dikembangkan oleh [Panita Research Gorontalo](https://huggingface.co/panitagorontalo)**

Aplikasi web interaktif untuk penerjemahan Bahasa Indonesia ↔ Bahasa Gorontalo (*Bahasa Hulontalo*), pencarian **32.039 kosakata kamus dwibahasa**, analisis **9.777 derivasi morfologi (imbuhan & 5 sistem kala)**, pencarian **29.546 kalimat paralel**, serta konversi ejaan ortografi berdasarkan **korpus terpadu v6** (rekonsiliasi & audit kurasi dari dua kamus **Prof. Dr. H. Mansoer Pateda**):
1. *Kamus Bahasa Gorontalo–Indonesia* (Balai Pustaka, 2001)
2. *Kamus Bahasa Indonesia–Gorontalo* (Pusat Bahasa, 1987)

### Tim Pengembang (Panita Research Gorontalo)

Proyek penerjemah dan kamus digital Bahasa Indonesia ↔ Bahasa Gorontalo ini dikembangkan oleh tim **Panita Research Gorontalo**:

* Funco Tanipu
* Richic Oct Sumual
* Budiyanto
* Thomas Jr

**Kontak Panita Research Gorontalo:**

* Website: [https://panita.web.id](https://panita.web.id)
* Email: [ai@panita.web.id](mailto:ai@panita.web.id), [richie@panita.web.id](mailto:richie@panita.web.id), [funco@panita.web.id](mailto:funco@panita.web.id)

### Model AI (v6), Notebook Google Colab, & Kompatibilitas HF Space
* **Notebook Google Colab Siap Pakai (`.ipynb`)**: [Unduh `Tutorial_Penerjemah_AI_Bahasa_Gorontalo_Panita_Research.ipynb`](./Tutorial_Penerjemah_AI_Bahasa_Gorontalo_Panita_Research.ipynb) lalu buka di [Google Colab](https://colab.research.google.com/) (`File -> Upload notebook`) untuk menjalankan NLLB-200, Qwen2.5-7B LoRA (`T4 GPU`), maupun GoTo Gemma 3 27B LoRA (`L4/A100 GPU`) beserta antarmuka web Gradio (`share=True`).
* **Paper Teknis (Edisi v6, Dwi-Bahasa)**: [Versi Bahasa Indonesia (`paper/paper_id.pdf`)](./paper/paper_id.pdf) | [English Version (`paper/paper_en.pdf`)](./paper/paper_en.pdf).
* [`panitagorontalo/nllb-200-gorontalo-bi-600M`](https://huggingface.co/panitagorontalo/nllb-200-gorontalo-bi-600M) — Mesin Penerjemah Neural Seq2Seq **v6** (`SacreBLEU = 40.59` / `chrF++ = 57.98` pada `ID -> GOR`; `SacreBLEU = 32.57` / `chrF++ = 56.80` pada `GOR (2001) -> ID`; `SacreBLEU = 35.60` / `chrF++ = 59.29` pada `GOR (1987) -> ID`; test set $n=1.136 / 618$). Paling cepat & paling akurat secara metrik otomatis di tingkat kalimat.
* [`panitagorontalo/qwen2.5-7b-gorontalo-lora`](https://huggingface.co/panitagorontalo/qwen2.5-7b-gorontalo-lora) — LLM Instruksi & Penalaran Morfologi 7B **v6** (`SacreBLEU = 32.86` / `chrF++ = 51.60` clean dan `32.28` / `50.99` all pada `ID -> GOR`; `SacreBLEU = 25.27` / `chrF++ = 50.50` clean pada `GOR (2001) -> ID`). Seimbang untuk GPU T4 15 GB (4-bit) dengan ortografi terjemahan kalimat `ID -> GOR` yang rapi.
* [`panitagorontalo/gemma3-27b-gotoai-gorontalo-lora`](https://huggingface.co/panitagorontalo/gemma3-27b-gotoai-gorontalo-lora) — LLM 27B **v6** di atas `GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct` (`SacreBLEU = 34.45` / `chrF++ = 53.98` clean dan `33.75` / `53.27` all pada `ID -> GOR`; `SacreBLEU = 29.20` / `chrF++ = 53.19` clean pada `GOR (2001) -> ID`; `SacreBLEU = 32.06` / `chrF++ = 56.42` clean pada `GOR (1987) -> ID`). Paling unggul di antara keluarga LLM untuk menjaga konteks paragraf pendek (`GOR -> ID`) dan penalaran Siasat A/B.
* Repositori Space ini menggunakan `sdk: static` (`index.html` + `corpus_gorontalo.json`) untuk penelusuran kamus & demo instan (dengan pemecahan otomatis per paragraf/kalimat) tanpa beban server, serta menyertakan [`app.py`](./app.py) untuk menjalankan server Gradio neural penuh.

#### Model Mana yang Bisa Dijalankan via Hugging Face Space?

| Model | HF Space Free Tier (`CPU Basic`: 2 vCPU / 16 GB RAM) | HF Space GPU / Colab Tier | Catatan Teknis |
| :--- | :---: | :---: | :--- |
| **`nllb-200-gorontalo-bi-600M`** | **Bisa (Dengan `sdk: gradio`)** | **Bisa (T4 / L4 / A100 Sangat Cepat)** | Ukuran ~2,46 GB (`safetensors`). Teruji memakai puncak RAM `~3,32 GB` (muat di batas 16 GB) & latensi `~1–3 detik/kalimat`. Wajib pecah per kalimat untuk paragraf panjang. |
| **`qwen2.5-7b-gorontalo-lora`** | **Tidak Bisa** | **Bisa (Minimal T4 Small 16 GB)** | *Base model* 7,6B butuh ~15,2 GB `bf16`; kuantisasi 4-bit (`bitsandbytes`) memakan ~5,5 GB VRAM GPU CUDA. |
| **`gemma3-27b-gotoai-gorontalo-lora`** | **Tidak Bisa** | **Bisa (L4 24 GB 4-bit / A100 80 GB `bf16`)** | *Base model* 27,5B (`51,1 GiB` `bf16`); memerlukan ~64 GiB VRAM dalam `bf16` atau ~18 GB VRAM dalam QLoRA 4-bit. |

### Keterbatasan Dataset & Harapan Pengembangan Bersama Komunitas

Dataset terpadu di balik model ini adalah **langkah awal**, bukan korpus final. Kami menyampaikan kekurangannya secara terbuka agar pengguna berhati-hati dan agar komunitas tahu di mana bantuan paling dibutuhkan.

#### Kekurangan yang diketahui

1. **Sumber utama terbatas pada kamus & kurasi korpus v6.** Data berasal dari dua kamus Prof. Dr. H. Mansoer Pateda (1987 dan 2001) ditambah perbaikan & perluasan v6 (`3.409` baris diperbaiki, `+311` paralel baru, `+5.241` leksikon baru). Kalimat paralel berasal dari entri kamus, sehingga cenderung pendek dan bergaya kamus; cakupan percakapan modern, teks hukum/teknis, cerita rakyat, dan media sosial masih terbatas.
2. **Dilatih pada tingkat kalimat tunggal.** Karena data latih disusun per kalimat (`MAX_LENGTH=128` untuk NLLB, `MAX_SEQ_LEN=256` untuk Qwen/Gemma 3), mengirim paragraf utuh secara langsung dapat menyebabkan **NLLB** dan **Qwen** berhenti sebelum kalimat terakhir (*sentence dropping*). Gunakan pemecahan kalimat otomatis (seperti pada [`app.py`](./app.py), [`index.html`](./index.html), dan `infer_gorontalo.py`).
3. **Masih ada sisa noise OCR/ekstraksi.** Meski 3.409 baris telah direkonsiliasi di v6, sebagian kecil entri masih menyisakan salah baca OCR atau potongan kalimat di data latih maupun *test set*. Subset "bersih" pada evaluasi LLM disaring otomatis lewat kode, bukan audit manual penuh oleh penutur.
4. **Dua ejaan dicampur.** Kamus 1987 dan 2001 memakai ejaan berbeda. Keluaran model distandardisasi ke ejaan 2001 (dengan titik dua `:` untuk vokal panjang khas kamus).
5. **Padanan kata & pertanyaan teori tata bahasa.** Beberapa entri kamus memiliki makna ganda. Untuk pertanyaan definisi teoretis terbuka di LLM (mis. menjelaskan *"5 sistem kala"*), Qwen maupun Gemma 3 masih dapat berhalusinasi; gunakan tab analisis/kamus berbasis lookup korpus sebagai rujukan aturan tata bahasa.
6. **Sebagian data dibangkitkan otomatis.** Augmentasi sinonim KBBI (`9.135`), fonologisasi kata serapan (`1.556`), parafrase analitis MPA (`699`), dan pasangan morfologi (`9.302`) dibantu aturan dan skrip linguistik.
7. **Evaluasi baru otomatis.** Skor memakai `SacreBLEU` dan `chrF++` dengan satu referensi per kalimat, belum disertai penilaian menyeluruh penutur asli.

> **Catatan penggunaan:** hasil terjemahan model ini adalah alat bantu, bukan rujukan baku. Mohon diperiksa penutur asli sebelum dipakai untuk dokumen resmi, bahan ajar, atau publikasi.

#### Harapan ke depan

Kami berharap dataset dan model ini terus dikembangkan bersama komunitas, baik penutur asli, guru, budayawan, peneliti bahasa, maupun mahasiswa:

* **Merapikan**: mengoreksi sisa salah OCR, salah ejaan, dan pasangan kalimat yang kurang tepat.
* **Melengkapi**: menambah kosakata modern, idiom, frasa, dialog alami, dan paragraf koheren multi-kalimat.
* **Memvalidasi**: menilai terjemahan model (kala, register hormat vs biasa, imbuhan, dan pilihan MPA).
* **Menambah sumber**: teks berbahasa Gorontalo yang sah dibagikan bersama izin pemiliknya.

Masukan dapat disampaikan lewat tab **Community** (Discussions) pada halaman repositori Hugging Face ini, atau hubungi tim Panita Research Gorontalo melalui [ai@panita.web.id](mailto:ai@panita.web.id) atau situs [https://panita.web.id](https://panita.web.id).
