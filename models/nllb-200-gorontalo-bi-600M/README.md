---
language:
  - id
  - gor
license: apache-2.0
library_name: transformers
pipeline_tag: translation
base_model: facebook/nllb-200-distilled-600M
tags:
  - translation
  - nllb-200
  - gorontalo
  - bahasa-gorontalo
  - hulontalo
  - indonesian
  - low-resource-language
  - austronesian
  - panita-research-gorontalo
  - mansoer-pateda
metrics:
  - sacrebleu
  - chrf
model-index:
  - name: nllb-200-gorontalo-bi-600M
    results:
      - task:
          type: translation
          name: Translation (Indonesia -> Gorontalo)
        dataset:
          name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
          type: panitagorontalo/gorontalo-indonesia-parallel
        metrics:
          - type: sacrebleu
            value: 40.59
            name: SacreBLEU (ID -> GOR Test v6)
          - type: chrf
            value: 57.98
            name: chrF++ (ID -> GOR Test v6)
      - task:
          type: translation
          name: Translation (Gorontalo -> Indonesia)
        dataset:
          name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
          type: panitagorontalo/gorontalo-indonesia-parallel
        metrics:
          - type: sacrebleu
            value: 32.57
            name: SacreBLEU (GOR 2001 -> ID Test v6)
          - type: chrf
            value: 56.80
            name: chrF++ (GOR 2001 -> ID Test v6)
      - task:
          type: translation
          name: Translation (Gorontalo 1987 -> Indonesia)
        dataset:
          name: Korpus Terpadu v6 Kamus Prof. Dr. H. Mansoer Pateda (1987 & 2001)
          type: panitagorontalo/gorontalo-indonesia-parallel
        metrics:
          - type: sacrebleu
            value: 35.60
            name: SacreBLEU (GOR 1987 -> ID Test v6)
          - type: chrf
            value: 59.29
            name: chrF++ (GOR 1987 -> ID Test v6)
---

# `panitagorontalo/nllb-200-gorontalo-bi-600M` (v6)
**Mesin Penerjemah Neural Dwibahasa Bahasa Indonesia (`ind_Latn`) $\leftrightarrow$ Bahasa Gorontalo (`gor_Latn`)**

[![Hugging Face Space](https://img.shields.io/badge/🦅_Hugging_Face_Space-Penerjemah_&_Kamus_Gorontalo-indigo)](https://huggingface.co/spaces/panitagorontalo/penerjemah-bahasa-gorontalo)
[![Notebook Tutorial Colab](https://img.shields.io/badge/📓_Google_Colab-Unduh_&_Jalankan_Notebook_(.ipynb)-orange)](https://huggingface.co/panitagorontalo/nllb-200-gorontalo-bi-600M/blob/main/Tutorial_Penerjemah_AI_Bahasa_Gorontalo_Panita_Research.ipynb)
[![Paper Teknis v6 ID](https://img.shields.io/badge/📄_Paper_Teknis_v6-Bahasa_Indonesia_(PDF)-blue)](https://huggingface.co/panitagorontalo/nllb-200-gorontalo-bi-600M/blob/main/paper/paper_id.pdf)
[![Technical Paper v6 EN](https://img.shields.io/badge/📄_Technical_Paper_v6-English_(PDF)-green)](https://huggingface.co/panitagorontalo/nllb-200-gorontalo-bi-600M/blob/main/paper/paper_en.pdf)

Dikembangkan oleh **Panita Research Gorontalo** sebagai bagian dari inisiatif pelestarian digital dan kecerdasan buatan untuk bahasa daerah **Bahasa Gorontalo (*Bahasa Hulontalo*)**. Model **v6** ini merupakan hasil **Full Fine-Tuning 4 Epochs (`bfloat16`)** di atas arsitektur **`facebook/nllb-200-distilled-600M`** dengan pembobotan 3× pada frasa leksikon/morfologi pendek ($\le 3$ kata) menggunakan **korpus terpadu v6** (**32.039** entri kosakata kamus / **28.727** unik bersih, **29.546** kalimat paralel mentah / **22.751** unik bersih, dan **9.777** pasangan morfologi) dari dua kamus **Prof. Dr. H. Mansoer Pateda** yang telah diaudit dan direkonsiliasi:
1. ***Kamus Bahasa Gorontalo–Indonesia*** (Pusat Bahasa Departemen Pendidikan Nasional & Balai Pustaka, 2001 — 302 halaman leksikon + pengantar).
2. ***Kamus Bahasa Indonesia–Gorontalo*** (Pusat Pembinaan dan Pengembangan Bahasa, 1987 — 489 halaman leksikon + pengantar).

---

## 1. Pembaruan Utama v6 (Rekonsiliasi Korpus + Peningkatan Skor Signifikan)

Pada rilis **v6** (`elapsed_minutes = 70.14`, `train_loss = 1.1307`), model dilatih pada korpus hasil audit & perbaikan:
1. **Korpus v6 Lebih Lengkap & Lebih Bersih**: Memperbaiki **3.409** baris paralel v5 (pemulihan pemotongan kolom & perbaikan OCR) serta menambah **+311** kalimat paralel terkurasi dan **+5.241** kosakata leksikon baru, menghasilkan **309.206** sampel pelatihan (`145.920` sampel dasar `seq2seq_train` + *oversampling* 3× pada leksikon & morfologi frasa pendek), serta pembagian `val=2.898` / `test=2.890` tanpa kebocoran data (*zero leakage*).
2. **Lonjakan Skor SacreBLEU & chrF++ (Test Set v6, $n=1.136$ / $618$)**:
   * `ID -> GOR` melonjak ke **`40.59` SacreBLEU / `57.98` chrF++** (dari baseline v4 `31.14 / 49.60`, v5 `35.65 / 53.87`, dan *zero-shot* `0.17 / 10.26`).
   * `GOR (2001) -> ID` meningkat ke **`32.57` SacreBLEU / `56.80` chrF++** (dari v4 `24.52 / 48.93` dan v5 `28.95 / 51.70`).
   * `GOR (1987) -> ID` meningkat ke **`35.60` SacreBLEU / `59.29` chrF++** (dari v4 `29.18 / 54.55` dan v5 `30.15 / 53.15`).

---

## 2. Inovasi Linguistik & Kaidah Mansoer Pateda

Bahasa Gorontalo (*Hulontalo*, ISO 639-3: `gor`) tergolong berdaya sumber rendah dan belum didukung oleh `NLLB-200` standar. **Panita Research Gorontalo** membangun jembatan komputasional dengan:
1. **Pendaftaran Token Bahasa Khusus `gor_Latn` (`ID = 256204`)**: Diinisialisasi dari rata-rata (*centroid*) representasi bahasa serumpun di `NLLB-200`: `ind_Latn`, `bug_Latn`, `tgl_Latn`, dan `ceb_Latn`.
2. **Jembatan Ortografi 1987 $\leftrightarrow$ 2001**:
   * **Ejaan 1987**: vokal ganda (`aa, ee, ii, oo, uu`), glotal `?` (`wa?u, lo?otapu`), tanpa luncuran `y/w` (`tio, boito, mai, wau`).
   * **Ejaan 2001 (Standar Target)**: titik dua `:` (`pa:tali, wa:tiya, di:la`), tanpa `?` (`wau, lootapu`), luncuran eksplisit (`tiyo, boyito, mayi, wawu`).
3. **Empat Siasat Mengatasi Keterbatasan Kosakata Bahasa Gorontalo (Kaidah Prof. Dr. H. Mansoer Pateda)**:
   * **Siasat A (Morphological Engine)**: pasangan derivasi morfologi, 31 sistem prefiks, infiks (`-um-`, `-il-`), sufiks (`-lo`, `-a`, `-i`), peluluhan fonem (`/t/->/l/, /s/->/n/, /p/->/m/, /k/->/ng/`), serta **5 Sistem Kala**.
   * **Siasat B (Loanword Phonologization `BI -> BG`)**: vokalisasi kata serapan agar berakhiran vokal (*dokter* $\rightarrow$ `dokuteri`, *adil* $\rightarrow$ `a:dili`).
   * **Siasat C (Analytical Paraphrasing & MPA)**: komposisi kerja + 4 morfem arah (`mayi`, `mao`, `mola`, `mota`).
   * **Siasat D (Lexical Synonym Bridging & Register)**: pemetaan sinonim KBBI serta register **Biasa (`wau / yio`)** vs **Hormat (`wa:tiya / ito`)**.

---

## 3. Hasil Evaluasi Uji Benchmark (`SacreBLEU` & `chrF++`)

Evaluasi dilakukan pada `Test Set` terstratifikasi resmi korpus v6 ($n=1.136$ untuk `ID->GOR` & `GOR 2001->ID`, $n=618$ untuk `GOR 1987->ID`):

| Arah Penerjemahan (`Test Set`) | *Zero-Shot* v6 (`SacreBLEU` / `chrF++`) | NLLB v4 (`SacreBLEU` / `chrF++`) | NLLB v5 (`SacreBLEU` / `chrF++`) | **NLLB v6 Terbaru (`SacreBLEU` / `chrF++`)** |
| :--- | :---: | :---: | :---: | :---: |
| **Indonesia $\rightarrow$ Gorontalo** | `0.17` / `10.26` | `31.14` / `49.60` | `35.65` / `53.87` | **`40.59` / `57.98`** |
| **Gorontalo (2001) $\rightarrow$ Indonesia** | `0.24` / `10.66` | `24.52` / `48.93` | `28.95` / `51.70` | **`32.57` / `56.80`** |
| **Gorontalo (1987) $\rightarrow$ Indonesia** | — | `29.18` / `54.55` | `30.15` / `53.15` | **`35.60` / `59.29`** |

---

## 4. Contoh Hasil Terjemahan Kualitatif (Terverifikasi `evaluasi_nllb_gorontalo.json` v6)

| Arah | Teks Masukan (*Input*) | Hasil Terjemahan Model (*Output*) | Keterangan |
| :---: | :--- | :--- | :--- |
| `ID -> GOR` | *matahari* | **`dulahu`** | Leksikon dasar (`Matahari sudah terbit` $\rightarrow$ `Matolodulahu ma lobutu`) |
| `ID -> GOR` | *air mata* | **`ponu`** | Padanan kamus GI (`ponu`) |
| `ID -> GOR` | *mata air* | **`mato lo taluhu`** | Dekomposisi semantik sumber air |
| `ID -> GOR` | *Saya pergi ke pasar kemarin.* | **`Wa:tiya lo:nao ode pa:tali ola:ngo`** | Pronomina hormat `Wa:tiya` + kala lampau `lo:nao` |
| `ID -> GOR` | *Dia sedang duduk di rumah.* | **`Tiyo hulo-huloa to bele`** | Aspek duratif (Kala Sedang) reduplikasi |
| `ID -> GOR` | *Rumah Pak Dula sangat bersih.* | **`Bele lemei Dula sambe beresi`** | Teknonimi `lemei` + fonologisasi vokalis `beresi` |
| `ID -> GOR` | *Jangan lupa berobat ke dokter.* | **`Di:la lipata mohunema oli dokuteri`** | Fonologisasi serapan vokalis `dokuteri` |
| `ID -> GOR` | *Ambillah ke sini buku itu.* | **`Hama:mayi buku boyito`** | Morfem Penunjuk Arah (`MPA`) `mayi` |
| `GOR -> ID` | *Wa:tiya loona:o ode pa:tali ola:ngo.* | **`Saya pergi ke pasar kemarin`** | Pemahaman vokal panjang + kala lampau |
| `GOR -> ID` | *Tiyo hulo-huloo to bele.* | **`Ia duduk di rumah`** | Penerjemahan aspek duratif |
| `GOR -> ID` | *Ngoolo haraga lo buku botiya?* | **`Berapa harga buku ini?`** | Kata tanya kuantitas `Ngoolo` |
| `GOR -> ID` | *Yilongola yio di:la losikola?* | **`Kenapa engkau tidak bersekolah?`** | Kata tanya kausalitas `Yilongola` |

---

## 5. Cara Menjalankan Inferensi (Disarankan Pemecahan Kalimat untuk Paragraf)

> **Catatan Paragraf:** Karena dilatih per kalimat tunggal (`MAX_LENGTH=128`), pecah teks paragraf per kalimat sebelum menerjemahkan agar kalimat ke-3 dan seterusnya tidak terpotong (*early EOS stopping*). Untuk Hugging Face Space Free Tier (`CPU Basic: 2 vCPU / 16 GB RAM`), model ini berjalan ringan (`~3,32 GB` puncak RAM, `~1–3 detik/kalimat` pada `num_beams=2`).

```python
import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MODEL_ID = "panitagorontalo/nllb-200-gorontalo-bi-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.bfloat16 if device == "cuda" else torch.float32

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, dtype=dtype).to(device)
model.eval()

def translate(text: str, src_lang="ind_Latn", tgt_lang="gor_Latn", num_beams=4) -> str:
    src_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    paragraphs = [p.strip() for p in re.split(r"\n+", text.strip()) if p.strip()]
    out_paras = []
    for para in paragraphs:
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        multi = len(sents) > 1
        parts = []
        for s in sents:
            sub_ids = tokenizer(s, add_special_tokens=False, truncation=True, max_length=250)["input_ids"]
            inp = torch.tensor([[src_id] + sub_ids + [tokenizer.eos_token_id]], device=device)
            with torch.inference_mode():
                gen = model.generate(
                    input_ids=inp,
                    forced_bos_token_id=tgt_id,
                    max_new_tokens=160,
                    num_beams=num_beams,
                    no_repeat_ngram_size=4,
                    early_stopping=True,
                )
            pred = tokenizer.decode(gen[0], skip_special_tokens=True).strip()
            if pred and not pred.endswith((".", "!", "?")):
                end_ch = s[-1] if s.endswith((".", "!", "?")) else "."
                if multi or end_ch in ("?", "!"):
                    pred += end_ch
            parts.append(pred)
        out_paras.append(" ".join(parts))
    return "\n\n".join(out_paras)

# Contoh:
print(translate("Saya pergi ke pasar kemarin.", "ind_Latn", "gor_Latn"))
# Output: Wa:tiya lo:nao ode pa:tali ola:ngo
```

---

## 6. Tim Pengembang (Panita Research Gorontalo)

* Funco Tanipu
* Richic Oct Sumual
* Budiyanto
* Thomas Jr

* Website: [https://panita.web.id](https://panita.web.id)
* Email: [ai@panita.web.id](mailto:ai@panita.web.id), [richie@panita.web.id](mailto:richie@panita.web.id), [funco@panita.web.id](mailto:funco@panita.web.id)

---

## 7. Keterbatasan Dataset & Harapan Pengembangan Bersama Komunitas

Dataset terpadu di balik model ini adalah **langkah awal**, bukan korpus final.

### Kekurangan yang diketahui

1. **Sumber terbatas pada kamus cetak (1987 & 2001) + rekonsiliasi v6.** Kalimat paralel berasal dari contoh kamus sehingga cenderung pendek dan bergaya kamus; percakapan sehari-hari, teks modern, dan dokumen hukum/teknis belum terwakili memadai.
2. **Dilatih di tingkat kalimat tunggal.** Input paragraf panjang perlu dipecah per kalimat menggunakan fungsi seperti di atas.
3. **Sisa noise OCR.** Walaupun **3.409** baris bermasalah telah diperbaiki pada v6, masih ada sebagian kecil salah baca OCR di pasangan kamus.
4. **Dua ejaan dicampur.** Ejaan 1987 dan 2001 diselaraskan dengan standar keluaran ejaan 2001 (`:` untuk vokal panjang).
5. **Sebagian data dibangkitkan otomatis.** Augmentasi sinonim KBBI, fonologisasi kata serapan, parafrase analitis MPA, dan pasangan morfologi dibantu skrip aturan.
6. **Cakupan ukuran (**`22.751` unik bersih / `29.546` paralel gabungan, `32.039` leksikon**).** Variasi dialek dan ragam tutur belum lengkap.
7. **Evaluasi otomatis.** Skor memakai `SacreBLEU` dan `chrF++` dengan satu referensi, belum disertai evaluasi penutur asli secara menyeluruh.

> **Catatan penggunaan:** hasil terjemahan model ini adalah bantuan, bukan rujukan baku. Mohon diperiksa penutur asli sebelum dipakai untuk dokumen resmi, bahan ajar, atau publikasi.

---

## 8. Sitasi & Penghargaan

```bibtex
@misc{panitagorontalo2026nllb,
  title        = {NLLB-200-Gorontalo-BI-600M v6: Bidirectional Neural Machine Translation for Indonesian and Gorontalo},
  author       = {{Panita Research Gorontalo}},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/panitagorontalo/nllb-200-gorontalo-bi-600M}}
}
```
