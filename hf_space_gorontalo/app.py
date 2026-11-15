#!/usr/bin/env python3
"""
app.py — Hugging Face Space: Penerjemah AI & Kamus Digital Bahasa Indonesia <-> Bahasa Gorontalo
Dikembangkan oleh Panita Research Gorontalo
Berjalan ringan dan cepat pada Hugging Face Spaces Free Tier (CPU Basic 2 vCPU / 16 GB RAM).
"""

import inspect
import os
import re
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import pandas as pd
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Optimasi thread CPU untuk Hugging Face Spaces Free Tier (2 vCPU)
if not torch.cuda.is_available():
    torch.set_num_threads(max(2, os.cpu_count() or 2))

MODEL_ID = os.environ.get("MODEL_ID", "panitagorontalo/nllb-200-gorontalo-bi-600M")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

print(f"Memuat model {MODEL_ID} pada {DEVICE} ({DTYPE})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
model.eval()
print("Model berhasil dimuat!")

# Muat korpus kamus terpadu untuk pencarian instan & penjelasan kosakata
df_lex = pd.read_parquet(DATA_DIR / "lexicon.parquet")
df_mor = pd.read_parquet(DATA_DIR / "morphology.parquet")
df_par = pd.read_parquet(DATA_DIR / "parallel_sentences.parquet")

# Indeks cepat kosakata untuk pendamping hasil terjemahan
id_lex_map = {}
gor_lex_map = {}
for _, row in df_lex.iterrows():
    id_w = str(row["indonesia"]).strip().lower()
    gor_w = str(row["gorontalo"]).strip()
    if id_w and len(id_w.split()) == 1:
        id_lex_map.setdefault(id_w, []).append(gor_w)
    gor_key = gor_w.lower().replace(":", "")
    if gor_key and len(gor_key.split()) == 1:
        gor_lex_map.setdefault(gor_key, []).append(f"{gor_w} = {row['indonesia']}")


def _restore_end_punctuation(src_sent: str, tgt_sent: str, force_period: bool) -> str:
    """Pertahankan tanda baca akhir kalimat saat potongan terjemahan digabung kembali."""
    out = tgt_sent.strip()
    if not out or out.endswith((".", "!", "?")):
        return out
    src_end = src_sent.strip()[-1:] if src_sent.strip() else ""
    if src_end in ("?", "!"):
        return f"{out}{src_end}"
    if force_period:
        return f"{out}{src_end if src_end == '.' else '.'}"
    return out


@lru_cache(maxsize=512)
def run_neural_translation(text: str, direction: str, num_beams: int = 2) -> tuple[str, float]:
    t0 = time.time()
    text = text.strip()
    if not text:
        return "", 0.0

    src_lang = "ind_Latn" if direction == "Indonesia → Gorontalo" else "gor_Latn"
    tgt_lang = "gor_Latn" if direction == "Indonesia → Gorontalo" else "ind_Latn"

    src_id = tokenizer.convert_tokens_to_ids(src_lang)
    tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    eos_id = tokenizer.eos_token_id

    # Terjemahkan per paragraf dan per kalimat agar input panjang tetap utuh
    paragraphs = [p.strip() for p in re.split(r"\n+", text) if p.strip()]
    translated_paragraphs = []

    with torch.inference_mode():
        for para in paragraphs:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
            multi = len(sentences) > 1
            translated_parts = []
            for sent in sentences:
                sub_ids = tokenizer(sent, add_special_tokens=False, truncation=True, max_length=250)["input_ids"]
                inp_ids = torch.tensor([[src_id] + sub_ids + [eos_id]], dtype=torch.long, device=DEVICE)
                gen_ids = model.generate(
                    input_ids=inp_ids,
                    forced_bos_token_id=tgt_id,
                    max_new_tokens=160,
                    num_beams=int(num_beams),
                    no_repeat_ngram_size=4,
                    early_stopping=True,
                )
                out = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
                translated_parts.append(_restore_end_punctuation(sent, out, force_period=multi))
            translated_paragraphs.append(" ".join(p for p in translated_parts if p))

    elapsed = time.time() - t0
    return "\n\n".join(translated_paragraphs), elapsed


def analyze_tokens_and_grammar(src_text: str, tgt_text: str, direction: str) -> str:
    """
    Menampilkan bedah kosakata, penanda 5 sistem kala, dan Morfem Penunjuk Arah (MPA)
    sesuai kaidah Prof. Dr. H. Mansoer Pateda.
    """
    gor_sentence = tgt_text if direction == "Indonesia → Gorontalo" else src_text
    id_sentence = src_text if direction == "Indonesia → Gorontalo" else tgt_text

    notes = []

    # 1. Deteksi 5 Sistem Kala & Morfem Penunjuk Arah pada kalimat Gorontalo
    gor_tokens = re.findall(r"[a-zA-Z:?'-]+", gor_sentence)
    tense_detected = []
    mpa_detected = []

    for tok in gor_tokens:
        tl = tok.lower()
        # Cek MPA
        if tl in {"mayi", "mai"} or tl.endswith("mayi") or tl.endswith("mai"):
            mpa_detected.append(f"`{tok}` → **MPA `mayi`**: arah menuju pembicara (*ke sini / datang*)")
        elif tl in {"mao", "ma?o"} or tl.endswith("mao") or tl.endswith("ma?o"):
            mpa_detected.append(f"`{tok}` → **MPA `mao`**: arah menjauh ke dekat pendengar (*ke situ / pergi*)")
        elif tl == "mola" or tl.endswith("mola"):
            mpa_detected.append(f"`{tok}` → **MPA `mola`**: arah menjauh ke udik/utara (*ke sana / pergi*)")
        elif tl == "mota" or tl.endswith("mota"):
            mpa_detected.append(f"`{tok}` → **MPA `mota`**: arah menjauh ke hilir/selatan (*ke sana / pergi*)")

        # Cek 5 Sistem Kala
        if "-" in tl and len(tl) >= 5:
            tense_detected.append(f"`{tok}` → **Kala Sedang** (reduplikasi kata kerja/sifat)")
        elif tl.startswith(("hemo", "hemoti", "hemopo")):
            tense_detected.append(f"`{tok}` → **Kala Sedang Masih** (`hemo-`)")
        elif tl.startswith(("helo", "heloti", "helopo")):
            tense_detected.append(f"`{tok}` → **Kala Sedang Lampau** (`helo-`)")
        elif tl.startswith(("lo:", "lopo", "loti", "loona", "lodelo", "lohama", "losikola", "lohutu", "le:", "lee")):
            tense_detected.append(f"`{tok}` → **Kala Lampau Aktif** (awalan `/l/`)")
        elif tl.startswith(("mo:", "mopo", "moti", "moona", "modelo", "mohama", "mosikola", "mohutu", "motali", "monga")):
            tense_detected.append(f"`{tok}` → **Kala Akan / Aktif** (awalan `/m/`)")
        elif tl.endswith(("yolo", "yalo", "olo", "alo")) and len(tl) >= 6:
            tense_detected.append(f"`{tok}` → **Bentuk Pasif / Kala Akan** (akhiran `-lo` dengan pergeseran vokal `-o -> -a`)")

    if tense_detected:
        notes.append("### ⏱️ Deteksi Sistem Kala & Morfologi\n" + "\n".join(f"- {x}" for x in dict.fromkeys(tense_detected)))
    if mpa_detected:
        notes.append("### 🧭 Deteksi Morfem Penunjuk Arah (MPA)\n" + "\n".join(f"- {x}" for x in dict.fromkeys(mpa_detected)))

    # 2. Pencarian Kosakata Terkait di Kamus Pateda
    lex_hits = []
    for w in re.findall(r"[a-zA-Z]+", id_sentence.lower()):
        if w in id_lex_map and len(w) >= 3:
            meanings = ", ".join(dict.fromkeys(id_lex_map[w][:4]))
            lex_hits.append(f"- **{w}** (ID) → `{meanings}` (GOR)")
    for w in gor_tokens:
        wk = w.lower().replace(":", "").replace("?", "")
        if wk in gor_lex_map and len(wk) >= 3:
            meanings = "; ".join(dict.fromkeys(gor_lex_map[wk][:3]))
            lex_hits.append(f"- **`{w}`** (GOR) → {meanings}")

    if lex_hits:
        notes.append("### 📖 Padanan Leksikon Kamus Pateda Terkait\n" + "\n".join(dict.fromkeys(lex_hits[:8])))

    if not notes:
        return "*Terjemahan dihasilkan secara neural oleh `panitagorontalo/nllb-200-gorontalo-bi-600M`.*"
    return "\n\n".join(notes)


def translate_ui(text: str, direction: str, num_beams: int):
    if not text or not text.strip():
        return "", "Silakan masukkan teks untuk diterjemahkan."
    out, elapsed = run_neural_translation(text, direction, int(num_beams))
    analysis = analyze_tokens_and_grammar(text, out, direction)
    header = f"**Waktu Inferensi ({DEVICE.upper()}, beam={int(num_beams)})**: `{elapsed:.2f} detik`\n\n"
    return out, header + analysis


def search_dictionary(query: str, search_mode: str):
    q = (query or "").strip().lower()
    if not q:
        return (
            df_lex[["indonesia", "gorontalo", "gorontalo_asli", "tipe", "sumber_kamus"]].head(15),
            df_mor.head(10),
            df_par[["src_indonesia", "tgt_gorontalo", "sumber_kamus"]].head(10),
        )

    if search_mode == "Cari dari Bahasa Indonesia":
        lex_sub = df_lex[df_lex["indonesia"].str.lower().str.contains(re.escape(q), na=False)].head(30)
        mor_sub = df_mor[
            df_mor["arti_turunan"].str.lower().str.contains(re.escape(q), na=False)
            | df_mor["arti_akar"].str.lower().str.contains(re.escape(q), na=False)
            | df_mor["akar_kata"].str.lower().str.contains(re.escape(q), na=False)
        ].head(20)
        par_sub = df_par[df_par["src_indonesia"].str.lower().str.contains(re.escape(q), na=False)].head(20)
    else:
        q_clean = q.replace(":", "").replace("?", "")
        lex_sub = df_lex[
            df_lex["gorontalo"].str.lower().str.replace(":", "", regex=False).str.contains(re.escape(q_clean), na=False)
            | df_lex["gorontalo_asli"].str.lower().str.contains(re.escape(q), na=False)
        ].head(30)
        mor_sub = df_mor[
            df_mor["kata_turunan"].str.lower().str.replace(":", "", regex=False).str.contains(re.escape(q_clean), na=False)
            | df_mor["akar_kata"].str.lower().str.contains(re.escape(q_clean), na=False)
        ].head(20)
        par_sub = df_par[
            df_par["tgt_gorontalo"].str.lower().str.replace(":", "", regex=False).str.contains(re.escape(q_clean), na=False)
        ].head(20)

    return (
        lex_sub[["indonesia", "gorontalo", "gorontalo_asli", "tipe", "sumber_kamus"]],
        mor_sub,
        par_sub[["src_indonesia", "tgt_gorontalo", "sumber_kamus"]],
    )


def convert_orthography(text: str, mode: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    if mode == "Ejaan 1987 → Ejaan Baku 2001":
        # 1. Vokal ganda -> titik dua (kecuali kata ulang)
        for v in ["a", "i", "u", "e", "o"]:
            t = re.sub(f"{v}{v}", f"{v}:", t, flags=re.IGNORECASE)
        # 2. Hapus tanda tanya glotal di tengah kata
        t = re.sub(r"(?<=[a-zA-Z])\?(?=[a-zA-Z])", "", t)
        # 3. Luncuran umum
        replacements = [
            (r"\btio\b", "tiyo"),
            (r"\bboito\b", "boyito"),
            (r"\bbotia\b", "botiya"),
            (r"\bwa:tia\b", "wa:tiya"),
            (r"\bmai\b", "mayi"),
            (r"\bdoi\b", "doyi"),
            (r"\btuau\b", "tuwawu"),
            (r"\bduluo\b", "duluwo"),
        ]
        for pat, rep in replacements:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        return t
    else:
        # Ejaan 2001 -> Ejaan 1987
        for v in ["a", "i", "u", "e", "o"]:
            t = re.sub(f"{v}:", f"{v}{v}", t, flags=re.IGNORECASE)
        replacements = [
            (r"\btiyo\b", "tio"),
            (r"\bboyito\b", "boito"),
            (r"\bbotiya\b", "botia"),
            (r"\bwaatiya\b", "waatia"),
            (r"\bmayi\b", "mai"),
            (r"\bmau\b", "ma?o"),
            (r"\bwau\b", "wa?u"),
            (r"\byio\b", "yi?o"),
        ]
        for pat, rep in replacements:
            t = re.sub(pat, rep, t, flags=re.IGNORECASE)
        return t


_APP_THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="blue")
_THEME_IN_LAUNCH = "theme" in inspect.signature(gr.Blocks.launch).parameters
_BLOCKS_KWARGS = {"title": "Penerjemah AI & Kamus Bahasa Gorontalo — Panita Research Gorontalo"}
if not _THEME_IN_LAUNCH:
    _BLOCKS_KWARGS["theme"] = _APP_THEME

with gr.Blocks(**_BLOCKS_KWARGS) as demo:
    gr.Markdown(
        """
        # 🦅 Penerjemah AI & Kamus Digital Bahasa Indonesia ↔ Bahasa Gorontalo
        **Dikembangkan oleh [Panita Research Gorontalo](https://huggingface.co/panitagorontalo)** | Berbasis Korpus Terpadu Kamus **Prof. Dr. H. Mansoer Pateda (1987 & 2001)**
        """
    )

    with gr.Tabs():
        # TAB 1: PENERJEMAH AI DUA ARAH
        with gr.TabItem("🤖 Penerjemah Kalimat AI"):
            with gr.Row():
                with gr.Column(scale=1):
                    direction_radio = gr.Radio(
                        choices=["Indonesia → Gorontalo", "Gorontalo → Indonesia"],
                        value="Indonesia → Gorontalo",
                        label="Arah Penerjemahan",
                    )
                    beam_slider = gr.Slider(
                        minimum=1,
                        maximum=4,
                        value=2,
                        step=1,
                        label="Kualitas Pencarian Beam (2 = Cepat untuk CPU Gratis, 4 = Akurasi Maksimal)",
                    )
                    input_box = gr.Textbox(
                        lines=4,
                        placeholder="Ketik kalimat Bahasa Indonesia atau Bahasa Gorontalo di sini...",
                        label="Teks Masukan",
                        value="Saya pergi ke pasar kemarin.",
                    )
                    translate_btn = gr.Button("Terjemahkan Sekarang 🚀", variant="primary")

                with gr.Column(scale=1):
                    output_box = gr.Textbox(
                        lines=4,
                        label="Hasil Terjemahan AI (Ejaan Baku Pateda 2001)",
                        interactive=False,
                    )
                    analysis_md = gr.Markdown(label="Analisis Linguistik & Kosakata")

            gr.Examples(
                examples=[
                    ["Saya pergi ke pasar kemarin.", "Indonesia → Gorontalo", 2],
                    ["Aku akan pergi membeli beras di pasar.", "Indonesia → Gorontalo", 2],
                    ["Dia sedang duduk di rumah.", "Indonesia → Gorontalo", 2],
                    ["Mereka sudah membawa buku itu ke sini.", "Indonesia → Gorontalo", 2],
                    ["Ambillah ke sini buku itu.", "Indonesia → Gorontalo", 2],
                    ["Rumah Pak Dula sangat bersih.", "Indonesia → Gorontalo", 2],
                    ["Jangan lupa berobat ke dokter.", "Indonesia → Gorontalo", 2],
                    ["Wa:tiya loona:o ode pa:tali ola:ngo.", "Gorontalo → Indonesia", 2],
                    ["Yilongola yio di:la losikola?", "Gorontalo → Indonesia", 2],
                    ["Ngoolo haraga lo buku botiya?", "Gorontalo → Indonesia", 2],
                    ["Tiyo hulo-huloo to bele.", "Gorontalo → Indonesia", 2],
                ],
                inputs=[input_box, direction_radio, beam_slider],
                outputs=[output_box, analysis_md],
                fn=translate_ui,
                cache_examples=False,
                label="Contoh Kalimat (Kaidah 5 Kala, MPA, Kata Serapan, & Sapaan)",
            )

            translate_btn.click(
                fn=translate_ui,
                inputs=[input_box, direction_radio, beam_slider],
                outputs=[output_box, analysis_md],
            )

        # TAB 2: KAMUS DIGITAL & MORFOLOGI
        _lex_str = f"{len(df_lex):,}".replace(",", ".")
        _mor_str = f"{len(df_mor):,}".replace(",", ".")
        _par_str = f"{len(df_par):,}".replace(",", ".")
        with gr.TabItem(f"📚 Kamus Digital & Morfologi ({_lex_str} Kosakata)"):
            gr.Markdown(
                f"Cari kosakata pokok, kata turunan berimbuhan (**{_mor_str}+ derivasi morfologi**), dan contoh kalimat paralel (**{_par_str} kalimat**) dari *Kamus Gorontalo–Indonesia (2001)*, *Kamus Indonesia–Gorontalo (1987)*, & kurasi rekonsiliasi **v6**."
            )
            with gr.Row():
                dict_query = gr.Textbox(
                    placeholder="Contoh: pasar, rumah, tulis, pa:tali, bele, tuladu, moluladu...",
                    label="Kata Kunci Pencarian",
                    scale=3,
                )
                dict_mode = gr.Radio(
                    choices=["Cari dari Bahasa Indonesia", "Cari dari Bahasa Gorontalo"],
                    value="Cari dari Bahasa Indonesia",
                    label="Mode Pencarian",
                    scale=2,
                )
                dict_btn = gr.Button("Cari di Kamus 🔍", variant="primary", scale=1)

            gr.Markdown("### 1. Entri Leksikon Dwibahasa")
            lex_table = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("### 2. Derivasi Imbuhan & Morfologi Produktif (Siasat A)")
            mor_table = gr.Dataframe(interactive=False, wrap=True)
            gr.Markdown("### 3. Contoh Kalimat Paralel di Kamus")
            par_table = gr.Dataframe(interactive=False, wrap=True)

            dict_btn.click(
                fn=search_dictionary,
                inputs=[dict_query, dict_mode],
                outputs=[lex_table, mor_table, par_table],
            )
            dict_query.submit(
                fn=search_dictionary,
                inputs=[dict_query, dict_mode],
                outputs=[lex_table, mor_table, par_table],
            )

        # TAB 3: KONVERTER ORTOGRAFI & KAIDAH 5 KALA
        with gr.TabItem("🔤 Konverter Ejaan (1987 ↔ 2001) & Kaidah Tata Bahasa"):
            with gr.Row():
                with gr.Column():
                    orth_mode = gr.Radio(
                        choices=["Ejaan 1987 → Ejaan Baku 2001", "Ejaan Baku 2001 → Ejaan 1987"],
                        value="Ejaan 1987 → Ejaan Baku 2001",
                        label="Arah Konversi Ortografi",
                    )
                    orth_in = gr.Textbox(
                        lines=3,
                        value="Waatia loona?o ode paatali olaango woli tio.",
                        label="Teks Masukan",
                    )
                    orth_btn = gr.Button("Konversi Ejaan 🔄", variant="primary")
                with gr.Column():
                    orth_out = gr.Textbox(lines=3, label="Hasil Konversi Ortografi", interactive=False)

            orth_btn.click(fn=convert_orthography, inputs=[orth_in, orth_mode], outputs=orth_out)

            gr.Markdown(
                """
                ---
                ### Ringkasan 5 Sistem Kala & 4 Morfem Penunjuk Arah (Prof. Dr. H. Mansoer Pateda)
                | Sistem Kala / Morfem | Penanda Morfologis | Contoh Bahasa Gorontalo | Terjemahan Bahasa Indonesia |
                | :--- | :--- | :--- | :--- |
                | **1. Kala Akan** | Awalan `/m/` (`mo-, mopo-, moti-`), sisipan `-um-`, akhiran pasif `-lo` | `modelo`, `deloolo`, `tumuoto` | *(akan) membawa, (akan) dibawa, (akan) masuk* |
                | **2. Kala Sedang Masih** | Awalan `hemo-`, `hemoti-` | `hemomake`, `hemotihuloo` | *sedang masih berpakaian, sedang masih duduk* |
                | **3. Kala Sedang** | Reduplikasi (Kata Ulang) | `hulo-huloo`, `bala-balato`, `tihu-tihulo` | *sedang duduk, sedang berguling, sedang berdiri* |
                | **4. Kala Sedang Lampau** | Awalan `helo-`, `heloti-` | `helodelo`, `helotihuloo` | *telah sedang membawa, telah sedang duduk* |
                | **5. Kala Lampau** | Awalan `/l/` (`lo-, lopo-, loti-, lee-`), sisipan pasif `-il-` | `lodelo`, `dilelo`, `leehuluto` | *telah membawa, telah dibawa, telah tergelincir* |
                | **MPA `mayi` (`mai`)** | Menuju pembicara (*ke sini / datang*) | `mayi mohama` / `deloamayi` | *datang mengambil / bawalah ke sini* |
                | **MPA `mao` (`ma?o`)** | Menjauh ke dekat pendengar (*ke situ / pergi*) | `mao modepito` / `deloamao` | *pergi mengantar / bawalah ke situ* |
                | **MPA `mola`** | Menjauh ke arah udik/utara (*ke sana / pergi*) | `mola motali` / `deloamola` | *pergi membeli / bawalah ke sana (udik)* |
                | **MPA `mota`** | Menjauh ke arah hilir/selatan (*ke sana / pergi*) | `mota mosikola` / `deloamota` | *pergi bersekolah / bawalah ke sana (hilir)* |
                """
            )

if __name__ == "__main__":
    _LAUNCH_KWARGS = {}
    if _THEME_IN_LAUNCH:
        _LAUNCH_KWARGS["theme"] = _APP_THEME
    demo.launch(**_LAUNCH_KWARGS)
