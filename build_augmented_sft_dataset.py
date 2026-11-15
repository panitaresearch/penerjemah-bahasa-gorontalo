#!/usr/bin/env python3
"""
build_augmented_sft_dataset.py
==============================
Membangun dataset terpadu yang diperkaya dengan 4 Siasat Linguistik Prof. Dr. H. Mansoer Pateda:
  - Pembersihan singkatan kamus (yg->yang, tdk->tidak, dgn->dengan, pd->pada, dr->dari, dll.)
  - Siasat A: Mesin Perakitan Imbuhan, 5 Kala, & Kaidah Morfofonemik Produktif
  - Siasat B: Kaidah Fonologisasi Kata Serapan (Loanword Phonologization Rule BI -> BG)
  - Siasat C: Dekomposisi Semantik & Parafrase Analitis dengan Morfem Penunjuk Arah (MPA)
  - Siasat D: Jembatan Sinonim Bahasa Indonesia (KBBI) & Register Biasa vs Hormat
Serta membagi dataset secara terstratifikasi (Train 90% / Val 5% / Test 5%) tanpa kebocoran data (zero leakage).
"""

import json
import os
import random
import re
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

BASE_DIR = Path("/content/drive/MyDrive/REPO/richie/modelgorontalo")
OUT_DIR = BASE_DIR / "dataset_terpadu_indonesia_gorontalo"

# 1. Kamus Ekspansi Singkatan Kamus Cetak ke Bahasa Indonesia Baku (KBBI)
ABBR_MAP = [
    (r"\byg\b", "yang"),
    (r"\btdk\b", "tidak"),
    (r"\bdgn\b", "dengan"),
    (r"\butk\b", "untuk"),
    (r"\bdlm\b", "dalam"),
    (r"\bkrn\b", "karena"),
    (r"\bsbg\b", "sebagai"),
    (r"\bpd\b", "pada"),
    (r"\bdr\b", "dari"),
    (r"\bspt\b", "seperti"),
    (r"\btsb\b", "tersebut"),
    (r"\bkp\b", "kepada"),
    (r"\bsdh\b", "sudah"),
    (r"\bblm\b", "belum"),
    (r"\bsesuatu\s+yg\b", "sesuatu yang"),
]


def clean_indonesian_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    for pat, repl in ABBR_MAP:
        t = re.sub(pat, repl, t, flags=re.IGNORECASE)
    # Normalisasi spasi ganda
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_gorontalo_text(text: str, src_id: str = "") -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    # Bersihkan jika ada kata Indonesia berimbuhan me-/pe-/-nya yang bocor di awal kalimat Gorontalo
    first_word = t.split()[0].lower() if t.split() else ""
    if len(t.split()) > 3 and (
        first_word.endswith("nya")
        or first_word.endswith("kan")
        or first_word in {"menyelesaikannya", "pekerjaan", "mengambilnya", "membuatnya"}
    ):
        t = " ".join(t.split()[1:])
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# Filter kualitas (v4): membuang baris hasil OCR rusak / salah-segmentasi
# ---------------------------------------------------------------------------
_NOISE_RE = re.compile(r"[~{}|\\«»£$@#*_=<>\[\]^]|(?<=[a-z])[A-Z]|\d[A-Za-z]|[A-Za-z]\d|\.{2,}")
_CONSONANT_RUN = re.compile(r"\b[bcdfghjklmnpqrstvwxz]{4,}\b")
_GOR_IN_ID = re.compile(r"[?:]|\b\w*(aa|ee|ii|oo|uu)\w*\b|\b(wau|yio|tiyo|boyito|botiye?a?|diila|di:la|lo|to|ma|bo|ti|ta|te|le|li)\b")
_ID_STOP = {"yang", "dan", "dengan", "untuk", "tidak", "sudah", "karena", "pada", "dari", "itu", "ini", "saya", "kami", "di", "ke", "akan"}


def _has_noise(text: str) -> bool:
    return bool(_NOISE_RE.search(text)) or bool(_CONSONANT_RUN.search(text.lower()))


def is_clean_parallel(src: str, tgt: str) -> bool:
    if not (6 <= len(src) <= 170 and 6 <= len(tgt) <= 170):
        return False
    if re.search(r"\s:\s|\s:$|\(\d\)|= =|==", tgt) or re.search(r"\s:\s|= =|\(\d\)", src):
        return False
    if _has_noise(src) or _has_noise(tgt):
        return False
    ws, wt = len(src.split()), len(tgt.split())
    if ws < 2 or wt < 2 or not (0.3 <= wt / ws <= 3.5):
        return False
    # sisi Indonesia tidak boleh didominasi token Gorontalo (salah segmentasi)
    src_tokens = re.findall(r"[a-z?:]+", src.lower())
    gor_hits = sum(1 for t in src_tokens if "?" in t or ":" in t or re.search(r"(aa|ee|ii|oo|uu)", t))
    if src_tokens and gor_hits / len(src_tokens) > 0.25:
        return False
    # sisi Gorontalo tidak boleh didominasi kata fungsi Indonesia
    tgt_tokens = re.findall(r"[a-z]+", tgt.lower())
    if tgt_tokens and sum(1 for t in tgt_tokens if t in _ID_STOP) / len(tgt_tokens) > 0.4:
        return False
    return True


def is_clean_lexicon(ind: str, gor: str) -> bool:
    if not (2 <= len(ind) <= 60 and 2 <= len(gor) <= 60):
        return False
    if len(ind.split()) > 4 or len(gor.split()) > 4:
        return False
    if re.search(r"\s:\s|= =|==|\(\d\)|[~{}|\\«»£$@#*_<>\[\]^]|\d", ind + " " + gor):
        return False
    if _CONSONANT_RUN.search(ind.lower()):
        return False
    return True


def extract_loanwords(lex_df: pd.DataFrame, mor_ig_df: pd.DataFrame) -> list[dict]:
    """
    Siasat B: Ekstraksi pasangan kata serapan (Loanword Phonologization BI -> BG)
    di mana kata Bahasa Indonesia diserap ke Bahasa Gorontalo secara vokalis
    (penambahan vokal akhir -i/-u/-o/-a, penyisipan vokal pada gugus konsonan,
    dan pelekatan imbuhan produktif Gorontalo).
    """
    loan_records = []
    seen = set()

    # 1. Pasangan kurasi eksplisit dari Kaidah Prof. Mansoer Pateda
    curated_loans = [
        ("adat", "a:dati", "a:datiyolo (diadatkan), monga:dati (menjalankan adat)", "Tambahkan vokal akhir -i dan panjangkan vokal awal a: karena berakhir konsonan /t/."),
        ("adil", "a:dili", "monga:dili (mengadili), mopoa:dili (mengadilkan)", "Tambahkan vokal akhir -i karena kata asli berakhir konsonan /l/."),
        ("acar", "a:cari", "a:cariyolo (dibuat acar), peia:cari (disuruh buat acar)", "Tambahkan vokal akhir -i karena kata asli berakhir konsonan /r/."),
        ("adres / alamat", "adresi", "popoadresiyolo (diadreskan)", "Tambahkan vokal akhir -i pada kata serapan berakhir konsonan /s/."),
        ("dokter", "dokuteri", "modokuteri (berobat ke dokter)", "Sisipkan vokal /u/ pada gugus konsonan -kt- dan tambahkan vokal akhir -i."),
        ("liter", "li:teri", "li:teriyolo (diliter / ditakar per liter)", "Panjangkan vokal penultima li: dan tambahkan vokal akhir -i."),
        ("mesel / tembok", "me:seli", "mome:seli (memesel), me:seliyolo (dimesel)", "Panjangkan vokal me: dan tambahkan vokal akhir -i."),
        ("kemeja", "kameja", "mohikameja (memakai kemeja), pilohikameja (dipakai sebagai kemeja)", "Adaptasi vokal e -> a pada suku awal menjadi kameja dan lekati prefiks mohi-/pilohi-."),
        ("polisi", "pulisi", "mopulisi (melapor ke polisi)", "Adaptasi vokal o -> u pada suku kata pertama."),
        ("perkara", "parakara", "moparakara (berperkara)", "Sisipkan vokal /a/ untuk memecah gugus konsonan -rk- menjadi parakara."),
        ("sabun", "sabongi", "mosabongi (menyabuni), sabongiyolo (disabuni)", "Ubah konsonan nasal akhir -n menjadi -ngi sesuai fonotaktik vokalis Gorontalo."),
        ("botol", "botulo", "ngobotulo (sebotol)", "Adaptasi akhiran -ol menjadi -ulo."),
        ("kantor", "kantori", "mokantori (berkantor)", "Tambahkan vokal akhir -i setelah konsonan /r/."),
        ("kabar", "habari", "mohabari (mengabarkan), habariyolo (dikabarkan)", "Adaptasi /k/ -> /h/ dan tambahkan vokal akhir -i."),
        ("pikir", "pikiri", "mopikiri (memikirkan), pikiriyolo (dipikirkan)", "Tambahkan vokal akhir -i setelah konsonan /r/."),
        ("ukur", "u:kuru", "mou:kuru (mengukur), u:kuruyolo (diukur)", "Tambahkan vokal akhir -u selaras vokal suku sebelumnya."),
        ("atur", "a:turu", "moa:turu (mengatur), a:turuyolo (diatur)", "Tambahkan vokal akhir -u selaras vokal suku sebelumnya."),
        ("kawin", "kawini", "mokawini (menikah), mopokawini (menikahkan)", "Tambahkan vokal akhir -i setelah konsonan /n/."),
        ("jamin", "jamini", "mojamini (menjamin), jaminiyolo (dijamin)", "Tambahkan vokal akhir -i setelah konsonan /n/."),
        ("cetak", "cetaki", "mocetaki (mencetak), cetakiylo (dicetak)", "Tambahkan vokal akhir -i setelah konsonan /k/."),
    ]
    for id_w, gor_w, deriv, rule in curated_loans:
        loan_records.append({
            "indonesia": id_w,
            "gorontalo": gor_w,
            "turunan_contoh": deriv,
            "kaidah": rule,
        })
        seen.add((id_w.lower(), gor_w.lower()))

    # 2. Ekstraksi otomatis dari leksikon gabungan
    for _, r in lex_df.iterrows():
        id_w = clean_indonesian_text(str(r["indonesia"])).lower()
        gor_w = str(r["gorontalo"]).strip().lower()
        # Ambil entri kata tunggal atau kata pertama sebelum koma
        if "," in id_w:
            id_w = id_w.split(",")[0].strip()
        if "," in gor_w:
            gor_w = gor_w.split(",")[0].strip()
        if " " not in id_w and " " not in gor_w and len(id_w) >= 4 and len(gor_w) >= 4:
            gor_clean = gor_w.replace(":", "")
            sim = SequenceMatcher(None, id_w, gor_clean).ratio()
            if sim >= 0.72 and id_w != gor_clean and gor_clean[-1] in "aiueo":
                key = (id_w, gor_w)
                if key not in seen:
                    seen.add(key)
                    # Tentukan pola fonologisasi
                    if id_w[-1] not in "aiueo" and gor_clean.startswith(id_w[:3]):
                        rule = f"Kata Indonesia '{id_w}' berakhir dengan konsonan '{id_w[-1]}', sehingga diadaptasi ke sifat vokalis Bahasa Gorontalo dengan akhiran vokal '-{gor_clean[-1]}' menjadi '{gor_w}'."
                    else:
                        rule = f"Adaptasi fonotaktik vokalis Bahasa Gorontalo dari kata '{id_w}' menjadi '{gor_w}' (berakhiran vokal terbuka '-{gor_clean[-1]}')."
                    loan_records.append({
                        "indonesia": id_w,
                        "gorontalo": gor_w,
                        "turunan_contoh": f"mo{gor_w} / {gor_w}yolo" if not gor_w.startswith("mo") else gor_w,
                        "kaidah": rule,
                    })

    # 3. Ekstraksi dari morfologi Indonesia-Gorontalo 1987
    for _, r in mor_ig_df.iterrows():
        akar_id = clean_indonesian_text(str(r["akar_indonesia"])).lower()
        tur_id = clean_indonesian_text(str(r["turunan_indonesia"])).lower()
        tur_gor = str(r["padanan_turunan_gorontalo"]).strip().lower()
        if " " not in akar_id and len(akar_id) >= 4 and len(tur_gor) >= 5:
            gor_clean = tur_gor.replace(":", "")
            if akar_id[:4] in gor_clean:
                key = (tur_id, tur_gor)
                if key not in seen:
                    seen.add(key)
                    loan_records.append({
                        "indonesia": f"{akar_id} -> {tur_id}",
                        "gorontalo": tur_gor,
                        "turunan_contoh": f"{r['contoh_indonesia']} = {r['contoh_gorontalo']}" if r["contoh_indonesia"] else tur_gor,
                        "kaidah": f"Kata dasar '{akar_id}' diserap secara vokalis dan dilekati imbuhan produktif Bahasa Gorontalo untuk membentuk '{tur_gor}' ('{tur_id}').",
                    })

    return loan_records


def build_grammar_rule_instructions() -> list[dict]:
    """
    Siasat A & C: Instruksi eksplisit Kaidah Tata Bahasa, 5 Kala, Morfofonemik,
    17 Kata Bantu Bilangan, MPA (mayi, mao, mola, mota), dan Ortografi Pateda.
    """
    with open(BASE_DIR / "gorontaloindonesia/dokumen_pengantar/07_tata_bahasa_dan_kaidah_linguistik.json") as f:
        gi_rules = json.load(f)
    with open(BASE_DIR / "indonesiagorontalo/dokumen_pengantar/04_kaidah_linguistik_dan_konversi_ejaan.json") as f:
        ig_rules = json.load(f)

    samples = []

    # 1. 5 Sistem Kala Bahasa Gorontalo
    samples.append({
        "task": "grammar_rule_gorontalo",
        "instruction": "Jelaskan 5 sistem kala (tense/aspect) dalam Bahasa Gorontalo menurut Prof. Dr. H. Mansoer Pateda beserta penanda morfologis dan contohnya.",
        "input": "Sistem Kala Bahasa Gorontalo",
        "output": (
            "Dalam Bahasa Gorontalo terdapat 5 sistem kala (waktu/aspek) yang wajib diperhatikan saat menerjemahkan kata kerja:\n"
            "1. Kala Akan: ditandai awalan berfonem /m/ (mo-, mopo-, moti-), akhiran pasif -lo, atau sisipan -um-. Contoh: modelo (akan membawa), deloolo (akan dibawa), tumuoto (akan masuk).\n"
            "2. Kala Sedang Masih: ditandai awalan hemo- atau hemoti-. Contoh: hemomake (sedang masih berpakaian), hemotihuloo (sedang masih mendudukkan diri).\n"
            "3. Kala Sedang: ditandai reduplikasi (kata ulang). Contoh: bala-balato (sedang berguling), hulo-huloo (sedang duduk), tihu-tihulo (sedang berdiri).\n"
            "4. Kala Sedang Lampau: ditandai awalan helo- atau heloti-. Contoh: helodelo (telah sedang membawa), helotihuloo (telah sedang duduk).\n"
            "5. Kala Lampau: ditandai awalan berfonem /l/ (lo-, lopo-, loti-, lee-) atau sisipan pasif lampau -il-. Contoh: lodelo (telah membawa), dilelo (telah dibawa), leehuluto (telah tergelincir)."
        ),
        "sumber_kamus": "kaidah_pateda_terpadu",
    })

    # 2. Morfem Penunjuk Arah (MPA - Siasat C)
    samples.append({
        "task": "grammar_rule_gorontalo",
        "instruction": "Jelaskan 4 Morfem Penunjuk Arah (MPA) dalam Bahasa Gorontalo beserta aturan penulisannya terhadap kata kerja.",
        "input": "Morfem Penunjuk Arah (mayi, mao, mola, mota)",
        "output": (
            "Terdapat 4 Morfem Penunjuk Arah (MPA) dalam Bahasa Gorontalo:\n"
            "1. mayi (mai): menuju pembicara ('ke sini / datang'), contoh: mayi mohama (datang mengambil / menjemput).\n"
            "2. mao (ma?o): menjauh dari pembicara ke samping kiri/kanan ('ke situ / pergi'), contoh: mao modepito (pergi mengantar).\n"
            "3. mola: menjauh dari pembicara ke arah utara/udik ('ke sana [udik] / pergi'), contoh: mola motali (pergi membeli).\n"
            "4. mota: menjauh dari pembicara ke arah selatan/hilir ('ke sana [hilir] / pergi'), contoh: mota mosikola (pergi bersekolah).\n"
            "Aturan Penulisan: MPA ditulis TERPISAH jika berada di depan kata kerja/lokasi (mis. 'mayi mohama'), tetapi ditulis SERANGKAI jika mengikuti kata kerja (mis. 'deloamayi' = bawalah ke sini, 'deloamao' = bawalah ke situ)."
        ),
        "sumber_kamus": "kaidah_pateda_terpadu",
    })

    # 3. Morfofonemik Peluluhan Konsonan & Pergeseran Vokal -o -> -a
    samples.append({
        "task": "grammar_rule_gorontalo",
        "instruction": "Bagaimana kaidah morfofonemik peluluhan konsonan awal dan perubahan vokal akhir -o menjadi -a dalam Bahasa Gorontalo?",
        "input": "Kaidah Morfofonemik Bahasa Gorontalo",
        "output": (
            "1. Peluluhan Konsonan Awal saat dilekati prefiks aktif/lampau (mo-/lo-/po-):\n"
            "   - Fonem /t/ luluh menjadi /l/: mo- + tuladu -> moluladu (akan menulis), lo- + tuladu -> loluladu (telah menulis).\n"
            "   - Fonem /s/ luluh menjadi /n/: mo- + sadiya -> monadiya (menyiapkan).\n"
            "   - Fonem /p/ luluh menjadi /m/: mo- + pututo -> momututo (memungut).\n"
            "   - Fonem /k/ atau vokal awal mendapat sengau /ng/: mo- + aato -> mongaato (menyapu).\n"
            "2. Pergeseran Vokal Akhir -o menjadi -a:\n"
            "   - Kata dasar atau kata berakhiran -o berubah menjadi -a ketika diikuti oleh akhiran (-lo, -po) atau kata penghubung 'lo' dalam frasa.\n"
            "   - Contoh tingkat kata: aato + -lo -> aatalo (disapu).\n"
            "   - Contoh tingkat frasa: lobuolo + lo didingga -> lobuola lo didingga (ditumbuk dengan alu)."
        ),
        "sumber_kamus": "kaidah_pateda_terpadu",
    })

    # 4. Pronomina & Akhiran Persona (Biasa vs Hormat)
    samples.append({
        "task": "grammar_rule_gorontalo",
        "instruction": "Sebutkan sistem kata ganti orang (pronomina) dan akhiran persona tingkat tutur Biasa vs Hormat dalam Bahasa Gorontalo.",
        "input": "Pronomina dan Akhiran Persona Bahasa Gorontalo",
        "output": (
            "Sistem Pronomina dan Akhiran Persona Bahasa Gorontalo membedakan tingkat tutur Biasa (akrab) dan Hormat (sopan):\n"
            "1. Orang I Tunggal ('aku/saya'): Bebas Biasa = wau (wa?u), Bebas Hormat = wa:tiya (waatia); Akhiran Biasa = -u (-?u), Akhiran Hormat = -la:tiya (-laatia).\n"
            "2. Orang I Jamak Eksklusif ('kami'): Bebas Biasa = ami, Bebas Hormat = amiya:tiya; Akhiran Biasa = -lami, Akhiran Hormat = -lamiya:tiya.\n"
            "3. Orang I Jamak Inklusif ('kita'): Bebas = ito / itowali; Akhiran = -nto (-ndo).\n"
            "4. Orang II Tunggal ('engkau/Anda/Bapak/Ibu'): Bebas Biasa = yio (yi?o), Bebas Hormat = ito; Akhiran Biasa = -mu, Akhiran Hormat = -nto (-ndo).\n"
            "5. Orang II Jamak ('kalian/kamu sekalian'): Bebas = timongoli; Akhiran = -limongoli.\n"
            "6. Orang III Tunggal ('ia/dia'): Bebas = tiyo (tio); Akhiran = -liyo (-lio).\n"
            "7. Orang III Jamak ('mereka'): Bebas = timongoliyo (timongolio); Akhiran = -limongoliyo."
        ),
        "sumber_kamus": "kaidah_pateda_terpadu",
    })

    # 5. Setiap Awalan Berpasangan & Tak Berpasangan
    for item in gi_rules["morfologi"]["awalan_berpasangan"]:
        samples.append({
            "task": "grammar_rule_gorontalo",
            "instruction": f"Jelaskan fungsi pasangan imbuhan awalan '{item['aktif']} / {item['lampau']} / {item['imperatif']}' dalam Bahasa Gorontalo.",
            "input": f"Prefiks {item['aktif']}, {item['lampau']}, {item['imperatif']}",
            "output": f"Pasangan awalan {item['aktif']} (kala akan/aktif), {item['lampau']} (kala lampau), dan {item['imperatif']} (imperatif/pelaku/pasif) berfungsi untuk menyatakan: {item['fungsi']}.",
            "sumber_kamus": "gorontalo_indonesia_2001",
        })

    # 6. 17 Penggolong Numeralia (Kata Bantu Bilangan)
    for num in gi_rules["penggolong_numeralia"]:
        samples.append({
            "task": "grammar_rule_gorontalo",
            "instruction": f"Apa kata bantu bilangan (penggolong numeralia) Bahasa Gorontalo untuk kategori '{num['kategori']}' beserta contohnya?",
            "input": f"Kategori benda: {num['kategori']}",
            "output": f"Kata bantu bilangan untuk '{num['kategori']}' adalah '{num['penggolong']}' (ngo{num['penggolong']} = satu {num['kategori']}). Contoh: {num['contoh']}.",
            "sumber_kamus": "gorontalo_indonesia_2001",
        })

    # 7. Jembatan Ortografi 1987 vs 2001
    samples.append({
        "task": "grammar_rule_gorontalo",
        "instruction": "Jelaskan perbedaan dan konversi antara Ejaan Kamus Pateda 1987 (Indonesia-Gorontalo) dan Ejaan Kamus Pateda 2001 (Gorontalo-Indonesia).",
        "input": "Konversi Ortografi Pateda 1987 dan 2001",
        "output": (
            "Terdapat 3 korespondensi sistematis antara Ejaan Pateda 1987 dan Ejaan Pateda 2001:\n"
            "1. Vokal Panjang: Ejaan 1987 menggunakan vokal ganda (aa, ii, uu, ee, oo seperti 'paatali, waatia, diila'), sedangkan Ejaan 2001 menggunakan titik dua (a:, i:, u:, e:, o: seperti 'pa:tali, wa:tiya, di:la').\n"
            "2. Hentian Glotal (/ʔ/): Ejaan 1987 menggunakan tanda tanya '?' ('wa?u, yi?o, ma?o, lo?otapu'), sedangkan Ejaan 2001 tidak menulis tanda tanya glotal ('wau, yio, mao, lootapu').\n"
            "3. Bunyi Luncuran (Glide /y/ dan /w/): Ejaan 1987 tidak menuliskan luncuran ('tio, boito, botia, mai, wau'), sedangkan Ejaan 2001 menuliskannya secara eksplisit ('tiyo, boyito, botiya, mayi, wawu')."
        ),
        "sumber_kamus": "kaidah_pateda_terpadu",
    })

    return samples


# Siasat D: Kamus Jembatan Sinonim Bahasa Indonesia (KBBI) & Register Tutur
SYNONYM_SUBSTITUTIONS = [
    (r"\btidak\b", ["tak", "tiada"]),
    (r"\bkarena\b", ["sebab", "lantaran"]),
    (r"\bsebab\b", ["karena", "lantaran"]),
    (r"\buntuk\b", ["bagi", "guna"]),
    (r"\bmelihat\b", ["memandang", "menyaksikan"]),
    (r"\bpergi\b", ["berangkat", "melangkah"]),
    (r"\bdatang\b", ["tiba", "hadir"]),
    (r"\bsangat\b", ["amat", "sungguh", "terlalu"]),
    (r"\bterlalu\b", ["amat", "sangat"]),
    (r"\bkalau\b", ["jika", "apabila", "bila"]),
    (r"\bjika\b", ["kalau", "apabila"]),
    (r"\btetapi\b", ["namun", "akan tetapi"]),
    (r"\bagar\b", ["supaya"]),
    (r"\bsupaya\b", ["agar"]),
    (r"\bmeskipun\b", ["walaupun", "kendatipun"]),
    (r"\bsudah\b", ["telah"]),
    (r"\btelah\b", ["sudah"]),
    (r"\bsedang\b", ["tengah", "lagi"]),
    (r"\bengkau\b", ["kamu", "kau"]),
    (r"\baku\b", ["saya"]),
    (r"\bia\b", ["dia"]),
    (r"\bdia\b", ["ia"]),
    (r"\bmempunyai\b", ["memiliki"]),
    (r"\bmendapat\b", ["memperoleh"]),
    (r"\bmembawa\b", ["mengangkut"]),
    (r"\bmembuat\b", ["membikin", "mengerjakan"]),
    (r"\bberkata\b", ["berujar", "berucap"]),
    (r"\bpandai\b", ["pintar", "cerdas"]),
    (r"\bsenang\b", ["gembira", "bahagia"]),
    (r"\bmarah\b", ["murka", "gusar"]),
    (r"\bcepat\b", ["lekas", "segera"]),
    (r"\brumah\b", ["tempat tinggal", "kediaman"]),
]


def generate_synonym_bridged_pairs(df_par: pd.DataFrame) -> list[dict]:
    """
    Siasat D: Menghasilkan variasi sinonim KBBI pada sisi Bahasa Indonesia (src_indonesia)
    yang memetakan ke target kalimat Bahasa Gorontalo yang sama, termasuk pembedaan
    register Biasa (wau/yio) vs Hormat (wa:tiya/ito).
    Hanya dipanggil pada split TRAIN agar 0% data leakage ke val/test!
    """
    augmented = []
    for _, row in df_par.iterrows():
        src = row["src_indonesia"]
        tgt = row["tgt_gorontalo"]
        sumber = row["sumber_kamus"]

        # Cek register Hormat vs Biasa berdasarkan pronomina Gorontalo di target
        tgt_tokens = set(re.findall(r"[a-z:?]+", tgt.lower()))
        if "wa:tiya" in tgt_tokens or "waatia" in tgt_tokens:
            # Jika target memakai wa:tiya (Hormat), pastikan varian formal 'saya'
            if re.search(r"\baku\b", src, flags=re.IGNORECASE):
                new_src = re.sub(r"\baku\b", "saya (hormat)", src, flags=re.IGNORECASE)
                augmented.append({
                    "src_indonesia": new_src,
                    "tgt_gorontalo": tgt,
                    "sumber_kamus": sumber,
                    "aug_type": "register_hormat",
                })
        elif "wau" in tgt_tokens or "wa?u" in tgt_tokens:
            if re.search(r"\bsaya\b", src, flags=re.IGNORECASE):
                new_src = re.sub(r"\bsaya\b", "aku", src, flags=re.IGNORECASE)
                augmented.append({
                    "src_indonesia": new_src,
                    "tgt_gorontalo": tgt,
                    "sumber_kamus": sumber,
                    "aug_type": "register_biasa",
                })

        # Substitusi sinonim KBBI
        for pat, syns in SYNONYM_SUBSTITUTIONS:
            if re.search(pat, src, flags=re.IGNORECASE):
                chosen_syn = random.choice(syns)
                new_src = re.sub(pat, chosen_syn, src, count=1, flags=re.IGNORECASE)
                if new_src.lower() != src.lower():
                    augmented.append({
                        "src_indonesia": new_src,
                        "tgt_gorontalo": tgt,
                        "sumber_kamus": sumber,
                        "aug_type": "synonym_kbbi",
                    })
                    break  # 1 variasi sinonim berkualitas tinggi per kalimat

    return augmented


def extract_mpa_analytical_samples(df_par: pd.DataFrame) -> list[dict]:
    """
    Siasat C: Mengekstrak kalimat yang menggunakan Morfem Penunjuk Arah (MPA: mayi, mao, mola, mota)
    atau konstruksi analitis sebagai tugas penalaran semantik & arah kegiatan.
    """
    mpa_samples = []
    mpa_pattern = re.compile(r"\b(mayi|mai|mao|ma\?o|mola|mota)\b", re.IGNORECASE)
    for _, row in df_par.iterrows():
        tgt = row["tgt_gorontalo"]
        src = row["src_indonesia"]
        m = mpa_pattern.search(tgt)
        if m:
            mpa_word = m.group(1).lower()
            arah_desc = {
                "mayi": "menuju pembicara (ke sini / datang)",
                "mai": "menuju pembicara (ke sini / datang)",
                "mao": "menjauh ke samping/tempat dekat (ke situ / pergi)",
                "ma?o": "menjauh ke samping/tempat dekat (ke situ / pergi)",
                "mola": "menjauh ke arah udik/utara (ke sana / pergi)",
                "mota": "menjauh ke arah hilir/selatan (ke sana / pergi)",
            }.get(mpa_word, "arah kegiatan")
            mpa_samples.append({
                "task": "analytical_paraphrase_mpa",
                "instruction": "Terjemahkan kalimat berikut ke dalam Bahasa Gorontalo dengan memperhatikan Morfem Penunjuk Arah (MPA: mayi, mao, mola, mota) dan kala kata kerjanya.",
                "input": src,
                "output": f"{tgt} (MPA '{mpa_word}': {arah_desc})",
                "sumber_kamus": row["sumber_kamus"],
            })
    return mpa_samples


def main():
    print("Memuat korpus terpadu dari:", OUT_DIR)
    df_par = pd.read_parquet(OUT_DIR / "combined_parallel_sentences.parquet")
    df_lex = pd.read_parquet(OUT_DIR / "combined_lexicon_id_to_gor.parquet")
    df_mor_gi = pd.read_parquet(BASE_DIR / "gorontaloindonesia/dataset/training_ready/train_morphology_grammar.parquet")
    df_mor_ig = pd.read_parquet(BASE_DIR / "indonesiagorontalo/dataset/training_ready/train_morphology_grammar.parquet")

    # 1. Bersihkan singkatan kamus pada kalimat paralel dan leksikon
    df_par["src_indonesia"] = df_par["src_indonesia"].apply(clean_indonesian_text)
    df_par["tgt_gorontalo"] = [
        clean_gorontalo_text(g, s) for g, s in zip(df_par["tgt_gorontalo"], df_par["src_indonesia"])
    ]
    df_par = df_par[(df_par["src_indonesia"].str.len() > 1) & (df_par["tgt_gorontalo"].str.len() > 1)].drop_duplicates(
        subset=["src_indonesia", "tgt_gorontalo"]
    ).reset_index(drop=True)

    df_lex["indonesia"] = df_lex["indonesia"].apply(clean_indonesian_text)
    df_lex["gorontalo"] = df_lex["gorontalo"].str.strip()
    df_lex = df_lex[(df_lex["indonesia"].str.len() > 0) & (df_lex["gorontalo"].str.len() > 0)].drop_duplicates(
        subset=["indonesia", "gorontalo"]
    ).reset_index(drop=True)

    n_par_before, n_lex_before = len(df_par), len(df_lex)
    df_par = df_par[[is_clean_parallel(s, t) for s, t in zip(df_par["src_indonesia"], df_par["tgt_gorontalo"])]].reset_index(drop=True)
    df_lex = df_lex[[is_clean_lexicon(i, g) for i, g in zip(df_lex["indonesia"], df_lex["gorontalo"])]].reset_index(drop=True)
    df_mor_ig = df_mor_ig[[
        is_clean_lexicon(str(a), str(b)) and is_clean_lexicon(str(c), str(d))
        for a, b, c, d in zip(df_mor_ig["akar_indonesia"], df_mor_ig["padanan_akar_gorontalo"],
                              df_mor_ig["turunan_indonesia"], df_mor_ig["padanan_turunan_gorontalo"])
    ]].reset_index(drop=True)
    print(f"Filter kualitas -> paralel {n_par_before}->{len(df_par)}, leksikon {n_lex_before}->{len(df_lex)}, morfologi IG -> {len(df_mor_ig)}")

    print(f"Kalimat paralel bersih unik: {len(df_par)}")
    print(f"Entri leksikon bersih unik : {len(df_lex)}")

    # 2. Stratified Split pada Kalimat Paralel (90% Train, 5% Val, 5% Test) berdasarkan sumber_kamus
    train_parts, val_parts, test_parts = [], [], []
    for sumber, grp in df_par.groupby("sumber_kamus"):
        grp_shuf = grp.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
        n = len(grp_shuf)
        n_test = int(n * 0.05)
        n_val = int(n * 0.05)
        test_parts.append(grp_shuf.iloc[:n_test])
        val_parts.append(grp_shuf.iloc[n_test : n_test + n_val])
        train_parts.append(grp_shuf.iloc[n_test + n_val :])

    par_train = pd.concat(train_parts, ignore_index=True)
    par_val = pd.concat(val_parts, ignore_index=True)
    par_test = pd.concat(test_parts, ignore_index=True)
    print(f"Split Kalimat Paralel -> Train: {len(par_train)}, Val: {len(par_val)}, Test: {len(par_test)}")

    # 3. Terapkan Siasat D (Jembatan Sinonim KBBI & Register) HANYA pada par_train
    syn_pairs = generate_synonym_bridged_pairs(par_train)
    df_syn = pd.DataFrame(syn_pairs)
    print(f"Siasat D (Augmentasi Sinonim KBBI & Register pada Train): +{len(df_syn)} pasangan kalimat")

    # 4. Terapkan Siasat B (Fonologisasi Kata Serapan BI -> BG)
    loan_records = extract_loanwords(df_lex, df_mor_ig)
    print(f"Siasat B (Fonologisasi Kata Serapan Vokalis BI -> BG): +{len(loan_records)} pasangan serapan")

    # 5. Terapkan Siasat A & C (Kaidah Tata Bahasa, 5 Kala, Morfofonemik, MPA)
    grammar_rules = build_grammar_rule_instructions()
    mpa_samples = extract_mpa_analytical_samples(par_train)
    print(f"Siasat A (Kaidah Eksplisit & 5.039 Morfologi): +{len(grammar_rules)} instruksi kaidah + {len(df_mor_gi)+len(df_mor_ig)} morfologi")
    print(f"Siasat C (Dekomposisi Semantik & MPA): +{len(mpa_samples)} sampel analitis MPA")

    # 6. Bangun Dataset Seq2Seq Dua Arah (ind_Latn <-> gor_Latn) untuk NLLB-200
    def make_seq2seq_df(par_split: pd.DataFrame, include_lex_and_syn: bool = False) -> pd.DataFrame:
        rows = []
        for _, r in par_split.iterrows():
            # Arah 1: Indonesia -> Gorontalo (Ejaan Baku 2001)
            rows.append({
                "src_lang": "ind_Latn",
                "tgt_lang": "gor_Latn",
                "src_text": r["src_indonesia"],
                "tgt_text": r["tgt_gorontalo"],
                "type": "parallel_sentence",
                "sumber_kamus": r["sumber_kamus"],
            })
            # Arah 2: Gorontalo (Ejaan 2001) -> Indonesia
            rows.append({
                "src_lang": "gor_Latn",
                "tgt_lang": "ind_Latn",
                "src_text": r["tgt_gorontalo"],
                "tgt_text": r["src_indonesia"],
                "type": "parallel_sentence",
                "sumber_kamus": r["sumber_kamus"],
            })
            # Jika ejaan asli 1987 berbeda dari ejaan 2001, tambahkan input ejaan 1987 -> Indonesia agar model memahami kedua ejaan!
            if (
                isinstance(r.get("tgt_gorontalo_asli"), str)
                and r["tgt_gorontalo_asli"].strip()
                and r["tgt_gorontalo_asli"].strip() != r["tgt_gorontalo"].strip()
            ):
                rows.append({
                    "src_lang": "gor_Latn",
                    "tgt_lang": "ind_Latn",
                    "src_text": r["tgt_gorontalo_asli"].strip(),
                    "tgt_text": r["src_indonesia"],
                    "type": "parallel_sentence_orth1987",
                    "sumber_kamus": r["sumber_kamus"],
                })

        if include_lex_and_syn:
            # Tambahkan Sinonim KBBI (Siasat D) ke Train Seq2Seq
            for _, r in df_syn.iterrows():
                rows.append({
                    "src_lang": "ind_Latn",
                    "tgt_lang": "gor_Latn",
                    "src_text": r["src_indonesia"],
                    "tgt_text": r["tgt_gorontalo"],
                    "type": f"synonym_{r['aug_type']}",
                    "sumber_kamus": r["sumber_kamus"],
                })
            # Tambahkan Leksikon Dua Arah ke Train Seq2Seq
            for _, r in df_lex.iterrows():
                rows.append({
                    "src_lang": "ind_Latn",
                    "tgt_lang": "gor_Latn",
                    "src_text": r["indonesia"],
                    "tgt_text": r["gorontalo"],
                    "type": "lexicon",
                    "sumber_kamus": r["sumber_kamus"],
                })
                rows.append({
                    "src_lang": "gor_Latn",
                    "tgt_lang": "ind_Latn",
                    "src_text": r["gorontalo"],
                    "tgt_text": r["indonesia"],
                    "type": "lexicon",
                    "sumber_kamus": r["sumber_kamus"],
                })
                if (
                    isinstance(r.get("gorontalo_asli"), str)
                    and r["gorontalo_asli"].strip()
                    and r["gorontalo_asli"].strip() != r["gorontalo"].strip()
                ):
                    rows.append({
                        "src_lang": "gor_Latn",
                        "tgt_lang": "ind_Latn",
                        "src_text": r["gorontalo_asli"].strip(),
                        "tgt_text": r["indonesia"],
                        "type": "lexicon_orth1987",
                        "sumber_kamus": r["sumber_kamus"],
                    })
            # Tambahkan Pasangan Morfologi (Siasat A) ke Train Seq2Seq
            for _, r in df_mor_gi.iterrows():
                if r["arti_turunan"] and r["kata_turunan"]:
                    rows.append({
                        "src_lang": "ind_Latn",
                        "tgt_lang": "gor_Latn",
                        "src_text": clean_indonesian_text(r["arti_turunan"]),
                        "tgt_text": r["kata_turunan"].strip(),
                        "type": "morphology",
                        "sumber_kamus": "gorontalo_indonesia_2001",
                    })
                    rows.append({
                        "src_lang": "gor_Latn",
                        "tgt_lang": "ind_Latn",
                        "src_text": r["kata_turunan"].strip(),
                        "tgt_text": clean_indonesian_text(r["arti_turunan"]),
                        "type": "morphology",
                        "sumber_kamus": "gorontalo_indonesia_2001",
                    })
            for _, r in df_mor_ig.iterrows():
                if r["turunan_indonesia"] and r["padanan_turunan_gorontalo"]:
                    rows.append({
                        "src_lang": "ind_Latn",
                        "tgt_lang": "gor_Latn",
                        "src_text": clean_indonesian_text(r["turunan_indonesia"]),
                        "tgt_text": r["padanan_turunan_gorontalo"].strip(),
                        "type": "morphology",
                        "sumber_kamus": "indonesia_gorontalo_1987",
                    })
                    rows.append({
                        "src_lang": "gor_Latn",
                        "tgt_lang": "ind_Latn",
                        "src_text": r["padanan_turunan_gorontalo"].strip(),
                        "tgt_text": clean_indonesian_text(r["turunan_indonesia"]),
                        "type": "morphology",
                        "sumber_kamus": "indonesia_gorontalo_1987",
                    })

        return pd.DataFrame(rows)

    seq2seq_train = make_seq2seq_df(par_train, include_lex_and_syn=True).sample(
        frac=1.0, random_state=RANDOM_SEED
    ).reset_index(drop=True)
    seq2seq_val = make_seq2seq_df(par_val, include_lex_and_syn=False)
    seq2seq_test = make_seq2seq_df(par_test, include_lex_and_syn=False)

    seq2seq_train.to_parquet(OUT_DIR / "seq2seq_train.parquet", index=False)
    seq2seq_val.to_parquet(OUT_DIR / "seq2seq_val.parquet", index=False)
    seq2seq_test.to_parquet(OUT_DIR / "seq2seq_test.parquet", index=False)
    print(f"Seq2Seq Dataset disimpan -> Train: {len(seq2seq_train)}, Val: {len(seq2seq_val)}, Test: {len(seq2seq_test)}")

    # 7. Bangun Dataset Instruksi SFT Multi-Tugas untuk LLM (Qwen2.5-7B-Instruct, dll.)
    def make_sft_from_parallel(par_split: pd.DataFrame) -> list[dict]:
        sft_rows = []
        for _, r in par_split.iterrows():
            sft_rows.append({
                "task": "translate_id_to_gor",
                "instruction": "Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Gorontalo yang baik dan benar sesuai kaidah Prof. Mansoer Pateda.",
                "input": r["src_indonesia"],
                "output": r["tgt_gorontalo"],
                "sumber_kamus": r["sumber_kamus"],
            })
            sft_rows.append({
                "task": "translate_gor_to_id",
                "instruction": "Terjemahkan kalimat bahasa Gorontalo berikut ke dalam bahasa Indonesia yang baik dan benar.",
                "input": r["tgt_gorontalo"],
                "output": r["src_indonesia"],
                "sumber_kamus": r["sumber_kamus"],
            })
            if (
                isinstance(r.get("tgt_gorontalo_asli"), str)
                and r["tgt_gorontalo_asli"].strip()
                and r["tgt_gorontalo_asli"].strip() != r["tgt_gorontalo"].strip()
            ):
                sft_rows.append({
                    "task": "translate_gor1987_to_id",
                    "instruction": "Terjemahkan kalimat bahasa Gorontalo (ejaan kamus 1987) berikut ke dalam bahasa Indonesia.",
                    "input": r["tgt_gorontalo_asli"].strip(),
                    "output": r["src_indonesia"],
                    "sumber_kamus": r["sumber_kamus"],
                })
        return sft_rows

    sft_val_rows = make_sft_from_parallel(par_val)
    sft_test_rows = make_sft_from_parallel(par_test)
    sft_train_rows = make_sft_from_parallel(par_train)

    # Tambahkan Siasat D (Sinonim KBBI & Register) ke SFT Train
    for _, r in df_syn.iterrows():
        sft_train_rows.append({
            "task": "translate_id_to_gor_synonym",
            "instruction": "Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Gorontalo yang tepat.",
            "input": r["src_indonesia"],
            "output": r["tgt_gorontalo"],
            "sumber_kamus": r["sumber_kamus"],
        })

    # Tambahkan Leksikon Dua Arah (ID -> GOR dan GOR -> ID) ke SFT Train
    for _, r in df_lex.iterrows():
        sft_train_rows.append({
            "task": "lexicon_id_to_gor",
            "instruction": "Apa kosakata atau istilah bahasa Gorontalo untuk kata/makna bahasa Indonesia berikut?",
            "input": r["indonesia"],
            "output": r["gorontalo"],
            "sumber_kamus": r["sumber_kamus"],
        })
        sft_train_rows.append({
            "task": "lexicon_gor_to_id",
            "instruction": "Apa arti kata atau istilah bahasa Gorontalo berikut dalam bahasa Indonesia?",
            "input": r["gorontalo"],
            "output": r["indonesia"],
            "sumber_kamus": r["sumber_kamus"],
        })

    # Tambahkan Siasat A: 5.039 Pasangan Morfologi + Kaidah Eksplisit ke SFT Train
    for _, r in df_mor_gi.iterrows():
        contoh_str = f" | Contoh: {r['contoh_kalimat']} = {clean_indonesian_text(r['arti_kalimat'])}" if r["contoh_kalimat"] else ""
        sft_train_rows.append({
            "task": "morphology_derivation",
            "instruction": f"Bentuklah kata turunan bahasa Gorontalo dari akar kata '{r['akar_kata']}' beserta artinya dalam bahasa Indonesia.",
            "input": f"Akar kata: {r['akar_kata']} -> Makna sasaran: {clean_indonesian_text(r['arti_turunan'])}",
            "output": f"Kata turunan: {r['kata_turunan']} (Artinya: {clean_indonesian_text(r['arti_turunan'])}){contoh_str}",
            "sumber_kamus": "gorontalo_indonesia_2001",
        })
    for _, r in df_mor_ig.iterrows():
        contoh_str = f" | Contoh: {clean_indonesian_text(r['contoh_indonesia'])} = {r['contoh_gorontalo']}" if r["contoh_indonesia"] else ""
        sft_train_rows.append({
            "task": "morphology_id_to_gor",
            "instruction": f"Terjemahkan kata turunan bahasa Indonesia '{clean_indonesian_text(r['turunan_indonesia'])}' (dari kata dasar '{clean_indonesian_text(r['akar_indonesia'])}') ke dalam bahasa Gorontalo.",
            "input": f"Kata dasar: {clean_indonesian_text(r['akar_indonesia'])} ({r['padanan_akar_gorontalo']}) -> Kata turunan: {clean_indonesian_text(r['turunan_indonesia'])}",
            "output": f"Padanan Gorontalo: {r['padanan_turunan_gorontalo']}{contoh_str}",
            "sumber_kamus": "indonesia_gorontalo_1987",
        })

    sft_train_rows.extend(grammar_rules)

    # Tambahkan Siasat B: Fonologisasi Kata Serapan ke SFT Train
    for item in loan_records:
        sft_train_rows.append({
            "task": "loanword_phonologization",
            "instruction": "Adaptasikan kata bahasa Indonesia berikut ke dalam bahasa Gorontalo sesuai kaidah fonotaktik vokalis Prof. Mansoer Pateda beserta contoh imbuhannya.",
            "input": item["indonesia"],
            "output": f"Bentuk serapan Gorontalo: {item['gorontalo']} | Contoh turunan: {item['turunan_contoh']} | Kaidah: {item['kaidah']}",
            "sumber_kamus": "siasat_b_pateda",
        })

    # Tambahkan Siasat C: Dekomposisi Semantik & MPA ke SFT Train
    sft_train_rows.extend(mpa_samples)

    df_sft_train = pd.DataFrame(sft_train_rows).sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    df_sft_val = pd.DataFrame(sft_val_rows)
    df_sft_test = pd.DataFrame(sft_test_rows)

    df_sft_train.to_parquet(OUT_DIR / "sft_augmented_train.parquet", index=False)
    df_sft_val.to_parquet(OUT_DIR / "sft_augmented_val.parquet", index=False)
    df_sft_test.to_parquet(OUT_DIR / "sft_augmented_test.parquet", index=False)

    print(f"SFT Augmented Dataset disimpan -> Train: {len(df_sft_train)}, Val: {len(df_sft_val)}, Test: {len(df_sft_test)}")
    print("\nDistribusi Tugas pada SFT Augmented Train:")
    print(df_sft_train["task"].value_counts())

    # Simpan ringkasan statistik pengayaan
    stats = {
        "parallel_clean_total": len(df_par),
        "parallel_splits": {
            "train": len(par_train),
            "val": len(par_val),
            "test": len(par_test),
        },
        "siasat_augmentation_counts": {
            "siasat_a_morphology_and_grammar": len(df_mor_gi) + len(df_mor_ig) + len(grammar_rules),
            "siasat_b_loanword_phonologization": len(loan_records),
            "siasat_c_analytical_mpa": len(mpa_samples),
            "siasat_d_synonym_and_register": len(df_syn),
        },
        "seq2seq_splits": {
            "train": len(seq2seq_train),
            "val": len(seq2seq_val),
            "test": len(seq2seq_test),
        },
        "sft_augmented_splits": {
            "train": len(df_sft_train),
            "val": len(df_sft_val),
            "test": len(df_sft_test),
        },
        "sft_train_task_breakdown": df_sft_train["task"].value_counts().to_dict(),
    }
    with open(OUT_DIR / "statistik_dataset_augmented.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("Statistik disimpan ke:", OUT_DIR / "statistik_dataset_augmented.json")


if __name__ == "__main__":
    main()
