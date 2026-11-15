#!/usr/bin/env python3
"""Rekonsiliasi & audit dry-run korpus Bahasa Indonesia <-> Bahasa Gorontalo (v5 -> kandidat v6).

Mengaudit workbook `othersource/ekstraksilain.xlsx` secara selektif dengan daftar putih
3 sheet (`Sheet12`, `GORONTALO V1`, `INDONESIA V1` + `INDONESIA 2` hanya sebagai
verifikator `#ERROR!`/ekspansi `==`), dan menolak 9 sheet lain yang mengandung
duplikat, halusinasi LLM, kehilangan tanda panjang `:`, atau pergeseran baris.

Mode default `--dry-run` bersifat READ-ONLY terhadap seluruh berkas v5 yang ada, dan
menulis laporan + pratinjau kandidat ke `reconcile_v6_preview/` untuk ditinjau user.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import re
from typing import Any

import openpyxl
import pandas as pd

import build_augmented_sft_dataset as bas


BASE_DIR = Path("/content/drive/MyDrive/REPO/richie/modelgorontalo")
XLSX_PATH = BASE_DIR / "othersource/ekstraksilain.xlsx"
DT_DIR = BASE_DIR / "dataset_terpadu_indonesia_gorontalo"
PREVIEW_DIR = DT_DIR / "reconcile_v6_preview"

BLACKLISTED_SHEETS = [
    "GORONTALO",
    "GORONTALO-INDONESIA",
    "GORONTALO SIMPLE",
    "INDONESIA",
    "INDONESIA GORONTALO",
    "INDONESIA 2 (raw tanpa verifikasi IV1)",
    "INDONESIA_GABUNG",
    "Copy of INDONESIA_GABUNG",
    "datasetkata",
]

# Kata Indonesia yang terkonfirmasi 0 kemunculan palsu pada 29.235 kalimat Gorontalo v5 valid.
_STRICT_ID_LEAKS_IN_GOR = {
    "yang", "tidak", "tak", "untuk", "dengan", "adalah", "sebagai", "karena",
    "sekarang", "kemarin", "diusut", "ambillah", "diikuti", "sebab",
    "kalau", "atau", "dalam", "bahwa", "akan", "sudah", "telah",
    "belum", "harus", "mesti", "hanya", "orang", "rumah", "pasar",
    "mengambil", "membeli", "menjual", "melihat", "membuat", "pekerjaan",
}

_GOR_LEAKS_IN_ID = {
    "wa:tiya", "waatia", "boyito", "botiya", "botiye", "tiyo", "timongoliyo",
    "timongolio", "lo:nao", "mo:nao", "moona?o", "loona?o", "yilonga",
    "lolimo", "lo?otapu", "lootapu", "oo?oondo", "pa:tali", "paatali",
    "di:la", "diila", "woluwo", "woluo", "hulontalo", "hulondalo",
    "tawu", "ilao", "mohama", "mowali", "tuwawu", "duluwo", "rasipedeu",
}

_META_PREFIX_RE = re.compile(
    r"^(?:Halaman\b|\d+[\s.:]|Baiklah|Tentu\b|Berikut\b|Catatan\b|Mohon\b|Maaf\b|"
    r"Kalimat Bahasa|Ini adalah|Sebagai AI|Penjelasan\b|Lema\b|Arti\b|Penerapan\b)",
    re.I,
)

_ID_ABBREV_RULES = [
    (re.compile(r"\btdk\b", re.I), "tidak"),
    (re.compile(r"\bdpt\b", re.I), "dapat"),
    (re.compile(r"\bttg\b", re.I), "tentang"),
    (re.compile(r"\bttp\b", re.I), "tetap"),
    (re.compile(r"\butk\b", re.I), "untuk"),
    (re.compile(r"\byg\b", re.I), "yang"),
    (re.compile(r"\bdgn\b", re.I), "dengan"),
    (re.compile(r"\bdng\b", re.I), "dengan"),
    (re.compile(r"\bdg\b", re.I), "dengan"),
    (re.compile(r"\bkrn\b", re.I), "karena"),
    (re.compile(r"\bdlm\b", re.I), "dalam"),
    (re.compile(r"\bdl\b", re.I), "dalam"),
    (re.compile(r"\bpd\b", re.I), "pada"),
    (re.compile(r"\bdp\b", re.I), "daripada"),
    (re.compile(r"\bsbg\b", re.I), "sebagai"),
    (re.compile(r"\bmsl\.?(?=\s|$|,|;|\))", re.I), "misalnya"),
    (re.compile(r"\bmis\.?(?=\s|$|,|;|\))", re.I), "misalnya"),
    (re.compile(r"\bdsb\.?(?=\s|$|,|;|\))", re.I), "dan sebagainya"),
    (re.compile(r"\bdst\.?(?=\s|$|,|;|\))", re.I), "dan seterusnya"),
    (re.compile(r"\bdll\.?(?=\s|$|,|;|\))", re.I), "dan lain-lain"),
    (re.compile(r"\bspt\b", re.I), "seperti"),
    (re.compile(r"\bthp\b", re.I), "terhadap"),
    (re.compile(r"\bkpd\b", re.I), "kepada"),
    (re.compile(r"\bdr\b", re.I), "dari"),
    (re.compile(r"\btlh\b", re.I), "telah"),
    (re.compile(r"\bsdh\b", re.I), "sudah"),
    (re.compile(r"\bblm\b", re.I), "belum"),
    (re.compile(r"\bbhw\b", re.I), "bahwa"),
    (re.compile(r"\bmsh\b", re.I), "masih"),
    (re.compile(r"\bjg\b", re.I), "juga"),
    (re.compile(r"(?<!\d\s)\bkm\b", re.I), "karena"),
    (re.compile(r"\bhny\b", re.I), "hanya"),
    (re.compile(r"\bybs\b", re.I), "yang bersangkutan"),
]


def gcol(row: list[Any], idx: int) -> str:
    if idx >= len(row) or row[idx] is None:
        return ""
    return str(row[idx]).strip()


def expand_indonesian_abbrev(text: str) -> str:
    """Ekspansi singkatan baku kamus Pateda tanpa mengubah kapitalisasi awal."""
    s = text
    for pat, repl in _ID_ABBREV_RULES:
        def _sub(m, r=repl):
            w = m.group(0)
            return r.capitalize() if w and w[0].isupper() else r
        s = pat.sub(_sub, s)
    return re.sub(r"\s+", " ", s).strip()


def convert_1987_to_2001_orthography(text: str) -> str:
    """Konversi ortografi Pateda 1987 (vokal ganda, glotal '?', tanpa glide) -> Pateda 2001."""
    if not text:
        return ""
    s = text
    glide_subs = [
        (r"\bwaatia\b", "wa:tiya"), (r"\bWaatia\b", "Wa:tiya"),
        (r"\bamiaatia\b", "amiya:tiya"), (r"\bAmiaatia\b", "Amiya:tiya"),
        (r"\btio\b", "tiyo"), (r"\bTio\b", "Tiyo"),
        (r"\btimongolio\b", "timongoliyo"), (r"\bTimongolio\b", "Timongoliyo"),
        (r"\bboito\b", "boyito"), (r"\bBoito\b", "Boyito"),
        (r"\bbotia\b", "botiya"), (r"\bBotia\b", "Botiya"),
        (r"\bbotie\b", "botiye"), (r"\bBotie\b", "Botiye"),
        (r"\butia\b", "utiya"), (r"\bUtia\b", "Utiya"),
        (r"\bmai\b", "mayi"), (r"\bMai\b", "Mayi"),
        (r"\bwau\b", "wawu"), (r"\bWau\b", "Wawu"),
        (r"\btuau\b", "tuwawu"), (r"\bduluo\b", "duluwo"),
        (r"\bdoi\b", "doyi"), (r"\bluahu\b", "luwahu"),
        (r"\bmoali\b", "mowali"), (r"\bloali\b", "lowali"),
        (r"\bdiila\b", "di:la"), (r"\bDiila\b", "Di:la"),
        (r"\bdiaalu\b", "diya:lu"), (r"\bDiaalu\b", "Diya:lu"),
        (r"lio\b", "liyo"), (r"mai\b", "mayi"),
    ]
    for pat, rep in glide_subs:
        s = re.sub(pat, rep, s)
    for v in ("a", "e", "i", "o", "u"):
        s = re.sub(v + v, v + ":", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<=[a-zA-Z:])\?(?=[a-zA-Z:])", "", s)
    return s


def norm_id(text: str) -> str:
    s = expand_indonesian_abbrev(text or "").lower()
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_gor_compare(text: str) -> str:
    """Normalisasi longgar untuk pencocokan lintas ortografi 1987/2001."""
    s = convert_1987_to_2001_orthography(text or "").lower()
    s = s.replace("?", "").replace(":", "").replace("'", "")
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def preserve_colons_from_ref(cand_gor: str, ref_gor: str) -> str:
    """Pulihkan tanda vokal panjang ':' dari ref v5 jika hilang di GORONTALO V1."""
    colon_map: dict[str, str] = {}
    for raw_tok in (ref_gor or "").split():
        core = raw_tok.strip(".,;:!?()\"'").lower()
        if ":" in core and not core.endswith(":"):
            bare = core.replace(":", "")
            colon_map[bare] = core
    if not colon_map:
        return cand_gor

    out_toks = []
    for tok in cand_gor.split():
        m = re.match(r"^([^a-zA-Z:]*)([a-zA-Z:]+)([^a-zA-Z:]*)$", tok)
        if m:
            pre, word, post = m.group(1), m.group(2), m.group(3)
            low = word.lower()
            if ":" not in word and low in colon_map:
                rep = colon_map[low]
                if word[0].isupper():
                    rep = rep[0].upper() + rep[1:]
                out_toks.append(f"{pre}{rep}{post}")
                continue
        out_toks.append(tok)
    return " ".join(out_toks)


def requires_indonesian_evidence(ind: str) -> bool:
    """Pastikan teks sisi Indonesia benar-benar berbahasa Indonesia (bukan kalimat Gorontalo nyasar)."""
    toks = [re.sub(r"[^\w-]", "", w.lower()) for w in ind.split()]
    toks = [t for t in toks if t]
    if not toks:
        return False
    if set(toks) & bas._ID_STOP:
        return True
    # Afiks produktif BI atau kata berkonsonan akhir khas BI (selain sufiks/partikel Gorontalo)
    id_affix = re.compile(r"^(?:meng|meny|mem|men|ber|ter|per|se)[a-z]{3,}$|(?:nya|kan|lah|kah)$")
    if any(id_affix.search(t) for t in toks):
        return True
    if any(t[-1] in "bcdfghjklmnpqrstvwxyz" and len(t) >= 3 and t not in {"tau", "mas", "pas"} for t in toks):
        return True
    return False


def has_strict_leak_or_artifact(ind: str, gor: str) -> str | None:
    """Kembalikan alasan penolakan jika ada artefak LLM, bahasa tertukar, non-ASCII, atau simbol rusak."""
    if not ind or not gor:
        return "empty_side"
    for side_name, txt in (("ind", ind), ("gor", gor)):
        if any(ord(ch) > 127 for ch in txt):
            return f"non_ascii_or_arrow_{side_name}"
        if "#ERROR!" in txt or "==" in txt or "---" in txt or "**" in txt or "->" in txt:
            return f"formula_or_placeholder_{side_name}"
        if re.search(r"[a-zA-Z]\d[a-zA-Z]|\d[a-zA-Z]{2,}|[a-zA-Z]{2,}\d", txt):
            return f"embedded_digit_ocr_{side_name}"
        if re.search(r"\[[^\]]*·[^\]]*\]", txt):
            return f"ipa_hallucination_{side_name}"
        if "(rujukan)" in txt.lower() or "tidak ada contoh" in txt.lower():
            return f"editorial_placeholder_{side_name}"
        if "contoh kalimat menggunakan anak kata" in txt.lower():
            return f"anak_kata_placeholder_{side_name}"
        if _META_PREFIX_RE.match(txt):
            return f"meta_or_page_header_{side_name}"
        if txt.count("(") != txt.count(")"):
            return f"unbalanced_parens_{side_name}"
        if re.match(r"^[0-9/\\|]+(?:\s|$)", txt) or re.match(r"^[^a-zA-Z\"']+[A-Za-z]", txt):
            return f"leading_nonalpha_ocr_{side_name}"
        if re.search(r"[|_]|(?<!\d)\b[a-z]{3,}[A-Z][a-zA-Z]*\b", txt):
            return f"ocr_symbol_or_case_{side_name}"
        if side_name == "gor" and re.match(r"^[B-DF-HJ-NP-TV-XZ]\s+[a-z]", txt):
            return "orphan_initial_char_gor"
        if side_name == "gor" and re.search(r"[bcdfghjklmnpqrstvwxyz]:[a-z]", txt.lower()):
            return "invalid_colon_after_consonant_gor"
        if side_name == "ind" and re.search(r"[a-z]:[a-z]", txt.lower()):
            return "glued_colon_in_id"

    if re.search(r"[a-zA-Z]\?[a-zA-Z]", ind):
        return "glottal_mark_in_id"

    nid_cmp = norm_id(ind)
    ngor_cmp = norm_gor_compare(gor)
    if nid_cmp == ngor_cmp and len(nid_cmp.split()) >= 2:
        return "untranslated_identical_pair"
    if not requires_indonesian_evidence(ind):
        return "missing_indonesian_evidence"

    gor_words = {re.sub(r"[^\w:?'-]", "", w.lower()) for w in gor.split()}
    ind_words = {re.sub(r"[^\w:?'-]", "", w.lower()) for w in ind.split()}
    bad_id_in_gor = gor_words & _STRICT_ID_LEAKS_IN_GOR
    if bad_id_in_gor:
        return f"indonesian_leak_in_gor:{sorted(bad_id_in_gor)[0]}"
    bad_gor_in_id = ind_words & _GOR_LEAKS_IN_ID
    if bad_gor_in_id:
        return f"gorontalo_leak_in_id:{sorted(bad_gor_in_id)[0]}"

    # Deteksi potongan Gorontalo terbawa di akhir sisi Indonesia (mis. '..., To bele')
    if "," in ind:
        tail = ind.rsplit(",", 1)[1].strip()
        tail_norm = norm_gor_compare(tail)
        tail_toks = tail_norm.split()
        if len(tail_toks) >= 2 and ngor_cmp.startswith(" ".join(tail_toks)):
            return "gorontalo_tail_leak_in_id"
    if re.match(r"^ya\b", ind.lower()) and re.search(r"\bya[,.]*$", gor.lower()):
        return "indonesian_ya_tail_leak_in_gor"
    if len(ind_words) <= 4 and (ind_words & {"lo", "to", "li", "te"}) and not (ind_words & bas._ID_STOP):
        return "gorontalo_phrase_in_id"
    return None


def parse_iv1_lexicon_headword(kata_induk: str) -> str | None:
    """Ambil lema/sub-lema Indonesia bersih dari format 'Kata Induk' INDONESIA V1."""
    k = re.sub(r"\s+", " ", (kata_induk or "").strip())
    if not k or any(ch in k for ch in ":=?#"):
        return None
    # Kasus 'Abadi (mengabadikan)' / 'Abjad (disuruh abjadkan)' -> ambil sub-entri imbuhan di kurung
    m_par = re.match(
        r"^[A-Za-z-]+\s*\(\s*((?:disuruh\s+|sedang\s+|telah\s+|akan\s+|meng|meny|mem|men|di|ber|ter|per|ke|pe)[a-z\s-]{2,35})\s*\)$",
        k,
        re.I,
    )
    if m_par:
        return re.sub(r"\s+", " ", m_par.group(1).strip().lower())
    k_nopar = re.sub(r"\s*\([^)]*\)", "", k).strip()
    m_sub = re.match(r"^[^,]+,\s*(?:[IVX]+\s+)?([A-Za-z][A-Za-z\s-]{1,40})$", k_nopar)
    if m_sub:
        return m_sub.group(1).strip()
    k_clean = re.sub(r"\s*,\s*[IVX]+$", "", k_nopar).strip()
    if re.match(r"^[A-Za-z][A-Za-z\s-]{1,40}$", k_clean):
        return k_clean.lower() if len(k_clean.split()) == 1 else k_clean
    return None


def clean_sentence_surface(text: str, is_gor: bool) -> str:
    s = re.sub(r"\s+", " ", (text or "").strip())
    s = re.sub(r"\s*;\s*$", "", s)
    if is_gor:
        return bas.clean_gorontalo_text(s)
    return bas.clean_indonesian_text(expand_indonesian_abbrev(s))


def extract_iv1_eq_replacement(kata_induk: str) -> str | None:
    """Ambil bentuk pengisi '==' yang aman dari kolom Kata Induk INDONESIA V1."""
    k = re.sub(r"\s+", " ", (kata_induk or "").strip())
    if not k or ":" in k or "==" in k:
        return None
    # Kasus: "Selop, I" -> "selop"; "Selonsong, I (berangus...)" -> "selonsong"
    m_num_only = re.match(r"^([A-Za-z-]+)\s*,\s*[IVX]+\s*(?:\([^)]*\))?$", k)
    if m_num_only:
        return m_num_only.group(1).lower()
    # Kasus: "Selesai, II diselesaikan" atau "Selongkar, menyelongkar (membongkar)"
    m_sub = re.match(r"^[A-Za-z-]+\s*,\s*(?:[IVX]+\s+)?([A-Za-z-]{3,})(?:\s*\([^)]*\))?$", k)
    if m_sub:
        sub = m_sub.group(1).lower()
        # Tolak fragmen tidak lengkap seperti "dipakaikan" / "dimasuki" jika tidak mandiri
        if sub in {"dipakaikan", "memakai", "diberi", "mencari"}:
            return None
        return sub
    # Kasus kata tunggal bersih ("terkira", "termakan", "abjad")
    if re.match(r"^[A-Za-z-]{2,}$", k):
        return k.lower()
    return None


def parse_gv1_headword(kata_induk: str) -> str:
    k = re.sub(r"\([^)]*\)", "", (kata_induk or "").strip())
    k = re.sub(r"\s+", " ", k).strip(" ,.;:")
    return k


def expand_gv1_sentence(raw_gor: str, headword: str) -> str | None:
    if "---" not in raw_gor:
        return raw_gor
    if not headword:
        return None
    return raw_gor.replace("---", headword)


def extract_whitelisted_candidates(wb: openpyxl.Workbook, rejected_log: list[dict]) -> dict[str, list[dict]]:
    """Ekstrak dan filter kandidat dari Sheet12, GORONTALO V1, serta INDONESIA V1 (+IND2 verifier)."""
    candidates: dict[str, list[dict]] = {"Sheet12": [], "GORONTALO V1": [], "INDONESIA V1": []}

    # --- 1. Sheet12 (IG 1987 A s.d. S hal. 376) ---
    s12_rows = [list(r) for r in wb["Sheet12"].iter_rows(values_only=True)][1:]
    for idx, r in enumerate(s12_rows, start=2):
        raw_gor = gcol(r, 0)
        raw_id = gcol(r, 1)
        if not raw_gor and not raw_id:
            continue
        id_clean = clean_sentence_surface(raw_id, is_gor=False)
        gor_asli = re.sub(r"\s*;\s*$", "", re.sub(r"\s+", " ", raw_gor.strip()))
        gor_std = clean_sentence_surface(convert_1987_to_2001_orthography(gor_asli), is_gor=True)
        reason = has_strict_leak_or_artifact(id_clean, gor_asli)
        if not reason and not bas.is_clean_parallel(id_clean, gor_std):
            reason = "failed_bas_is_clean_parallel"
        if reason:
            rejected_log.append({"sheet": "Sheet12", "row": idx, "reason": reason, "ind": raw_id, "gor": raw_gor})
            continue
        candidates["Sheet12"].append({
            "sheet": "Sheet12",
            "row": idx,
            "sumber_kamus": "indonesia_gorontalo_1987",
            "lema": "",
            "raw_id": raw_id,
            "raw_gor": raw_gor,
            "ind": id_clean,
            "gor_std": gor_std,
            "gor_asli": gor_asli,
            "had_dash_or_stem_ok": True,
        })

    # --- 2. GORONTALO V1 (GI 2001 A-Z) ---
    gv1_rows = [list(r) for r in wb["GORONTALO V1"].iter_rows(values_only=True)][1:]
    for idx, r in enumerate(gv1_rows, start=2):
        raw_lema = gcol(r, 0)
        raw_arti = gcol(r, 1)
        raw_gor = gcol(r, 2)
        raw_id = gcol(r, 3)
        if not raw_gor and not raw_id:
            continue
        head = parse_gv1_headword(raw_lema)
        had_dash = "---" in raw_gor
        gor_exp = expand_gv1_sentence(raw_gor, head)
        if gor_exp is None:
            rejected_log.append({"sheet": "GORONTALO V1", "row": idx, "reason": "unexpandable_dash", "ind": raw_id, "gor": raw_gor})
            continue
        id_clean = clean_sentence_surface(raw_id, is_gor=False)
        gor_std = clean_sentence_surface(gor_exp, is_gor=True)
        reason = has_strict_leak_or_artifact(id_clean, gor_std)
        if not reason and not bas.is_clean_parallel(id_clean, gor_std):
            reason = "failed_bas_is_clean_parallel"
        if reason:
            rejected_log.append({"sheet": "GORONTALO V1", "row": idx, "reason": reason, "lema": raw_lema, "ind": raw_id, "gor": raw_gor})
            continue

        # Anti-shift signal untuk entri novel: tanpa '---' WAJIB cek akar lema ada di kalimat
        stem = re.sub(r"[:?\s-]+", "", head.split()[0].lower()) if head else ""
        stem_in_gor = bool(len(stem) >= 3 and stem[: min(5, len(stem))] in norm_gor_compare(gor_std).replace(" ", ""))
        candidates["GORONTALO V1"].append({
            "sheet": "GORONTALO V1",
            "row": idx,
            "sumber_kamus": "gorontalo_indonesia_2001",
            "lema": head,
            "arti": raw_arti,
            "raw_id": raw_id,
            "raw_gor": raw_gor,
            "ind": id_clean,
            "gor_std": gor_std,
            "gor_asli": gor_std,
            "had_dash_or_stem_ok": had_dash or stem_in_gor,
        })

    # --- 3. Index verifikasi INDONESIA 2 untuk membantu #ERROR! / == di INDONESIA V1 ---
    ind2_rows = [list(r) for r in wb["INDONESIA 2"].iter_rows(values_only=True)][1:]
    ind2_by_lema: dict[str, list[tuple[str, str]]] = {}
    for r in ind2_rows:
        l, p = gcol(r, 0), gcol(r, 2)
        if not l or not p or "contoh kalimat menggunakan anak kata" in p.lower() or "tidak ada contoh" in p.lower():
            continue
        for seg in p.split(";"):
            parts = [x.strip() for x in seg.rsplit(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                ind2_by_lema.setdefault(l.split(",")[0].strip().lower(), []).append((parts[0], parts[1]))

    # --- 4. INDONESIA V1 (IG 1987 A-Z, fokus perbaikan & S akhir - Z) ---
    iv1_rows = [list(r) for r in wb["INDONESIA V1"].iter_rows(values_only=True)][1:]
    seen_selent = False
    for idx, r in enumerate(iv1_rows, start=2):
        c0, c1, c2, c3 = gcol(r, 0), gcol(r, 1), gcol(r, 2), gcol(r, 3)
        if "selent" in c0.lower() or "selent" in c1.lower():
            seen_selent = True
        if not c2 and not c3:
            continue

        # Deteksi pertukaran kolom C & D (seperti 'termakan')
        raw_gor, raw_id = c2, c3
        swapped = False
        if "==" in raw_gor and "==" not in raw_id and raw_id != "#ERROR!":
            raw_gor, raw_id = raw_id, raw_gor
            swapped = True

        # Perbaiki #ERROR! / == memakai verifikator silang INDONESIA 2 jika cocok Gorontalo >= 0.84
        verified_by_ind2 = False
        if (raw_id == "#ERROR!" or "==" in raw_id) and raw_gor and raw_gor != "#ERROR!":
            ng_iv1 = norm_gor_compare(raw_gor)
            keys = [c0.split(",")[0].strip().lower()]
            if "," in c0:
                keys.append(c0.split(",", 1)[1].strip().lstrip("IVX ").lower())
            best_seg, best_sim = None, 0.0
            for k in keys:
                for id_cand, gor_cand in ind2_by_lema.get(k, []):
                    sim = SequenceMatcher(None, ng_iv1, norm_gor_compare(gor_cand)).ratio()
                    if sim > best_sim:
                        best_sim = sim
                        best_seg = id_cand
            if best_seg and best_sim >= 0.84 and "==" not in best_seg:
                raw_id = best_seg
                verified_by_ind2 = True

        # Ekspansi == deterministik jika tidak diverifikasi IND2
        if "==" in raw_id:
            rep = extract_iv1_eq_replacement(c0)
            if not rep:
                rejected_log.append({"sheet": "INDONESIA V1", "row": idx, "reason": "ambiguous_eq_or_error", "lema": c0, "ind": raw_id, "gor": raw_gor})
                continue
            raw_id = re.sub(r"={2,}", rep, raw_id)

        if "#ERROR!" in raw_id or "#ERROR!" in raw_gor:
            rejected_log.append({"sheet": "INDONESIA V1", "row": idx, "reason": "unverified_excel_error", "lema": c0, "ind": raw_id, "gor": raw_gor})
            continue

        id_clean = clean_sentence_surface(raw_id, is_gor=False)
        gor_asli = re.sub(r"\s*;\s*$", "", re.sub(r"\s+", " ", raw_gor.strip()))
        gor_std = clean_sentence_surface(convert_1987_to_2001_orthography(gor_asli), is_gor=True)
        reason = has_strict_leak_or_artifact(id_clean, gor_asli)
        if not reason and not bas.is_clean_parallel(id_clean, gor_std):
            reason = "failed_bas_is_clean_parallel"
        if reason:
            rejected_log.append({"sheet": "INDONESIA V1", "row": idx, "reason": reason, "lema": c0, "ind": raw_id, "gor": raw_gor})
            continue

        candidates["INDONESIA V1"].append({
            "sheet": "INDONESIA V1",
            "row": idx,
            "sumber_kamus": "indonesia_gorontalo_1987",
            "lema": c0.split(",")[0].strip(),
            "arti": c1,
            "raw_id": raw_id,
            "raw_gor": raw_gor,
            "ind": id_clean,
            "gor_std": gor_std,
            "gor_asli": gor_asli,
            "in_s_end_tz": seen_selent or (bool(c0) and c0[0].upper() in "TUVWXYZ"),
            "swapped_fixed": swapped,
            "verified_by_ind2": verified_by_ind2,
            "had_dash_or_stem_ok": True,
        })

    return candidates


def build_v5_index(comb_df: pd.DataFrame):
    """Buat indeks pencocokan cepat terhadap seluruh 29.235 baris korpus v5."""
    rows_meta: list[dict] = []
    by_nid: dict[str, list[int]] = {}
    by_ngor: dict[str, list[int]] = {}
    tok_id_idx: dict[str, list[int]] = {}
    tok_gor_idx: dict[str, list[int]] = {}

    for idx, r in comb_df.iterrows():
        ind = str(r["src_indonesia"] or "")
        gor = str(r["tgt_gorontalo"] or "")
        nid = norm_id(ind)
        ngor = norm_gor_compare(gor)
        rows_meta.append({
            "idx": int(idx),
            "id": r["id"],
            "src_indonesia": ind,
            "tgt_gorontalo": gor,
            "tgt_gorontalo_asli": str(r["tgt_gorontalo_asli"] or ""),
            "lema": str(r["lema"] or ""),
            "sumber_kamus": str(r["sumber_kamus"] or ""),
            "nid": nid,
            "ngor": ngor,
        })
        if nid:
            by_nid.setdefault(nid, []).append(int(idx))
            for w in set(nid.split()):
                if len(w) >= 3:
                    tok_id_idx.setdefault(w, []).append(int(idx))
        if ngor:
            by_ngor.setdefault(ngor, []).append(int(idx))
            for w in set(ngor.split()):
                if len(w) >= 3:
                    tok_gor_idx.setdefault(w, []).append(int(idx))
    return rows_meta, by_nid, by_ngor, tok_id_idx, tok_gor_idx


def best_v5_matches(cand: dict, rows_meta, by_nid, by_ngor, tok_id_idx, tok_gor_idx):
    nid = norm_id(cand["ind"])
    ngor = norm_gor_compare(cand["gor_std"])

    cand_scores: dict[int, int] = {}
    for idx in by_nid.get(nid, []):
        cand_scores[idx] = cand_scores.get(idx, 0) + 100
    for idx in by_ngor.get(ngor, []):
        cand_scores[idx] = cand_scores.get(idx, 0) + 100
    for w in set(nid.split()):
        if len(w) >= 3:
            for idx in tok_id_idx.get(w, []):
                cand_scores[idx] = cand_scores.get(idx, 0) + 1
    for w in set(ngor.split()):
        if len(w) >= 3:
            for idx in tok_gor_idx.get(w, []):
                cand_scores[idx] = cand_scores.get(idx, 0) + 1

    top_idxs = sorted(cand_scores.keys(), key=lambda i: -cand_scores[i])[:10]
    best_idx, best_si, best_sg = None, 0.0, 0.0
    for idx in top_idxs:
        rm = rows_meta[idx]
        si = 1.0 if nid == rm["nid"] else SequenceMatcher(None, nid, rm["nid"]).ratio()
        sg = 1.0 if ngor == rm["ngor"] else SequenceMatcher(None, ngor, rm["ngor"]).ratio()
        if (si + sg) > (best_si + best_sg):
            best_idx, best_si, best_sg = idx, si, sg
    return best_idx, best_si, best_sg


def token_containment(shorter_norm: str, longer_norm: str) -> float:
    stoks = shorter_norm.split()
    ltoks = set(longer_norm.split())
    if not stoks:
        return 0.0
    return sum(1 for t in stoks if t in ltoks) / len(stoks)


def repair_side_needs_cleanup(txt: str, is_gor: bool) -> bool:
    """Deteksi artefak OCR/bocor kepala entri pada sisi v5 yang tadinya hendak dipertahankan."""
    if not txt or any(ord(ch) > 127 for ch in txt):
        return True
    if bas._CONSONANT_RUN.search(txt):
        return True
    if re.search(
        r"[|_=]|#ERROR!|---|\*\*|->|\b[a-z]{3,}[A-Z][a-zA-Z]*\b|"
        r"[a-zA-Z]\d[a-zA-Z]|\d[a-zA-Z]{2,}|[a-zA-Z]{2,}\d",
        txt,
    ):
        return True
    if re.match(r"^(?:[0-9/\\|]+(?:\s|$)|[^a-zA-Z\"']+[A-Za-z])", txt):
        return True
    if txt.count("(") != txt.count(")"):
        return True
    if not is_gor:
        if _META_PREFIX_RE.match(txt) or re.match(
            r"^(?:[A-Za-z]\s*:\s*|[a-z][a-z\s-]{1,25},\s+[A-Z]|[a-z-]{2,20}:\s+)",
            txt,
        ):
            return True
        if re.search(
            r"\b(?:diila|waaria|tampaj|adah|naka|scpeua|menyampaika|diturunka|diberika)\b",
            txt.lower(),
        ):
            return True
    else:
        if re.search(r"[°!]|\.[a-z]|\bu:ri\b|\(sehari\)|^\s*[B-DF-HJ-NP-TV-XZ]\s+[a-z]", txt):
            return True
    return False


def finalize_clean_repair_pair(
    new_ind: str,
    new_gor: str,
    new_gor_asli: str,
    rtype: str,
    rm: dict[str, Any],
    cand: dict[str, Any],
    si: float,
    sg: float,
) -> tuple[str, str, str, str] | None:
    """Pastikan kedua sisi perbaikan bersih ASCII, lolos is_clean_parallel & leak guard."""
    # Upgrade sisi Indonesia jika sisi v5 yang dipertahankan masih mengandung artefak/non-ASCII
    if new_ind == rm["src_indonesia"] and si >= 0.80:
        if (
            repair_side_needs_cleanup(new_ind, is_gor=False)
            or has_strict_leak_or_artifact(new_ind, new_gor) is not None
            or not bas.is_clean_parallel(new_ind, new_gor)
        ):
            new_ind = cand["ind"]
            rtype = "both_gor_and_id_repair"

    # Upgrade sisi Gorontalo jika sisi v5 yang dipertahankan masih mengandung artefak/non-ASCII
    if new_gor == rm["tgt_gorontalo"] and sg >= 0.85:
        if (
            repair_side_needs_cleanup(new_gor, is_gor=True)
            or any(ord(ch) > 127 for ch in new_gor_asli)
            or has_strict_leak_or_artifact(new_ind, new_gor) is not None
            or not bas.is_clean_parallel(new_ind, new_gor)
        ):
            new_gor = preserve_colons_from_ref(cand["gor_std"], rm["tgt_gorontalo"])
            new_gor_asli = cand["gor_asli"] if rm["sumber_kamus"] == "indonesia_gorontalo_1987" else new_gor
            rtype = "both_gor_and_id_repair"

    # Gerbang keras akhir: tolak jika masih ada non-ASCII atau gagal lolos validasi paralel
    if any(ord(ch) > 127 for ch in new_ind + new_gor + new_gor_asli):
        return None
    if has_strict_leak_or_artifact(new_ind, new_gor) is not None:
        return None
    if not bas.is_clean_parallel(new_ind, new_gor):
        return None
    return new_ind, new_gor, new_gor_asli, rtype


def reconcile_and_audit(apply: bool = False) -> dict[str, Any]:
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    comb_df = pd.read_parquet(DT_DIR / "combined_parallel_sentences.parquet")
    lex_df = pd.read_parquet(DT_DIR / "combined_lexicon_id_to_gor.parquet")
    split_frames = {
        split_name: pd.read_parquet(DT_DIR / f"seq2seq_{split_name}.parquet")
        for split_name in ("train", "val", "test")
    }
    split_lookup: dict[tuple[str, str], str] = {}
    for split_name, sdf in split_frames.items():
        par_sub = sdf[(sdf["type"] == "parallel_sentence") & (sdf["src_lang"] == "ind_Latn")]
        for _, r in par_sub.iterrows():
            split_lookup[(norm_id(r["src_text"]), norm_gor_compare(r["tgt_text"]))] = split_name

    wb = openpyxl.load_workbook(XLSX_PATH, read_only=True, data_only=True)
    rejected_log: list[dict] = []
    candidates = extract_whitelisted_candidates(wb, rejected_log)

    rows_meta, by_nid, by_ngor, tok_id_idx, tok_gor_idx = build_v5_index(comb_df)

    # Prioritas sumber perbaikan: Sheet12 (IG) -> GORONTALO V1 (GI) -> INDONESIA V1 (IG S-Z & umum)
    repairs_by_v5_idx: dict[int, dict] = {}
    novel_parallel: list[dict] = []
    seen_novel_keys: set[tuple[str, str]] = set()
    seen_ig_novel_ids: set[str] = set()
    id_ocr_review_queue: list[dict] = []

    ordered_cands = candidates["Sheet12"] + candidates["GORONTALO V1"] + candidates["INDONESIA V1"]
    for cand in ordered_cands:
        idx, si, sg = best_v5_matches(cand, rows_meta, by_nid, by_ngor, tok_id_idx, tok_gor_idx)
        cand_nid = norm_id(cand["ind"])
        cand_ngor = norm_gor_compare(cand["gor_std"])

        if idx is not None:
            rm = rows_meta[idx]
            v5_nid = rm["nid"]
            v5_ngor = rm["ngor"]
            in_split = split_lookup.get((v5_nid, v5_ngor), "filtered_out_of_clean_splits")

            # --- TRACK 1A: Perbaikan kalimat Gorontalo v5 yang terpotong (+ bersihkan bocor ID jika ada) ---
            if si >= 0.88 and (len(cand_ngor) - len(v5_ngor) >= 5):
                cont = token_containment(v5_ngor, cand_ngor)
                pref_sim = SequenceMatcher(None, v5_ngor, cand_ngor[: len(v5_ngor) + 6]).ratio()
                if cont >= 0.67 or pref_sim >= 0.72:
                    final_gor = preserve_colons_from_ref(cand["gor_std"], rm["tgt_gorontalo"])
                    # Jika v5_id ketambahan kata pengotor Gorontalo/gloss (mis. ', Koputusan', 'bangunan:', ', Diila'), pakai cand['ind']
                    id_leaked_boundary = bool(
                        len(v5_nid) > len(cand_nid)
                        and cand_nid in v5_nid
                        and (re.search(r"[:)]", rm["src_indonesia"][:25]) or ", " in rm["src_indonesia"][-20:])
                    )
                    id_truncated_suffix = bool(
                        len(cand_nid) > len(v5_nid)
                        and not bas._CONSONANT_RUN.search(cand["ind"])
                        and SequenceMatcher(None, v5_nid, cand_nid).ratio() >= 0.90
                        and re.search(r"\b(?:menyampaika|diturunka|diberika)\b", rm["src_indonesia"].lower())
                    )
                    fix_id_too = id_leaked_boundary or id_truncated_suffix
                    new_ind_val = cand["ind"] if fix_id_too else rm["src_indonesia"]
                    rtype = "both_gor_and_id_repair" if fix_id_too else "gor_truncation_rescue"
                    new_gor_asli = cand["gor_asli"] if rm["sumber_kamus"] == "indonesia_gorontalo_1987" else final_gor
                    cleaned = finalize_clean_repair_pair(
                        new_ind_val, final_gor, new_gor_asli, rtype, rm, cand, si, sg
                    )
                    if cleaned is None:
                        rejected_log.append({
                            "sheet": cand["sheet"],
                            "row": cand["row"],
                            "reason": "post_repair_pair_still_dirty",
                            "v5_id": rm["id"],
                            "ind": new_ind_val,
                            "gor": final_gor,
                        })
                        continue
                    new_ind_val, final_gor, new_gor_asli, rtype = cleaned
                    if idx not in repairs_by_v5_idx:
                        repairs_by_v5_idx[idx] = {
                            "repair_type": rtype,
                            "v5_idx": idx,
                            "v5_id": rm["id"],
                            "split": in_split,
                            "sumber_kamus": rm["sumber_kamus"],
                            "source_sheet": cand["sheet"],
                            "source_row": cand["row"],
                            "old_ind": rm["src_indonesia"],
                            "old_gor": rm["tgt_gorontalo"],
                            "new_ind": new_ind_val,
                            "new_gor": final_gor,
                            "new_gor_asli": new_gor_asli,
                            "sim_id": round(si, 3),
                            "sim_gor": round(sg, 3),
                        }
                    continue

            # --- TRACK 1B: Perbaikan cacat OCR sisi Indonesia (mis. 'tampaj dang', 'adah') ---
            if sg >= 0.90 and si < 0.97:
                has_v5_id_ocr_bug = bool(
                    bas._CONSONANT_RUN.search(rm["src_indonesia"])
                    or re.search(r"\b(?:tampaj|adah|naka|scpeua)\b", rm["src_indonesia"].lower())
                    or (len(cand_nid) - len(v5_nid) >= 6 and token_containment(v5_nid, cand_nid) >= 0.70)
                )
                cand_looks_bad = bool(
                    bas._CONSONANT_RUN.search(cand["ind"])
                    or re.search(r"\bmegurus\b", cand["ind"].lower())
                    or re.match(r"^(?:diikuti|diusut)\b", cand["ind"].lower())
                )
                if has_v5_id_ocr_bug and not cand_looks_bad and si >= 0.65:
                    v5_gor_noisy = bool(
                        re.search(r"[°!]|\.[a-z]|^[^a-zA-Z]|\bmomste\b|\bylo\b|\bwu\s+tatuu\b|\bAvia-Auanvo\b|\bu:ri\b|\(sehari\)", rm["tgt_gorontalo"])
                        or (len(cand_ngor) - len(v5_ngor) >= 4)
                    )
                    if v5_gor_noisy:
                        best_gor = preserve_colons_from_ref(cand["gor_std"], rm["tgt_gorontalo"])
                        best_gor_asli = cand["gor_asli"] if rm["sumber_kamus"] == "indonesia_gorontalo_1987" else best_gor
                        rtype = "both_gor_and_id_repair"
                    else:
                        best_gor = rm["tgt_gorontalo"]
                        best_gor_asli = rm["tgt_gorontalo_asli"]
                        rtype = "id_ocr_repair"
                    cleaned = finalize_clean_repair_pair(
                        cand["ind"], best_gor, best_gor_asli, rtype, rm, cand, si, sg
                    )
                    if cleaned is None:
                        rejected_log.append({
                            "sheet": cand["sheet"],
                            "row": cand["row"],
                            "reason": "post_repair_pair_still_dirty",
                            "v5_id": rm["id"],
                            "ind": cand["ind"],
                            "gor": best_gor,
                        })
                        continue
                    final_ind, best_gor, best_gor_asli, rtype = cleaned
                    if idx not in repairs_by_v5_idx:
                        repairs_by_v5_idx[idx] = {
                            "repair_type": rtype,
                            "v5_idx": idx,
                            "v5_id": rm["id"],
                            "split": in_split,
                            "sumber_kamus": rm["sumber_kamus"],
                            "source_sheet": cand["sheet"],
                            "source_row": cand["row"],
                            "old_ind": rm["src_indonesia"],
                            "old_gor": rm["tgt_gorontalo"],
                            "new_ind": final_ind,
                            "new_gor": best_gor,
                            "new_gor_asli": best_gor_asli,
                            "sim_id": round(si, 3),
                            "sim_gor": round(sg, 3),
                        }
                    elif idx in repairs_by_v5_idx and repairs_by_v5_idx[idx]["repair_type"] == "gor_truncation_rescue":
                        rep_prev = repairs_by_v5_idx[idx]
                        cleaned_merge = finalize_clean_repair_pair(
                            final_ind,
                            rep_prev["new_gor"],
                            rep_prev["new_gor_asli"],
                            "both_gor_and_id_repair",
                            rm,
                            cand,
                            si,
                            sg,
                        )
                        if cleaned_merge is not None:
                            rep_prev["new_ind"], rep_prev["new_gor"], rep_prev["new_gor_asli"], rep_prev["repair_type"] = cleaned_merge
                elif si < 0.85 and sg >= 0.92 and len(id_ocr_review_queue) < 250:
                    id_ocr_review_queue.append({
                        "v5_id": rm["id"],
                        "split": in_split,
                        "source_sheet": cand["sheet"],
                        "v5_ind": rm["src_indonesia"],
                        "cand_ind": cand["ind"],
                        "v5_gor": rm["tgt_gorontalo"],
                        "sim_id": round(si, 3),
                        "sim_gor": round(sg, 3),
                    })
                continue

            # --- TRACK 1C: Perbaikan OCR Gorontalo panjang serupa (karakter rusak di v5) ---
            if si >= 0.92 and abs(len(cand_ngor) - len(v5_ngor)) < 5 and sg < 0.96:
                v5_gor_corrupted = bool(
                    re.search(r"(?:^[^a-zA-Z]|mobcrenti|re:kentya|batlata|ra-wumu|nastthau|mutuluhe|dibaiaka)", rm["tgt_gorontalo"].lower())
                )
                if v5_gor_corrupted and sg >= 0.72 and idx not in repairs_by_v5_idx:
                    final_gor = preserve_colons_from_ref(cand["gor_std"], rm["tgt_gorontalo"])
                    # Pertahankan lema v5 jika asli benar (mis. 'muluhelo' bukan 'moluhelo' di GV1)
                    if rm["lema"] and len(rm["lema"]) >= 5 and rm["lema"].lower() in rm["tgt_gorontalo"].lower():
                        pat_m = re.sub(r"^[a-z]", "[a-z]", re.escape(rm["lema"]))
                        final_gor = re.sub(rf"\b{pat_m}\b", rm["lema"], final_gor, count=1, flags=re.I)
                    new_gor_asli = cand["gor_asli"] if rm["sumber_kamus"] == "indonesia_gorontalo_1987" else final_gor
                    cleaned = finalize_clean_repair_pair(
                        rm["src_indonesia"], final_gor, new_gor_asli, "gor_ocr_token_repair", rm, cand, si, sg
                    )
                    if cleaned is None:
                        rejected_log.append({
                            "sheet": cand["sheet"],
                            "row": cand["row"],
                            "reason": "post_repair_pair_still_dirty",
                            "v5_id": rm["id"],
                            "ind": rm["src_indonesia"],
                            "gor": final_gor,
                        })
                        continue
                    final_ind, final_gor, new_gor_asli, rtype = cleaned
                    repairs_by_v5_idx[idx] = {
                        "repair_type": rtype,
                        "v5_idx": idx,
                        "v5_id": rm["id"],
                        "split": in_split,
                        "sumber_kamus": rm["sumber_kamus"],
                        "source_sheet": cand["sheet"],
                        "source_row": cand["row"],
                        "old_ind": rm["src_indonesia"],
                        "old_gor": rm["tgt_gorontalo"],
                        "new_ind": final_ind,
                        "new_gor": final_gor,
                        "new_gor_asli": new_gor_asli,
                        "sim_id": round(si, 3),
                        "sim_gor": round(sg, 3),
                    }
                continue

        # --- TRACK 2: Kandidat kalimat baru (Novel Coverage) ---
        if si < 0.60 and sg < 0.60:
            if cand["sheet"] == "GORONTALO V1":
                if cand["row"] >= 8650:
                    rejected_log.append({
                        "sheet": "GORONTALO V1",
                        "row": cand["row"],
                        "reason": "gv1_tail_shift_zone",
                        "lema": cand["lema"],
                        "ind": cand["ind"],
                        "gor": cand["gor_std"],
                    })
                    continue
                if not cand.get("had_dash_or_stem_ok", False):
                    rejected_log.append({
                        "sheet": "GORONTALO V1",
                        "row": cand["row"],
                        "reason": "suspected_row_shift_no_dash_no_stem",
                        "lema": cand["lema"],
                        "ind": cand["ind"],
                        "gor": cand["gor_std"],
                    })
                    continue
                if (
                    ("," in cand["ind"] and len(cand_nid.split()) <= 3)
                    or len(cand_nid.split()) < 2
                    or len(cand_ngor.split()) < 2
                    or (len(cand_nid.split()) >= 4 and len(cand_ngor.split()) <= 2)
                    or re.search(r"\s(?:lo|to|u)$", cand["gor_std"].lower())
                ):
                    rejected_log.append({
                        "sheet": "GORONTALO V1",
                        "row": cand["row"],
                        "reason": "gv1_gloss_or_truncated_novel",
                        "lema": cand["lema"],
                        "ind": cand["ind"],
                        "gor": cand["gor_std"],
                    })
                    continue
            # Filter typo OCR spesifik sisa baris novel lintas sheet
            if re.search(
                r"nmb|\bwowu\b|kerastat|wuti musi|tetapimasih|mokaraja:hudungu|ii:langi|memdengarkan",
                (cand["ind"] + " " + cand["gor_std"]).lower(),
            ):
                rejected_log.append({
                    "sheet": cand["sheet"],
                    "row": cand["row"],
                    "reason": "novel_specific_ocr_typo",
                    "lema": cand.get("lema", ""),
                    "ind": cand["ind"],
                    "gor": cand["gor_std"],
                })
                continue
            # Untuk INDONESIA V1 novel, utamakan bagian pasca-Sheet12 + baris terverifikasi IND2/ekspansi aman
            if cand["sheet"] == "INDONESIA V1" and not (cand.get("in_s_end_tz") or cand.get("verified_by_ind2")):
                continue
            nkey = (cand_nid.replace(" ", ""), cand_ngor.replace(" ", ""))
            if nkey in seen_novel_keys:
                continue
            if cand["sumber_kamus"] == "indonesia_gorontalo_1987":
                if cand_nid in seen_ig_novel_ids:
                    continue
                seen_ig_novel_ids.add(cand_nid)
            seen_novel_keys.add(nkey)
            novel_parallel.append({
                "source_sheet": cand["sheet"],
                "source_row": cand["row"],
                "sumber_kamus": cand["sumber_kamus"],
                "lema": cand.get("lema", ""),
                "src_indonesia": cand["ind"],
                "tgt_gorontalo": cand["gor_std"],
                "tgt_gorontalo_asli": cand["gor_asli"],
                "max_v5_sim_id": round(si, 3),
                "max_v5_sim_gor": round(sg, 3),
            })

    # --- TRACK 3: Leksikon baru terkurasi ketat (hanya konsep ID yang belum ada di v5, bersih ASCII) ---
    v5_lex_keys = {
        (norm_id(str(r["indonesia"])), norm_gor_compare(str(r["gorontalo"])))
        for _, r in lex_df.iterrows()
    }
    v5_lex_compact = {
        (k[0].replace(" ", ""), k[1].replace(" ", ""))
        for k in v5_lex_keys
    }
    v5_lex_id_set = {k[0] for k in v5_lex_keys}
    novel_lexicon: list[dict] = []
    seen_lex: set[tuple[str, str]] = set()
    for sheet_name, cols in (("GORONTALO V1", (1, 0)), ("INDONESIA V1", (0, 1))):
        rows = [list(r) for r in wb[sheet_name].iter_rows(values_only=True)][1:]
        is_ig = sheet_name == "INDONESIA V1"
        for idx, r in enumerate(rows, start=2):
            if not is_ig and idx >= 8650:
                continue
            raw_ind = gcol(r, cols[0])
            raw_gor = gcol(r, cols[1])
            if not raw_ind or not raw_gor:
                continue
            if any(ord(ch) > 127 for ch in raw_ind + raw_gor):
                continue
            if any(bad in raw_ind + raw_gor for bad in ("#ERROR!", "(rujukan)", "tidak ada contoh", "==", "---")):
                continue
            if is_ig:
                parsed_head = parse_iv1_lexicon_headword(raw_ind)
                if not parsed_head:
                    continue
                ind_clean = bas.clean_indonesian_text(expand_indonesian_abbrev(parsed_head)).strip(" .,:;")
                raw_g_clean = re.sub(r"\s*\(\d+\)\s*", "; ", re.sub(r"\([^)]*\)", "", raw_gor))
                gor_candidates = [g.strip(" .,:;") for g in re.split(r"[;,]", raw_g_clean) if g.strip(" .,:;")]
            else:
                ind_clean = bas.clean_indonesian_text(expand_indonesian_abbrev(re.sub(r"\s*,\s*[IVX]+\b", "", raw_ind))).strip(" .,:;")
                gor_candidates = [re.sub(r"\([^)]*\)", "", raw_gor).strip(" .,:;")]

            if not ind_clean or len(ind_clean.split()) > 3 or len(ind_clean) < 3:
                continue
            if re.match(r"^(?:lih\.?|lihat)\b", ind_clean, re.I):
                continue
            nid_lex = norm_id(ind_clean)
            # Hanya tambah lema/makna Indonesia yang benar-benar belum ada di leksikon v5 agar tidak mengganggu rasio leksikon v5
            if not nid_lex or nid_lex in v5_lex_id_set:
                continue

            for gor_asli in gor_candidates:
                if len(gor_asli.split()) > 3 or any(ord(ch) > 127 for ch in gor_asli):
                    continue
                # Tolak mismatch agen 'ta ...' pada akar kata benda/kerja tunggal
                if is_ig and gor_asli.lower().startswith("ta ") and len(ind_clean.split()) == 1 and not ind_clean.lower().startswith(("pe", "peng", "peny", "pem", "pen")):
                    continue
                gor_std = bas.clean_gorontalo_text(
                    convert_1987_to_2001_orthography(gor_asli) if is_ig else gor_asli
                )
                if not bas.is_clean_lexicon(ind_clean, gor_std):
                    continue
                lkey = (nid_lex, norm_gor_compare(gor_std))
                ckey = (lkey[0].replace(" ", ""), lkey[1].replace(" ", ""))
                if not lkey[1] or lkey in v5_lex_keys or ckey in v5_lex_compact or lkey in seen_lex:
                    continue
                seen_lex.add(lkey)
                novel_lexicon.append({
                    "source_sheet": sheet_name,
                    "source_row": idx,
                    "indonesia": ind_clean,
                    "gorontalo": gor_std,
                    "gorontalo_asli": gor_asli,
                    "sumber_kamus": "indonesia_gorontalo_1987" if is_ig else "gorontalo_indonesia_2001",
                })

    repairs_list = list(repairs_by_v5_idx.values())
    repair_by_type: dict[str, int] = {}
    repair_by_split: dict[str, int] = {}
    repair_by_sheet: dict[str, int] = {}
    for rep in repairs_list:
        repair_by_type[rep["repair_type"]] = repair_by_type.get(rep["repair_type"], 0) + 1
        repair_by_split[rep["split"]] = repair_by_split.get(rep["split"], 0) + 1
        repair_by_sheet[rep["source_sheet"]] = repair_by_sheet.get(rep["source_sheet"], 0) + 1

    novel_by_sheet: dict[str, int] = {}
    for np_item in novel_parallel:
        novel_by_sheet[np_item["source_sheet"]] = novel_by_sheet.get(np_item["source_sheet"], 0) + 1

    lex_by_sheet: dict[str, int] = {}
    for nl_item in novel_lexicon:
        lex_by_sheet[nl_item["source_sheet"]] = lex_by_sheet.get(nl_item["source_sheet"], 0) + 1

    rej_counts: dict[str, int] = {}
    for rej in rejected_log:
        key = f"{rej['sheet']}::{rej['reason'].split(':')[0]}"
        rej_counts[key] = rej_counts.get(key, 0) + 1

    report = {
        "mode": "apply" if apply else "dry_run",
        "v5_baseline": {
            "combined_parallel_sentences": int(len(comb_df)),
            "combined_lexicon_id_to_gor": int(len(lex_df)),
            "seq2seq_test_ind2gor": int((split_frames["test"]["type"] == "parallel_sentence").sum() // 2),
        },
        "whitelisted_candidates_valid": {k: len(v) for k, v in candidates.items()},
        "blacklisted_sheets_skipped": BLACKLISTED_SHEETS,
        "repairs_summary": {
            "total_v5_rows_repaired": len(repairs_list),
            "by_repair_type": repair_by_type,
            "by_v5_split": repair_by_split,
            "by_source_sheet": repair_by_sheet,
            "id_ocr_manual_review_queue": len(id_ocr_review_queue),
        },
        "novel_parallel_summary": {
            "total_novel_parallel": len(novel_parallel),
            "by_source_sheet": novel_by_sheet,
        },
        "novel_lexicon_summary": {
            "total_novel_lexicon": len(novel_lexicon),
            "by_source_sheet": lex_by_sheet,
        },
        "rejected_summary": {
            "total_rejected_rows": len(rejected_log),
            "top_reasons": dict(sorted(rej_counts.items(), key=lambda kv: -kv[1])[:25]),
        },
    }

    # Simpan berkas pratinjau read-only
    def write_jsonl(path: Path, items: list[dict]):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")

    (PREVIEW_DIR / "dry_run_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_jsonl(PREVIEW_DIR / "repairs_preview.jsonl", repairs_list)
    write_jsonl(PREVIEW_DIR / "novel_parallel_preview.jsonl", novel_parallel)
    write_jsonl(PREVIEW_DIR / "novel_lexicon_preview.jsonl", novel_lexicon)
    write_jsonl(PREVIEW_DIR / "id_ocr_review_queue.jsonl", id_ocr_review_queue)
    write_jsonl(PREVIEW_DIR / "rejected_preview.jsonl", rejected_log)

    if apply:
        # Cadangkan v5 terlebih dahulu sebelum menulis berkas gabungan v6 (hanya saat --apply).
        backup_dir = DT_DIR / "v5_backup_before_reconcile"
        backup_dir.mkdir(exist_ok=True)
        for fname in ("combined_parallel_sentences.parquet", "combined_lexicon_id_to_gor.parquet"):
            src = DT_DIR / fname
            dst = backup_dir / fname
            if src.exists() and not dst.exists():
                dst.write_bytes(src.read_bytes())

        comb_updated = comb_df.copy()
        for rep in repairs_list:
            i = rep["v5_idx"]
            comb_updated.at[i, "src_indonesia"] = rep["new_ind"]
            comb_updated.at[i, "tgt_gorontalo"] = rep["new_gor"]
            comb_updated.at[i, "tgt_gorontalo_asli"] = rep["new_gor_asli"]

        next_par_id = len(comb_updated) + 1
        new_par_rows = []
        for offset, np_item in enumerate(novel_parallel):
            new_par_rows.append({
                "id": f"comb_par_v6_{next_par_id + offset:05d}",
                "src_indonesia": np_item["src_indonesia"],
                "tgt_gorontalo": np_item["tgt_gorontalo"],
                "tgt_gorontalo_asli": np_item["tgt_gorontalo_asli"],
                "lema": np_item["lema"],
                "kata_terkait": "",
                "sumber_kamus": np_item["sumber_kamus"],
            })
        if new_par_rows:
            comb_updated = pd.concat([comb_updated, pd.DataFrame(new_par_rows)], ignore_index=True)

        lex_updated = lex_df.copy()
        next_lex_id = len(lex_updated) + 1
        new_lex_rows = []
        for offset, nl_item in enumerate(novel_lexicon):
            new_lex_rows.append({
                "id": f"comb_lex_v6_{next_lex_id + offset:05d}",
                "indonesia": nl_item["indonesia"],
                "gorontalo": nl_item["gorontalo"],
                "gorontalo_asli": nl_item["gorontalo_asli"],
                "tipe": "kurasi_xlsx",
                "lema_induk": nl_item["gorontalo"] if nl_item["sumber_kamus"] == "gorontalo_indonesia_2001" else nl_item["indonesia"],
                "sumber_kamus": nl_item["sumber_kamus"],
            })
        if new_lex_rows:
            lex_updated = pd.concat([lex_updated, pd.DataFrame(new_lex_rows)], ignore_index=True)

        comb_updated.to_parquet(DT_DIR / "combined_parallel_sentences.parquet", index=False)
        lex_updated.to_parquet(DT_DIR / "combined_lexicon_id_to_gor.parquet", index=False)
        report["applied_counts"] = {
            "combined_parallel_sentences_new": int(len(comb_updated)),
            "combined_lexicon_id_to_gor_new": int(len(lex_updated)),
        }
        (PREVIEW_DIR / "dry_run_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    return report


def main():
    parser = argparse.ArgumentParser(description="Audit dry-run & rekonsiliasi korpus v5 dengan ekstraksilain.xlsx")
    parser.add_argument("--apply", action="store_true", help="Terapkan perbaikan & tambahan ke file combined_*.parquet (default: dry-run read-only)")
    args = parser.parse_args()

    report = reconcile_and_audit(apply=args.apply)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
