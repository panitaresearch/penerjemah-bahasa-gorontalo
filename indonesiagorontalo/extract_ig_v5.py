#!/usr/bin/env python3
"""
Visual Multimodal Extractor v5 for Kamus Bahasa Indonesia-Gorontalo (Pateda, 1987).

Uses 100% CPU-only OpenCV layout + strip-deskewing + Tesseract 5 (-l ind+eng --psm 6 tsv):
1. Strip-splits columns at minimal-ink horizontal valleys (~650-900px), avoiding stub strips,
   and deskews each strip locally ([-4.5 deg, +4.5 deg]) to cure book-spine curvature.
2. Feeds clean grayscale (specks <5px removed) to Tesseract for superior character recognition.
3. Detects visual italics via projection-variance slant ratio (Gorontalo words/examples = Italic;
   Indonesian headwords/gloss-tags/examples = Upright Roman) and visual bold via distanceTransform
   stroke width on headwords & sub-entries.
4. Visually recovers '= =' example delimiters in inter-word gaps, heals glottal '?' misreads
   ('[27]' between vowels -> '?'), fixes Pateda 1987 trailing italic ';'-as-'s' phonotactic
   glitches, and never splits Indonesian example sentences on internal commas ','.
5. Preserves clean Pateda 1987 orthography while also generating Pateda 2001 standarized form.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pymupdf

ROOT_DIR = Path('/content/drive/MyDrive/REPO/richie/modelgorontalo')
IG_DIR = ROOT_DIR / 'indonesiagorontalo'
DATASET_DIR = IG_DIR / 'dataset'
CACHE_V5_DIR = DATASET_DIR / 'ocr_cache_v5'
BATCHES_DIR = DATASET_DIR / 'batches'
HURUF_DIR = DATASET_DIR / 'huruf'

BATCH_GROUPS = {
    1: ['A', 'B', 'C'],
    2: ['D', 'E', 'F', 'G', 'H', 'I', 'J'],
    3: ['K', 'L'],
    4: ['M', 'N', 'O', 'P', 'Q'],
    5: ['R', 'S'],
    6: ['T', 'U', 'V', 'W', 'Y', 'Z'],
}


def convert_1987_to_2001_orthography(text: str) -> str:
    """Konversi ortografi Pateda 1987 (vokal ganda, glotal '?', tanpa glide) -> Pateda 2001."""
    if not text:
        return ''
    s = text
    glide_subs = [
        (r'\bwaatia\b', 'wa:tiya'), (r'\bWaatia\b', 'Wa:tiya'),
        (r'\bamiaatia\b', 'amiya:tiya'), (r'\bAmiaatia\b', 'Amiya:tiya'),
        (r'\btio\b', 'tiyo'), (r'\bTio\b', 'Tiyo'),
        (r'\btimongolio\b', 'timongoliyo'), (r'\bTimongolio\b', 'Timongoliyo'),
        (r'\bboito\b', 'boyito'), (r'\bBoito\b', 'Boyito'),
        (r'\bbotia\b', 'botiya'), (r'\bBotia\b', 'Botiya'),
        (r'\bbotie\b', 'botiye'), (r'\bBotie\b', 'Botiye'),
        (r'\butia\b', 'utiya'), (r'\bUtia\b', 'Utiya'),
        (r'\bmai\b', 'mayi'), (r'\bMai\b', 'Mayi'),
        (r'\bwau\b', 'wawu'), (r'\bWau\b', 'Wawu'),
        (r'\btuau\b', 'tuwawu'), (r'\bduluo\b', 'duluwo'),
        (r'\bdoi\b', 'doyi'), (r'\bluahu\b', 'luwahu'),
        (r'\bmoali\b', 'mowali'), (r'\bloali\b', 'lowali'),
        (r'\bdiila\b', 'di:la'), (r'\bDiila\b', 'Di:la'),
        (r'\bdiaalu\b', 'diya:lu'), (r'\bDiaalu\b', 'Diya:lu'),
        (r'lio\b', 'liyo'), (r'mai\b', 'mayi'),
    ]
    for pat, rep in glide_subs:
        s = re.sub(pat, rep, s)
    for v in ('a', 'e', 'i', 'o', 'u'):
        s = re.sub(v + v, v + ':', s, flags=re.IGNORECASE)
    s = re.sub(r'(?<=[a-zA-Z:])\?(?=[a-zA-Z:])', '', s)
    return s

LETTER_TO_PDF = {
    'A': 'Abjad A Kamus Bhsa Indo-Gtlo.pdf',
    'B': 'Abjad B Kamus Bhsa Indo-Gtlo.pdf',
    'C': 'Abjad C Kamus Bhsa Indo-Gtlo.pdf',
    'D': 'Abjad D Kamus Bhsa Indo-Gtlo.pdf',
    'E': 'Abjad E Kamus Bhsa Indo-Gtlo.pdf',
    'F': 'Abjad F Kamus Bhsa Indo-Gtlo.pdf',
    'G': 'Abjad G Kamus Bhsa Indo-Gtlo.pdf',
    'H': 'Abjad H Kamus Bhsa Indo-Gtlo.pdf',
    'I': 'Abjad I Kamus Bhsa Indo-Gtlo.pdf',
    'J': 'Abjad J Kamus Bhsa Indo-Gtlo.pdf',
    'K': 'Abjad K Kamus Bhsa Indo-Gtlo.pdf',
    'L': 'Abjad L Kamus Bhsa Indo-Gtlo.pdf',
    'M': 'Abjad M Kamus Bhsa Indo-Gtlo.pdf',
    'N': 'Abjad N Kamus Bhsa Indo-Gtlo.pdf',
    'O': 'Abjad O Kamus Bhsa Indo-Gtlo.pdf',
    'P': 'Abjad P Kamus Bhsa Indo-Gtlo.pdf',
    'Q': 'Abjad Q Kamus Bhsa Indo-Gtlo.pdf',
    'R': 'Abjad R Kamus Bhsa Indo-Gtlo.pdf',
    'S': 'Abjad S Kamus Bhsa Indo-Gtlo.pdf',
    'T': 'Abjad T kamus Bhsa Indo-Gtlo.pdf',
    'U': 'Abjad U Kamus Bhsa Indo-Gtlo.pdf',
    'V': 'Abjad V Kamus Bhsa Indo-Gtlo.pdf',
    'W': 'Abjad W Kamus Bhsa Indo-Gtlo.pdf',
    'Y': 'Abjad Y Kamus Bhsa Indo-Gtlo.pdf',
    'Z': 'Abjad Z Kamus Bhsa Indo-Gtlo.pdf',
}

CLEAR_INDONESIAN_WORDS = {
    'yang', 'yg', 'dan', 'atau', 'dengan', 'dg', 'dng', 'di', 'ke', 'dari', 'dr',
    'dalam', 'dl', 'dlm', 'pada', 'pd', 'untuk', 'utk', 'bagi', 'guna', 'sebab',
    'karena', 'krn', 'kalau', 'jika', 'bila', 'itu', 'ini', 'jangan', 'akan',
    'sudah', 'belum', 'masih', 'sedang', 'telah', 'tidak', 'tdk', 'tak', 'bukan',
    'saya', 'aku', 'kami', 'kita', 'engkau', 'kamu', 'dia', 'ia', 'mereka', 'beliau',
    'adalah', 'ialah', 'yaitu', 'yakni', 'seperti', 'spt', 'sebagai', 'sbg', 'oleh',
    'kepada', 'kpd', 'daripada', 'dp', 'mesti', 'harus', 'boleh', 'dapat', 'bisa',
    'mau', 'ingin', 'perlu', 'hanya', 'saja', 'juga', 'pula', 'baru', 'dulu', 'lagi',
    'lebih', 'kurang', 'sangat', 'sekali', 'tiap', 'setiap', 'biasanya', 'sering',
    'selalu', 'jarang', 'barangkali', 'mungkin', 'apakah', 'mengapa', 'bagaimana',
    'siapa', 'mana', 'kapan', 'berapa', 'semua', 'segala', 'seluruh', 'banyak',
    'sedikit', 'beberapa', 'orang', 'rumah', 'anak', 'ayah', 'ibu', 'bapak', 'air',
    'tanah', 'kayu', 'batu', 'api', 'hari', 'bulan', 'tahun', 'waktu', 'jam',
    'pukul', 'makan', 'minum', 'tidur', 'pergi', 'datang', 'pulang', 'duduk', 'berdiri',
    'jalan', 'lari', 'lihat', 'dengar', 'kata', 'ambil', 'beri', 'buat', 'kerja',
    'bawa', 'tahu', 'kenal', 'suka', 'cinta', 'marah', 'takut', 'senang', 'susah',
    'sakit', 'sehat', 'mati', 'hidup', 'besar', 'kecil', 'panjang', 'pendek', 'tinggi',
    'rendah', 'jauh', 'dekat', 'cepat', 'lambat', 'baik', 'buruk', 'jahat', 'benar',
    'salah', 'tetapi', 'tapi', 'namun', 'melainkan', 'sedangkan', 'walaupun', 'meskipun',
    'supaya', 'agar', 'hingga', 'sampai', 'sejak', 'selama', 'ketika', 'waktu',
    'dsb', 'dst', 'dll', 'mis', 'ump', 'lih', 'lihat', 'misalnya', 'umpamanya',
    'antara', 'tentang', 'terhadap', 'atas', 'bawah', 'depan', 'belakang', 'luar',
    'sini', 'situ', 'sana', 'sekarang', 'tadi', 'kemarin', 'besok', 'lusa', 'nanti',
    'pernah', 'serba', 'sama', 'lain', 'sendiri', 'satu', 'dua', 'tiga', 'empat',
    'lima', 'enam', 'tujuh', 'delapan', 'sembilan', 'sepuluh', 'seratus', 'seribu',
}

VALID_S_ENDINGS = {
    'es', 'bus', 'gas', 'kas', 'jas', 'pas', 'tas', 'pos', 'nos', 'las', 'mas', 'ros',
    'abses', 'absen', 'krisis', 'proses', 'dosis', 'tesis', 'kursus', 'kamus', 'modus',
    'bonus', 'virus', 'sinus', 'venus', 'plus', 'minus', 'kelas', 'gelas', 'beras',
    'keras', 'kertas', 'atlas', 'batas', 'bebas', 'luas', 'puas', 'panas', 'emas',
    'lemas', 'malas', 'balas', 'cemas', 'ikhlas', 'dinas', 'lunas', 'tegas', 'tugas',
    'bekas', 'lekas', 'nafas', 'napas', 'kipas', 'kuas', 'humus', 'rumus', 'bagus',
    'halus', 'lurus', 'kurus', 'terus', 'arus', 'harus', 'khusus', 'putus', 'ratus',
    'utus', 'mulus', 'tulus', 'alus', 'jenis', 'manis', 'tipis', 'baris', 'keris',
    'habis', 'tulis', 'lukis', 'amis', 'kumis', 'tangis', 'iris', 'garis', 'artis',
    'polis', 'karcis', 'buncis', 'kubis', 'iblis', 'kompas', 'kapas', 'ampas',
    'naik', 'tebas', 'rebas', 'alas', 'walas', 'malas', 'ulas', 'pulas', 'tuntas',
    'pantas', 'lintas', 'rentas', 'entas', 'getas', 'atas', 'petas', 'tes', 'les',
    'nos', 'os', 'ks', 'kb', 'kk', 'kd', 'kt', 'ki', 'kp', 'kbil', 'kg',
}

GORONTALO_1987_WORDS = {
    'wa?u', 'wau', 'waatia', 'yi?o', 'yio', 'ito', 'tio', 'timongolio', 'ami', 'amiaatia',
    'boito', 'botia', 'utia', 'yito', 'to', 'lo', 'mo', 'po', 'ma', 'ma?o', 'mao',
    'mai', 'mola', 'mota', 'o', 'u', 'ta', 'ti', 'te', 'olo', 'oli', 'ole', 'ode',
    'onto', 'wonu', 'wanu', 'sababu', 'diila', 'dila', 'diipo', 'dipo', 'didu',
    'diaalu', 'diaaluo', 'woluo', 'donggo', 'onggo', 'de', 'do', 'noonu', 'utoonu',
    'tatoono', 'openu', 'alihu', 'asali', 'potala', 'lapatao', 'musi', 'moali',
    'mowali', 'yinao', 'hilao', 'delo', 'odito', 'odia', 'odie', 'teea', 'teeto',
    'mealo', 'sambe', 'daadata', 'ngoidi', 'bolo', 'hemo', 'helo', 'tau', 'bele',
    'pale', 'hui', 'dulahu', 'dulahe', 'taluhu', 'taluhe', 'tuau', 'duluo', 'totolu',
    'wopato', 'limo', 'wolomo', 'pitu', 'walu', 'tioo', 'mopulu', 'ngo', 'ngoolo',
    'bo', 'yi', 'ya', 'te?e', 'moli', 'loli', 'poli', 'mopo', 'lopo', 'popo',
}

RN_EXCEPTIONS = {
    'karena', 'warna', 'berwarna', 'purna', 'sempurna', 'menyempurnakan', 'jernih',
    'pernah', 'ternak', 'beternak', 'peternak', 'gubernur', 'turnamen', 'jurnal',
    'kurnia', 'purnama', 'purbakala', 'sirna', 'bernas', 'kornel', 'alternatif',
    'internasional', 'modern', 'eksternal', 'internal', 'furna',
}

EDITORIAL_TAGS = {
    '(ks)', '(kb)', '(kk)', '(kd)', '(kt)', '(ki)', '(kp)', '(kbil)', '(kg)', '(kl)',
    'ks.', 'kb.', 'kk.', 'kd.', 'kt.', 'ki.', 'kp.', 'kbil.', 'kg.', 'kl.', 'kbb.',
    'ks', 'kb', 'kk', 'kd', 'kt', 'ki', 'kp', 'kbil', 'kg', 'kl', 'kbb',
}


def deskew_gray(gray: np.ndarray, max_ang: float = 4.5):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    small = cv2.resize(th, (0, 0), fx=0.5, fy=0.5)
    sh, sw = small.shape
    if sh < 20 or sw < 20:
        return gray, 0.0
    best_ang, best_var = 0.0, -1.0
    for ang in np.linspace(-max_ang, max_ang, 73):
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST)
        v = float(np.var(rot.sum(axis=1)))
        if v > best_var:
            best_var, best_ang = v, ang
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), best_ang


def heal_ig_token(tok: str) -> str:
    t = tok.replace('\xad', '-').replace('—', '-').replace('–', '-')
    # Heal glottal '?' misreads between vowels in Gorontalo 1987 ('wa7u' -> 'wa?u', 'lo2otapu' -> 'lo?otapu')
    t = re.sub(r'(?<=[aiueoAIUEO])[27](?=[aiueo])', '?', t)
    # Heal common OCR swaps
    subs = [
        (r'^1o$', 'lo'),
        (r'^0lo$', 'olo'),
        (r'^wa7u$', 'wa?u'),
        (r'^wa2u$', 'wa?u'),
        (r'^yi7o$', 'yi?o'),
        (r'^yi2o$', 'yi?o'),
        (r'^ma7o$', 'ma?o'),
        (r'^ma2o$', 'ma?o'),
        (r'^sababll$', 'sababu'),
        (r'^bualmya$', 'buahnya'),
        (r'^rUlllah$', 'rumah'),
        (r'^pekeIjaan$', 'pekerjaan'),
        (r'^yg$', 'yg'),
        (r'^boito$', 'boito'),
        (r'^botia$', 'botia'),
    ]
    for pat, rep in subs:
        t = re.sub(pat, rep, t)
    # Fix 'rn' -> 'm' inside words unless known Indonesian rn word
    c_low = t.strip('.,;:!?()').lower()
    if 'rn' in c_low and c_low not in RN_EXCEPTIONS:
        if re.match(r'^[a-z\?]+rn[a-z\?]+$', c_low):
            t = t.replace('rn', 'm')
    return t


def is_unambiguous_id(w: str) -> bool:
    c = w.strip('.,;:!?()"\'').lower()
    if not c or '?' in c:
        return False
    if c in CLEAR_INDONESIAN_WORDS:
        return True
    if re.match(r'^(?:meng|meny|mem|men|ber|ter|per|ke|pe|di)[a-z]{3,}$', c) and not c.startswith(('mo', 'lo', 'po', 'me', 'le', 'pe')):
        return True
    if re.match(r'^(?:meng|meny|mem|men|ber|ter)[a-z]{3,}$', c):
        return True
    if len(c) >= 5 and re.search(r'(?:nya|kan)$', c):
        return True
    if c[-1] in 'bcdfghjklmnpqrstvwxyz' and c not in {'tau', 'ha'}:
        return True
    return False


def has_eq_ink_in_gap(bw_sub: np.ndarray, y0: int, y1: int, x0: int, x1: int) -> bool:
    """Detect '= =' (double equals or double short dash) missed by Tesseract in a wide gap."""
    h, w = bw_sub.shape
    ry0 = max(0, y0 + 2)
    ry1 = min(h, y1 - 2)
    rx0 = max(0, x0 + 6)
    rx1 = min(w, x1 - 6)
    if ry1 <= ry0 + 4 or rx1 <= rx0 + 20:
        return False
    patch = (bw_sub[ry0:ry1, rx0:rx1] > 0).astype(np.uint8) * 255
    area = int((patch > 0).sum())
    if area < 16 or area > 450:
        return False
    nl_g, _, _, _ = cv2.connectedComponentsWithStats(patch)
    # '= =' has multiple horizontal bar strokes (>=2 CCs, e.g. 2 to 8 strokes)
    if nl_g - 1 < 2 or nl_g - 1 > 8:
        return False
    ys, xs = np.where(patch > 0)
    span_h = int(ys.max() - ys.min()) + 1
    span_w = int(xs.max() - xs.min()) + 1
    lh = max(1, y1 - y0)
    return span_w >= 16 and span_h <= int(lh * 0.72)


def parse_eq_token(tok: str) -> tuple[bool, str]:
    """Return (is_eq, trailing_punct) if token represents '= =' / '==' / '=' / '~' (with optional punct)."""
    t = tok.strip()
    if not t:
        return False, ''
    if t in {'=', '==', '= =', '===', '--', '---', '- -', '=-', '-=', '~=', '=~', '”', '“', '""'}:
        return True, ''
    m = re.match(r'^([=\-~_]{1,4})([,.;:!?]*)$', t)
    if m:
        stem, punct = m.group(1), m.group(2)
        if stem == '-' and not punct:
            return True, ''
        if stem in {'=', '==', '===', '--', '---', '=-', '-=', '~', '~=', '=~', '_', '__'}:
            return True, punct
    return False, ''


def is_eq_token(tok: str) -> bool:
    return parse_eq_token(tok)[0]


_ID_ABBREV_RULES = [
    (re.compile(r'\btdk\b', re.I), 'tidak'),
    (re.compile(r'\butk\b', re.I), 'untuk'),
    (re.compile(r'\byg\b', re.I), 'yang'),
    (re.compile(r'\bdgn\b', re.I), 'dengan'),
    (re.compile(r'\bdg\b', re.I), 'dengan'),
    (re.compile(r'\bkrn\b', re.I), 'karena'),
    (re.compile(r'\bdlm\b', re.I), 'dalam'),
    (re.compile(r'\bdl\b', re.I), 'dalam'),
    (re.compile(r'\bpd\b', re.I), 'pada'),
    (re.compile(r'\bsbg\b', re.I), 'sebagai'),
    (re.compile(r'\bmsl\.?(?=\s|$|,|;)', re.I), 'misalnya'),
    (re.compile(r'\bdsb\.?(?=\s|$|,|;)', re.I), 'dan sebagainya'),
    (re.compile(r'\bdst\.?(?=\s|$|,|;)', re.I), 'dan seterusnya'),
    (re.compile(r'\bdll\.?(?=\s|$|,|;)', re.I), 'dan lain-lain'),
    (re.compile(r'\bspt\b', re.I), 'seperti'),
    (re.compile(r'\bthp\b', re.I), 'terhadap'),
    (re.compile(r'\bkpd\b', re.I), 'kepada'),
    (re.compile(r'\bdr\b', re.I), 'dari'),
    (re.compile(r'\btlh\b', re.I), 'telah'),
    (re.compile(r'\bsdh\b', re.I), 'sudah'),
    (re.compile(r'\bblm\b', re.I), 'belum'),
    (re.compile(r'\bbhw\b', re.I), 'bahwa'),
    (re.compile(r'\bmsh\b', re.I), 'masih'),
    (re.compile(r'\bjg\b', re.I), 'juga'),
    (re.compile(r'(?<!\d\s)\bkm\b', re.I), 'karena'),
    (re.compile(r'\bhny\b', re.I), 'hanya'),
]


def expand_indonesian_abbrev(text: str) -> str:
    """Expand 1987 dictionary Indonesian shorthand inside example sentences cleanly."""
    s = text
    for pat, repl in _ID_ABBREV_RULES:
        def _sub(m, r=repl):
            w = m.group(0)
            return r.capitalize() if w and w[0].isupper() else r
        s = pat.sub(_sub, s)
    return re.sub(r'\s+', ' ', s).strip()


def ocr_ig_page(pdf_path: Path, page_idx_1based: int, dpi: int = 350):
    """Render single IG PDF page, crop borders, strip-deskew, OCR, and annotate indent/italic/bold."""
    doc = pymupdf.open(str(pdf_path))
    page = doc[page_idx_1based - 1]
    pix = page.get_pixmap(dpi=dpi)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
    h, w = gray.shape

    # 1. Crop white interior inside dark scanner margins
    col_p85 = np.percentile(gray[int(h * 0.12):int(h * 0.85), :], 85, axis=0)
    wc = np.where(col_p85 > 205)[0]
    px0 = int(wc[0]) + 14 if len(wc) > 100 else int(w * 0.05)
    px1 = int(wc[-1]) - 14 if len(wc) > 100 else int(w * 0.95)

    row_p85 = np.percentile(gray[:, max(0, px0 + 40):min(w, px1 - 40)], 85, axis=1)
    top_dark = [r for r in range(0, int(h * 0.15)) if row_p85[r] < 205]
    bot_dark = [r for r in range(int(h * 0.65), h) if row_p85[r] < 205]
    wr = np.where(row_p85 > 205)[0]
    py0 = (top_dark[-1] + 12) if top_dark else (int(wr[0]) + 10 if len(wr) > 100 else int(h * 0.02))
    py1 = (bot_dark[0] - 12) if bot_dark else (int(wr[-1]) - 10 if len(wr) > 100 else int(h * 0.98))
    if py1 <= py0 + 250:
        py0, py1 = int(h * 0.02), int(h * 0.98)

    page_gray = gray[py0:py1, px0:px1].copy()
    # Coarse whole-page deskew first so vertical column gutter is straight top-to-bottom
    page_gray, _ = deskew_gray(page_gray, max_ang=3.0)
    ph, pw = page_gray.shape
    bw_page = cv2.adaptiveThreshold(
        page_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 18
    )

    # Erase scanner frame bars, vertical dark lines, and tiny specks <5px
    nl_f, flab, fst, _ = cv2.connectedComponentsWithStats(bw_page)
    rules_x = []
    for lbl in range(1, nl_f):
        fw, fh, fa = fst[lbl, cv2.CC_STAT_WIDTH], fst[lbl, cv2.CC_STAT_HEIGHT], fst[lbl, cv2.CC_STAT_AREA]
        fx = fst[lbl, cv2.CC_STAT_LEFT]
        if (fw > pw * 0.35 and fh < 45) or (fh > ph * 0.12 and fw < 22) or (fh >= 35 and fw <= 8 and fh / max(1, fw) >= 5.0):
            if pw * 0.36 <= fx <= pw * 0.64 and fh > ph * 0.20:
                rules_x.append(int(fx + fw // 2))
            bw_page[flab == lbl] = 0
            page_gray[flab == lbl] = 255
        elif fa < 5 and fw <= 3 and fh <= 3:
            bw_page[flab == lbl] = 0
            page_gray[flab == lbl] = 255

    # 2. 2-Column split via central gutter in [0.38*pw, 0.62*pw]
    x_ink = (bw_page[int(ph * 0.08):int(ph * 0.88), :] > 0).sum(axis=0)
    tx_idx = np.where(x_ink > 12)[0]
    if len(tx_idx) < 40:
        return []
    tx0, tx1 = int(tx_idx[0]), int(tx_idx[-1])
    mlo, mhi = int(pw * 0.38), int(pw * 0.62)
    if rules_x:
        gutter = int(np.median(rules_x))
    else:
        col_dark = (bw_page[int(ph * 0.10):int(ph * 0.88), mlo:mhi] > 0).sum(axis=0)
        conv = np.convolve(col_dark, np.ones(25), mode='valid')
        gutter = mlo + int(np.argmin(conv)) + 12

    col_bounds = [
        (max(0, tx0 - 14), max(10, gutter - 4)),
        (min(pw - 10, gutter + 4), min(pw, tx1 + 14)),
    ]

    env = os.environ.copy()
    env['OMP_THREAD_LIMIT'] = '1'
    lines_out = []

    for c_idx, (cx0, cx1) in enumerate(col_bounds):
        col_g_full = page_gray[:, cx0:cx1]
        col_b_full = bw_page[:, cx0:cx1]
        ch_f, cw_f = col_g_full.shape
        if cw_f < 120 or (col_b_full > 0).sum() < 400:
            continue

        # Trim top header page-number/letter and bottom isolated page number per column
        row_ink = (col_b_full > 0).sum(axis=1)
        active_y = np.where(row_ink > 6)[0]
        if len(active_y) == 0:
            continue
        y_top = max(0, int(active_y[0]) - 8)
        y_bot = min(ch_f, int(active_y[-1]) + 8)

        for _trim_pass in range(3):
            in_seg, seg_start = False, y_top
            segs = []
            for r in range(y_top, y_bot):
                if row_ink[r] > 5:
                    if not in_seg:
                        in_seg, seg_start = True, r
                else:
                    if in_seg:
                        next_ink = np.where(row_ink[r:min(y_bot, r + 15)] > 5)[0]
                        if len(next_ink) == 0:
                            segs.append((seg_start, r))
                            in_seg = False
            if in_seg:
                segs.append((seg_start, y_bot))
            trimmed = False
            if len(segs) >= 2:
                first_s, first_e = segs[0]
                next_s = segs[1][0]
                if first_s < ch_f * 0.14 and next_s - first_e >= 22 and (first_e - first_s) < 68:
                    xs_f = np.where(col_b_full[first_s:first_e, :].sum(axis=0) > 0)[0]
                    if len(xs_f) == 0 or (int(xs_f[-1]) - int(xs_f[0])) < 200:
                        y_top = max(0, next_s - 8)
                        trimmed = True
                if not trimmed:
                    prev_end = segs[-2][1]
                    last_s, last_e = segs[-1]
                    if last_s - prev_end >= 22 and (last_e - last_s) < 65:
                        xs_l = np.where(col_b_full[last_s:last_e, :].sum(axis=0) > 0)[0]
                        if len(xs_l) == 0 or (int(xs_l[-1]) - int(xs_l[0])) < 180 or (last_e - last_s) <= 14:
                            y_bot = prev_end + 8
                            trimmed = True
            if not trimmed:
                break

        col_g = col_g_full[y_top:y_bot, :]
        col_b = col_b_full[y_top:y_bot, :]
        cch, cw = col_g.shape
        if cch < 40 or cw < 100:
            continue

        # Split column into horizontal strips at minimal-ink valleys (~580-920px, no stub <320px)
        r_sum = (col_b > 0).sum(axis=1).astype(np.float32)
        r_smooth = np.convolve(r_sum, np.ones(5, dtype=np.float32), mode='same')
        cuts = [0]
        cur_y = 0
        while cur_y + 1020 < cch:
            lo = cur_y + 560
            hi = min(cch - 340, cur_y + 920)
            if lo >= hi:
                break
            cut = lo + int(np.argmin(r_smooth[lo:hi]))
            cuts.append(cut)
            cur_y = cut
        cuts.append(cch)

        col_raw_lines = []
        for s_idx in range(len(cuts) - 1):
            sy0, sy1 = cuts[s_idx], cuts[s_idx + 1]
            if sy1 - sy0 < 20:
                continue
            sub_g = col_g[sy0:sy1, :].copy()
            max_ang = 3.8 if (sy1 - sy0) >= 320 else 1.8
            sub_g_desk, _ = deskew_gray(sub_g, max_ang=max_ang)

            bw_sub = cv2.adaptiveThreshold(
                sub_g_desk, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 18
            )
            nl, lab, st, _ = cv2.connectedComponentsWithStats(bw_sub)
            for lbl in range(1, nl):
                if st[lbl, cv2.CC_STAT_AREA] < 5 and st[lbl, cv2.CC_STAT_WIDTH] <= 3 and st[lbl, cv2.CC_STAT_HEIGHT] <= 3:
                    sub_g_desk[lab == lbl] = 255
                    bw_sub[lab == lbl] = 0

            dist = cv2.distanceTransform(bw_sub, cv2.DIST_L2, 3)

            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tf:
                tmp_path = tf.name
            cv2.imwrite(tmp_path, sub_g_desk)
            res = subprocess.run(
                ['tesseract', tmp_path, 'stdout', '-l', 'ind+eng', '--psm', '6', 'tsv'],
                capture_output=True, text=True, env=env
            )
            Path(tmp_path).unlink(missing_ok=True)

            strip_lines = {}
            for row in res.stdout.splitlines()[1:]:
                parts = row.split('\t')
                if len(parts) == 12 and int(parts[0]) == 5 and parts[11].strip():
                    key = (int(parts[2]), int(parts[3]), int(parts[4]))
                    wx, wy, ww, wh = int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])
                    try:
                        conf = float(parts[10])
                    except ValueError:
                        conf = 50.0
                    tok = parts[11].strip()
                    if ww < 2 or wh < 3:
                        continue
                    if tok in {'|', '!', '[', ']'} and ww <= 10:
                        continue
                    is_pure_punct = bool(re.match(r'^[=\-~_.,:;()?!]+$', tok))
                    if conf >= 0 and ((is_pure_punct and conf < 10.0) or (not is_pure_punct and (conf < 20.0 or (conf < 46.0 and len(tok) <= 3 and tok.lower() not in {'di', 'ke', 'yg', 'dg', 'si', 'te', 'ti', 'lo', 'to', 'ma', 'bo', 'ta', 'o', 'u', 'i', 'ii'})))):
                        continue
                    ink = bw_sub[wy:wy + wh, wx:wx + ww] > 0
                    sw_val = float(dist[wy:wy + wh, wx:wx + ww][ink].mean()) if ink.sum() >= 6 else 1.5

                    slant = 0.75
                    if ww >= 10 and wh >= 9 and ink.sum() >= 12:
                        crop_b = bw_sub[wy:wy + wh, wx:wx + ww]
                        v0 = float(np.var(crop_b.sum(axis=0).astype(np.float32)))
                        v_it = 0.0
                        for sh_ang in (13.0, 16.0):
                            sh_t = np.tan(np.deg2rad(sh_ang))
                            M_sh = np.float32([[1, sh_t, -sh_t * wh / 2], [0, 1, 0]])
                            warped = cv2.warpAffine(crop_b, M_sh, (ww + int(abs(sh_t) * wh) + 4, wh), flags=cv2.INTER_NEAREST)
                            v_it = max(v_it, float(np.var(warped.sum(axis=0).astype(np.float32))))
                        slant = v_it / (v0 + 1e-5)

                    tok = heal_ig_token(tok)
                    unambig_id = is_unambiguous_id(tok)
                    is_sym = is_eq_token(tok) or bool(re.match(r'^[\d\.,;:!?()\[\]"\'-]+$', tok))
                    t_low = tok.strip('.,;:!?()"\'').lower()
                    if is_sym or unambig_id or t_low in EDITORIAL_TAGS or re.match(r'^\([1-9]\)$', tok):
                        is_it = False
                    elif t_low in GORONTALO_1987_WORDS or '?' in t_low:
                        is_it = (slant > 0.88) or ('?' in t_low)
                    else:
                        is_it = (slant > 1.06) and (ww >= 10)

                    strip_lines.setdefault(key, []).append({
                        'x': wx, 'x1': wx + ww, 'y': wy, 'y1': wy + wh,
                        'tok': tok, 'it': is_it, 'sw': sw_val, 'slant': slant,
                        'abs_y': y_top + sy0 + wy, 'unambig_id': unambig_id, 'conf': conf
                    })

            for k in sorted(strip_lines.keys()):
                ws = sorted(strip_lines[k], key=lambda d: d['x'])
                while len(ws) >= 2 and ws[0]['tok'] in {"'", '"', '#', '·', '•', '.', '-', '—', "'#"}:
                    ws = ws[1:]
                # Drop faint back-of-page bleed/stamp lines (high low-conf gibberish ratio, e.g. Abjad Q)
                if len(ws) >= 3:
                    mean_conf = sum(max(0.0, w_d.get('conf', 50.0)) for w_d in ws) / len(ws)
                    noisy_cnt = sum(
                        1 for w_d in ws
                        if len(w_d['tok']) <= 2 or not any(c.islower() for c in w_d['tok']) or w_d.get('conf', 50.0) < 42.0
                    )
                    if mean_conf < 50.0 and (noisy_cnt / len(ws)) >= 0.55:
                        continue
                raw_joined = ' '.join(w_d['tok'] for w_d in ws).strip()
                if re.match(r'^(?:[A-Z]\s+)?\d{1,3}$', raw_joined) and (ws[0]['abs_y'] < ch_f * 0.14 or ws[0]['abs_y'] > ch_f * 0.84):
                    continue
                if re.match(r'^[A-Z]$', raw_joined) and ws[0]['abs_y'] < ch_f * 0.16:
                    continue
                if len(raw_joined) == 1 and not raw_joined.isalpha():
                    continue

                # Recover missing '= =' in wide inter-word gaps (30..160px) before a non-italic token
                recovered = []
                for idx_w, wd in enumerate(ws):
                    if idx_w > 0:
                        prev_w = ws[idx_w - 1]
                        gap = wd['x'] - prev_w['x1']
                        if 30 <= gap <= 160 and not wd['it'] and not is_eq_token(prev_w['tok']) and not is_eq_token(wd['tok']):
                            if has_eq_ink_in_gap(bw_sub, min(prev_w['y'], wd['y']), max(prev_w['y1'], wd['y1']), prev_w['x1'], wd['x']):
                                recovered.append({
                                    'x': prev_w['x1'] + 3, 'x1': wd['x'] - 3,
                                    'y': wd['y'], 'y1': wd['y1'],
                                    'tok': '==EQ==', 'it': False, 'sw': 1.4, 'slant': 0.8,
                                    'abs_y': wd['abs_y'], 'unambig_id': False
                                })
                    recovered.append(wd)

                # Collapse consecutive '=', '=', '--' tokens (even with trailing punct '=,'/'=.') into canonical '==EQ=='
                collapsed = []
                i_w = 0
                while i_w < len(recovered):
                    w0 = recovered[i_w]
                    eq0, p0 = parse_eq_token(w0['tok'])
                    if eq0:
                        if (i_w + 1 < len(recovered)):
                            eq1, p1 = parse_eq_token(recovered[i_w + 1]['tok'])
                            if eq1:
                                w0 = dict(w0)
                                w0['tok'] = '==EQ==' + (p1 or p0)
                                w0['it'] = False
                                w0['x1'] = recovered[i_w + 1]['x1']
                                collapsed.append(w0)
                                i_w += 2
                                continue
                        if w0['tok'].startswith('==EQ==') or p0 or w0['tok'] in {'==', '= =', '===', '--', '---', '- -', '='}:
                            w0 = dict(w0)
                            w0['tok'] = '==EQ==' + p0
                            w0['it'] = False
                            collapsed.append(w0)
                            i_w += 1
                            continue
                    collapsed.append(w0)
                    i_w += 1

                # Fix Pateda 1987 italic ';'-as-'s' phonotactic misreads (e.g. 'wahus' -> 'wahu;')
                for idx_w, wd in enumerate(collapsed):
                    if wd['it'] and wd['tok'] != '==EQ==':
                        m_s = re.match(r'^([a-zA-Z\?\-]+)s([,\.]?)$', wd['tok'])
                        if m_s:
                            stem = m_s.group(1)
                            full_w = (stem + 's').lower()
                            if len(stem) >= 2 and full_w not in VALID_S_ENDINGS and stem[-1].lower() in 'aiueo':
                                next_tok = collapsed[idx_w + 1] if idx_w + 1 < len(collapsed) else None
                                if (next_tok is None) or (not next_tok['it']) or re.match(r'^\(\d+\)$', next_tok['tok']):
                                    wd['tok'] = stem + ';'

                col_raw_lines.append((collapsed, bw_sub))

        if not col_raw_lines:
            continue

        # 3. Per-column left-margin reference M0 & median stroke width
        x0s = [ws[0]['x'] for ws, _ in col_raw_lines if len(ws) >= 2 or len(ws[0]['tok']) > 1]
        if not x0s:
            continue
        x0_low = float(np.percentile(x0s, 8))
        m0_candidates = [x for x in x0s if x <= x0_low + 22]
        M0 = float(np.median(m0_candidates)) if m0_candidates else x0_low
        all_sws = [wd['sw'] for ws, _ in col_raw_lines for wd in ws if len(wd['tok']) >= 2 and wd['tok'] != '==EQ==']
        med_sw = float(np.median(all_sws)) if all_sws else 1.5

        for ws, _bw in col_raw_lines:
            line_x0 = ws[0]['x']
            indent = 0 if (line_x0 <= M0 + 18 and not ws[0]['it']) else 4
            toks_out = []
            for wd in ws:
                is_bold = (wd['sw'] > med_sw * 1.07) and (not wd['it']) and (wd['tok'] != '==EQ==')
                for sub_t in wd['tok'].split():
                    toks_out.append([sub_t, bool(wd['it']), is_bold])
            if toks_out:
                lines_out.append([indent, toks_out])
    return lines_out


def ocr_and_cache_ig_page(args):
    letter, pdf_path_str, pno = args
    cache_p = CACHE_V5_DIR / f'{letter}_p{pno:03d}.json'
    if cache_p.exists():
        return letter, pno, json.loads(cache_p.read_text())
    lines = ocr_ig_page(Path(pdf_path_str), pno, dpi=350)
    cache_p.write_text(json.dumps(lines, ensure_ascii=False))
    return letter, pno, lines


# ---------------------------------------------------------------------------
# IG 1987 Linguistic & Structural Parsing (v5)
# ---------------------------------------------------------------------------

def join_ig_hyphenated_wraps(lines):
    if not lines:
        return []
    merged = []
    i = 0
    while i < len(lines):
        indent, toks = lines[i]
        while toks and toks[-1][0].endswith('-') and toks[-1][0] != '==EQ==' and not toks[-1][0].endswith('--') and (i + 1) < len(lines):
            next_indent, next_toks = lines[i + 1]
            if not next_toks or next_indent == 0:
                break
            first_next, next_it, next_bold = next_toks[0]
            prev_word, prev_it, prev_bold = toks[-1]
            prev_base = prev_word[:-1].lower()
            if len(prev_base) >= 3 and first_next.lower().startswith(prev_base[:3]):
                joined = heal_ig_token(prev_word + first_next)
            else:
                joined = heal_ig_token(prev_word[:-1] + first_next)
            toks[-1] = [joined, bool(prev_it or next_it), bool(prev_bold and next_bold)]
            if len(next_toks) > 1:
                lines[i + 1] = [next_indent, next_toks[1:]]
                break
            else:
                i += 1
        merged.append((indent, toks))
        i += 1
    return merged


def is_valid_ig_headword(toks, letter: str, allow_indented: bool = False) -> tuple[bool, str, list]:
    """
    Check if a line opens a main Indonesian headword matching letter `letter`.
    Handles both 'Headword : ...' and 'Headword, [I|II] subword : ...' patterns.
    """
    if not toks:
        return False, '', toks
    # Skip stray leading noise symbols like "'#"
    start_idx = 0
    while start_idx < len(toks) - 1 and toks[start_idx][0] in {"'", '"', '#', '·', '•', '.', '-', '—', "'#"}:
        start_idx += 1
    if start_idx >= len(toks):
        return False, '', toks

    toks_s = list(toks[start_idx:])
    # Un-glue 'Word,I' or 'Word,II' -> ['Word,', 'I'] before checking Pattern A
    m_glue = re.match(r'^([A-Za-z\-]+),(I{1,3}|[a-z]{2,})([.:;]?)$', toks_s[0][0])
    if m_glue:
        toks_s = [(m_glue.group(1) + ',', toks_s[0][1], toks_s[0][2]),
                  (m_glue.group(2) + m_glue.group(3), toks_s[0][1], toks_s[0][2])] + toks_s[1:]

    w0_raw, w0_it, w0_b = toks_s[0]
    if w0_it:
        return False, '', toks

    w0_clean = w0_raw.lstrip('\'"#.-•·(')
    if not w0_clean or not w0_clean[0].isalpha() or w0_clean[0].upper() != letter.upper():
        return False, '', toks
    # IG 1987 main headwords ALWAYS begin with a capital letter
    if not w0_clean[0].isupper():
        return False, '', toks

    # Pattern A: 'Headword, <subword/I/II/italic> ...'
    if w0_clean.endswith(',') and len(toks_s) >= 2:
        hw_cand = w0_clean.rstrip(',.:; ')
        if re.match(r'^[A-Za-z][A-Za-z\-\'\/]{0,30}$', hw_cand) and hw_cand.lower() not in CLEAR_INDONESIAN_WORDS:
            next_t, next_it, next_b = toks_s[1]
            has_near_colon = any((':' in t[0] or ';' in t[0]) for t in toks_s[1:6] if not t[1])
            next_c = next_t.strip('.,;:')
            looks_entry = next_it or next_b or has_near_colon or (next_c in {'I', 'II', 'III'}) or (
                next_c and next_c[0].islower() and re.match(r'^(?:me|mem|men|meng|meny|ber|di|ter|ke|pe|per|se)', next_c)
            )
            if looks_entry and (not allow_indented or has_near_colon or next_b or next_c in {'I', 'II', 'III'}):
                return True, hw_cand, toks_s[1:]

    # Pattern B: 'Headword [Word2] : ...' or 'Headword [I|II] : ...'
    cand_toks = []
    rest_idx = len(toks_s)
    found_colon = False
    for idx, (t, is_it, is_b) in enumerate(toks_s):
        if is_it and idx > 0:
            rest_idx = idx
            break
        if idx > 0 and t.strip('.,;:') in {'I', 'II', 'III'}:
            rest_idx = idx
            break
        if t in {':', ';', '->'}:
            found_colon = True
            rest_idx = idx + 1
            break
        if t.endswith((':', ';')):
            cand_toks.append(t[:-1])
            found_colon = True
            rest_idx = idx + 1
            break
        if idx > 0 and (t.startswith('(') or t.startswith('==EQ==') or re.match(r'^lih\.?$', t, re.I)):
            rest_idx = idx
            break
        if idx >= 4:
            break
        cand_toks.append(t)

    if not cand_toks:
        return False, '', toks

    hw = ' '.join(cand_toks).strip('.,;:!?()"\' ')
    hw_clean = re.sub(r'\s+', ' ', hw)
    hw_clean = re.sub(r'[,\s]+(?:I{1,3}|[0-9]+)$', '', hw_clean).strip('.,;:!?()"\' ')
    hw_alpha = hw_clean.lstrip('-"\'(')
    if not hw_alpha or hw_alpha[0].upper() != letter.upper():
        return False, '', toks
    if not hw_alpha[0].isupper():
        return False, '', toks
    if not re.match(r'^[A-Za-z][A-Za-z\s\-\'\.,\/]{0,42}$', hw_clean):
        return False, '', toks
    if hw_clean.lower() in CLEAR_INDONESIAN_WORDS and not found_colon:
        return False, '', toks
    if allow_indented and not found_colon:
        return False, '', toks
    if not found_colon and not w0_b and len(cand_toks) > 2:
        return False, '', toks
    if not found_colon and hw_clean.split()[0].lower() in CLEAR_INDONESIAN_WORDS:
        if rest_idx < len(toks_s) and not toks_s[rest_idx][1]:
            return False, '', toks

    return True, hw_clean, toks_s[rest_idx:]


def strip_editorial_tags(text: str) -> str:
    s = re.sub(r'\((?:ks|kb|kk|kd|kt|ki|kp|kbil|kg|kl|lih\.?[^)]*)\)', '', text, flags=re.I)
    s = re.sub(r'\b(?:ks|kb|kk|kd|kt|ki|kp|kbil)\.', '', s)
    s = re.sub(r'\s+', ' ', s).strip(' ,;:-.')
    return s


def extract_gor_glosses_and_examples(toks, active_word: str):
    """
    Parse token stream [tok, is_it, is_bold] of one IG 1987 sense/entry block into:
    - padanan_gor (1987), padanan_standar (2001)
    - contoh [{'kalimat_indonesia', 'kalimat_gorontalo', 'kalimat_gorontalo_standar'}]
    Note: '==EQ==' is the headword/sub-entry placeholder inside Indonesian example sentences!
    """
    if not toks:
        return [], [], []

    # Replace '==EQ==' / '=' placeholder with active_word & clean editorial tags
    rep_word = re.sub(r'^[I]+\s+', '', active_word).strip().lower()
    cleaned = []
    for idx_t, (t, is_it, is_b) in enumerate(toks):
        m_eq = re.match(r'^(?:==EQ==|[=\-~]{2,4}|[=~])([,.;:!?]*)$', t)
        if m_eq:
            eq_punct = m_eq.group(1)
            # Skip stray placeholder at very start of gloss before anything else
            if idx_t == 0 or not cleaned:
                if eq_punct and cleaned:
                    cleaned[-1][0] += eq_punct
                continue
            prev_t = cleaned[-1][0]
            prev_clean = prev_t.strip(',.;:!?()').lower()
            # Avoid duplicating rep_word if OCR already rendered 'kelintil = ,' -> keep only punct
            if prev_clean == rep_word or prev_clean.endswith(rep_word):
                if eq_punct:
                    cleaned[-1][0] += eq_punct
                continue
            if prev_t.endswith((';', '.')) or re.match(r'^\(\d+\)$', prev_t):
                w_sub = rep_word.capitalize()
            else:
                w_sub = rep_word
            cleaned.append([w_sub + eq_punct, False])
        elif not is_it and t.lower() in EDITORIAL_TAGS:
            continue
        else:
            cleaned.append([t, bool(is_it)])

    if not cleaned:
        return [], [], []

    # Bridge 1-2 Gorontalo-like / vowel-final tokens inside italic spans
    def is_ig_bridgeable(ts: str) -> bool:
        wc = ts.strip('.,;:!?()"\'').lower()
        if not wc:
            return False
        if '?' in wc or wc in GORONTALO_1987_WORDS:
            return True
        if is_unambiguous_id(wc):
            return False
        return (wc[-1] in 'aiueo') or (len(wc) <= 3 and wc.isalpha())

    for idx in range(1, len(cleaned) - 1):
        if not cleaned[idx][1] and cleaned[idx - 1][1]:
            if cleaned[idx + 1][1] and is_ig_bridgeable(cleaned[idx][0]):
                cleaned[idx][1] = True
            elif (idx + 2) < len(cleaned) and not cleaned[idx + 1][1] and cleaned[idx + 2][1]:
                if is_ig_bridgeable(cleaned[idx][0]) and is_ig_bridgeable(cleaned[idx + 1][0]):
                    cleaned[idx][1] = True
                    cleaned[idx + 1][1] = True

    # Split into sense chunks at '(1)', '(2)', ... and within each sense at ';' / '.'
    sense_blocks = []
    cur_blk = []
    for tok, is_it in cleaned:
        if not is_it and re.match(r'^\([1-9]\)$', tok):
            if cur_blk:
                sense_blocks.append(cur_blk)
            cur_blk = []
        else:
            cur_blk.append([tok, is_it])
    if cur_blk:
        sense_blocks.append(cur_blk)

    padanan = []
    contoh = []

    for blk in sense_blocks:
        if not blk:
            continue
        # Sub-split blk into clauses at ';' or '.' (NEVER at internal comma ',')
        clauses = []
        buf = []
        for idx_t, (tok, is_it) in enumerate(blk):
            buf.append([tok, is_it])
            if tok.endswith((';', '.')) and idx_t + 1 < len(blk):
                if tok.lower() not in {'dsb.', 'dst.', 'dll.', 'mis.', 'ump.', 'lih.', 'dr.', 'yg.', 'tdk.'}:
                    clauses.append(buf)
                    buf = []
        if buf:
            clauses.append(buf)

        in_examples = False
        for cl in clauses:
            # Partition alternating [RM_run, IT_run] segments inside this clause
            runs = []
            cur_r_it = cl[0][1]
            cur_r = []
            for tok, is_it in cl:
                if is_it != cur_r_it:
                    if cur_r:
                        runs.append((cur_r_it, cur_r))
                    cur_r_it = is_it
                    cur_r = [tok]
                else:
                    cur_r.append(tok)
            if cur_r:
                runs.append((cur_r_it, cur_r))

            # Before first multi-word non-italic Indonesian example sentence, leading IT runs are glosses
            idx_r = 0
            if not in_examples:
                while idx_r < len(runs):
                    r_is_it, r_toks = runs[idx_r]
                    if r_is_it:
                        for sub_p in re.split(r'[;,]', ' '.join(r_toks)):
                            p = re.sub(r'\s+', ' ', sub_p).strip(' ,;:.()-"\'')
                            if p and len(p) >= 1 and not any(c.isdigit() for c in p) and p not in padanan:
                                padanan.append(p)
                        idx_r += 1
                    else:
                        # Non-italic run: gloss note if parenthetical or single short word without IT example after
                        txt = ' '.join(r_toks).strip(' ,;:-.')
                        has_next_it = (idx_r + 1 < len(runs) and runs[idx_r + 1][0])
                        is_paren_note = txt.startswith('(') and txt.endswith(')')
                        # Wait: is this non-italic run an Indonesian example sentence (or just 1-word gloss)?
                        looks_example = (not is_paren_note) and has_next_it and (
                            len(r_toks) >= 2 or (len(txt) >= 3 and txt[0].isupper()) or rep_word in txt.lower()
                        )
                        if looks_example:
                            in_examples = True
                            break
                        else:
                            # Skip single non-italic gloss tag/parenthetical note, or save known Gor fallback
                            for tk in r_toks:
                                tc = tk.strip(',;:.()').lower()
                                if (tc in GORONTALO_1987_WORDS or '?' in tc) and tc not in CLEAR_INDONESIAN_WORDS and tc not in padanan:
                                    padanan.append(tc)
                            idx_r += 1

            # Extract remaining alternating (Non-italic ID, Italic GOR) pairs as examples
            while idx_r < len(runs):
                if runs[idx_r][0]:
                    # Leading italic run in example zone without preceding RM -> gloss if still short & single word
                    idx_r += 1
                    continue
                if idx_r + 1 >= len(runs):
                    break
                id_toks = runs[idx_r][1]
                gor_toks = runs[idx_r + 1][1]
                idx_r += 2

                id_raw_joined = ' '.join(id_toks).replace('==EQ==', ' ' + rep_word)
                id_sent = expand_indonesian_abbrev(strip_editorial_tags(id_raw_joined))
                id_sent = re.sub(r'^[=\-~,\s]+', '', id_sent).strip(' ,;:-')
                gor_sent = re.sub(r'\s+', ' ', ' '.join(gor_toks).replace('==EQ==', '')).strip(' ,;:-.=~')
                gor_sent = re.sub(r'^[=\-~,\s]+', '', gor_sent).strip(' ,;:-.')
                if rep_word in {'qari', 'qariah', 'quran'}:
                    id_sent = re.sub(r'\boari(ah)?\b', lambda m: 'qari' + (m.group(1) or ''), id_sent, flags=re.I)
                if id_sent and gor_sent and len(id_sent) >= 3 and len(gor_sent) >= 3:
                    if id_sent.strip().isdigit() and not re.search(r'\b(?:hetuto|lihu|juta|riwu|mopulu)\b', gor_sent, re.I):
                        continue
                    if ':' in id_sent and re.search(r'[A-Z][a-z]+\s*:', id_sent):
                        continue
                    contoh.append({
                        'kalimat_indonesia': id_sent,
                        'kalimat_gorontalo': gor_sent,
                        'kalimat_gorontalo_standar': convert_1987_to_2001_orthography(gor_sent),
                    })

    padanan_std = [convert_1987_to_2001_orthography(p) for p in padanan]
    return padanan, padanan_std, contoh


def split_entry_and_subentries(toks, headword: str):
    """
    Split an entry's token stream into:
    - main_toks (for headword itself)
    - subentries [(sub_word, sub_toks), ...] when derived Indonesian words appear as 'sub_word : ...'
    """
    hw_stem = re.sub(r'[^a-z]', '', headword.lower())
    sub_breaks = []
    i = 0
    while i < len(toks):
        tok, is_it, is_bold = toks[i]
        if is_it:
            i += 1
            continue
        # Handle optional Roman numeral marker before sub-word: 'I renggutan :', 'II direnggut :'
        sub_prefix = ''
        w_idx = i
        if tok.strip('.,') in {'I', 'II', 'III'} and (i + 1 < len(toks)) and not toks[i + 1][1]:
            sub_prefix = tok.strip('.,') + ' '
            w_idx = i + 1
            tok, is_it, is_bold = toks[w_idx]

        t_clean = tok.strip('.,;:!?()"\'')
        if not t_clean or not t_clean[0].isalpha():
            i += 1
            continue

        # Allow 1-word sub-entry or 2-word sub-entry (e.g. 'mengadaptasikan diri :')
        colon_idx = None
        sub_words = [t_clean]
        if tok.endswith((':', ';')):
            colon_idx = w_idx
        elif w_idx + 1 < len(toks) and toks[w_idx + 1][0] in {':', ';'} and not toks[w_idx + 1][1]:
            colon_idx = w_idx + 1
        elif w_idx + 2 < len(toks) and not toks[w_idx + 1][1]:
            w2 = toks[w_idx + 1][0]
            if w2.endswith((':', ';')):
                sub_words.append(w2[:-1])
                colon_idx = w_idx + 1
            elif w2.startswith('('):
                # e.g. 'merengus (masam muka) :'
                j = w_idx + 1
                while j < min(len(toks), w_idx + 6):
                    if toks[j][0].endswith((':', ';')) or toks[j][0] in {':', ';'}:
                        colon_idx = j
                        break
                    j += 1

        if colon_idx is None:
            i += 1
            continue

        at_boundary = (i == 0) or toks[i - 1][1] or toks[i - 1][0].endswith((';', '.', ',')) or toks[i - 1][0].strip('.,') in {'I', 'II', 'III'}
        if not at_boundary:
            i += 1
            continue

        sub_w = re.sub(r'\s*\([^)]*\).*$', '', ' '.join(sub_words)).strip(' .,:;')
        t_low = t_clean.lower()
        shares_stem = len(hw_stem) >= 3 and hw_stem[:3] in t_low
        is_deriv = bool(re.match(r'^(?:me|mem|men|meng|meny|ber|di|ter|ke|pe|per|se)[a-z\-]{3,}$', t_low)) or ('-' in t_low)
        if sub_w and (is_bold or is_deriv or shares_stem or sub_prefix) and t_low not in CLEAR_INDONESIAN_WORDS and t_low not in EDITORIAL_TAGS and t_low not in {'kbb', 'lih', 'dsb', 'dst', 'dll'} and t_low != hw_stem:
            sub_breaks.append((i, sub_w, colon_idx + 1))
            i = colon_idx + 1
            continue
        i += 1

    if not sub_breaks:
        return toks, []

    main_toks = toks[:sub_breaks[0][0]]
    subs = []
    for k, (b_idx, sub_w, start_idx) in enumerate(sub_breaks):
        end_idx = sub_breaks[k + 1][0] if k + 1 < len(sub_breaks) else len(toks)
        subs.append((sub_w, toks[start_idx:end_idx]))
    return main_toks, subs


def parse_letter_pdf(letter: str, pdf_pages_data: list[tuple[int, list]]):
    """Parse all pages of one IG letter PDF into clean lexical entries matching IG batch schema."""
    entries = []
    cur_entry = None

    def flush(e):
        if not e:
            return
        toks = e.pop('_toks', [])
        main_toks, sub_list = split_entry_and_subentries(toks, e['lema_indonesia'])
        pad, pad_std, exs = extract_gor_glosses_and_examples(main_toks, e['lema_indonesia'])
        e['padanan_gorontalo'] = pad
        e['padanan_gorontalo_standar'] = pad_std
        e['contoh'] = exs
        turunan = []
        for sub_w, s_toks in sub_list:
            if sub_w.lower() == 'oariah' and e['lema_indonesia'].lower() == 'qari':
                sub_w = 'Qariah'
            spad, spad_std, sexs = extract_gor_glosses_and_examples(s_toks, sub_w)
            if spad or sexs:
                turunan.append({
                    'kata_indonesia': sub_w,
                    'padanan_gorontalo': spad,
                    'padanan_gorontalo_standar': spad_std,
                    'contoh': sexs,
                })
        e['turunan'] = turunan
        if e['padanan_gorontalo'] or e['contoh'] or e['turunan']:
            entries.append(e)

    for pno, raw_lines in pdf_pages_data:
        lines = join_ig_hyphenated_wraps(raw_lines)
        for indent, toks in lines:
            if not toks:
                continue
            ok_hw, hw, rest = is_valid_ig_headword(toks, letter, allow_indented=(indent != 0))
            if ok_hw:
                flush(cur_entry)
                cur_entry = {
                    'lema_header_asli': hw,
                    'lema_induk': hw,
                    'lema_indonesia': hw,
                    'tipe_lema': 'lema_pokok',
                    'huruf': letter,
                    'halaman_pdf': pno,
                    '_toks': list(rest),
                }
                continue
            if cur_entry is not None:
                cur_entry['_toks'].extend(toks)

    flush(cur_entry)
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--test-pages', nargs='*', default=None, help='e.g. A:1 A:2 B:22 K:28 P:24 Q:1 R:13')
    parser.add_argument('--batches', nargs='*', type=int, default=[1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    CACHE_V5_DIR.mkdir(parents=True, exist_ok=True)
    BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    HURUF_DIR.mkdir(parents=True, exist_ok=True)

    if args.test_pages:
        t0 = time.time()
        by_letter = {}
        tasks = []
        for spec in args.test_pages:
            let, p_str = spec.split(':')
            let = let.upper()
            pno = int(p_str)
            pdf_p = IG_DIR / LETTER_TO_PDF[let]
            tasks.append((let, str(pdf_p), pno))
        with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks))) as ex:
            results = list(ex.map(ocr_and_cache_ig_page, tasks))
        for let, pno, lines in results:
            by_letter.setdefault(let, []).append((pno, lines))
        total_e = 0
        for let, plist in by_letter.items():
            plist.sort(key=lambda x: x[0])
            parsed = parse_letter_pdf(let, plist)
            total_e += len(parsed)
            for e in parsed:
                print(f"[{let} p.{e['halaman_pdf']}] {e['lema_indonesia']} | pad={e['padanan_gorontalo']} | std={e['padanan_gorontalo_standar']} | ex={e['contoh']}")
                for t in e['turunan']:
                    print(f"    -> {t['kata_indonesia']} | pad={t['padanan_gorontalo']} | ex={t['contoh']}")
        print(f"TEST extracted {total_e} entries in {time.time() - t0:.1f}s")
        return

    # Full run across requested batches
    target_letters = []
    for b in args.batches:
        for let in BATCH_GROUPS[b]:
            if let in LETTER_TO_PDF:
                target_letters.append(let)

    all_tasks = []
    for let in target_letters:
        pdf_p = IG_DIR / LETTER_TO_PDF[let]
        doc = pymupdf.open(str(pdf_p))
        for pno in range(1, len(doc) + 1):
            all_tasks.append((let, str(pdf_p), pno))

    cached_cnt = sum(1 for let, _, pno in all_tasks if (CACHE_V5_DIR / f'{let}_p{pno:03d}.json').exists())
    print(f"Running IG v5 OCR on {len(all_tasks) - cached_cnt} remaining pages (out of {len(all_tasks)}) with {args.workers} workers...")
    t0 = time.time()
    pages_by_letter = {let: [] for let in target_letters}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, (let, pno, lines) in enumerate(ex.map(ocr_and_cache_ig_page, all_tasks), 1):
            pages_by_letter[let].append((pno, lines))
            if i % 40 == 0 or i == len(all_tasks):
                print(f"  OCR IG v5 progress: {i}/{len(all_tasks)} ({time.time() - t0:.1f}s)", flush=True)

    letter_entries = {}
    for let in target_letters:
        plist = sorted(pages_by_letter[let], key=lambda x: x[0])
        parsed = parse_letter_pdf(let, plist)
        letter_entries[let] = parsed
        huruf_out = HURUF_DIR / f'kamus_indo_gorontalo_huruf_{let}.json'
        huruf_out.write_text(json.dumps(parsed, ensure_ascii=False, indent=2))

    for b in args.batches:
        b_entries = []
        for let in BATCH_GROUPS[b]:
            b_entries.extend(letter_entries.get(let, []))
        clean_p = BATCHES_DIR / f'batch_{b}_clean.jsonl'
        par_p = BATCHES_DIR / f'batch_{b}_parallel.jsonl'
        rep_p = BATCHES_DIR / f'batch_{b}_report.json'

        par_rows = []
        sub_cnt = 0
        with clean_p.open('w') as fc:
            for e in b_entries:
                fc.write(json.dumps(e, ensure_ascii=False) + '\n')
                for ex in e['contoh']:
                    par_rows.append({
                        'src_indonesia': ex['kalimat_indonesia'],
                        'tgt_gorontalo': ex['kalimat_gorontalo_standar'],
                        'tgt_gorontalo_asli': ex['kalimat_gorontalo'],
                        'lema_indonesia': e['lema_indonesia'],
                        'huruf': e['huruf'],
                        'halaman_pdf': e['halaman_pdf'],
                    })
                for sub in e['turunan']:
                    sub_cnt += 1
                    for ex in sub['contoh']:
                        par_rows.append({
                            'src_indonesia': ex['kalimat_indonesia'],
                            'tgt_gorontalo': ex['kalimat_gorontalo_standar'],
                            'tgt_gorontalo_asli': ex['kalimat_gorontalo'],
                            'lema_indonesia': sub['kata_indonesia'],
                            'huruf': e['huruf'],
                            'halaman_pdf': e['halaman_pdf'],
                        })
        with par_p.open('w') as fp:
            for r in par_rows:
                fp.write(json.dumps(r, ensure_ascii=False) + '\n')

        rep = {
            'batch': b,
            'huruf': BATCH_GROUPS[b],
            'total_entries': len(b_entries),
            'total_turunan': sub_cnt,
            'total_parallel': len(par_rows),
        }
        rep_p.write_text(json.dumps(rep, ensure_ascii=False, indent=2))
        print(f"Batch {b} {BATCH_GROUPS[b]}: {len(b_entries)} entries, {sub_cnt} turunan, {len(par_rows)} parallel")


if __name__ == '__main__':
    main()
