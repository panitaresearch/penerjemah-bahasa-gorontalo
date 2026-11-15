#!/usr/bin/env python3
"""
extract_gi_v5.py
================
Multimodal Vision + Tesseract 5 (ind+eng) + PyMuPDF Span Fusion extractor
for Kamus Bahasa Gorontalo-Indonesia (2001, 302 lexicon pages: PDF pp. 38-339).

Improvements over v4:
1. Runs clean 300 DPI grayscale strip-deskewed Tesseract 5 (-l ind+eng --psm 6)
   on ALL 302 pages instead of trusting the noisy 2005 embedded PDF OCR text.
2. Uses text-centered column gutter detection with vertical-rule erasure so
   right-column characters never bleed into the left column.
3. Splits each column at zero/min-ink horizontal gaps into strips and locally
   deskews each strip to eliminate paper-curl line merging.
4. Measures per-word visual stroke width (bold) and shear variance (italic),
   fusing with PyMuPDF span italic flags and colon ':' hints.
5. Recovers missing '---' dash placeholders from wide inter-word ink gaps.
6. Preserves multi-word bold sub-headwords, full un-truncated definitions,
   and clean multi-sense alignment.
"""

import argparse
import csv
import json
import os
import re
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pymupdf

BASE_DIR = Path('/content/drive/MyDrive/REPO/richie/modelgorontalo/gorontaloindonesia')
PDF_PATH = BASE_DIR / 'Kamus Bahasa Gorontalo-Indonesia   341a.pdf'
DATASET_DIR = BASE_DIR / 'dataset'
BATCH_DIR = DATASET_DIR / 'batches'
HURUF_DIR = DATASET_DIR / 'huruf'
TRAIN_DIR = DATASET_DIR / 'training_ready'
CACHE_V5_DIR = DATASET_DIR / 'ocr_cache_v5'

for d in [DATASET_DIR, BATCH_DIR, HURUF_DIR, TRAIN_DIR, CACHE_V5_DIR]:
    d.mkdir(parents=True, exist_ok=True)

DEGRADED_PDF_PAGES = {
    54, 58, 60, 62, 64, 99, 100, 101, 103, 104, 107, 108, 111, 112, 129, 143, 273, 287
}

INDONESIAN_STOP_STARTS = {
    'yang', 'yg', 'dan', 'atau', 'untuk', 'utk', 'dengan', 'dng', 'dg', 'dalam', 'dlm', 'dl',
    'dari', 'dr', 'dp', 'pada', 'pd', 'ke', 'di', 'karena', 'krn', 'sebab', 'jika', 'kalau',
    'bila', 'apabila', 'supaya', 'agar', 'sehingga', 'sampai', 'hingga', 'seperti', 'spt',
    'misalnya', 'mis', 'yaitu', 'ialah', 'adalah', 'merupakan', 'mempunyai', 'memiliki',
    'menggunakan', 'melakukan', 'membuat', 'menjadi', 'memberi', 'membawa', 'mengambil',
    'melihat', 'mendapat', 'menurut', 'mengenai', 'melalui', 'melainkan', 'meskipun',
    'walaupun', 'tetapi', 'namun', 'tidak', 'tak', 'tdk', 'bukan', 'belum', 'sudah', 'telah',
    'sedang', 'akan', 'dapat', 'bisa', 'boleh', 'harus', 'mesti', 'perlu', 'sering', 'selalu',
    'biasa', 'biasanya', 'kadang', 'tempat', 'orang', 'barang', 'sesuatu', 'seseorang',
    'anak', 'ayah', 'ibu', 'rumah', 'pekerjaan', 'perbuatan', 'perkataan', 'perasaan',
    'keadaan', 'bagian', 'bahagian', 'jenis', 'macam', 'pohon', 'buah', 'daun', 'kayu',
    'bambu', 'air', 'tanah', 'batu', 'besi', 'kain', 'baju', 'celana', 'kepala', 'tangan',
    'kaki', 'mata', 'mulut', 'hati', 'perut', 'badan', 'tubuh', 'suara', 'bunyi', 'warna',
    'rasa', 'bau', 'waktu', 'hari', 'malam', 'pagi', 'sore', 'siang', 'tahun', 'bulan',
    'minggu', 'jam', 'pukul', 'uang', 'harga', 'pasar', 'sekolah', 'jalan', 'kota', 'kampung',
    'desa', 'negeri', 'laut', 'sungai', 'gunung', 'hutan', 'kebun', 'sawah', 'ladang',
    'binatang', 'hewan', 'burung', 'ikan', 'ayam', 'kuda', 'sapi', 'kambing', 'anjing',
    'kucing', 'ular', 'semut', 'nyamuk', 'lalat', 'lebah', 'alat', 'benda', 'cara', 'hal',
    'kata', 'bahasa', 'nama', 'gelar', 'panggilan', 'ukuran', 'jumlah', 'hitungan',
    'potongan', 'lembar', 'helai', 'ikat', 'tangkai', 'butir', 'biji', 'ekor', 'batang',
    'rumpun', 'sisir', 'tandan', 'piring', 'mangkuk', 'gelas', 'botol', 'kaleng', 'karung',
    'keranjang', 'perahu', 'kapal', 'mobil', 'roda', 'lantai', 'dinding', 'atap', 'pintu',
    'jendela', 'tiang', 'tangga', 'dapur', 'kamar', 'halaman', 'pagar', 'sumur', 'lebih',
    'kurang', 'paling', 'sangat', 'amat', 'terlalu', 'cukup', 'agak', 'sekali', 'benar',
    'betul', 'sungguh', 'pasti', 'tentu', 'mungkin', 'barangkali', 'sekarang', 'kemarin',
    'besok', 'lusa', 'tadi', 'nanti', 'segera', 'cepat', 'lambat', 'lama', 'baru', 'tua',
    'muda', 'besar', 'kecil', 'panjang', 'pendek', 'tinggi', 'rendah', 'jauh', 'dekat',
    'luas', 'sempit', 'tebal', 'tipis', 'berat', 'ringan', 'keras', 'lembek', 'lembut',
    'kasar', 'halus', 'tajam', 'tumpul', 'panas', 'dingin', 'sejuk', 'hangat', 'basah',
    'kering', 'bersih', 'kotor', 'terang', 'gelap', 'putih', 'hitam', 'merah', 'kuning',
    'hijau', 'biru', 'manis', 'pahit', 'asam', 'asin', 'pedas', 'tawar', 'gurih', 'enak',
    'lezat', 'busuk', 'harum', 'wangi', 'amis', 'anyir', 'baik', 'buruk', 'jahat', 'bagus',
    'indah', 'cantik', 'elok', 'jelek', 'bodoh', 'pintar', 'pandai', 'cerdik', 'rajin',
    'malas', 'berani', 'takut', 'malu', 'marah', 'senang', 'gembira', 'sedih', 'susah',
    'payah', 'lelah', 'capek', 'sakit', 'sehat', 'hidup', 'mati', 'kaya', 'miskin', 'penuh',
    'kosong', 'banyak', 'sedikit', 'semua', 'seluruh', 'sebagian', 'separuh', 'setengah',
    'masing-masing', 'tiap', 'tiap-tiap', 'setiap', 'beberapa', 'berbagai', 'engkau',
    'kamu', 'anda', 'dia', 'ia', 'mereka', 'kami', 'kita', 'saya', 'aku', 'beliau', 'ini',
    'itu', 'sini', 'situ', 'sana', 'mana', 'siapa', 'apa', 'mengapa', 'kenapa', 'bagaimana',
    'berapa', 'kapan', 'bilamana', 'sesudah', 'sebelum', 'ketika', 'sambil', 'seraya',
    'tanpa', 'kecuali', 'selain', 'antara', 'tengah', 'atas', 'bawah', 'depan', 'belakang',
    'samping', 'sebelah', 'luar', 'ujung', 'pangkal', 'batas', 'tepi', 'pinggir', 'sudut',
    'pojok', 'arah', 'sebangsa', 'sejenis', 'upacara', 'penyakit', 'tumbuhan', 'kelihatan',
    'bekerja', 'berdarah', 'ditambah', 'menambah', 'duri', 'darah', 'pelangi', 'gobek',
    'tukul', 'dada', 'kemarau', 'dibiarkan', 'gadis', 'kesiangan', 'jatuh', 'naik', 'turun',
    'jangan', 'hanya', 'jagung', 'padi', 'beras', 'nasi', 'pisang', 'kelapa', 'mangga',
    'jeruk', 'lemon', 'sirih', 'pinang', 'kapur', 'gambir', 'tembakau', 'rokok', 'kopi',
    'gula', 'garam', 'minyak', 'arang', 'abu', 'asap', 'embun', 'kabut', 'awan', 'langit',
    'bintang', 'petir', 'guntur', 'angin', 'hujan', 'ombak', 'arus', 'pasang', 'surut',
    'danau', 'rawa', 'kolam', 'parit', 'selokan', 'jembatan', 'layar', 'dayung', 'kemudi',
    'jangkar', 'pukat', 'jaring', 'kail', 'pancing', 'bubu', 'tombak', 'parang', 'pisau',
    'kapak', 'cangkul', 'sabit', 'bajak', 'garu', 'lesung', 'alu', 'nyiru', 'tapisan',
    'kukusan', 'periuk', 'belanga', 'kuali', 'wajan', 'tungku', 'sendok', 'garpu', 'cangkir',
    'teko', 'ceret', 'ember', 'timba', 'gayung', 'tikar', 'bantal', 'kasur', 'kelambu',
    'selimut', 'sarung', 'kebaya', 'kemeja', 'topi', 'kopiah', 'songkok', 'sabuk', 'cincin',
    'gelang', 'kalung', 'anting', 'subang', 'jarum', 'benang', 'gunting', 'setrika', 'sisir',
    'cermin', 'bedak', 'sabun', 'ucapan', 'taktik', 'kerangka', 'joran', 'kutu', 'balok',
    'umbi', 'tobat', 'buta', 'huruf', 'mengungkit', 'mengumumkan', 'lamban', 'abang',
    'sapu', 'tugas', 'peti', 'buah', 'kurap', 'disapu', 'menyapu', 'diungkit', 'belajar',
    'diajar', 'diselam', 'menyelam', 'mengadu', 'pengaduan', 'musyawarah', 'serambi',
    'mari', 'marilah', 'tenang', 'tiang', 'haluan', 'pelana', 'kertas', 'dipegang', 'memegang',
    'menyeberang', 'diseberangi', 'belanja', 'ongkos', 'dibelanjai', 'berbelanja',
    'menolong', 'ditolong', 'jongos', 'bulan', 'juri', 'juta', 'tali', 'takbir',
}

GORONTALO_PARTICLES = {
    'bo', 'to', 'lo', 'mo', 'po', 'ma', 'mao', 'mayi', 'mola', 'mota', 'ta', 'ti', 'te',
    'tiyo', 'yio', 'wau', 'wa:tiya', 'ami', 'amiya:tiya', 'ito', 'timongoliyo', 'timongoli',
    'tomongoli', 'u', 'o', 'olo', 'oli', 'ole', 'ode', 'onto', 'wonu', 'wanu', 'sababu',
    'babo', 'di:la', 'dila', 'di:po', 'dipo', 'didu', 'diya:lu', 'diya:luwo', 'woluwo',
    'donggo', 'onggo', 'de', 'do', 'yito', 'boyito', 'botiya', 'botiye', 'utiya', 'u:ti',
    'uti', 'no:nu', 'uto:nu', 'tato:nu', 'openu', 'alihu', 'asali', 'potala', 'lapatao',
    'timi-timiidu', 'dumo-dupo', 'dumodupo', 'ngo:lo', 'ngohuyi', 'wee:wo', 'we:wo',
    'musi', 'mowali', 'yinawo', 'hila', 'delo', 'odito', 'odiya', 'odiye', 'te:ya', 'te:to',
    'wawu', 'mealo', 'sambe', 'da:da:ta', 'ngo:idi', 'poodaha', 'bolo', 'hemo', 'helo',
    'tawu', 'bele', 'pale', 'huyi', 'dulahu', 'dulahe', 'taluhu', 'taluhe', 'bongulo',
    'tuwawu', 'taliya', 'hama:lo', 'potali', 'pohamalo'
}

CLEAR_INDONESIAN_STOP = {
    'yang', 'yg', 'dan', 'atau', 'dengan', 'dg', 'dng', 'di', 'ke', 'dari', 'dr', 'dalam', 'dlm',
    'pada', 'pd', 'untuk', 'utk', 'bagi', 'guna', 'sebab', 'karena', 'krn', 'kalau', 'jika', 'bila',
    'itu', 'ini', 'jangan', 'akan', 'sudah', 'belum', 'masih', 'sedang', 'telah', 'tidak', 'tdk', 'tak',
    'saya', 'aku', 'kami', 'kita', 'engkau', 'kamu', 'dia', 'ia', 'mereka', 'beliau',
    'adalah', 'ialah', 'yaitu', 'yakni', 'seperti', 'spt', 'sebagai', 'sbg', 'oleh', 'kepada', 'kpd',
    'mesti', 'harus', 'boleh', 'dapat', 'bisa', 'mau', 'ingin', 'perlu', 'hanya', 'saja', 'juga', 'pula',
    'baru', 'dulu', 'lagi', 'lebih', 'kurang', 'sangat', 'sekali', 'tiap', 'tiap-tiap', 'setiap',
    'biasanya', 'sering', 'selalu', 'jarang', 'barangkali', 'mungkin', 'apakah', 'mengapa', 'bagaimana',
}

RN_EXCEPTIONS = {
    'karena', 'warna', 'berwarna', 'purna', 'sempurna', 'menyempurnakan', 'jernih', 'menjernihkan',
    'pernah', 'ternak', 'beternak', 'peternak', 'gubernur', 'turnamen', 'jurnal', 'kurnia',
    'purnama', 'purbakala', 'sirna', 'bernas', 'kornel', 'alternatif', 'internasional', 'modern'
}


def heal_ocr_token(tok: str) -> str:
    t = tok.replace('\xad', '-')
    # Normalize dash sequences of >=2 dash-like chars
    if re.match(r'^[—–\-~_]{2,}$', t) or re.match(r'^[—–\-~_]{2,}[,;:.!?]?$', t):
        punct = t[-1] if t[-1] in ',;:.!?' else ''
        return '---' + punct
    if t in {'———', '—-—', '~-—', '-—-'}:
        return '---'
    subs = [
        (r'^fimi-timiidu$', 'timi-timiidu'),
        (r'^fimirimiidu$', 'timi-timiidu'),
        (r'^timttimiidu$', 'timi-timiidu'),
        (r'^tangkartangkai$', 'tangkai-tangkai'),
        (r'^tangkattangkai$', 'tangkai-tangkai'),
        (r'^hainpir$', 'hampir'),
        (r'^bualmya$', 'buahnya'),
        (r'^bualnya$', 'buahnya'),
        (r'^10$', 'lo'),
        (r'^0lo$', 'olo'),
        (r'^sababll$', 'sababu'),
        (r'^wa-tiya$', 'wa:tiya'),
        (r'^wa-liva$', 'wa:tiya'),
        (r'^wa-live$', 'wa:tiya'),
        (r'^wa:lira$', 'wa:tiya'),
        (r'^amiva:tiva$', 'amiya:tiya'),
        (r'^tivo$', 'tiyo'),
        (r'^tiya$', 'tiyo'),
        (r'^bovito$', 'boyito'),
        (r'^boviye$', 'boyiye'),
        (r'^potalimavri$', 'potalimayi'),
        (r'^potalimavi$', 'potalimayi'),
        (r'^polnmamayi$', 'pohamamayi'),
        (r'^pofw-$', 'poha-'),
        (r'^salivo$', 'saliyo'),
        (r'^delomivo$', 'delomiyo'),
        (r'^viol$', 'yio'),
        (r'^vio$', 'yio'),
        (r'^taliva$', 'taliya'),
        (r'^wont$', 'wonu'),
        (r'^pa:ngo$', 'pa:ngo'),
        (r'^pasngo$', 'pa:ngo'),
        (r'^messeli$', 'me:seli'),
        (r'^me-seli$', 'me:seli'),
        (r'^momaS$', 'momaso'),
        (r'^beii$', 'beli'),
        (r'^rUlllah$', 'rumah'),
        (r'^pekeIjaan$', 'pekerjaan'),
        (r'^beIjauhan$', 'berjauhan'),
        (r'^beIjalan$', 'berjalan'),
        (r'^d\.i:la$', 'di:la'),
        (r'^abjadiyoto$', 'abjadiyolo'),
        (r'^abjadkan$', 'abjadkan'),
        (r'^dasda:ta$', 'da:da:ta'),
        (r'^dacda:ta$', 'da:da:ta'),
        (r'^howumu$', 'howumu'),
        (r'^bowuntu$', 'howumu'),
        (r'^pote:mao$', 'pate:mao'),
        (r'^pofe:mao$', 'pate:mao'),
        (r'^Uopulita$', 'ilopulita'),
        (r'^@opulita$', 'ilopulita'),
        (r'^hepoYitohe$', 'hepoyitohe'),
        (r'^ka:peli$', 'ka:pali'),
        (r'^boyiro$', 'boyito'),
        (r'^diseborangi$', 'diseberangi'),
        (r'^bere$', 'bele'),
        (r'^beis$', 'bele'),
        (r'^axpanga$', 'aa:panga'),
        (r'^balsakali$', 'balaakali'),
        (r'^menyindi$', 'menyindir'),
        (r'^balanja:to$', 'balanja:lo'),
        (r'^belarja$', 'belanja'),
        (r'^mya$', 'saya'),
        (r'^selesaL$', 'selesai'),
        (r'^dir\)$', 'diri)'),
        (r'^har$', 'hari'),
        (r'^mix$', 'mis.'),
        (r'^ditcnung$', 'ditenung'),
        (r'^berpesang$', 'berpegang'),
        (r'^berpeyangan$', 'berpegangan'),
        (r'^\(bak$', '(baik'),
        (r'^dicla$', 'di:la'),
        (r'^poralimavi$', 'potalimayi'),
        (r'^potalimari$', 'potalimayi'),
        (r'^isidki$', 'isiriki'),
        (r'^isiriklyolo$', 'isirikiyolo'),
        (r'^ishisiciki$', 'isi-isiriki'),
        (r'^Auyi$', 'huyi'),
        (r'^wterika$', 'seterika'),
        (r'^kasri$', 'kasti'),
        (r'^pegera$', 'segera'),
        (r'^dinyatakan$', 'dinyalakan'),
        (r'^yinuiyo$', 'yinuliyo'),
        (r'^1iyo$', 'tiyo'),
        (r'^hamnpir$', 'hampir'),
    ]
    for pat, rep in subs:
        t = re.sub(pat, rep, t)
    # Fix rn -> m inside words except known Indonesian rn words
    if 'rn' in t.lower() and t.strip('.,;:!?()').lower() not in RN_EXCEPTIONS:
        if re.match(r'^[a-z:]+rn[a-z:]+$', t.lower()):
            t = t.replace('rn', 'm')
    return t


def expand_indonesian_abbrev(text: str) -> str:
    abbr = [
        (r'\byg\b', 'yang'), (r'\bdg\b', 'dengan'), (r'\bdng\b', 'dengan'),
        (r'\bpd\b', 'pada'), (r'\bdp\b', 'daripada'), (r'\butk\b', 'untuk'),
        (r'\bkrn\b', 'karena'), (r'\btdk\b', 'tidak'), (r'\bsbg\b', 'sebagai'),
        (r'\bdl\b', 'dalam'), (r'\bdlm\b', 'dalam'), (r'\bspt\b', 'seperti'),
        (r'\bdsb\b', 'dan sebagainya'), (r'\bdst\b', 'dan seterusnya'),
        (r'\bdll\b', 'dan lain-lain'), (r'\bmis\b\.?', 'misalnya'),
        (r'\bybs\b', 'yang bersangkutan'),
    ]
    out = text
    for pat, rep in abbr:
        out = re.sub(pat, rep, out)
    out = re.sub(r'\bmem\s+punyai\b', 'mempunyai', out)
    out = re.sub(r'\bhamnpir\b', 'hampir', out)
    return re.sub(r'\s+', ' ', out).strip()


def deskew_gray(gray: np.ndarray, max_ang: float = 3.0):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    small = cv2.resize(th, (0, 0), fx=0.5, fy=0.5)
    sh, sw = small.shape
    if sh < 20 or sw < 20:
        return gray, 0.0
    best_ang, best_var = 0.0, -1.0
    for ang in np.linspace(-max_ang, max_ang, 61):
        M = cv2.getRotationMatrix2D((sw / 2, sh / 2), ang, 1.0)
        rot = cv2.warpAffine(small, M, (sw, sh), flags=cv2.INTER_NEAREST)
        v = float(np.var(rot.sum(axis=1)))
        if v > best_var:
            best_var, best_ang = v, ang
    h, w = gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), best_ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), best_ang


def get_pymupdf_spans(page):
    """Extract PDF-coordinate spans with italic/bold flags."""
    d = page.get_text('dict')
    spans = []
    for b in d.get('blocks', []):
        for l in b.get('lines', []):
            for s in l.get('spans', []):
                txt = s['text'].strip()
                if not txt:
                    continue
                is_it = bool(s['flags'] & 2) or ('Italic' in s['font']) or ('Oblique' in s['font'])
                x0, y0, x1, y1 = s['bbox']
                spans.append((x0, y0, x1, y1, txt, is_it))
    return spans


def match_pymupdf_span(pdf_x, pdf_y, pdf_spans):
    best = None
    best_dist = 1e9
    for sx0, sy0, sx1, sy1, txt, is_it in pdf_spans:
        if sy0 - 4 <= pdf_y <= sy1 + 4 and sx0 - 6 <= pdf_x <= sx1 + 6:
            cx = (sx0 + sx1) / 2.0
            cy = (sy0 + sy1) / 2.0
            dist = abs(pdf_x - cx) + abs(pdf_y - cy) * 2.0
            if dist < best_dist:
                best_dist = dist
                best = (txt, is_it)
    return best


def is_unambiguous_indonesian_word(w: str) -> bool:
    c = w.strip('.,;:!?()"\'').lower()
    if not c or ':' in c:
        return False
    if c in CLEAR_INDONESIAN_STOP:
        return True
    if re.match(r'^(?:meng|meny|mem|men|ber|ter|per|ke|pe|di)[a-z]{3,}$', c):
        return True
    if len(c) >= 5 and re.search(r'(?:nya|kan|an)$', c) and c not in {'tawu', 'bulan'}:
        return True
    if c[-1] in 'bcdfghjklmnpqrstvwxyz' and c not in {'tau', 'ha'}:
        return True
    return False


def is_likely_indonesian_word(w: str) -> bool:
    c = w.strip('.,;:!?()"\'').lower()
    if not c or ':' in c:
        return False
    if c in INDONESIAN_STOP_STARTS:
        return True
    return is_unambiguous_indonesian_word(c)


def recover_colon_from_pymupdf(tess_tok: str, pdf_txt: str) -> str:
    """
    If PyMuPDF span has a colon ':' for long vowel (e.g., 'a:ato', 'pa:ngo', 'wa:tiya', 'aa:panga')
    and Tesseract misread ':' as a thin letter ('ato', 'atata', 'paingo', 'axpanga'),
    selectively restore ':' when skeleton matches!
    """
    if ':' not in pdf_txt or ':' in tess_tok:
        return tess_tok
    if len(tess_tok) < 3 or is_likely_indonesian_word(tess_tok):
        return tess_tok
    t_low = tess_tok.strip('.,;:!?()').lower()
    for pw in pdf_txt.split():
        p_clean = pw.strip('.,;:!?()').lower()
        if ':' not in p_clean:
            continue
        p_nocolon = p_clean.replace(':', '')
        if t_low == p_nocolon:
            punct_pre = tess_tok[:len(tess_tok) - len(tess_tok.lstrip('('))]
            punct_post = tess_tok[len(tess_tok.rstrip('.,;:!?()')):]
            return punct_pre + p_clean + punct_post
        if len(p_clean) == len(t_low):
            diffs = [i for i in range(len(p_clean)) if p_clean[i] != t_low[i]]
            if len(diffs) == 1 and p_clean[diffs[0]] == ':' and t_low[diffs[0]] in 'tilrsx.':
                punct_pre = tess_tok[:len(tess_tok) - len(tess_tok.lstrip('('))]
                punct_post = tess_tok[len(tess_tok.rstrip('.,;:!?()')):]
                return punct_pre + p_clean + punct_post
        # Also catch 'a:ato' -> 'ato' or 'aa:panga' -> 'axpanga'
        if abs(len(p_clean) - len(t_low)) <= 2 and p_clean[:1] == t_low[:1] and p_clean[-3:] == t_low[-3:]:
            if len(t_low) >= 3:
                punct_pre = tess_tok[:len(tess_tok) - len(tess_tok.lstrip('('))]
                punct_post = tess_tok[len(tess_tok.rstrip('.,;:!?()')):]
                return punct_pre + p_clean + punct_post
    return tess_tok


def has_dash_ink_in_gap(bw_strip: np.ndarray, y0: int, y1: int, gx0: int, gx1: int) -> bool:
    if gx1 - gx0 < 24:
        return False
    h, w = bw_strip.shape
    ym0 = max(0, y0 + int((y1 - y0) * 0.25))
    ym1 = min(h, y0 + int((y1 - y0) * 0.80))
    xx0 = max(0, gx0 + 4)
    xx1 = min(w, gx1 - 4)
    if ym1 <= ym0 or xx1 <= xx0:
        return False
    patch = bw_strip[ym0:ym1, xx0:xx1] > 0
    col_cov = (patch.sum(axis=0) > 0).mean()
    # Dash occupies vertical middle band, not full ascender/descender height
    row_sum = patch.sum(axis=1)
    active_rows = (row_sum > 0).sum()
    return col_cov > 0.30 and active_rows <= max(8, int((y1 - y0) * 0.45))


def ocr_gi_page(pdf_page: int, dpi: int = 350):
    """
    Extracts structured lines for one GI PDF page:
    returns list of [indent (0 or 4), list of [token, is_italic, is_bold]]
    """
    doc = pymupdf.open(str(PDF_PATH))
    page = doc[pdf_page - 1]
    is_degraded = pdf_page in DEGRADED_PDF_PAGES
    pdf_spans = [] if is_degraded else get_pymupdf_spans(page)

    pix = page.get_pixmap(dpi=dpi)
    scale_to_pdf = 72.0 / dpi
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0]
    h, w = gray.shape

    # 1. White page bounds (exclude dark scanner margin & horizontal frame bars)
    col_p85 = np.percentile(gray[int(h * 0.10):int(h * 0.65), :], 85, axis=0)
    white_cols = np.where(col_p85 > 215)[0]
    px0 = int(white_cols[0]) + 14 if len(white_cols) > 100 else int(w * 0.15)
    px1 = int(white_cols[-1]) - 14 if len(white_cols) > 100 else int(w * 0.88)

    row_p85 = np.percentile(gray[:, max(0, px0 + 40):min(w, px1 - 40)], 85, axis=1)
    top_dark = [r for r in range(0, int(h * 0.15)) if row_p85[r] < 215]
    bot_dark = [r for r in range(int(h * 0.50), h) if row_p85[r] < 215]
    white_rows = np.where(row_p85 > 215)[0]
    py0 = (top_dark[-1] + 12) if top_dark else (int(white_rows[0]) + 10 if len(white_rows) > 100 else int(h * 0.02))
    py1 = (bot_dark[0] - 12) if bot_dark else (int(white_rows[-1]) - 10 if len(white_rows) > 100 else int(h * 0.96))
    if py1 <= py0 + 200:
        py0, py1 = int(h * 0.02), int(h * 0.96)

    page_gray = gray[py0:py1, px0:px1].copy()
    ph, pw = page_gray.shape
    bw_page = cv2.adaptiveThreshold(
        page_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 18
    )

    # Erase any remaining horizontal/vertical scanner frame bars & long rules, recording central column rules
    num_fl, flab, fst, _ = cv2.connectedComponentsWithStats(bw_page)
    rules_x = []
    for lbl in range(1, num_fl):
        fw, fh = fst[lbl, cv2.CC_STAT_WIDTH], fst[lbl, cv2.CC_STAT_HEIGHT]
        fx = fst[lbl, cv2.CC_STAT_LEFT]
        if (fw > pw * 0.30 and fh < 45) or (fh > ph * 0.12 and fw < 22) or (fh >= 35 and fw <= 8 and fh / max(1, fw) >= 5.0):
            if pw * 0.32 <= fx <= pw * 0.68 and fh > ph * 0.25:
                rules_x.append(int(fx + fw // 2))
            bw_page[flab == lbl] = 0
            page_gray[flab == lbl] = 255

    # 2. Text x-bounds & gutter in central [0.35*pw, 0.65*pw] band
    x_ink = (bw_page[int(ph * 0.08):int(ph * 0.86), :] > 0).sum(axis=0)
    tx_idx = np.where(x_ink > 15)[0]
    if len(tx_idx) < 50:
        return []
    tx0, tx1 = int(tx_idx[0]), int(tx_idx[-1])
    mid_lo = max(0, int(pw * 0.35))
    mid_hi = min(pw, int(pw * 0.65))

    if rules_x:
        gutter = int(np.median(rules_x))
    else:
        col_dark = (bw_page[int(ph * 0.08):int(ph * 0.86), mid_lo:mid_hi] > 0).sum(axis=0)
        conv = np.convolve(col_dark, np.ones(35), mode='valid')
        gutter = mid_lo + int(np.argmin(conv)) + 17

    col_bounds = [
        (max(0, tx0 - 16), max(10, gutter - 12)),
        (min(pw - 10, gutter + 12), min(pw, tx1 + 16)),
    ]

    env = os.environ.copy()
    env['OMP_THREAD_LIMIT'] = '1'
    all_col_lines = []

    for c_idx, (cx0, cx1) in enumerate(col_bounds):
        col_g = page_gray[:, cx0:cx1]
        col_b = bw_page[:, cx0:cx1]
        ch, cw = col_g.shape
        if cw < 100:
            continue

        # Clean tiny specks in col_b before computing row_ink profiles for trims & cuts
        nl_c, lab_c, st_c, _ = cv2.connectedComponentsWithStats(col_b)
        for lbl_c in range(1, nl_c):
            if st_c[lbl_c, cv2.CC_STAT_AREA] < 5 and st_c[lbl_c, cv2.CC_STAT_WIDTH] <= 3 and st_c[lbl_c, cv2.CC_STAT_HEIGHT] <= 3:
                col_b[lab_c == lbl_c] = 0

        # Find vertical text extent, skipping top header letter and bottom page number / frame noise
        row_ink = (col_b > 0).sum(axis=1)
        active_y = np.where(row_ink > 6)[0]
        if len(active_y) == 0:
            continue
        y_top = max(0, int(active_y[0]) - 8)
        y_bot = min(ch, int(active_y[-1]) + 8)

        # Group active row segments across full column to trim isolated bottom page numbers or top single-letter headers
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
                prev_end = segs[-2][1]
                last_s, last_e = segs[-1]
                if last_s - prev_end >= 22 and (last_e - last_s) < 70:
                    xs_last = np.where(col_b[last_s:last_e, :].sum(axis=0) > 0)[0]
                    if len(xs_last) == 0 or (int(xs_last[-1]) - int(xs_last[0])) < 200 or (last_e - last_s) <= 16:
                        y_bot = prev_end + 8
                        trimmed = True
                if not trimmed and len(segs) >= 2:
                    first_s, first_e = segs[0]
                    next_s = segs[1][0]
                    if first_s < ch * 0.12 and next_s - first_e >= 25 and (first_e - first_s) < 60:
                        xs_first = np.where(col_b[first_s:first_e, :].sum(axis=0) > 0)[0]
                        if len(xs_first) == 0 or (int(xs_first[-1]) - int(xs_first[0])) < 140:
                            y_top = max(0, next_s - 8)
                            trimmed = True
            if not trimmed:
                break

        col_g_crop = col_g[y_top:y_bot, :]
        col_b_crop = col_b[y_top:y_bot, :]
        cch = col_g_crop.shape[0]
        if cch < 40:
            continue

        # Split into strips at minimal-ink horizontal whitespace valleys (~700-1050px per strip, never a stub <350px)
        r_sum = (col_b_crop > 0).sum(axis=1).astype(np.float32)
        r_smooth = np.convolve(r_sum, np.ones(5, dtype=np.float32), mode='same')
        cuts = [0]
        cur_y = 0
        while cur_y + 1080 < cch:
            lo = cur_y + 580
            hi = min(cch - 360, cur_y + 940)
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
            sub_g = col_g_crop[sy0:sy1, :].copy()
            max_ang = 3.0 if (sy1 - sy0) >= 350 else 1.5
            sub_g_desk, ang = deskew_gray(sub_g, max_ang=max_ang)
            # Clean tiny isolated dust (<4 px and <3x3) on grayscale
            bw_sub = cv2.adaptiveThreshold(
                sub_g_desk, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 35, 18
            )
            nl, lab, st, _ = cv2.connectedComponentsWithStats(bw_sub)
            for lbl in range(1, nl):
                if st[lbl, cv2.CC_STAT_AREA] < 4 and st[lbl, cv2.CC_STAT_WIDTH] < 3 and st[lbl, cv2.CC_STAT_HEIGHT] < 3:
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

            # Inverse rotation matrix to map strip coords back to page_gray coords
            sh_s, sw_s = sub_g_desk.shape
            M_inv = cv2.getRotationMatrix2D((sw_s / 2, sh_s / 2), -ang, 1.0)

            strip_lines = {}
            for row in res.stdout.splitlines()[1:]:
                parts = row.split('\t')
                if len(parts) == 12 and int(parts[0]) == 5 and parts[11].strip():
                    key = (int(parts[2]), int(parts[3]), int(parts[4]))
                    wx, wy, ww, wh = int(parts[6]), int(parts[7]), int(parts[8]), int(parts[9])
                    tok = parts[11].strip()
                    if ww < 2 or wh < 4:
                        continue
                    # Skip stray vertical-rule/punctuation artifacts
                    if tok in {'|', '!', '[', ']', '—'} and ww <= 12:
                        continue
                    if tok in {'|', 'l', 'i', '.'} and ww < 8 and (wx > cw - 30 or wx < 18):
                        continue
                    ink = bw_sub[wy:wy + wh, wx:wx + ww] > 0
                    sw_val = float(dist[wy:wy + wh, wx:wx + ww][ink].mean()) if ink.sum() >= 6 else 1.65

                    # Visual slant ratio for italic detection
                    slant = 0.8
                    if ww >= 12 and wh >= 10 and ink.sum() >= 15:
                        crop_b = bw_sub[wy:wy + wh, wx:wx + ww]
                        v0 = float(np.var(crop_b.sum(axis=0).astype(np.float32)))
                        v_it = 0.0
                        for sh_ang in (13.0, 16.0):
                            sh_t = np.tan(np.deg2rad(sh_ang))
                            M_sh = np.float32([[1, sh_t, -sh_t * wh / 2], [0, 1, 0]])
                            warped = cv2.warpAffine(crop_b, M_sh, (ww + int(abs(sh_t) * wh) + 4, wh), flags=cv2.INTER_NEAREST)
                            v_it = max(v_it, float(np.var(warped.sum(axis=0).astype(np.float32))))
                        slant = v_it / (v0 + 1e-5)

                    # Map center back to PDF coordinates
                    cx_s = wx + ww / 2.0
                    cy_s = wy + wh / 2.0
                    orig_cx = M_inv[0, 0] * cx_s + M_inv[0, 1] * cy_s + M_inv[0, 2]
                    orig_cy = M_inv[1, 0] * cx_s + M_inv[1, 1] * cy_s + M_inv[1, 2]
                    pdf_x = (px0 + cx0 + orig_cx) * scale_to_pdf
                    pdf_y = (py0 + y_top + sy0 + orig_cy) * scale_to_pdf

                    span_match = match_pymupdf_span(pdf_x, pdf_y, pdf_spans) if pdf_spans else None
                    pdf_it = span_match[1] if span_match else False
                    if span_match:
                        tok = recover_colon_from_pymupdf(tok, span_match[0])

                    tok = heal_ocr_token(tok)
                    # Replace weird all-caps dash misreads (e.g. 'TA ESSE') if wide thin aspect or known noise
                    if tok in {'ee!', 'SS', 'ESSE'} or (len(tok) >= 2 and tok.isupper() and ww > wh * 2.5):
                        tok = '---'
                    is_dash = ('---' in tok)
                    id_like = is_likely_indonesian_word(tok)
                    unambig_id = is_unambiguous_indonesian_word(tok)
                    if is_dash:
                        is_it = True
                    elif unambig_id:
                        # Unambiguous Indonesian words are NEVER italic on their own
                        is_it = False
                    elif not is_degraded and span_match:
                        is_it = bool(pdf_it and slant > 0.85)
                    elif id_like:
                        is_it = False
                    else:
                        is_it = (slant > 1.15) and (ww >= 12)

                    strip_lines.setdefault(key, []).append({
                        'x': wx, 'x1': wx + ww, 'y': wy, 'y1': wy + wh,
                        'tok': tok, 'it': is_it, 'sw': sw_val, 'slant': slant,
                        'abs_y': y_top + sy0 + wy, 'id_like': unambig_id
                    })

            for k in sorted(strip_lines.keys()):
                ws = sorted(strip_lines[k], key=lambda d: d['x'])
                # Skip standalone page numbers at bottom, single letter headers, or publisher stamps
                raw_joined = ' '.join(w_d['tok'] for w_d in ws).strip()
                if re.match(r'^(?:[A-Za-z]{1,3}\s+)?\d{1,3}$', raw_joined) and ws[0]['abs_y'] > ch * 0.72:
                    continue
                if re.match(r'^[A-Z]$', raw_joined) and ws[0]['abs_y'] < ch * 0.08:
                    continue
                if len(raw_joined) == 1 and not raw_joined.isalpha():
                    continue
                if any(st in raw_joined.upper() for st in ('DIDIKAN', 'NDIKAN', 'BALAI PUSTAKA', 'PUSAT BAHASA', 'DEPARTEMEN', 'PERPU')):
                    continue
                if re.match(r'^PE\s*\)\s*\d*\??$', raw_joined.strip()):
                    continue
                if raw_joined.replace('---', '').strip(' |') == '' and ws[0]['abs_y'] > ch * 0.70:
                    continue

                # Recover missing '---' in wide inter-word gaps
                recovered = []
                for idx_w, wd in enumerate(ws):
                    if idx_w > 0:
                        prev_w = ws[idx_w - 1]
                        gap = wd['x'] - prev_w['x1']
                        if 42 <= gap <= 280 and '---' not in prev_w['tok'] and '---' not in wd['tok']:
                            if has_dash_ink_in_gap(bw_sub, min(prev_w['y'], wd['y']), max(prev_w['y1'], wd['y1']), prev_w['x1'], wd['x']):
                                recovered.append({
                                    'x': prev_w['x1'] + 4, 'x1': wd['x'] - 4,
                                    'y': wd['y'], 'y1': wd['y1'],
                                    'tok': '---', 'it': True, 'sw': 1.6, 'slant': 1.0,
                                    'abs_y': wd['abs_y'], 'id_like': False
                                })
                    recovered.append(wd)

                # Smooth isolated 1-word italic spikes on degraded pages
                for idx_w, wd in enumerate(recovered):
                    if wd['it'] and '---' not in wd['tok'] and wd['id_like']:
                        prev_it = recovered[idx_w - 1]['it'] if idx_w > 0 else False
                        next_it = recovered[idx_w + 1]['it'] if idx_w + 1 < len(recovered) else False
                        if not (prev_it and next_it):
                            wd['it'] = False

                col_raw_lines.append((recovered, bw_sub))

        if not col_raw_lines:
            continue

        # Compute column-level p70 for indent==0 vs indent==4 and median sw for bold
        x0s = [ws[0]['x'] for ws, _ in col_raw_lines if len(ws) > 1 or len(ws[0]['tok']) > 1]
        p70 = float(np.percentile(x0s, 70)) if x0s else 0.0
        all_sws = [wd['sw'] for ws, _ in col_raw_lines for wd in ws if len(wd['tok']) >= 2]
        med_sw = float(np.median(all_sws)) if all_sws else 1.7

        for ws, bw_sub in col_raw_lines:
            line_x0 = ws[0]['x']
            # A short 1-char headword like 'a' at left margin followed by Roman Indonesian is NEVER italic
            if line_x0 < p70 - 16 and len(ws) >= 2 and len(ws[0]['tok']) <= 2 and not ws[1]['it']:
                ws[0]['it'] = False
            first_it = ws[0]['it']
            indent = 0 if (line_x0 < p70 - 16 and not first_it) else 4
            # Check leading '---' in wide indent gap
            if indent == 4 and p70 + 42 <= line_x0 <= p70 + 280 and '---' not in ws[0]['tok']:
                if has_dash_ink_in_gap(bw_sub, ws[0]['y'], ws[0]['y1'], int(p70), line_x0):
                    ws.insert(0, {
                        'x': int(p70), 'x1': line_x0 - 4, 'y': ws[0]['y'], 'y1': ws[0]['y1'],
                        'tok': '---', 'it': True, 'sw': 1.6, 'slant': 1.0, 'abs_y': ws[0]['abs_y'],
                        'id_like': False
                    })

            toks_out = []
            for wd in ws:
                is_bold = (wd['sw'] > med_sw * 1.09) and (not wd['it'])
                for sub_t in wd['tok'].split():
                    toks_out.append([sub_t, bool(wd['it'] or '---' in sub_t), is_bold])
            if toks_out:
                all_col_lines.append([indent, toks_out])

    return all_col_lines


# ---------------------------------------------------------------------------
# Linguistic & Structural Parsing (v5)
# ---------------------------------------------------------------------------

def join_styled_hyphenated_wraps(col_lines):
    if not col_lines:
        return []
    merged = []
    i = 0
    while i < len(col_lines):
        indent, toks = col_lines[i]
        while toks and toks[-1][0].endswith('-') and not toks[-1][0].endswith('---') and (i + 1) < len(col_lines):
            next_indent, next_toks = col_lines[i + 1]
            if not next_toks:
                i += 1
                continue
            first_next, next_it, next_bold = next_toks[0]
            prev_word, prev_it, prev_bold = toks[-1]
            prev_base = prev_word[:-1].lower()
            # Preserve hyphen in true reduplications (e.g. 'timi-' + 'timiidu')
            if len(prev_base) >= 3 and first_next.lower().startswith(prev_base[:3]):
                joined = heal_ocr_token(prev_word + first_next)
            else:
                joined = heal_ocr_token(prev_word[:-1] + first_next)
            toks[-1] = [joined, bool(prev_it or next_it), bool(prev_bold and next_bold)]
            if len(next_toks) > 1:
                col_lines[i + 1] = [4, next_toks[1:]]
                break
            else:
                i += 1
        merged.append((indent, toks))
        i += 1
    return merged


def is_gorontalo_word(w: str, headword_root: str = '') -> bool:
    w_clean = w.strip('.,;:!?()"\'').lower()
    if not w_clean:
        return False
    if '---' in w_clean:
        return True
    if w_clean in INDONESIAN_STOP_STARTS or w_clean in CLEAR_INDONESIAN_STOP:
        return False
    if w_clean in GORONTALO_PARTICLES:
        return True
    if ':' in w_clean:
        return True
    if is_likely_indonesian_word(w_clean):
        return False
    if headword_root and len(headword_root) >= 3 and headword_root in w_clean.replace(':', ''):
        return True
    if re.match(r'^(?:mong|long|pong|mopo|lopo|popo|moti|loti|poti|mohi|lohi|pohi|molo|lolo|polo|motu|lotu|potu|mongo|ilo|pil|til|bil|dil|me:|le:|pe:|mo:|lo:|po:|mo|lo|po|ma|le|pe)[a-z:]{3,}', w_clean):
        return True
    if re.match(r'^([a-z:]{2,})-\1', w_clean) and w_clean.split('-')[0] not in INDONESIAN_STOP_STARTS:
        return True
    if re.search(r'(?:liyo|nto|mu|limongoliyo|laa|ma:o|mayi|mola|mota|yalo|walo|lo|po)$', w_clean):
        return True
    return False


def is_valid_headword(first_word: str, valid_letters=None) -> bool:
    w = first_word.strip('.,;:!?()"\'')
    if len(w) <= 1 and w.lower() not in {'a', 'e', 'i', 'o', 'u'}:
        return False
    if not re.match(r'^[a-zA-Z\'-][a-zA-Z:\'-]*$', w):
        return False
    w_low = w.lower().lstrip('\'-:')
    if not w_low:
        return False
    if w_low in INDONESIAN_STOP_STARTS or w_low in {'sababu', 'onggo', 'donggo', 'timi-timiidu', 'dumodupo', 'layito', 'odito', 'odiye', 'te:ya', 'te:to'}:
        return False
    if '-' in w_low and w_low.split('-')[0] in INDONESIAN_STOP_STARTS:
        return False
    w_base = w_low.rstrip('0123456789 ')
    if not w_base or (w_base[-1] not in 'aiueo-'):
        return False
    if valid_letters and w_base[0].isalpha():
        if w_base[0].upper() not in valid_letters:
            return False
    return True


def check_shares_root(w1: str, current_lema: str) -> bool:
    lema_clean = re.sub(r'[^a-z]', '', current_lema.split()[0].lower())
    w1_clean = re.sub(r'[^a-z]', '', w1.lower())
    if not lema_clean or not w1_clean:
        return False
    stem = lema_clean[:-1] if len(lema_clean) >= 4 else lema_clean
    if len(stem) >= 3 and stem in w1_clean:
        return True
    if len(lema_clean) >= 4:
        if w1_clean.startswith(lema_clean[:2]) and lema_clean[-2:] in w1_clean:
            return True
        if lema_clean[0] in 'tspk' and len(stem) >= 3 and stem[1:] in w1_clean:
            return True
    if len(lema_clean) == 3 and lema_clean[:-1] in w1_clean and re.match(r'^(?:mo|lo|po|me|le|pe|he|hi|ta|ti|te)', w1_clean):
        return True
    return False


def extract_subentry(styled_toks, current_lema: str):
    if not styled_toks:
        return None, styled_toks
    w1_raw, w1_it, w1_bold = styled_toks[0]
    if w1_it:
        return None, styled_toks
    w1 = w1_raw.strip('.,;:!?()"\'')
    if not re.match(r'^[a-z][a-z:\'-]{1,}$', w1):
        return None, styled_toks
    if w1 in INDONESIAN_STOP_STARTS or w1 in {'sababu', 'wonu', 'wanu', 'alihu', 'asali', 'openu', 'lapatao'}:
        return None, styled_toks
    if '-' in w1 and w1.split('-')[0] in INDONESIAN_STOP_STARTS:
        return None, styled_toks
    if w1[-1] not in 'aiueo':
        return None, styled_toks
    if len(styled_toks) < 2:
        return None, styled_toks
    if '---' in styled_toks[1][0]:
        return None, styled_toks

    # Check whether line starts with a valid sub-headword (either root-sharing or bold Gorontalo)
    shares = check_shares_root(w1, current_lema)
    is_particle_compound = False
    if w1 in {'ta', 'ti', 'te', 'u', 'o'} and len(styled_toks) >= 3:
        w2_raw, w2_it, _ = styled_toks[1]
        w2 = w2_raw.strip('.,;:!?()"\'')
        if not w2_it and re.match(r'^[a-z][a-z:\'-]{2,}$', w2) and check_shares_root(w2, current_lema):
            is_particle_compound = True

    if is_particle_compound:
        w2 = styled_toks[1][0].strip('.,;:!?()"\'')
        return f"{w1} {w2}", styled_toks[2:]

    if not shares and not (w1_bold and is_gorontalo_word(w1, current_lema)):
        return None, styled_toks

    stem = re.sub(r'[^a-z]', '', current_lema.split()[0].lower())[:-1]

    # 3-word compound '<w1> lo/to <w3>'
    if len(styled_toks) >= 4:
        w2_raw, w2_it, w2_b = styled_toks[1]
        w3_raw, w3_it, w3_b = styled_toks[2]
        w2 = w2_raw.strip('.,;:!?()"\'').lower()
        w3 = w3_raw.strip('.,;:!?()"\'')
        if not w2_it and not w3_it and w2 in {'lo', 'to'} and re.match(r'^[a-z][a-z:\'-]{2,}$', w3) and (w3[-1] in 'aiueo' or w3 == 'tau') and w3.lower() not in INDONESIAN_STOP_STARTS:
            return f"{w1} {w2} {w3}", styled_toks[3:]

    # 2-word compound '<w1> <w2>' (either w2 is bold, or shares Gorontalo phonotactics and NOT Indonesian)
    if len(styled_toks) >= 3:
        w2_raw, w2_it, w2_b = styled_toks[1]
        w2 = w2_raw.strip('.,;:!?()"\'')
        w2_low = w2.lower()
        if not w2_it and re.match(r'^[a-z][a-z:\'-]{2,}$', w2) and w2[-1] in 'aiueo' and w2_low not in INDONESIAN_STOP_STARTS:
            if not re.match(r'^(?:meng|meny|mem|men|me|di|ber|ter|pe|per|ke|se)[a-z]{3,}$', w2_low):
                if w2_b or is_gorontalo_word(w2, stem) or w2_low in {'bele', 'dunula', 'hiyo', 'pa:ngo', 'duhu', 'dulahu', 'dulahe', 'taluhu', 'taluhe', 'bongo', 'huta', 'ayu', 'tawu', 'mayihulalo', 'bule', 'bulotu'}:
                    return f"{w1} {w2}", styled_toks[2:]

    return w1, styled_toks[1:]


def extract_def_and_examples(word_form: str, styled_toks, current_lema: str = ''):
    if not styled_toks:
        return [], []

    # Normalize dash sequences
    norm_toks = []
    for tok, is_it, _ in styled_toks:
        t = tok.strip()
        if not t or t in {'·', '•', '|'}:
            continue
        norm_toks.append([t, bool(is_it or '---' in t)])

    if not norm_toks:
        return [], []

    # Bridge 1-to-2 non-Indonesian vowel-final tokens or '---' surrounded by italic tokens
    def is_bridgeable(tok_s: str) -> bool:
        wc = tok_s.strip('.,;:!?()"\'').lower()
        if not wc:
            return False
        if '---' in wc or wc in GORONTALO_PARTICLES or is_gorontalo_word(wc, current_lema):
            return True
        if wc in CLEAR_INDONESIAN_STOP or is_unambiguous_indonesian_word(wc):
            return False
        return (wc[-1] in 'aiueo') or (len(wc) <= 3 and wc.isalpha())

    for idx in range(1, len(norm_toks) - 1):
        if not norm_toks[idx][1] and norm_toks[idx - 1][1]:
            if norm_toks[idx + 1][1] and is_bridgeable(norm_toks[idx][0]):
                norm_toks[idx][1] = True
            elif (idx + 2) < len(norm_toks) and not norm_toks[idx + 1][1] and norm_toks[idx + 2][1]:
                if is_bridgeable(norm_toks[idx][0]) and is_bridgeable(norm_toks[idx + 1][0]):
                    norm_toks[idx][1] = True
                    norm_toks[idx + 1][1] = True

    # Split into sense segments at '1.', '2.', '3.', ... (and '|.' / '|,' OCR of '1.')
    sense_segments = []
    cur_seg = []
    for idx, (tok, is_it) in enumerate(norm_toks):
        is_sense_num = (not is_it) and bool(re.match(r'^(?:[1-9]|I|II|III|\|)[\.,]$', tok))
        if is_sense_num and idx > 0 and norm_toks[idx - 1][0].lower().strip('.,') in {'pukul', 'jam', 'no', 'nomor'}:
            is_sense_num = False
        if is_sense_num:
            if cur_seg:
                sense_segments.append(cur_seg)
            cur_seg = []
        else:
            cur_seg.append([tok, is_it])
    if cur_seg:
        sense_segments.append(cur_seg)

    definitions = []
    examples = []
    lema_root = re.sub(r'[^a-z]', '', (current_lema or word_form).split()[0].lower())

    for seg in sense_segments:
        if not seg:
            continue
        has_it = any(is_it for _, is_it in seg)
        if not has_it:
            d_str = expand_indonesian_abbrev(' '.join(t for t, _ in seg).replace('---', '').replace('|', '').strip(' ,;:-.'))
            if d_str and len(d_str) >= 2:
                definitions.append(d_str)
            continue

        # Locate start of first italic / Gorontalo example (never inside unclosed parentheses!)
        paren_depth = 0
        first_it = len(seg)
        for i, (tok_s, is_it) in enumerate(seg):
            if is_it and paren_depth == 0:
                first_it = i
                break
            paren_depth += tok_s.count('(') - tok_s.count(')')
            if paren_depth < 0:
                paren_depth = 0
        if first_it == len(seg):
            d_str = expand_indonesian_abbrev(' '.join(t for t, _ in seg).replace('---', '').replace('|', '').strip(' ,;:-.'))
            if d_str and len(d_str) >= 2:
                definitions.append(d_str)
            continue
        while first_it > 0:
            prev_w = seg[first_it - 1][0]
            if prev_w.endswith(')') or prev_w.endswith(';'):
                break
            if is_gorontalo_word(prev_w, lema_root) and first_it >= 2:
                first_it -= 1
                seg[first_it][1] = True
            else:
                break

        def_tokens = [t for t, _ in seg[:first_it]]
        d_str = expand_indonesian_abbrev(' '.join(def_tokens).replace('---', '').replace('|', '').strip(' ,;:-.'))
        if d_str and len(d_str) >= 2:
            definitions.append(d_str)

        # Also bridge short gaps between two italic runs if they are Gorontalo words (not Indonesian!)
        for idx in range(1, len(seg) - 1):
            if not seg[idx][1] and seg[idx - 1][1] and seg[idx + 1][1]:
                wc = seg[idx][0].strip('.,;:!?()"\'').lower()
                if is_gorontalo_word(wc, lema_root):
                    seg[idx][1] = True

        pos = first_it
        m = len(seg)
        while pos < m:
            while pos < m and not seg[pos][1]:
                pos += 1
            if pos >= m:
                break
            it_start = pos
            while pos < m and seg[pos][1]:
                pos += 1
            it_end = pos
            rm_start = pos
            while pos < m and not seg[pos][1]:
                pos += 1
            rm_end = pos

            gor_raw = ' '.join(t for t, _ in seg[it_start:it_end])
            id_raw = ' '.join(t for t, _ in seg[rm_start:rm_end])
            gor_sent = gor_raw.replace('---', word_form).replace('|', '').strip(' ,;:-.')
            id_sent = expand_indonesian_abbrev(id_raw.replace('---', '').replace('|', '').strip(' ,;:-.'))
            gor_sent = re.sub(r'\s+', ' ', gor_sent).strip()
            id_sent = re.sub(r'\s+', ' ', id_sent).strip()

            if (
                gor_sent and id_sent and len(gor_sent) >= 3 and len(id_sent) >= 3
                and not re.match(r'^[\d\W]+$', id_sent)
                and not re.match(r'^[\d\W]+$', gor_sent)
            ):
                examples.append({
                    'kalimat_gorontalo': gor_sent,
                    'terjemahan_indonesia': id_sent
                })
            elif not d_str and id_sent and len(id_sent) >= 2:
                definitions.append(id_sent)

    return definitions, examples


def clean_gloss_for_lexicon(def_str: str):
    s = re.sub(r'^(?:[1-9]|I|II|III)[\.,]\s*', '', def_str).strip(' ,;:-.')
    if not s or len(s) < 2:
        return []
    results = []
    s_balanced = re.sub(r'\([^)]*$', '', s).strip(' ,;:-.')
    if s_balanced and len(s_balanced) >= 2:
        results.append(s_balanced)
    core = re.sub(r'\s*\([^)]*\)', '', s_balanced).strip(' ,;:-.')
    core = re.sub(r'\s+', ' ', core).strip()
    if core and core != s_balanced and len(core) >= 2:
        results.append(core)
    if ',' in core or ';' in core:
        parts = [p.strip(' ,;:-.') for p in re.split(r'[;,]', core) if p.strip(' ,;:-.')]
        if 2 <= len(parts) <= 4 and all(1 <= len(p.split()) <= 4 and len(p) >= 3 for p in parts):
            for p in parts:
                if p.lower() not in {'dan sebagainya', 'dan seterusnya', 'misalnya', 'ump'} and not p.startswith('pukul '):
                    if p not in results:
                        results.append(p)
    return results


def ocr_and_cache_page(pdf_page: int):
    cache_p = CACHE_V5_DIR / f'page_{pdf_page:03d}.json'
    if cache_p.exists():
        return pdf_page, json.loads(cache_p.read_text())
    lines = ocr_gi_page(pdf_page, dpi=350)
    cache_p.write_text(json.dumps(lines, ensure_ascii=False))
    return pdf_page, lines


def parse_all_pages(pages_list):
    page_data = []
    page_dominant_letter = []

    for pdf_page in pages_list:
        kamus_page = pdf_page - 37
        _, raw_lines = ocr_and_cache_page(pdf_page)
        col_lines = join_styled_hyphenated_wraps(raw_lines)

        letter_counts = Counter()
        for ind, toks in col_lines:
            if ind == 0 and toks and not toks[0][1]:
                w0 = toks[0][0].strip('.,;:!?()"\'').lstrip('-:\'')
                if w0 and w0[0].isalpha() and is_valid_headword(toks[0][0], None):
                    letter_counts[w0[0].upper()] += 1
        dom = letter_counts.most_common(1)[0][0] if letter_counts else (page_dominant_letter[-1] if page_dominant_letter else 'A')
        page_dominant_letter.append(dom)
        page_data.append((kamus_page, pdf_page, col_lines, letter_counts))

    all_entries = []
    cur_entry = None
    cur_sub = None

    def flush_entry(entry):
        if not entry:
            return
        toks = entry.pop('_styled_toks', [])
        defs, exs = extract_def_and_examples(entry['lema'], toks, entry['lema'])
        entry['definisi'] = defs
        entry['contoh'] = exs
        clean_turunan = []
        for sub in entry['turunan']:
            stoks = sub.pop('_styled_toks', [])
            sdefs, sexs = extract_def_and_examples(sub['kata'], stoks, entry['lema'])
            sub['definisi'] = sdefs
            sub['contoh'] = sexs
            if sub['definisi'] or sub['contoh']:
                clean_turunan.append(sub)
        entry['turunan'] = clean_turunan
        if entry['definisi'] or entry['contoh'] or entry['turunan']:
            all_entries.append(entry)

    for idx, (kamus_page, pdf_page, col_lines, letter_counts) in enumerate(page_data):
        valid_letters = {page_dominant_letter[idx]}
        if idx > 0:
            valid_letters.add(page_dominant_letter[idx - 1])
        if idx + 1 < len(page_dominant_letter):
            valid_letters.add(page_dominant_letter[idx + 1])
        for ch, cnt in letter_counts.items():
            if cnt >= 2:
                valid_letters.add(ch)

        for indent, styled_toks in col_lines:
            if not styled_toks:
                continue
            first_word, first_it, first_bold = styled_toks[0]
            rest_toks = styled_toks[1:]

            if indent == 0 and not first_it and len(styled_toks) == 1 and re.match(r'^[A-Z]$', first_word.strip('.')):
                continue

            if indent == 0 and not first_it and is_valid_headword(first_word, valid_letters):
                lema = first_word.rstrip('.,;:')
                # Handle multi-word bold headword like 'aa: panga' -> 'aa:panga'
                if lema.endswith(':') and rest_toks and not rest_toks[0][1] and rest_toks[0][0].isalpha():
                    lema = lema + rest_toks[0][0].rstrip('.,;:')
                    rest_toks = rest_toks[1:]
                first_alpha = lema.lstrip('-:\'')
                entry_letter = first_alpha[0].upper() if (first_alpha and first_alpha[0].isalpha()) else page_dominant_letter[idx]
                flush_entry(cur_entry)
                cur_entry = {
                    'lema': lema,
                    'huruf': entry_letter,
                    'halaman_kamus': kamus_page,
                    'halaman_pdf': pdf_page,
                    'sumber_ocr': 'multimodal_tess5_pymupdf_v5',
                    '_styled_toks': rest_toks,
                    'definisi': [],
                    'contoh': [],
                    'turunan': []
                }
                cur_sub = None
            else:
                if not cur_entry:
                    continue
                sub_word, rem_toks = extract_subentry(styled_toks, cur_entry['lema'])
                if sub_word:
                    cur_sub = {
                        'kata': sub_word,
                        '_styled_toks': rem_toks,
                        'definisi': [],
                        'contoh': []
                    }
                    cur_entry['turunan'].append(cur_sub)
                else:
                    if cur_sub is not None:
                        cur_sub['_styled_toks'].extend(styled_toks)
                    else:
                        cur_entry['_styled_toks'].extend(styled_toks)

    flush_entry(cur_entry)
    return all_entries


def export_all_datasets(all_entries):
    for b_idx in range(6):
        p_start = b_idx * 50 + 1
        p_end = 302 if b_idx == 5 else (b_idx + 1) * 50
        b_entries = [e for e in all_entries if p_start <= e['halaman_kamus'] <= p_end]
        b_parallel = []
        for e in b_entries:
            for c in e['contoh']:
                b_parallel.append({
                    'lema': e['lema'], 'kata_terkait': e['lema'],
                    'kalimat_gorontalo': c['kalimat_gorontalo'],
                    'terjemahan_indonesia': c['terjemahan_indonesia'],
                    'halaman_kamus': e['halaman_kamus']
                })
            for sub in e['turunan']:
                for c in sub['contoh']:
                    b_parallel.append({
                        'lema': e['lema'], 'kata_terkait': sub['kata'],
                        'kalimat_gorontalo': c['kalimat_gorontalo'],
                        'terjemahan_indonesia': c['terjemahan_indonesia'],
                        'halaman_kamus': e['halaman_kamus']
                    })
        with open(BATCH_DIR / f'batch_{b_idx+1}_clean.jsonl', 'w') as f:
            for e in b_entries:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        with open(BATCH_DIR / f'batch_{b_idx+1}_parallel.jsonl', 'w') as f:
            for p in b_parallel:
                f.write(json.dumps(p, ensure_ascii=False) + '\n')
        rep = {
            'batch': b_idx + 1,
            'halaman_kamus_range': f'{p_start}-{p_end}',
            'jumlah_lema': len(b_entries),
            'jumlah_sub_lema': sum(len(e['turunan']) for e in b_entries),
            'jumlah_kalimat_paralel': len(b_parallel)
        }
        (BATCH_DIR / f'batch_{b_idx+1}_report.json').write_text(json.dumps(rep, indent=2, ensure_ascii=False))

    parallel_rows, lexicon_rows, morph_rows, sft_rows = [], [], [], []
    seen_lex = set()

    for e in all_entries:
        definisi_full = '; '.join(e['definisi']).strip()
        for d_sense in e['definisi']:
            for gloss in clean_gloss_for_lexicon(d_sense):
                key = (e['lema'].lower(), gloss.lower())
                if key not in seen_lex:
                    seen_lex.add(key)
                    lexicon_rows.append({
                        'id': f"lex_{len(lexicon_rows)+1:05d}",
                        'gorontalo': e['lema'],
                        'indonesia': gloss,
                        'tipe': 'lema_pokok',
                        'lema_induk': e['lema'],
                        'halaman_buku': e['halaman_kamus']
                    })
        for c in e['contoh']:
            parallel_rows.append({
                'id': f"par_{len(parallel_rows)+1:05d}",
                'src_indonesia': c['terjemahan_indonesia'],
                'tgt_gorontalo': c['kalimat_gorontalo'],
                'lema': e['lema'],
                'kata_terkait': e['lema'],
                'halaman_buku': e['halaman_kamus']
            })
        for sub in e['turunan']:
            sub_def_full = '; '.join(sub['definisi']).strip()
            for sd_sense in sub['definisi']:
                for gloss in clean_gloss_for_lexicon(sd_sense):
                    key = (sub['kata'].lower(), gloss.lower())
                    if key not in seen_lex:
                        seen_lex.add(key)
                        lexicon_rows.append({
                            'id': f"lex_{len(lexicon_rows)+1:05d}",
                            'gorontalo': sub['kata'],
                            'indonesia': gloss,
                            'tipe': 'kata_turunan',
                            'lema_induk': e['lema'],
                            'halaman_buku': e['halaman_kamus']
                        })
            if sub_def_full and len(sub_def_full) >= 2:
                morph_rows.append({
                    'id': f"mor_{len(morph_rows)+1:05d}",
                    'akar_kata': e['lema'],
                    'arti_akar': definisi_full,
                    'kata_turunan': sub['kata'],
                    'arti_turunan': sub_def_full,
                    'contoh_kalimat': sub['contoh'][0]['kalimat_gorontalo'] if sub['contoh'] else '',
                    'arti_kalimat': sub['contoh'][0]['terjemahan_indonesia'] if sub['contoh'] else '',
                    'halaman_buku': e['halaman_kamus']
                })
            for c in sub['contoh']:
                parallel_rows.append({
                    'id': f"par_{len(parallel_rows)+1:05d}",
                    'src_indonesia': c['terjemahan_indonesia'],
                    'tgt_gorontalo': c['kalimat_gorontalo'],
                    'lema': e['lema'],
                    'kata_terkait': sub['kata'],
                    'halaman_buku': e['halaman_kamus']
                })

    for p in parallel_rows:
        sft_rows.append({
            'id': f"sft_{len(sft_rows)+1:05d}",
            'task': 'translate_id_to_gor',
            'instruction': 'Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Gorontalo yang baik dan benar.',
            'input': p['src_indonesia'],
            'output': p['tgt_gorontalo'],
            'lema': p['lema'],
            'halaman_buku': p['halaman_buku']
        })
        sft_rows.append({
            'id': f"sft_{len(sft_rows)+1:05d}",
            'task': 'translate_gor_to_id',
            'instruction': 'Terjemahkan kalimat bahasa Gorontalo berikut ke dalam bahasa Indonesia.',
            'input': p['tgt_gorontalo'],
            'output': p['src_indonesia'],
            'lema': p['lema'],
            'halaman_buku': p['halaman_buku']
        })
    for lx in lexicon_rows:
        sft_rows.append({
            'id': f"sft_{len(sft_rows)+1:05d}",
            'task': 'lexicon_id_to_gor',
            'instruction': 'Apa kosakata atau istilah bahasa Gorontalo untuk makna bahasa Indonesia berikut?',
            'input': lx['indonesia'],
            'output': lx['gorontalo'],
            'lema': lx['lema_induk'],
            'halaman_buku': lx['halaman_buku']
        })
    for m in morph_rows:
        sft_rows.append({
            'id': f"sft_{len(sft_rows)+1:05d}",
            'task': 'morphology_derivation',
            'instruction': f"Bentuklah kata turunan bahasa Gorontalo dari akar kata '{m['akar_kata']}' beserta artinya dalam bahasa Indonesia.",
            'input': f"Akar kata: {m['akar_kata']} ({m['arti_akar'] or 'kata dasar'}) -> Makna sasaran: {m['arti_turunan']}",
            'output': f"Kata turunan: {m['kata_turunan']} (Artinya: {m['arti_turunan']})" + (f" | Contoh: {m['contoh_kalimat']} = {m['arti_kalimat']}" if m['contoh_kalimat'] else ""),
            'lema': m['akar_kata'],
            'halaman_buku': m['halaman_buku']
        })

    with open(DATASET_DIR / 'kamus_gorontalo_indonesia.jsonl', 'w') as f:
        for e in all_entries:
            f.write(json.dumps(e, ensure_ascii=False) + '\n')
    (DATASET_DIR / 'kamus_gorontalo_indonesia.json').write_text(json.dumps(all_entries, indent=2, ensure_ascii=False))
    with open(DATASET_DIR / 'korpus_kalimat_paralel.jsonl', 'w') as f:
        for p in parallel_rows:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')

    by_letter = {}
    for e in all_entries:
        h = e['huruf'] if e['huruf'].isalpha() else 'A'
        by_letter.setdefault(h, []).append(e)
    for h, entries_h in sorted(by_letter.items()):
        (HURUF_DIR / f'kamus_huruf_{h}.json').write_text(json.dumps(entries_h, indent=2, ensure_ascii=False))

    def save_jsonl_and_parquet(rows, stem):
        with open(TRAIN_DIR / f'{stem}.jsonl', 'w') as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
        pd.DataFrame(rows).to_parquet(TRAIN_DIR / f'{stem}.parquet', index=False)

    save_jsonl_and_parquet(parallel_rows, 'train_parallel_sentences')
    save_jsonl_and_parquet(lexicon_rows, 'train_lexicon_id_to_gor')
    save_jsonl_and_parquet(morph_rows, 'train_morphology_grammar')
    save_jsonl_and_parquet(sft_rows, 'train_sft_instruction')

    pd.DataFrame(parallel_rows).to_csv(DATASET_DIR / 'korpus_kalimat_paralel.csv', index=False)
    pd.DataFrame(parallel_rows).to_parquet(DATASET_DIR / 'korpus_kalimat_paralel.parquet', index=False)
    pd.DataFrame(lexicon_rows).to_csv(DATASET_DIR / 'leksikon_kosakata_lengkap.csv', index=False)
    pd.DataFrame(lexicon_rows).to_parquet(DATASET_DIR / 'leksikon_kosakata_lengkap.parquet', index=False)

    stats = {
        'total_halaman_kamus': 302,
        'metode_ekstraksi': 'Multimodal Strip-Deskewed Tesseract 5 (ind+eng) + Visual Bold/Italic + PyMuPDF Fusion + Gap-Dash Recovery (v5)',
        'total_entri_lema_pokok': len(all_entries),
        'total_sub_lema_turunan': sum(len(e['turunan']) for e in all_entries),
        'total_leksikon_bersih': len(lexicon_rows),
        'total_kalimat_paralel_bersih': len(parallel_rows),
        'total_pasangan_morfologi': len(morph_rows),
        'total_instruksi_sft': len(sft_rows),
        'distribusi_huruf': {k: len(v) for k, v in sorted(by_letter.items())}
    }
    (DATASET_DIR / 'statistik_dataset.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print("GI v5 DONE! Stats:", json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-pages', nargs='+', type=int, default=None)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    t0 = time.time()
    if args.test_pages:
        pages = args.test_pages
        for p in pages:
            ocr_and_cache_page(p)
        entries = parse_all_pages(pages)
        print(f"TEST extracted {len(entries)} entries in {time.time()-t0:.1f}s")
        for e in entries:
            print(f"[{e['halaman_pdf']}] {e['lema']} | def={e['definisi']} | ex={e['contoh']}")
            for s in e['turunan']:
                print(f"    -> {s['kata']} | def={s['definisi']} | ex={s['contoh']}")
        return

    all_pages = list(range(38, 340))
    missing = [p for p in all_pages if not (CACHE_V5_DIR / f'page_{p:03d}.json').exists()]
    print(f"Running OCR v5 on {len(missing)} remaining pages (out of {len(all_pages)}) with {args.workers} workers...")
    if missing:
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for p, _ in ex.map(ocr_and_cache_page, missing):
                done += 1
                if done % 30 == 0 or done == len(missing):
                    print(f"  OCR progress: {done}/{len(missing)} ({time.time()-t0:.1f}s)")

    print("Parsing structured entries across all 302 pages...")
    entries = parse_all_pages(all_pages)
    export_all_datasets(entries)


if __name__ == '__main__':
    main()
