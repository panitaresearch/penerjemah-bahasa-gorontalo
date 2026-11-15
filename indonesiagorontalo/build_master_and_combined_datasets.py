import json, re
import pandas as pd
from pathlib import Path

BASE_IG = Path('/content/drive/MyDrive/REPO/richie/modelgorontalo/indonesiagorontalo')
DATASET_IG = BASE_IG / 'dataset'
TRAIN_IG = DATASET_IG / 'training_ready'
BATCH_IG = DATASET_IG / 'batches'
HURUF_IG = DATASET_IG / 'huruf'

BASE_GI = Path('/content/drive/MyDrive/REPO/richie/modelgorontalo/gorontaloindonesia/dataset/training_ready')
COMBINED_DIR = Path('/content/drive/MyDrive/REPO/richie/modelgorontalo/dataset_terpadu_indonesia_gorontalo')
COMBINED_DIR.mkdir(parents=True, exist_ok=True)
TRAIN_IG.mkdir(parents=True, exist_ok=True)

# 1. Gather all entries across all 6 v5-extracted batches of Kamus Indonesia-Gorontalo
all_entries = []
for b_idx in range(1, 7):
    b_file = BATCH_IG / f'batch_{b_idx}_clean.jsonl'
    for line in b_file.read_text().splitlines():
        if line.strip():
            all_entries.append(json.loads(line))

parallel_rows = []
lexicon_rows = []
morph_rows = []
sft_rows = []
seen_lex = set()


def add_lex_pair(ind_word, gor_asli, gor_std, tipe, lema_induk, huruf, hal):
    ind_c = ind_word.strip(' ,;:.').lower() if not ind_word.istitle() else ind_word.strip(' ,;:.')
    ind_low = ind_c.lower()
    g_asli = gor_asli.strip(' ,;:.')
    g_std = gor_std.strip(' ,;:.')
    if not ind_low or not g_std or len(g_std) < 2:
        return
    key = (ind_low, g_std.lower())
    if key in seen_lex:
        return
    seen_lex.add(key)
    lexicon_rows.append({
        'id': f"ig_lex_{len(lexicon_rows)+1:05d}",
        'indonesia': ind_low,
        'gorontalo_asli': g_asli,
        'gorontalo_standar': g_std,
        'tipe': tipe,
        'lema_induk': lema_induk,
        'huruf': huruf,
        'halaman_pdf': hal
    })


for e in all_entries:
    lema_induk = e.get('lema_induk', e['lema_indonesia']).lower()
    lema_id = e['lema_indonesia'].lower()
    tipe_lema = e.get('tipe_lema', 'lema_pokok')
    pads_asli = [p.strip(' ,;:.') for p in e['padanan_gorontalo'] if p.strip(' ,;:.')]
    pads_std = [p.strip(' ,;:.') for p in e['padanan_gorontalo_standar'] if p.strip(' ,;:.')]

    # Add individual clean synonym pairs as well as combined if multiple
    for pa, ps in zip(pads_asli, pads_std):
        add_lex_pair(lema_id, pa, ps, tipe_lema, lema_induk, e['huruf'], e['halaman_pdf'])
    if len(pads_std) > 1:
        add_lex_pair(lema_id, '; '.join(pads_asli), '; '.join(pads_std), tipe_lema, lema_induk, e['huruf'], e['halaman_pdf'])

    if tipe_lema == 'kata_turunan' and lema_id != lema_induk and pads_std:
        morph_rows.append({
            'id': f"ig_mor_{len(morph_rows)+1:05d}",
            'akar_indonesia': lema_induk,
            'padanan_akar_gorontalo': pads_std[0],
            'turunan_indonesia': lema_id,
            'padanan_turunan_gorontalo': '; '.join(pads_std),
            'padanan_turunan_asli_1987': '; '.join(pads_asli),
            'contoh_indonesia': e['contoh'][0]['kalimat_indonesia'] if e['contoh'] else '',
            'contoh_gorontalo': e['contoh'][0]['kalimat_gorontalo_standar'] if e['contoh'] else '',
            'huruf': e['huruf'],
            'halaman_pdf': e['halaman_pdf']
        })

    for c in e['contoh']:
        parallel_rows.append({
            'id': f"ig_par_{len(parallel_rows)+1:05d}",
            'src_indonesia': c['kalimat_indonesia'],
            'tgt_gorontalo': c['kalimat_gorontalo_standar'],
            'tgt_gorontalo_asli_1987': c['kalimat_gorontalo'],
            'lema_indonesia': lema_induk,
            'kata_terkait': lema_id,
            'huruf': e['huruf'],
            'halaman_pdf': e['halaman_pdf']
        })

    for sub in e['turunan']:
        sub_id = sub['kata_indonesia'].strip(' ,;:.').lower()
        spads_asli = [p.strip(' ,;:.') for p in sub['padanan_gorontalo'] if p.strip(' ,;:.')]
        spads_std = [p.strip(' ,;:.') for p in sub['padanan_gorontalo_standar'] if p.strip(' ,;:.')]
        for spa, sps in zip(spads_asli, spads_std):
            add_lex_pair(sub_id, spa, sps, 'kata_turunan', lema_induk, e['huruf'], e['halaman_pdf'])
        if len(spads_std) > 1:
            add_lex_pair(sub_id, '; '.join(spads_asli), '; '.join(spads_std), 'kata_turunan', lema_induk, e['huruf'], e['halaman_pdf'])

        if spads_std:
            morph_rows.append({
                'id': f"ig_mor_{len(morph_rows)+1:05d}",
                'akar_indonesia': lema_induk,
                'padanan_akar_gorontalo': pads_std[0] if pads_std else spads_std[0],
                'turunan_indonesia': sub_id,
                'padanan_turunan_gorontalo': '; '.join(spads_std),
                'padanan_turunan_asli_1987': '; '.join(spads_asli),
                'contoh_indonesia': sub['contoh'][0]['kalimat_indonesia'] if sub['contoh'] else '',
                'contoh_gorontalo': sub['contoh'][0]['kalimat_gorontalo_standar'] if sub['contoh'] else '',
                'huruf': e['huruf'],
                'halaman_pdf': e['halaman_pdf']
            })
        for c in sub['contoh']:
            parallel_rows.append({
                'id': f"ig_par_{len(parallel_rows)+1:05d}",
                'src_indonesia': c['kalimat_indonesia'],
                'tgt_gorontalo': c['kalimat_gorontalo_standar'],
                'tgt_gorontalo_asli_1987': c['kalimat_gorontalo'],
                'lema_indonesia': lema_induk,
                'kata_terkait': sub_id,
                'huruf': e['huruf'],
                'halaman_pdf': e['halaman_pdf']
            })

# Build SFT Instruction dataset for indonesiagorontalo
for p in parallel_rows:
    sft_rows.append({
        'id': f"ig_sft_{len(sft_rows)+1:05d}",
        'task': 'translate_id_to_gor',
        'instruction': 'Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Gorontalo yang baik dan benar.',
        'input': p['src_indonesia'],
        'output': p['tgt_gorontalo'],
        'output_ejaan_1987': p['tgt_gorontalo_asli_1987'],
        'lema': p['lema_indonesia']
    })
    sft_rows.append({
        'id': f"ig_sft_{len(sft_rows)+1:05d}",
        'task': 'translate_gor_to_id',
        'instruction': 'Terjemahkan kalimat bahasa Gorontalo berikut ke dalam bahasa Indonesia.',
        'input': p['tgt_gorontalo'],
        'output': p['src_indonesia'],
        'output_ejaan_1987': p['src_indonesia'],
        'lema': p['lema_indonesia']
    })
for lx in lexicon_rows:
    sft_rows.append({
        'id': f"ig_sft_{len(sft_rows)+1:05d}",
        'task': 'lexicon_id_to_gor',
        'instruction': 'Apa padanan kata atau istilah bahasa Gorontalo untuk kata bahasa Indonesia berikut?',
        'input': lx['indonesia'],
        'output': lx['gorontalo_standar'],
        'output_ejaan_1987': lx['gorontalo_asli'],
        'lema': lx['lema_induk']
    })
for m in morph_rows:
    sft_rows.append({
        'id': f"ig_sft_{len(sft_rows)+1:05d}",
        'task': 'morphology_id_to_gor',
        'instruction': f"Terjemahkan kata turunan bahasa Indonesia '{m['turunan_indonesia']}' (dari kata dasar '{m['akar_indonesia']}') ke dalam bahasa Gorontalo beserta contoh penggunaannya.",
        'input': f"Kata dasar: {m['akar_indonesia']} ({m['padanan_akar_gorontalo']}) -> Kata turunan: {m['turunan_indonesia']}",
        'output': f"Padanan Gorontalo: {m['padanan_turunan_gorontalo']}" + (f" | Contoh: {m['contoh_indonesia']} = {m['contoh_gorontalo']}" if m['contoh_indonesia'] else ""),
        'output_ejaan_1987': m['padanan_turunan_asli_1987'],
        'lema': m['akar_indonesia']
    })

# Save master files in indonesiagorontalo/dataset/
with open(DATASET_IG / 'kamus_indonesia_gorontalo.jsonl', 'w') as f:
    for e in all_entries:
        f.write(json.dumps(e, ensure_ascii=False) + '\n')
(DATASET_IG / 'kamus_indonesia_gorontalo.json').write_text(json.dumps(all_entries, indent=2, ensure_ascii=False))
with open(DATASET_IG / 'korpus_kalimat_paralel.jsonl', 'w') as f:
    for p in parallel_rows:
        f.write(json.dumps(p, ensure_ascii=False) + '\n')

pd.DataFrame(parallel_rows).to_csv(DATASET_IG / 'korpus_kalimat_paralel.csv', index=False)
pd.DataFrame(parallel_rows).to_parquet(DATASET_IG / 'korpus_kalimat_paralel.parquet', index=False)
pd.DataFrame(lexicon_rows).to_csv(DATASET_IG / 'leksikon_kosakata_lengkap.csv', index=False)
pd.DataFrame(lexicon_rows).to_parquet(DATASET_IG / 'leksikon_kosakata_lengkap.parquet', index=False)


def save_jsonl_parquet(rows, directory, stem):
    with open(directory / f'{stem}.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    pd.DataFrame(rows).to_parquet(directory / f'{stem}.parquet', index=False)


save_jsonl_parquet(parallel_rows, TRAIN_IG, 'train_parallel_sentences')
save_jsonl_parquet(lexicon_rows, TRAIN_IG, 'train_lexicon_id_to_gor')
save_jsonl_parquet(morph_rows, TRAIN_IG, 'train_morphology_grammar')
save_jsonl_parquet(sft_rows, TRAIN_IG, 'train_sft_instruction')

by_letter = {}
for e in all_entries:
    by_letter[e['huruf']] = by_letter.get(e['huruf'], 0) + 1

ig_stats = {
    'sumber': 'Kamus Bahasa Indonesia-Gorontalo (Prof. Dr. H. Mansoer Pateda)',
    'total_halaman_pengantar': 19,
    'total_halaman_leksikon': 489,
    'total_berkas_pdf': 26,
    'total_entri_lema_indonesia': len(all_entries),
    'total_sub_lema_turunan': sum(len(e['turunan']) for e in all_entries),
    'total_leksikon_bersih': len(lexicon_rows),
    'total_kalimat_paralel_bersih': len(parallel_rows),
    'total_pasangan_morfologi': len(morph_rows),
    'total_instruksi_sft': len(sft_rows),
    'distribusi_abjad': dict(sorted(by_letter.items()))
}
(DATASET_IG / 'statistik_dataset.json').write_text(json.dumps(ig_stats, indent=2, ensure_ascii=False))
print("Kamus Indonesia-Gorontalo Stats:", json.dumps(ig_stats, indent=2))

# 3. Build Unified Combined Dataset (Kamus Gorontalo-Indonesia + Kamus Indonesia-Gorontalo)
gi_par = [json.loads(line) for line in open(BASE_GI / 'train_parallel_sentences.jsonl') if line.strip()]
gi_lex = [json.loads(line) for line in open(BASE_GI / 'train_lexicon_id_to_gor.jsonl') if line.strip()]
gi_mor = [json.loads(line) for line in open(BASE_GI / 'train_morphology_grammar.jsonl') if line.strip()]
gi_sft = [json.loads(line) for line in open(BASE_GI / 'train_sft_instruction.jsonl') if line.strip()]

comb_par = []
for r in gi_par:
    comb_par.append({
        'id': f"comb_par_{len(comb_par)+1:05d}",
        'src_indonesia': r['src_indonesia'],
        'tgt_gorontalo': r['tgt_gorontalo'],
        'tgt_gorontalo_asli': r['tgt_gorontalo'],
        'lema': r['lema'],
        'kata_terkait': r.get('kata_terkait', r['lema']),
        'sumber_kamus': 'gorontalo_indonesia_2001'
    })
for r in parallel_rows:
    comb_par.append({
        'id': f"comb_par_{len(comb_par)+1:05d}",
        'src_indonesia': r['src_indonesia'],
        'tgt_gorontalo': r['tgt_gorontalo'],
        'tgt_gorontalo_asli': r['tgt_gorontalo_asli_1987'],
        'lema': r['lema_indonesia'],
        'kata_terkait': r['kata_terkait'],
        'sumber_kamus': 'indonesia_gorontalo_1987'
    })

comb_lex = []
for r in gi_lex:
    comb_lex.append({
        'id': f"comb_lex_{len(comb_lex)+1:05d}",
        'indonesia': r['indonesia'],
        'gorontalo': r['gorontalo'],
        'gorontalo_asli': r['gorontalo'],
        'tipe': r['tipe'],
        'lema_induk': r['lema_induk'],
        'sumber_kamus': 'gorontalo_indonesia_2001'
    })
for r in lexicon_rows:
    comb_lex.append({
        'id': f"comb_lex_{len(comb_lex)+1:05d}",
        'indonesia': r['indonesia'],
        'gorontalo': r['gorontalo_standar'],
        'gorontalo_asli': r['gorontalo_asli'],
        'tipe': r['tipe'],
        'lema_induk': r['lema_induk'],
        'sumber_kamus': 'indonesia_gorontalo_1987'
    })

comb_sft = []
for r in gi_sft:
    comb_sft.append({
        'id': f"comb_sft_{len(comb_sft)+1:05d}",
        'task': r['task'],
        'instruction': r['instruction'],
        'input': r['input'],
        'output': r['output'],
        'sumber_kamus': 'gorontalo_indonesia_2001'
    })
for r in sft_rows:
    comb_sft.append({
        'id': f"comb_sft_{len(comb_sft)+1:05d}",
        'task': r['task'],
        'instruction': r['instruction'],
        'input': r['input'],
        'output': r['output'],
        'sumber_kamus': 'indonesia_gorontalo_1987'
    })

save_jsonl_parquet(comb_par, COMBINED_DIR, 'combined_parallel_sentences')
save_jsonl_parquet(comb_lex, COMBINED_DIR, 'combined_lexicon_id_to_gor')
save_jsonl_parquet(comb_sft, COMBINED_DIR, 'combined_sft_instruction')

comb_stats = {
    'total_kalimat_paralel_gabungan': len(comb_par),
    'rincian_kalimat_paralel': {
        'kamus_gorontalo_indonesia_2001': len(gi_par),
        'kamus_indonesia_gorontalo_1987': len(parallel_rows)
    },
    'total_leksikon_gabungan': len(comb_lex),
    'rincian_leksikon': {
        'kamus_gorontalo_indonesia_2001': len(gi_lex),
        'kamus_indonesia_gorontalo_1987': len(lexicon_rows)
    },
    'total_pasangan_morfologi_gabungan': len(gi_mor) + len(morph_rows),
    'total_instruksi_sft_gabungan': len(comb_sft)
}
(COMBINED_DIR / 'statistik_dataset_terpadu.json').write_text(json.dumps(comb_stats, indent=2, ensure_ascii=False))
print("Combined Dataset Stats:", json.dumps(comb_stats, indent=2))
