#!/usr/bin/env python3
"""
infer_gorontalo.py
==================
Utilitas Inferensi & Demo Penerjemah Dwibahasa Bahasa Indonesia <-> Bahasa Gorontalo
Mendukung model yang dilatih di A100 GPU (Korpus v6):
  1. NLLB-200-600M (Full Fine-Tuned Seq2Seq: `models/nllb-200-gorontalo-bi-600M`)
  2. Qwen2.5-7B-Instruct LoRA (`models/qwen2.5-7b-gorontalo-lora`)
  3. GoTo Gemma 3 27B CPT GoTo-AI v3 LoRA (`models/gemma3-27b-gotoai-gorontalo-lora`)

Mendukung pemecahan otomatis per paragraf dan per kalimat agar penerjemahan teks panjang
tetap utuh (tanpa sentence dropping) dan bebas pengulangan kata berlebih.

Contoh pemakaian CLI:
  python3 infer_gorontalo.py --model nllb --direction id2gor --text "Saya pergi ke pasar kemarin."
  python3 infer_gorontalo.py --model qwen --direction gor2id --text "Wa:tiya loona:o ode pa:tali ola:ngo."
  python3 infer_gorontalo.py --model gemma3 --direction id2gor --text "Saya pergi ke pasar kemarin."
"""

import argparse
import os
import re
from pathlib import Path
from typing import Callable, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

BASE_DIR = Path(__file__).resolve().parent
NLLB_DIR = BASE_DIR / "models" / "nllb-200-gorontalo-bi-600M"
NLLB_HF_REPO = "panitagorontalo/nllb-200-gorontalo-bi-600M"
QWEN_BASE = "Qwen/Qwen2.5-7B-Instruct"
QWEN_LORA_DIR = BASE_DIR / "models" / "qwen2.5-7b-gorontalo-lora"
QWEN_HF_REPO = "panitagorontalo/qwen2.5-7b-gorontalo-lora"
GEMMA3_BASE = "GoToCompany/gemma3-27b-cpt-gotoai-v3-instruct"
GEMMA3_LORA_DIR = BASE_DIR / "models" / "gemma3-27b-gotoai-gorontalo-lora"
GEMMA3_HF_REPO = "panitagorontalo/gemma3-27b-gotoai-gorontalo-lora"


def _resolve_model_source(local_dir: Path, hf_repo_id: str, env_key: str) -> str:
    """Gunakan path/ID env bila diset, path lokal jika bobot ada, atau fallback ke Hugging Face Hub."""
    override = os.environ.get(env_key, "").strip()
    if override:
        return override
    if (local_dir / "config.json").exists() or (local_dir / "adapter_config.json").exists():
        return str(local_dir)
    return hf_repo_id


def _resolve_dtype() -> torch.dtype:
    """Pilih bfloat16 pada GPU modern (Ampere+), float16 pada GPU CUDA lama (T4), dan float32 di CPU."""
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def _build_4bit_config(compute_dtype: torch.dtype):
    """Bangun konfigurasi NF4 4-bit untuk GPU VRAM terbatas (mis. T4 15 GB / L4 24 GB)."""
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )

SYSTEM_PROMPT = (
    "Anda adalah pakar bahasa dan penerjemah dwibahasa Bahasa Indonesia <-> Bahasa Gorontalo "
    "yang menguasai tata bahasa, 5 sistem kala, morfofonemik, morfem penunjuk arah (mayi, mao, mola, mota), "
    "dan kaidah leksikografi Prof. Dr. H. Mansoer Pateda."
)


def split_paragraph_chunks(text: str, sentences_per_chunk: int = 1) -> list[list[str]]:
    """Pecah teks menjadi paragraf dan kelompok kalimat berukuran `sentences_per_chunk`."""
    paragraphs = [p.strip() for p in re.split(r"\n+", text.strip()) if p.strip()]
    chunk_step = max(1, int(sentences_per_chunk))
    result: list[list[str]] = []
    for para in paragraphs:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        chunks = [
            " ".join(sentences[i : i + chunk_step])
            for i in range(0, len(sentences), chunk_step)
        ]
        result.append(chunks)
    return result


def restore_chunk_punctuation(src_chunk: str, tgt_chunk: str, force_period: bool) -> str:
    """Pertahankan tanda baca akhir kalimat saat menggabungkan potongan terjemahan."""
    out = tgt_chunk.strip()
    if not out:
        return ""
    if out.endswith((".", "!", "?")):
        return out
    src_end = src_chunk.strip()[-1:] if src_chunk.strip() else ""
    if src_end in ("?", "!"):
        return f"{out}{src_end}"
    if force_period:
        return f"{out}{src_end if src_end == '.' else '.'}"
    return out


def translate_chunked(
    chunk_fn: Callable[[str], str],
    text: str,
    auto_split: bool = True,
    sentences_per_chunk: int = 1,
) -> str:
    """Jalankan penerjemahan per paragraf/kalimat dengan restorasi tanda baca."""
    cleaned = text.strip()
    if not cleaned:
        return ""
    if not auto_split:
        return chunk_fn(cleaned)

    para_groups = split_paragraph_chunks(cleaned, sentences_per_chunk=sentences_per_chunk)
    translated_paragraphs: list[str] = []
    for chunks in para_groups:
        multi = len(chunks) > 1
        parts: list[str] = []
        for chunk in chunks:
            raw_out = chunk_fn(chunk)
            parts.append(restore_chunk_punctuation(chunk, raw_out, force_period=multi))
        translated_paragraphs.append(" ".join(p for p in parts if p))
    return "\n\n".join(translated_paragraphs)


def _default_instruction(direction: str, custom_instruction: str = "") -> str:
    if custom_instruction:
        return custom_instruction
    if direction == "id2gor":
        return "Terjemahkan kalimat bahasa Indonesia berikut ke dalam bahasa Gorontalo yang baik dan benar sesuai kaidah Prof. Mansoer Pateda."
    return "Terjemahkan kalimat bahasa Gorontalo berikut ke dalam bahasa Indonesia yang baik dan benar."


def _load_lora_model(base_id: str, lora_src: str, load_in_4bit: bool = False):
    """Muat base model dan adapter LoRA secara adaptif (BF16/FP16 atau QLoRA 4-bit)."""
    dtype = _resolve_dtype()
    kwargs = {"dtype": dtype}
    if load_in_4bit:
        if not torch.cuda.is_available():
            raise RuntimeError("Mode --load-in-4bit membutuhkan GPU CUDA.")
        kwargs["quantization_config"] = _build_4bit_config(dtype)
        kwargs["device_map"] = "auto"
    else:
        kwargs["device_map"] = "auto" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(lora_src)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
    model = PeftModel.from_pretrained(base_model, lora_src)
    model.eval()
    return tokenizer, model


def build_translator(
    model_name: str,
    direction: str = "id2gor",
    custom_instruction: str = "",
    max_new_tokens: int = 192,
    num_beams: int = 4,
    auto_split: bool = True,
    sentences_per_chunk: int = 0,
    load_in_4bit: bool = False,
) -> Callable[[str], str]:
    """Muat model pilihan sekali lalu kembalikan fungsi penerjemah siap pakai."""
    resolved_chunk = (
        sentences_per_chunk
        if sentences_per_chunk > 0
        else (2 if (model_name == "gemma3" and direction == "gor2id") else 1)
    )
    instruction = _default_instruction(direction, custom_instruction)
    use_split = auto_split and not bool(custom_instruction)

    if model_name == "nllb":
        nllb_src = _resolve_model_source(NLLB_DIR, NLLB_HF_REPO, "NLLB_MODEL_ID")
        tokenizer = AutoTokenizer.from_pretrained(nllb_src)
        model = AutoModelForSeq2SeqLM.from_pretrained(nllb_src, dtype=_resolve_dtype()).to(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        src_lang = "ind_Latn" if direction == "id2gor" else "gor_Latn"
        tgt_lang = "gor_Latn" if direction == "id2gor" else "ind_Latn"
        src_id = tokenizer.convert_tokens_to_ids(src_lang)
        tgt_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        eos_id = tokenizer.eos_token_id

        @torch.inference_mode()
        def run_chunk(chunk: str) -> str:
            sub_ids = tokenizer(chunk, add_special_tokens=False, truncation=True, max_length=250)["input_ids"]
            inp_ids = torch.tensor([[src_id] + sub_ids + [eos_id]], dtype=torch.long, device=model.device)
            gen_ids = model.generate(
                input_ids=inp_ids,
                forced_bos_token_id=tgt_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                no_repeat_ngram_size=4,
                early_stopping=True,
            )
            return tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        return lambda txt: translate_chunked(run_chunk, txt, auto_split=auto_split, sentences_per_chunk=resolved_chunk)

    if model_name == "qwen":
        qwen_src = _resolve_model_source(QWEN_LORA_DIR, QWEN_HF_REPO, "QWEN_LORA_ID")
        tokenizer, model = _load_lora_model(QWEN_BASE, qwen_src, load_in_4bit=load_in_4bit)

        @torch.inference_mode()
        def run_chunk(chunk: str) -> str:
            prompt = (
                f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
                f"<|im_start|>user\n{instruction}\n\nInput: {chunk}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
            enc = tokenizer(prompt, return_tensors="pt").to(model.device)
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            out_ids = gen_ids[0, enc["input_ids"].shape[1] :]
            return tokenizer.decode(out_ids, skip_special_tokens=True).strip()

        return lambda txt: translate_chunked(run_chunk, txt, auto_split=use_split, sentences_per_chunk=resolved_chunk)

    if model_name == "gemma3":
        gemma_src = _resolve_model_source(GEMMA3_LORA_DIR, GEMMA3_HF_REPO, "GEMMA3_LORA_ID")
        tokenizer, model = _load_lora_model(GEMMA3_BASE, gemma_src, load_in_4bit=load_in_4bit)
        eos_ids = [
            i
            for i in [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<end_of_turn>")]
            if isinstance(i, int) and i >= 0
        ]

        @torch.inference_mode()
        def run_chunk(chunk: str) -> str:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"{instruction}\n\nInput: {chunk}"},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to(model.device)
            gen_ids = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                top_k=None,
                top_p=None,
                no_repeat_ngram_size=4,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=eos_ids or tokenizer.eos_token_id,
            )
            out_ids = gen_ids[0, enc["input_ids"].shape[1] :]
            return tokenizer.decode(out_ids, skip_special_tokens=True).strip()

        return lambda txt: translate_chunked(run_chunk, txt, auto_split=use_split, sentences_per_chunk=resolved_chunk)

    raise ValueError(f"Model tidak dikenal: {model_name}")


def translate_nllb(
    text: str,
    direction: str = "id2gor",
    max_new_tokens: int = 192,
    num_beams: int = 4,
    auto_split: bool = True,
    sentences_per_chunk: int = 1,
) -> str:
    """Terjemahkan teks dengan NLLB-200-600M sambil memecah teks panjang per kalimat."""
    return build_translator(
        "nllb",
        direction=direction,
        max_new_tokens=max_new_tokens,
        num_beams=num_beams,
        auto_split=auto_split,
        sentences_per_chunk=sentences_per_chunk,
    )(text)


def translate_qwen(
    text: str,
    direction: str = "id2gor",
    custom_instruction: str = "",
    max_new_tokens: int = 192,
    auto_split: bool = True,
    sentences_per_chunk: int = 1,
    load_in_4bit: bool = False,
) -> str:
    """Terjemahkan teks dengan Qwen2.5-7B-Instruct LoRA dan guard pengulangan."""
    return build_translator(
        "qwen",
        direction=direction,
        custom_instruction=custom_instruction,
        max_new_tokens=max_new_tokens,
        auto_split=auto_split,
        sentences_per_chunk=sentences_per_chunk,
        load_in_4bit=load_in_4bit,
    )(text)


def translate_gemma3(
    text: str,
    direction: str = "id2gor",
    custom_instruction: str = "",
    max_new_tokens: int = 192,
    auto_split: bool = True,
    sentences_per_chunk: int = 0,
    load_in_4bit: bool = False,
) -> str:
    """Terjemahkan teks dengan GoTo Gemma 3 27B LoRA dan guard pengulangan."""
    return build_translator(
        "gemma3",
        direction=direction,
        custom_instruction=custom_instruction,
        max_new_tokens=max_new_tokens,
        auto_split=auto_split,
        sentences_per_chunk=sentences_per_chunk,
        load_in_4bit=load_in_4bit,
    )(text)


def main():
    parser = argparse.ArgumentParser(description="Inferensi Model Penerjemah Bahasa Indonesia <-> Gorontalo")
    parser.add_argument("--model", choices=["nllb", "qwen", "gemma3"], default="nllb")
    parser.add_argument("--direction", choices=["id2gor", "gor2id"], default="id2gor")
    parser.add_argument("--text", type=str, default="", help="Teks masukan untuk diterjemahkan.")
    parser.add_argument("--instruction", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument(
        "--chunk-sentences",
        type=int,
        default=0,
        help="Jumlah kalimat per potongan (0 = otomatis: 2 untuk gemma3 gor2id, 1 untuk lainnya).",
    )
    parser.add_argument(
        "--no-split",
        action="store_true",
        help="Nonaktifkan pemecahan paragraf/kalimat otomatis.",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Aktifkan kuantisasi NF4 4-bit (bitsandbytes) untuk model qwen/gemma3 pada GPU 15–24 GB.",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Jalankan mode interaktif di terminal agar model hanya dimuat satu kali.",
    )
    args = parser.parse_args()

    if not args.interactive and not args.text.strip():
        parser.error("Berikan --text \"...\" atau aktifkan --interactive.")

    translate_fn = build_translator(
        args.model,
        direction=args.direction,
        custom_instruction=args.instruction,
        max_new_tokens=args.max_new_tokens,
        auto_split=not args.no_split,
        sentences_per_chunk=args.chunk_sentences,
        load_in_4bit=args.load_in_4bit,
    )

    if args.text.strip():
        res = translate_fn(args.text)
        print(f"[{args.model.upper()} | {args.direction}] {args.text} -> {res}")

    if args.interactive:
        print(f"Mode interaktif aktif ({args.model.upper()} | {args.direction}). Ketik 'exit' untuk keluar.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not line or line.lower() in {"exit", "quit"}:
                break
            print(translate_fn(line))


if __name__ == "__main__":
    main()

