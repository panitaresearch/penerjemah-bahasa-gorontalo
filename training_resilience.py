#!/usr/bin/env python3
"""
training_resilience.py
======================
Utilitas mitigasi crash / sesi putus bersama untuk seluruh skrip pelatihan GPU
(`train_nllb_gorontalo.py`, `train_qwen_lora_gorontalo.py`, `train_gemma3_27b_lora_gorontalo.py`):
  1. Validasi integritas isi checkpoint (`find_latest_valid_checkpoint`) agar sesi resume
     tidak pernah crash akibat folder `checkpoint-*` yang baru setengah tertulis saat putus.
  2. Callback progres atomik + penangkapan sinyal `SIGTERM`/`SIGUSR1` (`ProgressAndGracefulStopCallback`)
     sehingga `Trainer` menyelesaikan langkah berjalan, menulis checkpoint aman, lalu berhenti bersih.
  3. Penanda selesai pelatihan (`training_completed.json`) agar jika crash terjadi pada fase evaluasi
     test set, proses lanjutan langsung melewati training ulang dan melanjutkan evaluasi.
  4. Cache evaluasi parsial per arah/tugas (`PartialEvalStore`) agar prediksi test set yang sudah
     selesai di-generate tidak terbuang saat sesi terputus.
"""

from __future__ import annotations

import json
import os
import signal
import time
from pathlib import Path
from typing import Any

from transformers import TrainerCallback


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def _has_nonempty_weights(ckpt_dir: Path, is_lora: bool) -> bool:
    if is_lora:
        candidates = [
            ckpt_dir / "adapter_model.safetensors",
            ckpt_dir / "adapter_model.bin",
        ]
        if any(p.is_file() and p.stat().st_size > 0 for p in candidates):
            return True
    candidates = [
        ckpt_dir / "model.safetensors",
        ckpt_dir / "pytorch_model.bin",
    ]
    if any(p.is_file() and p.stat().st_size > 0 for p in candidates):
        return True
    shard_files = list(ckpt_dir.glob("model-*.safetensors")) + list(ckpt_dir.glob("pytorch_model-*.bin"))
    return bool(shard_files and all(p.is_file() and p.stat().st_size > 0 for p in shard_files))


def is_checkpoint_complete(ckpt_dir: Path, is_lora: bool) -> bool:
    """Pastikan folder checkpoint berisi state JSON, bobot, dan state optimizer yang utuh."""
    if not ckpt_dir.is_dir():
        return False

    state_file = ckpt_dir / "trainer_state.json"
    if not state_file.is_file() or state_file.stat().st_size == 0:
        return False
    try:
        json.loads(state_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False

    if not _has_nonempty_weights(ckpt_dir, is_lora=is_lora):
        return False

    opt_ok = (ckpt_dir / "optimizer.pt").is_file() and (ckpt_dir / "optimizer.pt").stat().st_size > 0
    if not opt_ok:
        opt_ok = (ckpt_dir / "optimizer.bin").is_file() and (ckpt_dir / "optimizer.bin").stat().st_size > 0
    sched_ok = (ckpt_dir / "scheduler.pt").is_file() and (ckpt_dir / "scheduler.pt").stat().st_size > 0
    return opt_ok and sched_ok


def find_latest_valid_checkpoint(ckpt_dir: Path, is_lora: bool) -> str | None:
    """Pilih checkpoint terbaru yang lolos verifikasi kelengkapan berkas untuk auto-resume aman."""
    if not ckpt_dir.exists():
        return None
    candidates: list[tuple[int, Path]] = []
    for p in ckpt_dir.glob("checkpoint-*"):
        if not p.is_dir():
            continue
        try:
            step = int(p.name.split("-")[-1])
        except ValueError:
            continue
        candidates.append((step, p))

    candidates.sort(key=lambda x: x[0], reverse=True)
    for step, p in candidates:
        if is_checkpoint_complete(p, is_lora=is_lora):
            return str(p)
        print(f"[PERINGATAN] Mengabaikan checkpoint tidak lengkap/korup: {p} (step={step})")
    return None


def is_training_already_completed(final_dir: Path, is_lora: bool) -> bool:
    """Cek apakah model final sudah selesai dilatih & tersimpan utuh di Google Drive."""
    marker = final_dir / "training_completed.json"
    if not marker.is_file() or marker.stat().st_size == 0:
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not data.get("completed"):
        return False
    return _has_nonempty_weights(final_dir, is_lora=is_lora)


def mark_training_completed(final_dir: Path, metadata: dict[str, Any]) -> None:
    payload = {
        "completed": True,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **metadata,
    }
    _atomic_write_json(final_dir / "training_completed.json", payload)


class PartialEvalStore:
    """Menyimpan hasil evaluasi test set secara bertahap per arah agar tahan putus saat generasi."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {}
        if self.path.is_file():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data = loaded
            except (OSError, json.JSONDecodeError):
                self.data = {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def put(self, key: str, value: Any) -> None:
        self.data[key] = value
        _atomic_write_json(self.path, self.data)

    def clear(self) -> None:
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                pass


class ProgressAndGracefulStopCallback(TrainerCallback):
    """Mencatat progres training secara atomik ke Drive dan memicu save aman bila menerima SIGTERM/SIGUSR1."""

    def __init__(self, progress_path: Path, start_time: float):
        self.progress_path = progress_path
        self.start_time = start_time
        self.stop_requested = False
        self.stop_signal: int | None = None
        self._prev_handlers: dict[int, Any] = {}
        self.progress_path.parent.mkdir(parents=True, exist_ok=True)
        self._install_signal_handlers()

    def _install_signal_handlers(self) -> None:
        for sig_name in ("SIGTERM", "SIGUSR1"):
            sig = getattr(signal, sig_name, None)
            if sig is None:
                continue
            try:
                prev = signal.getsignal(sig)

                def _handler(signum, _frame):
                    self.stop_requested = True
                    self.stop_signal = signum
                    print(f"\n[SINYAL {signum}] Permintaan henti terdeteksi; menyimpan checkpoint aman setelah langkah aktif...")

                self._prev_handlers[sig] = prev
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                # Tidak berada di main thread; lewati registrasi sinyal tanpa gagal.
                pass

    def restore_signal_handlers(self) -> None:
        for sig, prev in self._prev_handlers.items():
            try:
                signal.signal(sig, prev)
            except (ValueError, OSError):
                pass
        self._prev_handlers.clear()

    def _write_state(self, event: str, state, logs=None):
        elapsed_min = round((time.time() - self.start_time) / 60.0, 2)
        payload = {
            "event": event,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "global_step": int(state.global_step),
            "max_steps": int(state.max_steps),
            "progress_pct": round(100.0 * state.global_step / max(1, state.max_steps), 2),
            "epoch": round(float(state.epoch or 0.0), 4),
            "best_metric": getattr(state, "best_metric", None),
            "best_model_checkpoint": getattr(state, "best_model_checkpoint", None),
            "elapsed_minutes_this_run": elapsed_min,
            "stop_requested": self.stop_requested,
            "last_logs": logs or {},
        }
        _atomic_write_json(self.progress_path, payload)

    def on_train_begin(self, args, state, control, **kwargs):
        self._write_state("train_begin", state)

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_requested:
            control.should_save = True
            control.should_training_stop = True

    def on_log(self, args, state, control, logs=None, **kwargs):
        self._write_state("log", state, logs=logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._write_state("evaluate", state, logs=metrics)

    def on_save(self, args, state, control, **kwargs):
        self._write_state("save", state)
