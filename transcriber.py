from __future__ import annotations

import csv
import re
import ssl
import threading
import warnings
import wave
from pathlib import Path

import numpy as np
import whisper
from jiwer import wer

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = "base"
DEFAULT_MANIFEST = BASE_DIR / "custom_dataset" / "metadata.csv"

_model_lock = threading.Lock()
_model_cache: dict[str, whisper.Whisper] = {}


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_manifest_rows(manifest_path: Path | str = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {path}")

    with open(path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        required = {"audio_path", "transcript", "split", "speaker_id", "language", "command_id"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_text = ", ".join(sorted(missing))
            raise ValueError(f"Manifest is missing required columns: {missing_text}")

        rows: list[dict[str, str]] = []
        for row in reader:
            if not row.get("audio_path"):
                continue
            rows.append(row)
        return rows


def get_model(model_name: str = DEFAULT_MODEL) -> whisper.Whisper:
    with _model_lock:
        if model_name not in _model_cache:
            try:
                _model_cache[model_name] = whisper.load_model(model_name)
            except ssl.SSLCertVerificationError as exc:
                raise RuntimeError(
                    "Whisper could not download the model because SSL certificate "
                    "verification failed on this machine."
                ) from exc
        return _model_cache[model_name]


def load_wav_audio(audio_path: Path | str, target_rate: int = 16000) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV files are supported: {audio_path}")

        audio_bytes = wav_file.readframes(frame_count)

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if sample_rate != target_rate:
        duration = len(audio) / sample_rate
        target_length = int(duration * target_rate)
        source_positions = np.linspace(0, duration, num=len(audio), endpoint=False)
        target_positions = np.linspace(0, duration, num=target_length, endpoint=False)
        audio = np.interp(target_positions, source_positions, audio).astype(np.float32)

    return audio


def transcribe_file(audio_path: Path | str, model_name: str = DEFAULT_MODEL) -> str:
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {path}")

    model = get_model(model_name)

    if path.suffix.lower() == ".wav":
        audio_input = load_wav_audio(path)
    else:
        audio_input = str(path)

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="FP16 is not supported on CPU; using FP32 instead",
        )
        result = model.transcribe(audio_input, fp16=False)

    return result["text"].strip()


def evaluate_manifest(
    manifest_path: Path | str = DEFAULT_MANIFEST,
    split: str | None = "test",
    model_name: str = DEFAULT_MODEL,
) -> list[dict[str, str | float]]:
    rows = load_manifest_rows(manifest_path)
    results: list[dict[str, str | float]] = []

    for row in rows:
        if split and row["split"] != split:
            continue

        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = BASE_DIR / audio_path

        if not audio_path.exists():
            results.append({"file": row["audio_path"], "error": f"File not found: {audio_path}"})
            continue

        predicted = transcribe_file(audio_path, model_name=model_name)
        ground_truth = row["transcript"]
        gt_clean = normalize_text(ground_truth)
        pred_clean = normalize_text(predicted)

        results.append(
            {
                "file": row["audio_path"],
                "ground_truth": ground_truth,
                "predicted": predicted,
                "wer": wer(gt_clean, pred_clean),
                "split": row["split"],
                "speaker_id": row["speaker_id"],
                "language": row["language"],
                "command_id": row["command_id"],
            }
        )

    return results


def allowed_audio_file(filename: str) -> bool:
    return Path(filename).suffix.lower() in {".wav", ".mp3", ".m4a", ".mp4", ".mpeg", ".mpga", ".webm"}
