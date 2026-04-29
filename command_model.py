from __future__ import annotations

import csv
import json
import math
import random
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

SAMPLE_RATE = 16000
CLIP_SECONDS = 3.0
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
DEFAULT_CHECKPOINT = Path("artifacts/command_model/best_command_model.pt")
COMMAND_TEXT = {
    "light_on": "Turn on the light",
    "light_off": "Turn off the light",
    "door_open": "Open the door",
    "door_close": "Close the door",
    "music_play": "Play the music",
    "music_stop": "Stop the music",
    "volume_up": "Increase the volume",
    "volume_down": "Decrease the volume",
    "what_time": "What is the time",
    "call_friend": "Call my friend",
}

_predictor_lock = threading.Lock()
_predictor_cache: dict[str, tuple[nn.Module, dict[int, str], torch.device]] = {}


def read_wav_mono(audio_path: Path | str, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        source_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()

        if sample_width != 2:
            raise ValueError(f"Only 16-bit PCM WAV files are supported: {audio_path}")

        audio_bytes = wav_file.readframes(frame_count)

    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    if source_rate != sample_rate:
        duration = len(audio) / source_rate
        target_length = int(duration * sample_rate)
        src_positions = np.linspace(0, duration, num=len(audio), endpoint=False)
        dst_positions = np.linspace(0, duration, num=target_length, endpoint=False)
        audio = np.interp(dst_positions, src_positions, audio).astype(np.float32)

    return audio


def trim_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    if len(audio) == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-5:
        return audio

    threshold = max(0.005, peak * 0.08)
    voiced = np.flatnonzero(np.abs(audio) >= threshold)
    if len(voiced) == 0:
        return audio

    margin = int(sample_rate * 0.08)
    start = max(0, int(voiced[0]) - margin)
    end = min(len(audio), int(voiced[-1]) + margin)
    return audio[start:end]


def normalize_volume(audio: np.ndarray, target_rms: float = 0.08) -> np.ndarray:
    if len(audio) == 0:
        return audio

    rms = float(np.sqrt(np.mean(np.square(audio))))
    if rms < 1e-5:
        return audio

    gain = min(10.0, target_rms / rms)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def augment_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    gain = random.uniform(0.85, 1.15)
    audio = np.clip(audio * gain, -1.0, 1.0).astype(np.float32)

    max_shift = int(sample_rate * 0.12)
    shift = random.randint(-max_shift, max_shift)
    if shift > 0:
        audio = np.pad(audio, (shift, 0), mode="constant")[: len(audio)]
    elif shift < 0:
        audio = np.pad(audio[-shift:], (0, -shift), mode="constant")

    if random.random() < 0.4:
        noise_level = random.uniform(0.001, 0.004)
        noise = np.random.normal(0.0, noise_level, size=len(audio)).astype(np.float32)
        audio = np.clip(audio + noise, -1.0, 1.0).astype(np.float32)

    return audio


def fix_length(audio: np.ndarray, sample_rate: int = SAMPLE_RATE, clip_seconds: float = CLIP_SECONDS) -> np.ndarray:
    target = int(sample_rate * clip_seconds)
    if len(audio) < target:
        padded = np.zeros(target, dtype=np.float32)
        padded[: len(audio)] = audio
        return padded
    return audio[:target]


def prepare_audio(audio: np.ndarray, augment: bool = False) -> np.ndarray:
    audio = trim_silence(audio)
    audio = normalize_volume(audio)
    audio = fix_length(audio)
    if augment:
        audio = augment_audio(audio)
    return audio


def audio_to_feature(audio: np.ndarray) -> torch.Tensor:
    waveform = torch.from_numpy(audio)
    window = torch.hann_window(WIN_LENGTH)
    stft = torch.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window=window,
        return_complex=True,
    )
    spec = stft.abs()
    spec = torch.log1p(spec)
    spec = (spec - spec.mean()) / (spec.std() + 1e-6)
    return spec.unsqueeze(0)


def load_manifest(manifest_path: Path | str) -> list[dict[str, str]]:
    with open(manifest_path, "r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        rows = [row for row in reader if row.get("audio_path")]
    if not rows:
        raise ValueError("No rows were found in the manifest.")
    return rows


def build_label_mapping(rows: list[dict[str, str]]) -> dict[str, int]:
    labels = sorted({row["command_id"] for row in rows})
    return {label: index for index, label in enumerate(labels)}


def save_label_mapping(label_to_index: dict[str, int], output_path: Path | str) -> None:
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(label_to_index, file, indent=2, ensure_ascii=True)


def load_label_mapping(input_path: Path | str) -> dict[str, int]:
    with open(input_path, "r", encoding="utf-8") as file:
        return json.load(file)


@dataclass
class Metrics:
    loss: float
    accuracy: float


class CommandDataset(Dataset):
    def __init__(
        self,
        rows: list[dict[str, str]],
        label_to_index: dict[str, int],
        base_dir: Path,
        augment: bool = False,
    ) -> None:
        self.rows = rows
        self.label_to_index = label_to_index
        self.base_dir = base_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        audio_path = Path(row["audio_path"])
        if not audio_path.is_absolute():
            audio_path = self.base_dir / audio_path

        audio = read_wav_mono(audio_path)
        audio = prepare_audio(audio, augment=self.augment)
        feature = audio_to_feature(audio)
        label = torch.tensor(self.label_to_index[row["command_id"]], dtype=torch.long)
        return feature, label


class CommandCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs))


def split_rows(rows: list[dict[str, str]], split_name: str) -> list[dict[str, str]]:
    return [row for row in rows if row["split"] == split_name]


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def evaluate_model(model: nn.Module, loader, loss_fn, device: torch.device) -> Metrics:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = loss_fn(logits, labels)

            total_loss += loss.item() * labels.size(0)
            total_correct += int((logits.argmax(dim=1) == labels).sum().item())
            total_count += labels.size(0)

    if total_count == 0:
        return Metrics(loss=math.nan, accuracy=math.nan)

    return Metrics(loss=total_loss / total_count, accuracy=total_correct / total_count)


def load_predictor(checkpoint_path: Path | str = DEFAULT_CHECKPOINT) -> tuple[nn.Module, dict[int, str], torch.device]:
    checkpoint_path = Path(checkpoint_path)
    cache_key = str(checkpoint_path.resolve())

    with _predictor_lock:
        if cache_key in _predictor_cache:
            return _predictor_cache[cache_key]

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        label_to_index = checkpoint["label_to_index"]
        index_to_label = {value: key for key, value in label_to_index.items()}
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CommandCNN(num_classes=len(label_to_index)).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        bundle = (model, index_to_label, device)
        _predictor_cache[cache_key] = bundle
        return bundle


def predict_command(audio_path: Path | str, checkpoint_path: Path | str = DEFAULT_CHECKPOINT) -> dict[str, str | float]:
    audio_path = Path(audio_path)
    model, index_to_label, device = load_predictor(checkpoint_path)
    audio = read_wav_mono(audio_path)
    audio = prepare_audio(audio)
    feature = audio_to_feature(audio).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(feature)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = int(probabilities.argmax(dim=1).item())
        confidence = float(probabilities[0, predicted_index].item())

    command_id = index_to_label[predicted_index]
    return {
        "command_id": command_id,
        "command_text": COMMAND_TEXT.get(command_id, command_id),
        "confidence": confidence,
    }
