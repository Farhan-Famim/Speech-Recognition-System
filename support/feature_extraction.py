from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
N_MFCC = 13


def extract_features(file_path: Path | str) -> np.ndarray | None:
    """Return a 39-value MFCC feature vector for speaker recognition."""
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError(
            "Speaker recognition needs librosa. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        audio, sample_rate = librosa.load(str(file_path), sr=SAMPLE_RATE, mono=True)
        if audio.size == 0:
            raise ValueError("Audio file is empty.")

        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=N_MFCC)
        mfcc_mean = np.mean(mfccs.T, axis=0)
        mfcc_std = np.std(mfccs.T, axis=0)

        delta_mfccs = librosa.feature.delta(mfccs)
        delta_mean = np.mean(delta_mfccs.T, axis=0)

        return np.concatenate([mfcc_mean, mfcc_std, delta_mean]).astype(np.float32)
    except Exception as exc:
        print(f"Error processing {file_path}: {exc}")
        return None
