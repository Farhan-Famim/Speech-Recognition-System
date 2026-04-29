from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .feature_extraction import extract_features
from .transcriber import DEFAULT_MANIFEST, load_manifest_rows

DEFAULT_OUTPUT_DIR = Path("artifacts/speaker_model")
DEFAULT_MODEL_NAME = "speaker_model.pkl"


def build_parser():
    parser = argparse.ArgumentParser(description="Train an SVM speaker-recognition model from the project manifest.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="CSV manifest with a speaker_id column.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Where to save the speaker model.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME, help="Filename for the saved joblib model.")
    parser.add_argument(
        "--train-splits",
        default="train,val",
        help="Comma-separated splits used for training. Ignored when --train-all is set.",
    )
    parser.add_argument("--eval-split", default="test", help="Split used for evaluation after training.")
    parser.add_argument("--train-all", action="store_true", help="Train on every manifest row and skip evaluation.")
    return parser


def split_names(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def resolve_audio_path(row: dict[str, str], base_dir: Path) -> Path:
    audio_path = Path(row["audio_path"])
    if not audio_path.is_absolute():
        audio_path = base_dir / audio_path
    return audio_path


def build_dataset(rows: list[dict[str, str]], base_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    features_list = []
    labels = []

    for row in rows:
        audio_path = resolve_audio_path(row, base_dir)
        if not audio_path.exists():
            print(f"Skipping missing file: {audio_path}")
            continue

        features = extract_features(audio_path)
        if features is None:
            continue

        features_list.append(features)
        labels.append(row["speaker_id"])

    if not features_list:
        raise ValueError("No speaker features could be extracted.")

    return np.array(features_list, dtype=np.float32), np.array(labels)


def main():
    args = build_parser().parse_args()

    try:
        import joblib
        from sklearn.metrics import accuracy_score
        from sklearn.svm import SVC
    except ImportError as exc:
        raise RuntimeError(
            "Speaker training needs librosa, scikit-learn, and joblib. "
            "Install dependencies with: pip install -r requirements.txt"
        ) from exc

    base_dir = Path(__file__).resolve().parent.parent
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path

    rows = load_manifest_rows(manifest_path)
    if args.train_all:
        train_rows = rows
        eval_rows = []
    else:
        train_splits = split_names(args.train_splits)
        train_rows = [row for row in rows if row["split"] in train_splits]
        eval_rows = [row for row in rows if row["split"] == args.eval_split]

    if not train_rows:
        raise ValueError("No training rows found for the requested split selection.")

    print("Extracting speaker features...")
    X_train, y_train = build_dataset(train_rows, base_dir)
    speakers = sorted(set(y_train))
    if len(speakers) < 2:
        raise ValueError("Speaker training needs at least two different speaker_id values.")

    model = SVC(kernel="linear", probability=True, class_weight="balanced")
    model.fit(X_train, y_train)

    output_dir = base_dir / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / args.model_name

    bundle = {
        "model": model,
        "speakers": speakers,
        "feature_version": "mfcc_mean_std_delta_39",
        "manifest": str(manifest_path.relative_to(base_dir) if manifest_path.is_relative_to(base_dir) else manifest_path),
    }
    joblib.dump(bundle, model_path)

    print(f"\nSpeaker model saved to: {model_path}")
    print(f"Training samples: {len(y_train)}")
    print(f"Speakers: {', '.join(speakers)}")

    if eval_rows:
        X_eval, y_eval = build_dataset(eval_rows, base_dir)
        y_pred = model.predict(X_eval)
        print(f"Evaluation split: {args.eval_split}")
        print(f"Evaluation samples: {len(y_eval)}")
        print(f"Speaker accuracy: {accuracy_score(y_eval, y_pred):.4f}")
    else:
        print("No held-out speaker evaluation was calculated because --train-all used every row.")


if __name__ == "__main__":
    main()
