import argparse

from .transcriber import DEFAULT_MANIFEST, DEFAULT_MODEL, evaluate_manifest


def print_results(rows):
    if not rows:
        print("No rows matched the requested split.")
        return

    for row in rows:
        print(f"FILE: {row['file']}")
        if "error" in row:
            print(f"ERROR: {row['error']}")
        else:
            print(f"CMD  : {row['command_id']}")
            print(f"LANG : {row['language']}")
            print(f"SPK  : {row['speaker_id']}")
            print(f"GT   : {row['ground_truth']}")
            print(f"PRED : {row['predicted']}")
            print(f"WER  : {row['wer']:.2f}")
        print("-" * 50)


def build_parser():
    parser = argparse.ArgumentParser(description="Evaluate Whisper on your custom manifest.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size to load.")
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="CSV manifest with audio_path, transcript, split, speaker_id, language, command_id.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate. Use empty string to evaluate every row.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    print("Whisper Evaluation System\n")
    split = args.split or None
    rows = evaluate_manifest(args.manifest, split=split, model_name=args.model)
    print_results(rows)


if __name__ == "__main__":
    main()
