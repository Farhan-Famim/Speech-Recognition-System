import argparse
from pathlib import Path

from .transcriber import DEFAULT_MODEL, transcribe_file


def build_parser():
    parser = argparse.ArgumentParser(description="Transcribe one audio file with Whisper.")
    parser.add_argument("--input", required=True, help="Path to the audio file.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Whisper model size to load.")
    return parser


def main():
    args = build_parser().parse_args()
    audio_path = Path(args.input)

    if not audio_path.exists():
        print("Audio file not found.")
        return

    print("Whisper Speech Recognition System\n")
    print("Transcribing audio...\n")
    print("Recognized Text:")
    print("----------------")
    print(transcribe_file(audio_path, model_name=args.model))


if __name__ == "__main__":
    main()
