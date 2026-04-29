# Speech Recognition Project

This project now supports two tracks:

- Whisper transcription for uploaded audio and live speech
- a custom trainable command-recognition model built from your own recordings

## What exists right now

- `temp.py`: transcribe one audio file with Whisper
- `main.py`: evaluate Whisper against your own manifest file
- `app.py`: web UI for file upload and live microphone recording
- `transcriber.py`: shared backend service used by both CLI scripts and the web app
- `train_command_model.py`: train a speech-command classifier on your own dataset
- `evaluate_command_model.py`: evaluate the trained command model
- `command_model.py`: audio loading, feature extraction, dataset, and model code
- `custom_dataset/metadata.csv`: your own dataset manifest

## macOS / Linux run steps

From the project root:

```bash
cd speech_recognition_1
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python temp.py --input path/to/file.wav
python main.py --manifest custom_dataset/metadata.csv --split test
python train_command_model.py --manifest custom_dataset/metadata.csv --resplit-by-take --output-dir artifacts/command_model_resplit
python evaluate_command_model.py --manifest custom_dataset/metadata.csv
python app.py
```

Then open `http://127.0.0.1:5050`.

## Training and evaluation checkpoints

Use the resplit checkpoint for report accuracy:

```bash
python train_command_model.py \
  --manifest custom_dataset/metadata.csv \
  --output-dir artifacts/command_model_resplit \
  --resplit-by-take \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4

python evaluate_command_model.py --manifest custom_dataset/metadata.csv --split test
```

Use the train-all checkpoint for the final demo app:

```bash
python train_command_model.py \
  --manifest custom_dataset/metadata.csv \
  --output-dir artifacts/command_model \
  --train-all \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4
```

The train-all checkpoint uses every recording, so it is useful for the demo but should not be reported as held-out test accuracy.

## Notes

- The bundled `venv/` folder in this repository is a Windows virtual environment, so it should not be used on macOS.
- The web app runs on port `5050` to avoid the macOS port `5000` AirPlay conflict.
- Local `.wav` files can now run without `ffmpeg`.
- Live microphone recording is captured in the browser and encoded to WAV before upload, so it also avoids `ffmpeg` for the browser recording path.
- Whisper still needs to download a model such as `base` the first time you run it, unless that model is already cached.
- If you have a certificate issue while downloading the model, fix that first or pre-download the model on a trusted network.
- The custom command model expects `16-bit PCM .wav` files.

## About your own training data

For a fixed 10-command project, training a custom command classifier is a strong practical choice. It is smaller, faster, and more realistic than full end-to-end ASR fine-tuning on only a few hundred recordings.

Recommended workflow:

1. Record your own 10 phrases for multiple speakers.
2. Store them as WAV files.
3. Fill `custom_dataset/metadata.csv`.
4. Train the command model with `train_command_model.py`.
5. Evaluate it with `evaluate_command_model.py`.
6. Use Whisper and the command model together in your report:
   baseline transcription plus your own trained recognizer.

## Existing sample dataset

The old `dataset/` folder is no longer used by the main evaluation flow. You can keep it as legacy reference data or delete it manually once your own `custom_dataset` is ready.
