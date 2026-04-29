# Speech Recognition System

This project is a web-based speech recognition demo for CSE445. It supports:

- command recognition with a custom trained command model
- speaker recognition with an SVM voice model
- Whisper baseline transcription for comparison
- browser audio upload and live microphone recording

## Project Structure

```text
speech_recognition_1/
├── main.py                         # Main file to run the web app
├── README.md                       # Project explanation and run guide
├── requirements.txt                # Python dependencies
├── data/
│   └── custom_dataset/             # Dataset manifest and WAV recordings
├── support/                        # Supporting Python modules and training scripts
├── static/                         # Browser JavaScript and CSS
├── templates/                      # Flask HTML template
├── artifacts/                      # Trained model files used by the app
└── others/                         # PPTX, reports, and demo video go here
```

## Included Dataset

The dataset is stored in:

```text
data/custom_dataset/
```

Important files:

- `data/custom_dataset/metadata.csv`: dataset manifest
- `data/custom_dataset/audio/`: WAV recordings

The manifest columns are:

```text
audio_path, transcript, split, speaker_id, language, command_id
```

## Included Models

The app is ready to run after cloning because the trained model files are included:

- `artifacts/command_model/best_command_model.pt`
- `artifacts/command_model/labels.json`
- `artifacts/speaker_model/speaker_model.pkl`

The resplit checkpoint used for report-style held-out evaluation is also included:

- `artifacts/command_model_resplit/best_command_model.pt`
- `artifacts/command_model_resplit/labels.json`

## Setup

Use Python 3.10 or newer.

```bash
cd speech_recognition_1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
cd speech_recognition_1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run The Project

Start the web app:

```bash
python main.py
```

Then open:

```text
http://127.0.0.1:5050
```

The UI lets you choose between:

- `Trained Model`: predicts one of the 10 fixed commands and the speaker
- `Whisper Baseline`: transcribes the speech and predicts the speaker

## Train Or Retrain Models

Train the command model:

```bash
python -m support.train_command_model \
  --manifest data/custom_dataset/metadata.csv \
  --output-dir artifacts/command_model \
  --train-all \
  --epochs 60 \
  --batch-size 16 \
  --learning-rate 5e-4 \
  --weight-decay 1e-4
```

Train the speaker recognition model:

```bash
python -m support.train_svm --manifest data/custom_dataset/metadata.csv
```

Evaluate the command model on the test split:

```bash
python -m support.evaluate_command_model \
  --manifest data/custom_dataset/metadata.csv \
  --checkpoint artifacts/command_model_resplit/best_command_model.pt \
  --split test
```

Evaluate Whisper on the manifest:

```bash
python -m support.evaluate_whisper \
  --manifest data/custom_dataset/metadata.csv \
  --split test
```

## Notes

- The app runs on port `5050`.
- Live microphone recording is captured in the browser and sent as WAV.
- The trained command model supports WAV input.
- Optional: install `ffmpeg` for non-WAV Whisper uploads such as MP3, M4A, MP4, or WebM.
- Whisper may download the `base` model the first time it is used.
- Do not commit `.venv/`, `venv/`, `__pycache__/`, or `uploads/`.

## Other Deliverables

Put these files in the `others/` folder before final submission:

- final presentation PPTX
- final report PDF
- update presentation PPTX
- update report PDF
- one-minute demo video
