from __future__ import annotations

import tempfile
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from command_model import DEFAULT_CHECKPOINT, predict_command
from transcriber import DEFAULT_MODEL, allowed_audio_file, transcribe_file

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024
app.config["UPLOAD_FOLDER"] = BASE_DIR / "uploads"
app.config["UPLOAD_FOLDER"].mkdir(exist_ok=True)
app.config["CHECKPOINT_PATH"] = BASE_DIR / DEFAULT_CHECKPOINT


@app.get("/")
def index():
    return render_template(
        "index.html",
        checkpoint_path=str(app.config["CHECKPOINT_PATH"].relative_to(BASE_DIR)),
        default_model=DEFAULT_MODEL,
    )


def error_response(message: str, status_code: int = 400):
    return jsonify({"ok": False, "error": message}), status_code


def run_inference(audio_path: Path, mode: str, whisper_model: str = DEFAULT_MODEL):
    if mode == "trained":
        if audio_path.suffix.lower() != ".wav":
            raise ValueError("The trained command model only supports WAV input.")

        prediction = predict_command(audio_path, checkpoint_path=app.config["CHECKPOINT_PATH"])
        return {
            "mode": "trained",
            "model": "trained_command_model",
            "command_id": prediction["command_id"],
            "command_text": prediction["command_text"],
            "confidence": prediction["confidence"],
        }

    if mode == "whisper":
        transcript = transcribe_file(audio_path, model_name=whisper_model)
        return {
            "mode": "whisper",
            "model": whisper_model,
            "transcript": transcript,
        }

    raise ValueError("Unsupported mode. Choose 'trained' or 'whisper'.")


@app.post("/api/predict-file")
def predict_uploaded_file():
    audio_file = request.files.get("audio")
    mode = request.form.get("mode", "trained").strip().lower()
    whisper_model = request.form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if audio_file is None or audio_file.filename == "":
        return error_response("Please choose an audio file first.")

    if not allowed_audio_file(audio_file.filename):
        return error_response("Unsupported file type. Upload WAV, MP3, M4A, MP4, or WebM audio.")

    safe_name = secure_filename(audio_file.filename)
    target_path = app.config["UPLOAD_FOLDER"] / safe_name
    audio_file.save(target_path)

    try:
        payload = run_inference(target_path, mode=mode, whisper_model=whisper_model)
        payload.update({"ok": True, "filename": audio_file.filename})
        return jsonify(payload)
    except Exception as exc:
        return error_response(str(exc), 500)
    finally:
        target_path.unlink(missing_ok=True)


@app.post("/api/predict-recording")
def predict_recording():
    audio_file = request.files.get("audio")
    mode = request.form.get("mode", "trained").strip().lower()
    whisper_model = request.form.get("model", DEFAULT_MODEL).strip() or DEFAULT_MODEL

    if audio_file is None:
        return error_response("No recording was received.")

    suffix = Path(audio_file.filename or "recording.wav").suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=app.config["UPLOAD_FOLDER"]) as temp_file:
        temp_path = Path(temp_file.name)

    try:
        audio_file.save(temp_path)
        payload = run_inference(temp_path, mode=mode, whisper_model=whisper_model)
        payload.update({"ok": True, "filename": audio_file.filename or "recording.wav"})
        return jsonify(payload)
    except Exception as exc:
        return error_response(str(exc), 500)
    finally:
        temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)
