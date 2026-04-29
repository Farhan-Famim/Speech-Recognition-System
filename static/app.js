const uploadForm = document.getElementById("upload-form");
const audioFileInput = document.getElementById("audio-file");
const selectedFile = document.getElementById("selected-file");
const resultText = document.getElementById("result-text");
const resultMeta = document.getElementById("result-meta");
const feedback = document.getElementById("feedback");
const startButton = document.getElementById("start-recording");
const stopButton = document.getElementById("stop-recording");
const recordStatus = document.getElementById("record-status");
const meterFill = document.getElementById("meter-fill");
const inferenceMode = document.getElementById("inference-mode");
const uploadHelp = document.getElementById("upload-help");
const uploadTitle = document.getElementById("upload-title");
const uploadSubmit = document.getElementById("upload-submit");
const recordHelp = document.getElementById("record-help");
const resultTitle = document.getElementById("result-title");
const modeStatus = document.getElementById("mode-status");

let audioContext = null;
let audioStream = null;
let processor = null;
let recordingFrames = [];
let recordingSampleRate = 44100;
let isRecording = false;
const defaultWhisperModel = "base";

audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files[0];
  selectedFile.textContent = file ? file.name : "No file selected";
});

inferenceMode.addEventListener("change", syncModeUi);
syncModeUi();

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const file = audioFileInput.files[0];
  if (!file) {
    showFeedback("Please choose a file before submitting.", "error");
    return;
  }

  const mode = inferenceMode.value;
  setBusyState(mode === "trained" ? "Uploading audio and predicting command..." : "Uploading audio and transcribing with Whisper...");

  const formData = new FormData();
  formData.append("audio", file);
  formData.append("mode", mode);
  formData.append("model", defaultWhisperModel);

  try {
    const response = await fetch("/api/predict-file", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    handleTranscriptResponse(payload, file.name);
  } catch (error) {
    showFeedback(error.message || "Upload failed.", "error");
  }
});

startButton.addEventListener("click", async () => {
  try {
    await startRecording();
  } catch (error) {
    showFeedback(error.message || "Unable to access microphone.", "error");
  }
});

stopButton.addEventListener("click", async () => {
  try {
    const wavBlob = await stopRecording();
    const mode = inferenceMode.value;
    setBusyState(mode === "trained" ? "Sending recording for command prediction..." : "Sending recording for Whisper transcription...");

    const formData = new FormData();
    formData.append("audio", wavBlob, "live-recording.wav");
    formData.append("mode", mode);
    formData.append("model", defaultWhisperModel);

    const response = await fetch("/api/predict-recording", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    handleTranscriptResponse(payload, "live-recording.wav");
  } catch (error) {
    showFeedback(error.message || "Recording failed.", "error");
  }
});

function setBusyState(message) {
  resultMeta.textContent = message;
  resultText.textContent = "Working...";
  feedback.hidden = true;
}

function handleTranscriptResponse(payload, fallbackName) {
  if (!payload.ok) {
    showFeedback(payload.error || "Prediction failed.", "error");
    return;
  }

  feedback.hidden = true;

  if (payload.mode === "trained") {
    const confidence = typeof payload.confidence === "number" ? `${(payload.confidence * 100).toFixed(1)}%` : "n/a";
    resultMeta.textContent = `${payload.filename || fallbackName} / ${payload.command_id} / confidence: ${confidence}`;
    resultText.textContent = `${payload.command_text}\n\nCommand ID: ${payload.command_id}`;
    showFeedback("Prediction completed.", "success");
  } else {
    resultMeta.textContent = `${payload.filename || fallbackName} / whisper:${payload.model}`;
    resultText.textContent = payload.transcript || "(No text returned)";
    showFeedback("Transcription completed.", "success");
  }
}

function showFeedback(message, type) {
  feedback.hidden = false;
  feedback.className = `feedback ${type}`;
  feedback.textContent = message;

  if (type === "error") {
    resultMeta.textContent = "Request failed";
    resultText.textContent = "No command prediction available.";
  }
}

async function startRecording() {
  if (isRecording) {
    return;
  }

  audioStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  audioContext = new AudioContext();
  recordingSampleRate = audioContext.sampleRate;
  const source = audioContext.createMediaStreamSource(audioStream);
  processor = audioContext.createScriptProcessor(4096, 1, 1);
  recordingFrames = [];

  processor.onaudioprocess = (event) => {
    if (!isRecording) {
      return;
    }

    const channelData = event.inputBuffer.getChannelData(0);
    recordingFrames.push(new Float32Array(channelData));

    let peak = 0;
    for (let i = 0; i < channelData.length; i += 1) {
      peak = Math.max(peak, Math.abs(channelData[i]));
    }
    meterFill.style.width = `${Math.max(8, Math.min(100, peak * 130))}%`;
  };

  source.connect(processor);
  processor.connect(audioContext.destination);

  isRecording = true;
  startButton.disabled = true;
  stopButton.disabled = false;
  recordStatus.textContent = "Recording...";
  resultMeta.textContent = "Listening";
  resultText.textContent = inferenceMode.value === "trained"
    ? "Speak one of your trained commands, then stop the recording."
    : "Speak into your microphone, then stop the recording for Whisper transcription.";
}

function syncModeUi() {
  const trainedMode = inferenceMode.value === "trained";

  uploadHelp.textContent = trainedMode
    ? "WAV only"
    : "Upload WAV, MP3, M4A, MP4, or WebM";
  uploadTitle.textContent = trainedMode ? "Choose WAV" : "Choose audio";
  audioFileInput.accept = trainedMode
    ? ".wav,audio/wav"
    : ".wav,.mp3,.m4a,.mp4,.mpeg,.mpga,.webm,audio/*";
  uploadSubmit.textContent = trainedMode ? "Predict Command" : "Transcribe with Whisper";
  recordHelp.textContent = trainedMode ? "Browser WAV" : "Browser audio";
  modeStatus.textContent = trainedMode ? "Trained model" : "Whisper";
  resultTitle.textContent = trainedMode ? "Predicted Command" : "Transcript";
  resultText.textContent = trainedMode
    ? "Your predicted command will appear here."
    : "Your transcript will appear here.";
}

async function stopRecording() {
  if (!isRecording) {
    throw new Error("Recording has not started.");
  }

  isRecording = false;
  startButton.disabled = false;
  stopButton.disabled = true;
  recordStatus.textContent = "Microphone idle";
  meterFill.style.width = "8%";

  processor.disconnect();
  audioStream.getTracks().forEach((track) => track.stop());
  await audioContext.close();

  const wavBlob = encodeWav(recordingFrames, recordingSampleRate);

  audioContext = null;
  audioStream = null;
  processor = null;
  recordingFrames = [];

  return wavBlob;
}

function encodeWav(frames, sampleRate) {
  const merged = mergeFrames(frames);
  const view = new DataView(new ArrayBuffer(44 + merged.length * 2));

  writeAscii(view, 0, "RIFF");
  view.setUint32(4, 36 + merged.length * 2, true);
  writeAscii(view, 8, "WAVE");
  writeAscii(view, 12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  writeAscii(view, 36, "data");
  view.setUint32(40, merged.length * 2, true);

  floatTo16BitPCM(view, 44, merged);
  return new Blob([view], { type: "audio/wav" });
}

function mergeFrames(frames) {
  let totalLength = 0;
  frames.forEach((frame) => {
    totalLength += frame.length;
  });

  const merged = new Float32Array(totalLength);
  let offset = 0;
  frames.forEach((frame) => {
    merged.set(frame, offset);
    offset += frame.length;
  });
  return merged;
}

function floatTo16BitPCM(view, offset, input) {
  for (let i = 0; i < input.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, input[i]));
    view.setInt16(offset + i * 2, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
  }
}

function writeAscii(view, offset, text) {
  for (let i = 0; i < text.length; i += 1) {
    view.setUint8(offset + i, text.charCodeAt(i));
  }
}
