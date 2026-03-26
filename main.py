
# VERSION 2.1
# - Normalization of texts in order to improve wer scores

import whisper
import os
import re
from jiwer import wer  # used to measure accuracy

# Load model once
model = whisper.load_model("base")

DATASET_PATH = "dataset"
TRANSCRIPT_FILE = "dataset/transcripts.txt"

TEST_FILES = [
    "speaker_3/audio_1.wav",
    "speaker_3/audio_2.wav",
    "speaker_3/audio_3.wav",
    "speaker_3/audio_4.wav",
    "speaker_3/audio_5.wav"
]


# 🔧 Normalize text (IMPORTANT for fair WER)
def normalize_text(text):
    text = text.lower()
    
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_transcripts(file_path):
    transcripts = {}
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            path, text = line.split("|", 1)
            transcripts[path.strip()] = text.strip()
    
    return transcripts

# Transcribe all the samples in the dataset
def transcribe_and_compare(transcripts):
    print("Running Whisper on dataset...\n")
    
    for audio_path, ground_truth in transcripts.items():
        full_path = os.path.join(DATASET_PATH, audio_path)
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
        
        result = model.transcribe(full_path)
        predicted = result["text"].strip()
        
        error = wer(ground_truth.lower(), predicted.lower())

        print(f"FILE: {audio_path}")
        print(f"GT   : {ground_truth}")
        print(f"PRED : {predicted}")
        print('\nScore: ')
        print(f"WER  : {error:.2f}")
        print("-" * 50)


def transcribe_selected(transcripts, selected_files):
    print("Running Whisper on selected files...\n")
    
    for audio_path in selected_files:
        if audio_path not in transcripts:
            print(f"Missing transcript for {audio_path}")
            continue
        
        full_path = os.path.join(DATASET_PATH, audio_path)
        
        if not os.path.exists(full_path):
            print(f"File not found: {full_path}")
            continue
        
        result = model.transcribe(full_path)
        predicted = result["text"].strip()
        ground_truth = transcripts[audio_path]
        
        # 🔥 Normalize before WER
        gt_clean = normalize_text(ground_truth)
        pred_clean = normalize_text(predicted)
        
        error = wer(gt_clean, pred_clean)
        
        print(f"FILE: {audio_path}")
        print(f"GT   : {ground_truth}")
        print(f"PRED : {predicted}")
        print(f"WER  : {error:.2f}")
        print("-" * 50)


def main():
    print("Whisper Evaluation System\n")
    
    transcripts = load_transcripts(TRANSCRIPT_FILE)
    transcribe_selected(transcripts, TEST_FILES)
    #transcribe_and_compare(transcripts)


if __name__ == "__main__":
    main()
