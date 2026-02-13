import speech_recognition as sr
from pydub import AudioSegment
import io
import os

# -------- SETTINGS ----------
INPUT_FILE = "input.wav"   # change if needed
LANGUAGE = "en-US"         # "bn-BD" for Bangla
# ----------------------------

def load_audio(file_path):
    """
    Loads audio file and ensures it is WAV format.
    If already WAV, just read it.
    """
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".wav":
        return file_path

    # Convert to wav if other format
    print("Converting audio to wav...")
    sound = AudioSegment.from_file(file_path)
    wav_path = "converted_audio.wav"
    sound.export(wav_path, format="wav")
    return wav_path


def recognize_speech(audio_path):
    recognizer = sr.Recognizer()

    with sr.AudioFile(audio_path) as source:
        print("Reading audio...")
        audio_data = recognizer.record(source)

    try:
        print("Recognizing speech...")
        text = recognizer.recognize_google(audio_data, language=LANGUAGE)
        return text

    except sr.UnknownValueError:
        return "Speech could not be understood."

    except sr.RequestError:
        return "API unavailable or internet problem."


def main():
    print("Speech Recognition System Started\n")

    wav_file = load_audio(INPUT_FILE)
    result = recognize_speech(wav_file)

    print("\nRecognized Text:")
    print("----------------")
    print(result)


if __name__ == "__main__":
    main()
