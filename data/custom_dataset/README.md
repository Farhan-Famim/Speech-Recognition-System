# Custom Dataset

This folder is for your own recordings and metadata.

## Expected structure

Place your WAV files under `data/custom_dataset/audio/`, for example:

- `data/custom_dataset/audio/Speaker1/speaker_01_phrase_01_take_01.wav`
- `data/custom_dataset/audio/Speaker2/speaker_02_phrase_01_take_01.wav`

## Metadata format

Edit `data/custom_dataset/metadata.csv` with one row per audio file.

Required columns:

- `audio_path`: relative path from the project root
- `transcript`: exact spoken text
- `split`: `train`, `val`, or `test`
- `speaker_id`: speaker name or ID
- `language`: for example `en` or `bn`
- `command_id`: normalized command label such as `light_on`

## Recommended split

Since each speaker says each phrase 3 times:

- repetition 1 and 2 -> `train`
- repetition 3 -> `test`

Move a small part of the training data into `val` for validation.

Example:

- first repetition -> `train`
- second repetition -> `val`
- third repetition -> `test`
