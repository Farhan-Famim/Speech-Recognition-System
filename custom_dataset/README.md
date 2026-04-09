# Custom Dataset

This folder is for your own recordings and metadata.

## Expected structure

Place your WAV files anywhere under this project, for example:

- `custom_dataset/audio/english/speaker_01/light_on_01.wav`
- `custom_dataset/audio/bengali/speaker_01/light_on_01.wav`

## Metadata format

Edit `custom_dataset/metadata.csv` with one row per audio file.

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
