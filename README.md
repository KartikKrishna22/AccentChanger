# Accent Changer

CPU-only Streamlit app that converts speech to a selected accent using pretrained models:
- Faster-Whisper (`base`) for transcription
- Coqui XTTS v2 for synthesis

## 1) Project Layout

```text
.
|-- streamlit_app.py
|-- ml_pipeline.py
|-- requirements.txt
|-- .streamlit/config.toml
`-- data/reference_voices/
    |-- american.wav
    |-- british.wav
    |-- indian.wav
    `-- australian.wav
```

Notes:
- No backend service is required.
- The app runs fully inside Streamlit.

## 2) Prerequisites

- Python 3.11 (recommended)
- Windows, macOS, or Linux

## 3) Local Setup

From project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

## 4) Required Reference Voices

Place these files in [data/reference_voices](data/reference_voices):
- `american.wav`
- `british.wav`
- `indian.wav`
- `australian.wav`

Recommended quality:
- 6-10 seconds each
- Clear single-speaker speech
- English voice clips

## 5) Run Locally

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run streamlit_app.py
```

## 6) Deploy on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. In Streamlit Community Cloud, click New app.
3. Select your repository and branch.
4. Set Main file path to `streamlit_app.py`.
5. Use dependencies from `requirements.txt`.
6. Keep `runtime.txt` in the repo to force Python 3.11.
7. Deploy.

Important:
- Commit [data/reference_voices](data/reference_voices) files so they are available in cloud runtime.
- Keep upload length under 15 seconds.
- If logs show Python 3.14, verify `runtime.txt` and `.python-version` are present at repo root, then reboot the app.

## 7) How to Use

1. Upload input audio (WAV/MP3/M4A/OGG/FLAC).
2. Select target accent (American/British/Indian/Australian).
3. Select voice gender.
4. Click Convert.
5. Play or download the output WAV.

## 8) Performance Notes

- CPU-only inference path is enforced.
- Models are loaded lazily and cached.
- First conversion is slower due to initial model loading.
- For Streamlit Cloud stability, defaults are tuned for lower memory:
  - `WHISPER_MODEL_SIZE=tiny`
  - `CACHE_XTTS_MODEL=0` (XTTS is unloaded after each conversion)
- If you want faster repeat conversions and have enough memory, set `CACHE_XTTS_MODEL=1`.
- Streamlit Cloud safety default: app uses a lightweight fallback TTS model.
  This prevents frequent memory-related crashes caused by full XTTS on free-tier containers.
- To enable full XTTS on Cloud, set `ENABLE_XTTS_IN_CLOUD=1` (may still OOM on small instances).
- Optional fallback controls:
  - `LIGHT_TTS_MODEL_NAME` (default: `tts_models/en/ljspeech/tacotron2-DDC`)
  - `CACHE_LIGHT_TTS_MODEL` (default: `1`)

## 9) Troubleshooting

- Missing reference voice error:
  - Verify all four base accent files exist in [data/reference_voices](data/reference_voices).

- Audio too long error:
  - Keep input to 15 seconds or less.

- Import/module errors locally:
  - Recreate virtual environment and reinstall:

```powershell
Remove-Item -Recurse -Force .venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

- Streamlit Cloud dependency install fails with Python 3.14:
  - Ensure the deployed branch contains both `runtime.txt` and `.python-version` with `3.11`.
  - In Streamlit Cloud: Manage app -> Settings -> Advanced -> Python version -> 3.11.
  - Clear cache and reboot app.
