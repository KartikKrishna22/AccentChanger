"""Streamlit app entrypoint."""

import os
import wave
from pathlib import Path
from uuid import uuid4

import numpy as np
import streamlit as st

os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

IS_STREAMLIT_CLOUD = Path("/mount/src").exists() or os.getenv("STREAMLIT_SHARING_MODE") == "1"
ENABLE_XTTS_IN_CLOUD = os.getenv("ENABLE_XTTS_IN_CLOUD", "0") == "1"

try:
    import soundfile as sf
except ModuleNotFoundError:
    sf = None

try:
    import librosa
except ModuleNotFoundError:
    librosa = None

ACCENTS = ["American", "British", "Indian", "Australian"]
GENDERS = ["Female", "Male"]
MAX_AUDIO_SECONDS = 15.0
SUPPORTED_INPUT_TYPES = ["wav", "mp3", "m4a", "ogg", "flac"]

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_voices"

REFERENCE_FILE_MAP = {
    "American": REFERENCE_DIR / "american.wav",
    "British": REFERENCE_DIR / "british.wav",
    "Indian": REFERENCE_DIR / "indian.wav",
    "Australian": REFERENCE_DIR / "australian.wav",
}


def _audio_duration_seconds(file_path: Path) -> float:
    if sf is not None:
        info = sf.info(str(file_path))
        return float(info.frames) / float(info.samplerate)

    with wave.open(str(file_path), "rb") as wav_file:
        frames = wav_file.getnframes()
        sample_rate = wav_file.getframerate()
        if sample_rate <= 0:
            raise ValueError("Invalid WAV sample rate.")
        return float(frames) / float(sample_rate)


def _write_mono_wav_22050(target_path: Path, audio_samples: np.ndarray) -> None:
    """Write float32 audio as PCM16 WAV without requiring soundfile."""
    mono = np.asarray(audio_samples, dtype=np.float32).reshape(-1)
    if mono.size == 0:
        raise ValueError("Empty decoded audio stream.")
    pcm16 = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm16 * 32767.0).astype(np.int16)
    with wave.open(str(target_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(22050)
        wav_file.writeframes(pcm16.tobytes())


def _save_reference_uploads() -> None:
    st.subheader("Reference Voices")
    st.caption("Upload base WAV reference per accent. Base files are used as Female voices by default.")
    for accent, target_path in REFERENCE_FILE_MAP.items():
        if target_path.exists():
            st.caption(f"{accent}: ready")
            continue
        upload = st.file_uploader(f"{accent} reference WAV", type=["wav"], key=f"ref_{accent}")
        if upload is not None:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            target_path.write_bytes(upload.getvalue())
            st.success(f"Saved {target_path.name}")


def _all_references_ready() -> bool:
    return all(path.exists() for path in REFERENCE_FILE_MAP.values())


def _convert_locally(input_wav_bytes: bytes, accent: str, gender: str) -> bytes:
    from ml_pipeline import generate_speech, transcribe_audio

    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    upload_id = uuid4().hex
    input_path = UPLOAD_DIR / f"{upload_id}.wav"
    input_path.write_bytes(input_wav_bytes)

    duration = _audio_duration_seconds(input_path)
    if duration > MAX_AUDIO_SECONDS:
        input_path.unlink(missing_ok=True)
        raise ValueError(f"Audio too long. Max length is {int(MAX_AUDIO_SECONDS)} seconds.")

    text = transcribe_audio(input_path)
    output_path = OUTPUT_DIR / f"{upload_id}_{accent.lower()}.wav"
    generated_path = generate_speech(
        text=text,
        accent=accent,
        gender=gender,
        style=None,
        intensity=0.0,
        output_path=output_path,
    )

    return Path(generated_path).read_bytes()


def _convert_locally_from_any_audio(
    input_audio_bytes: bytes,
    suffix: str,
    accent: str,
    gender: str,
) -> bytes:
    """Accept common formats and normalize to WAV for local conversion."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    upload_id = uuid4().hex
    raw_path = UPLOAD_DIR / f"{upload_id}{suffix}"
    wav_path = UPLOAD_DIR / f"{upload_id}.wav"
    raw_path.write_bytes(input_audio_bytes)

    try:
        if suffix == ".wav":
            wav_path = raw_path
        else:
            if librosa is None:
                raise ValueError("Non-WAV input requires librosa. Install dependencies or upload WAV.")
            audio_samples, _ = librosa.load(str(raw_path), sr=22050, mono=True)
            if audio_samples.size == 0:
                raise ValueError("Empty decoded audio stream.")
            if sf is not None:
                sf.write(str(wav_path), audio_samples, 22050, subtype="PCM_16")
            else:
                _write_mono_wav_22050(wav_path, audio_samples)
            raw_path.unlink(missing_ok=True)

        return _convert_locally(input_wav_bytes=wav_path.read_bytes(), accent=accent, gender=gender)
    finally:
        raw_path.unlink(missing_ok=True)


def main() -> None:
    st.set_page_config(page_title="Accent Changer", layout="centered")
    st.title("Accent Changer")
    st.caption("Upload a short WAV file, choose a target accent, and convert.")
    st.caption("Exact-wording mode is enabled. Style text injection is disabled.")
    st.caption("Mode: local Streamlit inference")

    if IS_STREAMLIT_CLOUD and not ENABLE_XTTS_IN_CLOUD:
        st.warning(
            "Cloud safe mode is active. Using lightweight fallback TTS (accent/gender fidelity will be limited). "
            "Set ENABLE_XTTS_IN_CLOUD=1 in Streamlit app settings to enable full XTTS."
        )

    _save_reference_uploads()

    uploaded_file = st.file_uploader(
        "Input audio (WAV/MP3/M4A/OGG/FLAC, max 15s)",
        type=SUPPORTED_INPUT_TYPES,
    )
    accent = st.selectbox("Target accent", ACCENTS)
    gender = st.selectbox("Voice gender", GENDERS)

    if uploaded_file is not None:
        input_audio_bytes = uploaded_file.getvalue()
        st.subheader("Input audio")
        file_suffix = Path(uploaded_file.name or "input.wav").suffix.lower() or ".wav"
        mime = "audio/wav" if file_suffix == ".wav" else "audio/mpeg"
        st.audio(input_audio_bytes, format=mime)

    if st.button("Convert", type="primary", use_container_width=True):
        if uploaded_file is None:
            st.error("Please upload a WAV file first.")
            return
        if not _all_references_ready():
            st.error("Missing reference WAV files. Upload all four reference accents above.")
            return

        try:
            with st.spinner("Converting locally with Whisper + XTTS..."):
                file_suffix = Path(uploaded_file.name or "input.wav").suffix.lower() or ".wav"
                output_audio = _convert_locally_from_any_audio(
                    input_audio_bytes=uploaded_file.getvalue(),
                    suffix=file_suffix,
                    accent=accent,
                    gender=gender.lower(),
                )

            st.subheader("Output audio")
            st.audio(output_audio, format="audio/wav")
            st.download_button(
                label="Download output WAV",
                data=output_audio,
                file_name="accent_changed.wav",
                mime="audio/wav",
                use_container_width=True,
            )
        except Exception as exc:
            st.error(str(exc))


if __name__ == "__main__":
    main()
