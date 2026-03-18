"""ML pipeline: transcription and accent conversion."""

import gc
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import soundfile as sf
import torch
import torchaudio
from faster_whisper import WhisperModel

WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
XTTS_MODEL_NAME = os.getenv("XTTS_MODEL_NAME", "tts_models/multilingual/multi-dataset/xtts_v2")
CACHE_XTTS_MODEL = os.getenv("CACHE_XTTS_MODEL", "0") == "1"
ENABLE_XTTS_IN_CLOUD = os.getenv("ENABLE_XTTS_IN_CLOUD", "0") == "1"
LIGHT_TTS_MODEL_NAME = os.getenv("LIGHT_TTS_MODEL_NAME", "tts_models/en/ljspeech/tacotron2-DDC")
CACHE_LIGHT_TTS_MODEL = os.getenv("CACHE_LIGHT_TTS_MODEL", "1") == "1"

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "outputs"
REFERENCE_DIR = PROJECT_ROOT / "data" / "reference_voices"

IS_STREAMLIT_CLOUD = Path("/mount/src").exists() or os.getenv("STREAMLIT_SHARING_MODE") == "1"

# Force CPU path even on machines with CUDA installed.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("COQUI_TOS_AGREED", "1")
# XTTS checkpoints require full torch.load for trusted local model files.
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

CPU_THREADS = max(1, min((os.cpu_count() or 2), 2))
torch.set_num_threads(CPU_THREADS)
torch.set_num_interop_threads(1)


def _patch_torchaudio_load() -> None:
	"""Use soundfile-backed loading to avoid torchcodec dependency issues."""
	def _safe_load(filepath: str, *args, **kwargs):
		data, sample_rate = sf.read(filepath, dtype="float32", always_2d=True)
		tensor = torch.from_numpy(data.T)
		return tensor, sample_rate

	torchaudio.load = _safe_load

ACCENT_REFERENCE_WAVS = {
	"american": REFERENCE_DIR / "american.wav",
	"british": REFERENCE_DIR / "british.wav",
	"indian": REFERENCE_DIR / "indian.wav",
	"australian": REFERENCE_DIR / "australian.wav",
}

SUPPORTED_GENDERS = {"female", "male"}

STYLE_PREFIXES = {
	"podcast": "Welcome back to the show. ",
	"news": "This is a news update. ",
}


@lru_cache(maxsize=1)
def _get_whisper_model() -> WhisperModel:
	return WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")


def _load_xtts_model() -> Any:
	if IS_STREAMLIT_CLOUD and not ENABLE_XTTS_IN_CLOUD:
		raise RuntimeError(
			"XTTS is disabled on Streamlit Cloud by default to prevent memory crashes. "
			"Set ENABLE_XTTS_IN_CLOUD=1 in app settings to enable it (may still OOM)."
		)

	# Import lazily so Streamlit startup does not trigger heavy TTS initialization.
	from TTS.api import TTS

	_patch_torchaudio_load()
	return TTS(XTTS_MODEL_NAME).to("cpu")


def _load_light_tts_model() -> Any:
	from TTS.api import TTS

	_patch_torchaudio_load()
	return TTS(LIGHT_TTS_MODEL_NAME).to("cpu")


@lru_cache(maxsize=1)
def _get_light_tts_model_cached() -> Any:
	return _load_light_tts_model()


def _get_light_tts_model() -> Any:
	if CACHE_LIGHT_TTS_MODEL:
		return _get_light_tts_model_cached()
	return _load_light_tts_model()


@lru_cache(maxsize=1)
def _get_xtts_model_cached() -> Any:
	return _load_xtts_model()


def _get_xtts_model() -> Any:
	if CACHE_XTTS_MODEL:
		return _get_xtts_model_cached()
	return _load_xtts_model()


def warmup_models() -> None:
	"""Load models once during app startup to avoid first-request latency."""
	_get_whisper_model()
	_get_xtts_model()


def transcribe_audio(audio_path: str | Path) -> str:
	"""Transcribe short audio with Faster-Whisper on CPU."""
	src = Path(audio_path)
	model = _get_whisper_model()
	segments, _ = model.transcribe(
		str(src),
		language="en",
		beam_size=1,
		vad_filter=True,
		temperature=0.0,
		condition_on_previous_text=False,
	)
	text = " ".join(segment.text.strip() for segment in segments).strip()
	text = _sanitize_transcript(text)
	if not text:
		raise ValueError("No speech detected in input audio.")
	return text


def _sanitize_transcript(text: str) -> str:
	"""Normalize transcript to printable ASCII punctuation and spacing."""
	# Normalize unicode variants (smart quotes, compatibility forms, etc.).
	clean = unicodedata.normalize("NFKC", text)
	clean = clean.replace("\u2018", "'").replace("\u2019", "'")
	clean = clean.replace("\u201c", '"').replace("\u201d", '"')
	clean = clean.replace("\u2013", "-").replace("\u2014", "-")

	# Remove non-printable / non-ASCII symbols that can cause odd TTS behavior.
	clean = re.sub(r"[^\x20-\x7E]", " ", clean)

	# Collapse repeated whitespace and tidy spacing around punctuation.
	clean = re.sub(r"\s+", " ", clean).strip()
	clean = re.sub(r"\s+([,.;:!?])", r"\1", clean)

	return clean


def _apply_style(text: str, style: Optional[str], intensity: float) -> str:
	# Exact-wording mode: never inject style phrases.
	return text


def generate_speech(
	text: str,
	accent: str,
	gender: str = "female",
	style: Optional[str] = None,
	intensity: float = 1.0,
	output_path: Optional[str | Path] = None,
) -> str:
	"""Generate speech in a selected accent using XTTS v2 and reference voices."""
	key = accent.strip().lower()
	if key not in ACCENT_REFERENCE_WAVS:
		raise ValueError("Unsupported accent. Use: American, British, Indian, Australian.")

	gender_key = (gender or "female").strip().lower()
	if gender_key not in SUPPORTED_GENDERS:
		raise ValueError("Unsupported gender. Use: male or female.")

	if intensity < 0.0 or intensity > 1.0:
		raise ValueError("Intensity must be between 0.0 and 1.0.")

	gender_specific_wav = REFERENCE_DIR / f"{key}_{gender_key}.wav"
	speaker_wav = gender_specific_wav if gender_specific_wav.exists() else ACCENT_REFERENCE_WAVS[key]
	if not speaker_wav.exists():
		raise FileNotFoundError(
			f"Missing reference voice for accent={key}, gender={gender_key}. "
			f"Expected one of: {gender_specific_wav} or {ACCENT_REFERENCE_WAVS[key]}"
		)

	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
	out_path = Path(output_path) if output_path else OUTPUT_DIR / f"xtts_{key}_{uuid4().hex}.wav"
	styled_text = _apply_style(text=text, style=style, intensity=intensity)

	if IS_STREAMLIT_CLOUD and not ENABLE_XTTS_IN_CLOUD:
		try:
			light_tts = _get_light_tts_model()
		except Exception as exc:
			raise RuntimeError(
				"Cloud fallback TTS failed to load. "
				"Set LIGHT_TTS_MODEL_NAME to another lightweight model or enable XTTS explicitly. "
				f"Original error: {exc}"
			) from exc

		try:
			light_tts.tts_to_file(text=styled_text, file_path=str(out_path))
		finally:
			if not CACHE_LIGHT_TTS_MODEL:
				del light_tts
				gc.collect()

		return str(out_path)

	try:
		tts = _get_xtts_model()
	except Exception as exc:
		raise RuntimeError(
			"Failed to load XTTS model on this host. "
			"Set XTTS_MODEL_NAME to a smaller model or retry later. "
			f"Original error: {exc}"
		) from exc

	try:
		tts.tts_to_file(
			text=styled_text,
			speaker_wav=str(speaker_wav),
			language="en",
			file_path=str(out_path),
		)
	finally:
		# In low-memory deployments (e.g., Streamlit Cloud), avoid retaining XTTS in memory.
		if not CACHE_XTTS_MODEL:
			del tts
			gc.collect()

	return str(out_path)


def process_audio(input_path: str | Path, output_path: str | Path) -> str:
	"""Step-2 compatibility helper: passthrough WAV write."""
	src = Path(input_path)
	dst = Path(output_path)
	audio_samples, sample_rate = sf.read(str(src))
	sf.write(str(dst), audio_samples, sample_rate)
	return str(dst)
