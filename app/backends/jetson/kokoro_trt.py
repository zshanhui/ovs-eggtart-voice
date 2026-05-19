"""Kokoro TTS backend for Jetson TensorRT.

Supports: BASIC_TTS, STREAMING (chunked PCM from synthesized audio),
MULTI_SPEAKER.

The hot path keeps text normalization/tokenization and voice lookup in Python,
then runs the Kokoro acoustic model with either a prebuilt TensorRT engine or
the CPU ONNX Runtime fallback. We deliberately do not depend on ORT GPU /
TensorRT Execution Provider here; the production acceleration path should be a
hand-written TensorRT+CPU hybrid, not provider-level graph partitioning.
"""

from __future__ import annotations

import io
import logging
import os
import re
import struct
import time
from typing import Optional

import numpy as np

from app.backends.jetson.matcha_trt import CudaMemoryPool
from app.core.tts_backend import TTSBackend, TTSCapability
from app.core.tts_speakers import resolve_speaker_kwargs

logger = logging.getLogger(__name__)

_MODEL_BASE = os.environ.get("KOKORO_MODEL_BASE", "/opt/models/kokoro-multi-lang-v1_0")
_MODEL_ONNX = os.environ.get("KOKORO_ONNX", os.path.join(_MODEL_BASE, "model.onnx"))
_ENGINE_PATH = os.environ.get(
    "KOKORO_TRT_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_fp16.engine"),
)
_HYBRID_DIR = os.environ.get("KOKORO_HYBRID_DIR", os.path.join(_MODEL_BASE, "hybrid"))
_HYBRID_PREFIX_ENGINE_ENV = os.environ.get("KOKORO_HYBRID_PREFIX_ENGINE")
_HYBRID_PREFIX_ENGINE_DYN = os.path.join(_HYBRID_DIR, "kokoro_prefix_encoder_dyn4_128_fp16.engine")
_HYBRID_PREFIX_ENGINE_FIXED = os.path.join(_HYBRID_DIR, "kokoro_prefix_encoder_s96_fp16.engine")
_HYBRID_SUFFIX_ONNX = os.environ.get(
    "KOKORO_HYBRID_SUFFIX_ONNX",
    os.path.join(_HYBRID_DIR, "kokoro_suffix_encoder.onnx"),
)
_SPLIT_ENCODER_ENGINE = os.environ.get(
    "KOKORO_SPLIT_ENCODER_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_prefix_encoder_dyn4_128_fp16.engine"),
)
_SPLIT_LENGTH_ONNX = os.environ.get(
    "KOKORO_SPLIT_LENGTH_ONNX",
    os.path.join(_MODEL_BASE, "engines", "cpu_length_regulator.onnx"),
)
_SPLIT_DECODER_ENGINE = os.environ.get(
    "KOKORO_SPLIT_DECODER_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_decoder_backbone_dyn64_256_fp16.engine"),
)
_SPLIT_DECODER_ENGINE_LONG = os.environ.get(
    "KOKORO_SPLIT_DECODER_ENGINE_LONG",
    os.path.join(_MODEL_BASE, "engines", "kokoro_decoder_backbone_dyn256_512_fp16.engine"),
)
_SPLIT_SOURCE_ENGINE = os.environ.get(
    "KOKORO_SPLIT_SOURCE_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_source_dyn128_512_bf16.engine"),
)
_SPLIT_SOURCE_ENGINE_LONG = os.environ.get(
    "KOKORO_SPLIT_SOURCE_ENGINE_LONG",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_source_dyn512_1024_bf16.engine"),
)
_SPLIT_SOURCE_ONNX = os.environ.get(
    "KOKORO_SPLIT_SOURCE_ONNX",
    os.path.join(_MODEL_BASE, "engines", "cpu_generator_source.onnx"),
)
_SPLIT_GENERATOR_ENGINE = os.environ.get(
    "KOKORO_SPLIT_GENERATOR_ENGINE",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_rest_preexp_dyn64_256_fp16.engine"),
)
_SPLIT_GENERATOR_ENGINE_LONG = os.environ.get(
    "KOKORO_SPLIT_GENERATOR_ENGINE_LONG",
    os.path.join(_MODEL_BASE, "engines", "kokoro_generator_rest_preexp_dyn256_512_fp16.engine"),
)
_SPLIT_ISTFT_ONNX = os.environ.get(
    "KOKORO_SPLIT_ISTFT_ONNX",
    os.path.join(_MODEL_BASE, "engines", "cpu_postspec_istft.onnx"),
)
_VOICES_BIN = os.environ.get("KOKORO_VOICES", os.path.join(_MODEL_BASE, "voices.bin"))
_TOKENS_PATH = os.environ.get("KOKORO_TOKENS", os.path.join(_MODEL_BASE, "tokens.txt"))

SAMPLE_RATE = 24000
MAX_TOKENS = int(os.environ.get("KOKORO_MAX_TOKENS", "510"))
DEFAULT_SPEAKER_ID = int(os.environ.get("KOKORO_DEFAULT_SID", os.environ.get("TTS_DEFAULT_SID", "52")))
DEFAULT_SPEED = float(os.environ.get("TTS_DEFAULT_SPEED", "1.0"))
VOICE_STYLES = 510
STYLE_DIM = 256
STYLE_BYTES = VOICE_STYLES * STYLE_DIM * 4
STREAM_SEGMENT_TOKENS = int(os.environ.get("KOKORO_STREAM_MAX_SEGMENT_TOKENS", "64"))
STREAM_SEGMENT_TEXT = os.environ.get("KOKORO_STREAM_SEGMENT_TEXT", "1").lower() not in ("0", "false", "no")
SYNTH_SEGMENT_TEXT = os.environ.get("KOKORO_SYNTH_SEGMENT_TEXT", "1").lower() not in ("0", "false", "no")


def _hybrid_prefix_engine_path() -> str:
    if _HYBRID_PREFIX_ENGINE_ENV:
        return _HYBRID_PREFIX_ENGINE_ENV
    if os.path.exists(_HYBRID_PREFIX_ENGINE_DYN):
        return _HYBRID_PREFIX_ENGINE_DYN
    return _HYBRID_PREFIX_ENGINE_FIXED


def _samples_to_wav(samples: np.ndarray, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    arr = np.asarray(samples, dtype=np.float32).reshape(-1)
    np.clip(arr, -1.0, 1.0, out=arr)
    pcm = (arr * 32767).astype(np.int16)
    data_size = pcm.nbytes
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<I", 16))
    buf.write(struct.pack("<HHIIHH", 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm.tobytes())
    return buf.getvalue()


class KokoroTRTBackend(TTSBackend):
    """English Kokoro v1.0 TTS accelerated with TensorRT on Jetson."""

    def __init__(self):
        self._token_to_id: dict[str, int] = {}
        self._runtime_mode = os.environ.get("KOKORO_TRT_RUNTIME", "auto").strip().lower()
        self._engine = None
        self._ctx = None
        self._pool: CudaMemoryPool | None = None
        self._ort_sess = None
        self._suffix_sess = None
        self._split_length_sess = None
        self._split_source_sess = None
        self._split_istft_sess = None
        self._split_engines = {}
        self._split_ctxs = {}
        self._split_long_engines = {}
        self._split_long_ctxs = {}
        self._token_input_name = "input_ids"
        self._output_name = None
        self._hybrid_fixed_seq_len: int | None = None
        self._hybrid_max_seq_len: int | None = None
        self._ready = False

    @property
    def name(self) -> str:
        return "kokoro_trt"

    @property
    def capabilities(self) -> set[TTSCapability]:
        return {
            TTSCapability.BASIC_TTS,
            TTSCapability.STREAMING,
            TTSCapability.MULTI_SPEAKER,
        }

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def is_ready(self) -> bool:
        return self._ready

    def preload(self) -> None:
        self._load_tokens()
        if self._runtime_mode in ("cpu", "ort", "ort_cpu", "onnxruntime"):
            self._load_ort()
        elif self._runtime_mode in ("split", "split_generator", "trt_split", "trt_cpu_split"):
            self._load_split_generator()
        elif self._runtime_mode in ("hybrid", "trt_cpu", "trt_prefix"):
            self._load_hybrid()
        elif os.path.exists(_ENGINE_PATH):
            self._load_engine()
        elif self._split_generator_assets_exist():
            self._load_split_generator()
        elif os.path.exists(_hybrid_prefix_engine_path()) and os.path.exists(_HYBRID_SUFFIX_ONNX):
            self._load_hybrid()
        else:
            logger.warning("Kokoro TRT engine missing at %s; using CPU ORT fallback", _ENGINE_PATH)
            self._load_ort()
        try:
            self._warmup()
        except Exception as exc:
            if self._runtime_mode == "engine":
                logger.warning(
                    "Kokoro direct TensorRT warmup failed (%s); falling back to CPU ORT",
                    exc,
                )
                self._engine = None
                self._ctx = None
                self._pool = None
                self._load_ort()
                self._warmup()
            else:
                raise
        self._ready = True

    def _load_tokens(self) -> None:
        if not os.path.exists(_TOKENS_PATH):
            raise FileNotFoundError(f"Kokoro tokens not found: {_TOKENS_PATH}")
        with open(_TOKENS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                raw = line.rstrip("\n").rstrip("\r")
                if not raw:
                    continue
                rsep = max(raw.rfind(" "), raw.rfind("\t"))
                if rsep < 0:
                    continue
                tok = raw[:rsep] or " "
                try:
                    self._token_to_id[tok] = int(raw[rsep + 1:])
                except ValueError:
                    continue
        logger.info("Loaded %d Kokoro tokens from %s", len(self._token_to_id), _TOKENS_PATH)

    def _load_engine(self) -> None:
        import tensorrt as trt

        t0 = time.time()
        with open(_ENGINE_PATH, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize Kokoro engine: {_ENGINE_PATH}")
        self._ctx = self._engine.create_execution_context()
        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        if "tokens" in names:
            self._token_input_name = "tokens"
        elif "input_ids" in names:
            self._token_input_name = "input_ids"
        self._pool = CudaMemoryPool()
        self._runtime_mode = "engine"
        logger.info("Kokoro TRT engine loaded: %s (%.1fs)", _ENGINE_PATH, time.time() - t0)

    def _load_hybrid(self) -> None:
        import onnxruntime as ort
        import tensorrt as trt

        prefix_engine = _hybrid_prefix_engine_path()
        if not os.path.exists(prefix_engine):
            raise FileNotFoundError(f"Kokoro hybrid prefix engine not found: {prefix_engine}")
        if not os.path.exists(_HYBRID_SUFFIX_ONNX):
            raise FileNotFoundError(f"Kokoro hybrid suffix ONNX not found: {_HYBRID_SUFFIX_ONNX}")

        t0 = time.time()
        with open(prefix_engine, "rb") as f:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            self._engine = runtime.deserialize_cuda_engine(f.read())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize Kokoro hybrid prefix engine: {prefix_engine}")
        self._ctx = self._engine.create_execution_context()
        self._pool = CudaMemoryPool()
        self._configure_hybrid_token_profile()
        self._suffix_sess = ort.InferenceSession(_HYBRID_SUFFIX_ONNX, providers=["CPUExecutionProvider"])
        self._token_input_name = "tokens"
        self._runtime_mode = "hybrid"
        logger.info(
            "Kokoro hybrid loaded: prefix=%s suffix=%s token_profile=fixed:%s max:%s (%.1fs)",
            prefix_engine,
            _HYBRID_SUFFIX_ONNX,
            self._hybrid_fixed_seq_len,
            self._hybrid_max_seq_len,
            time.time() - t0,
        )

    def _split_generator_assets_exist(self) -> bool:
        required = (
            _SPLIT_ENCODER_ENGINE,
            _SPLIT_LENGTH_ONNX,
            _SPLIT_DECODER_ENGINE,
            _SPLIT_GENERATOR_ENGINE,
            _SPLIT_ISTFT_ONNX,
        )
        if not all(os.path.exists(path) for path in required):
            return False
        return os.path.exists(_SPLIT_SOURCE_ENGINE) or os.path.exists(_SPLIT_SOURCE_ONNX)

    def _load_split_generator(self) -> None:
        import onnxruntime as ort
        import tensorrt as trt

        required = {
            "encoder": _SPLIT_ENCODER_ENGINE,
            "decoder": _SPLIT_DECODER_ENGINE,
            "generator": _SPLIT_GENERATOR_ENGINE,
        }
        if os.path.exists(_SPLIT_SOURCE_ENGINE):
            required["source"] = _SPLIT_SOURCE_ENGINE
        for name, path in required.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kokoro split {name} engine not found: {path}")
        for name, path in {
            "length regulator": _SPLIT_LENGTH_ONNX,
            "ISTFT": _SPLIT_ISTFT_ONNX,
        }.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"Kokoro split {name} ONNX not found: {path}")
        if "source" not in required and not os.path.exists(_SPLIT_SOURCE_ONNX):
            raise FileNotFoundError(
                f"Kokoro split source engine/ONNX not found: {_SPLIT_SOURCE_ENGINE} / {_SPLIT_SOURCE_ONNX}"
            )

        t0 = time.time()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self._split_engines = {}
        self._split_ctxs = {}
        self._split_long_engines = {}
        self._split_long_ctxs = {}
        for name, path in required.items():
            with open(path, "rb") as f:
                engine = runtime.deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError(f"Failed to deserialize Kokoro split {name} engine: {path}")
            self._split_engines[name] = engine
            self._split_ctxs[name] = engine.create_execution_context()
        long_required = {
            "decoder": _SPLIT_DECODER_ENGINE_LONG,
            "source": _SPLIT_SOURCE_ENGINE_LONG,
            "generator": _SPLIT_GENERATOR_ENGINE_LONG,
        }
        if all(os.path.exists(path) for path in long_required.values()):
            for name, path in long_required.items():
                with open(path, "rb") as f:
                    engine = runtime.deserialize_cuda_engine(f.read())
                if engine is None:
                    raise RuntimeError(f"Failed to deserialize Kokoro split long {name} engine: {path}")
                self._split_long_engines[name] = engine
                self._split_long_ctxs[name] = engine.create_execution_context()
        elif any(os.path.exists(path) for path in long_required.values()):
            missing = [path for path in long_required.values() if not os.path.exists(path)]
            logger.warning("Ignoring incomplete Kokoro 256-512 bucket; missing: %s", missing)
        self._pool = CudaMemoryPool()
        self._configure_split_token_profile()
        self._split_length_sess = ort.InferenceSession(_SPLIT_LENGTH_ONNX, providers=["CPUExecutionProvider"])
        self._split_istft_sess = ort.InferenceSession(_SPLIT_ISTFT_ONNX, providers=["CPUExecutionProvider"])
        if "source" not in required:
            self._split_source_sess = ort.InferenceSession(_SPLIT_SOURCE_ONNX, providers=["CPUExecutionProvider"])
        self._token_input_name = "tokens"
        self._runtime_mode = "split_generator"
        logger.info(
            "Kokoro split-generator loaded: encoder=%s decoder=%s source=%s generator=%s "
            "long_bucket=%s length=%s istft=%s token_profile=fixed:%s max:%s (%.1fs)",
            _SPLIT_ENCODER_ENGINE,
            _SPLIT_DECODER_ENGINE,
            _SPLIT_SOURCE_ENGINE if "source" in required else _SPLIT_SOURCE_ONNX,
            _SPLIT_GENERATOR_ENGINE,
            bool(self._split_long_engines),
            _SPLIT_LENGTH_ONNX,
            _SPLIT_ISTFT_ONNX,
            self._hybrid_fixed_seq_len,
            self._hybrid_max_seq_len,
            time.time() - t0,
        )

    def _configure_split_token_profile(self) -> None:
        engine = self._split_engines.get("encoder")
        if engine is None:
            return
        try:
            min_shape, _opt_shape, max_shape = engine.get_tensor_profile_shape("tokens", 0)
            min_seq = int(tuple(min_shape)[1])
            max_seq = int(tuple(max_shape)[1])
            self._hybrid_max_seq_len = max_seq
            self._hybrid_fixed_seq_len = max_seq if min_seq == max_seq else None
        except Exception:
            self._hybrid_fixed_seq_len = None
            self._hybrid_max_seq_len = int(os.environ.get("KOKORO_SPLIT_MAX_SEQ_LEN", "128"))

    def _configure_hybrid_token_profile(self) -> None:
        assert self._engine is not None
        try:
            min_shape, _opt_shape, max_shape = self._engine.get_tensor_profile_shape("tokens", 0)
            min_seq = int(tuple(min_shape)[1])
            max_seq = int(tuple(max_shape)[1])
            self._hybrid_max_seq_len = max_seq
            self._hybrid_fixed_seq_len = max_seq if min_seq == max_seq else None
        except Exception:
            fixed = int(os.environ.get("KOKORO_HYBRID_TOKEN_LEN", "0"))
            self._hybrid_fixed_seq_len = fixed or None
            self._hybrid_max_seq_len = fixed or int(os.environ.get("KOKORO_HYBRID_MAX_SEQ_LEN", "128"))

    def _load_ort(self) -> None:
        import onnxruntime as ort

        if not os.path.exists(_MODEL_ONNX):
            raise FileNotFoundError(f"Kokoro ONNX not found: {_MODEL_ONNX}")
        providers = ["CPUExecutionProvider"]
        sess_opt = ort.SessionOptions()
        sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._ort_sess = ort.InferenceSession(_MODEL_ONNX, sess_opt, providers=providers)
        input_names = {item.name for item in self._ort_sess.get_inputs()}
        if "tokens" in input_names:
            self._token_input_name = "tokens"
        elif "input_ids" in input_names:
            self._token_input_name = "input_ids"
        else:
            raise RuntimeError(f"Kokoro ONNX missing token input; inputs={sorted(input_names)}")
        outputs = self._ort_sess.get_outputs()
        self._output_name = outputs[0].name if outputs else None
        active = self._ort_sess.get_providers()
        self._runtime_mode = "ort_cpu"
        logger.info("Kokoro ORT providers: %s", active)

    def _warmup(self) -> None:
        start = time.time()
        for text in ("OK.", "Hello."):
            self.synthesize(text)
        logger.info("Kokoro warmup: %.1fs", time.time() - start)

    def synthesize(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
        pitch_shift: Optional[float] = None,
        language: Optional[str] = None,
        **kwargs,
    ) -> tuple[bytes, dict]:
        del pitch_shift, language
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, speaker_id=speaker_id, **kwargs)
        sid = voice.get("speaker_id", DEFAULT_SPEAKER_ID)
        if SYNTH_SEGMENT_TEXT and self._runtime_mode in ("hybrid", "split_generator"):
            max_tokens = max(1, (self._hybrid_max_seq_len or 128) - 2)
            token_count = len(self._text_to_token_ids(text))
            if token_count > max_tokens:
                segment_limit = int(os.environ.get("KOKORO_SYNTH_MAX_SEGMENT_TOKENS", str(STREAM_SEGMENT_TOKENS)))
                return self._synthesize_segments(text, segment_limit, speaker_id=sid, speed=speed)
        return self._synthesize_one(text, speaker_id=sid, speed=speed)

    def _synthesize_segments(
        self,
        text: str,
        max_tokens: int,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, dict]:
        t_start = time.time()
        segments = self._split_stream_text(text, max_tokens)
        pcm_parts: list[bytes] = []
        metas: list[dict] = []
        for segment in segments:
            wav, meta = self._synthesize_one(segment, speaker_id=speaker_id, speed=speed)
            metas.append(meta)
            if len(wav) > 44:
                pcm_parts.append(wav[44:])
        pcm = b"".join(pcm_parts)
        samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32767.0
        wav = _samples_to_wav(samples, SAMPLE_RATE)
        duration = len(samples) / SAMPLE_RATE
        elapsed = time.time() - t_start
        return wav, {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": SAMPLE_RATE,
            "num_tokens": sum(int(meta.get("num_tokens", 0)) for meta in metas),
            "infer_ms": round(sum(float(meta.get("infer_ms", 0.0)) for meta in metas), 1),
            "language": "en",
            "runtime": self._runtime_mode,
            "segments": len(segments),
            "truncated": False,
        }

    def _synthesize_one(
        self,
        text: str,
        speaker_id: Optional[int] = None,
        speed: Optional[float] = None,
    ) -> tuple[bytes, dict]:
        sid = DEFAULT_SPEAKER_ID if speaker_id is None else int(speaker_id)
        spd = DEFAULT_SPEED if speed is None else float(speed)
        t_start = time.time()
        token_ids = self._text_to_token_ids(text)
        if not token_ids:
            silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
            return _samples_to_wav(silence, SAMPLE_RATE), {
                "duration": 0.1,
                "inference_time": 0.0,
                "sample_rate": SAMPLE_RATE,
                "language": "en",
                "runtime": self._runtime_mode,
            }

        max_tokens = MAX_TOKENS
        if self._runtime_mode in ("hybrid", "split_generator"):
            max_tokens = max(1, (self._hybrid_max_seq_len or 128) - 2)
        truncated = len(token_ids) > max_tokens
        token_ids = token_ids[:max_tokens]
        ids = [0, *token_ids, 0]
        if self._runtime_mode in ("hybrid", "split_generator") and self._hybrid_fixed_seq_len:
            ids = ids + [0] * max(0, self._hybrid_fixed_seq_len - len(ids))
        input_ids = np.array([ids], dtype=np.int64)
        style = self._load_style(sid, len(token_ids))
        speed_arr = np.array([spd], dtype=np.float32)

        t_infer = time.time()
        if self._runtime_mode == "engine":
            audio = self._run_engine(input_ids, style, speed_arr)
        elif self._runtime_mode == "hybrid":
            audio = self._run_hybrid(input_ids, style, speed_arr)
        elif self._runtime_mode == "split_generator":
            try:
                audio = self._run_split_generator(input_ids, style, speed_arr)
            except ValueError as exc:
                if os.environ.get("KOKORO_SPLIT_CPU_FALLBACK", "1").lower() in ("0", "false", "no"):
                    raise
                logger.warning("Kokoro split-generator shape mismatch; falling back to CPU ORT: %s", exc)
                self._load_ort()
                audio = self._run_ort(input_ids, style, speed_arr)
        else:
            audio = self._run_ort(input_ids, style, speed_arr)
        infer_ms = (time.time() - t_infer) * 1000

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        wav = _samples_to_wav(audio, SAMPLE_RATE)
        duration = len(audio) / SAMPLE_RATE
        elapsed = time.time() - t_start
        return wav, {
            "duration": round(duration, 3),
            "inference_time": round(elapsed, 3),
            "rtf": round(elapsed / duration, 3) if duration > 0 else 0,
            "sample_rate": SAMPLE_RATE,
            "num_tokens": len(token_ids),
            "infer_ms": round(infer_ms, 1),
            "language": "en",
            "runtime": self._runtime_mode,
            "truncated": truncated,
        }

    def generate_streaming(self, text: str, **kwargs):
        voice = resolve_speaker_kwargs(self.model_id, allow_embedding=False, **kwargs)
        sid = voice.get("speaker_id", DEFAULT_SPEAKER_ID)
        segments = [text]
        if STREAM_SEGMENT_TEXT and kwargs.get("segment_text", True):
            segments = self._split_stream_text(text, kwargs.get("segment_max_tokens"))
        try:
            chunk_ms = int(os.environ.get("KOKORO_STREAM_CHUNK_MS", "40"))
        except ValueError:
            chunk_ms = 40
        chunk_ms = max(10, min(200, chunk_ms))
        chunk_bytes = max(2, int(SAMPLE_RATE * chunk_ms / 1000) * 2)
        for segment in segments:
            wav, _meta = self.synthesize(
                segment,
                speaker_id=sid,
                speed=kwargs.get("speed"),
            )
            if len(wav) <= 44:
                continue
            pcm = wav[44:]
            for offset in range(0, len(pcm), chunk_bytes):
                chunk = pcm[offset:offset + chunk_bytes]
                if chunk:
                    yield chunk

    def _split_stream_text(self, text: str, max_tokens: Optional[int] = None) -> list[str]:
        text = " ".join((text or "").split())
        if not text:
            return []
        if max_tokens is None:
            max_tokens = STREAM_SEGMENT_TOKENS
        try:
            max_tokens = int(max_tokens)
        except (TypeError, ValueError):
            max_tokens = STREAM_SEGMENT_TOKENS
        if max_tokens <= 0:
            return [text]
        max_tokens = max(16, max_tokens)

        parts = [part.strip() for part in re.split(r"(?<=[.!?;:])\s+", text) if part.strip()]
        if not parts:
            parts = [text]
        segments: list[str] = []
        for part in parts:
            segments.extend(self._split_text_by_token_count(part, max_tokens))
        return segments or [text]

    def _split_text_by_token_count(self, text: str, max_tokens: int) -> list[str]:
        if len(self._text_to_token_ids(text)) <= max_tokens:
            return [text]
        words = text.split()
        if not words:
            return [text]
        segments: list[str] = []
        current_words: list[str] = []
        for word in words:
            candidate_words = [*current_words, word]
            candidate = " ".join(candidate_words)
            if current_words and len(self._text_to_token_ids(candidate)) > max_tokens:
                segments.append(" ".join(current_words))
                current_words = [word]
            else:
                current_words = candidate_words
            current = " ".join(current_words)
            if current and len(self._text_to_token_ids(current)) > max_tokens:
                segments.extend(self._split_long_word(current, max_tokens))
                current_words = []
        if current_words:
            segments.append(" ".join(current_words))
        return segments

    def _split_long_word(self, text: str, max_tokens: int) -> list[str]:
        parts: list[str] = []
        current = ""
        for ch in text:
            candidate = f"{current}{ch}"
            if current and len(self._text_to_token_ids(candidate)) > max_tokens:
                parts.append(current)
                current = ch
            else:
                current = candidate
        if current:
            parts.append(current)
        return parts or [text]

    def _text_to_token_ids(self, text: str) -> list[int]:
        import piper_phonemize

        text = text.strip()
        if not text:
            return []
        sentences = piper_phonemize.phonemize_espeak(text, "en-us")
        ids: list[int] = []
        for sent_idx, phonemes in enumerate(sentences or []):
            if sent_idx > 0:
                self._append_token(ids, " ")
            joined = "".join(p for p in phonemes if p)
            for ch in joined:
                self._append_token(ids, ch)
        if ids:
            return ids

        # Last-resort ASCII path keeps smoke tests debuggable if espeak data is
        # missing; quality is not expected to be acceptable in this branch.
        for ch in re.sub(r"\s+", " ", text.lower()):
            self._append_token(ids, ch)
        return ids

    def _append_token(self, ids: list[int], token: str) -> None:
        tid = self._token_to_id.get(token)
        if tid is not None:
            ids.append(tid)

    def _load_style(self, speaker_id: int, token_count: int) -> np.ndarray:
        if not os.path.exists(_VOICES_BIN):
            raise FileNotFoundError(f"Kokoro voices not found: {_VOICES_BIN}")
        style_idx = max(0, min(VOICE_STYLES - 1, int(token_count)))
        offset = speaker_id * STYLE_BYTES + style_idx * STYLE_DIM * 4
        size = os.path.getsize(_VOICES_BIN)
        if offset + STYLE_DIM * 4 > size:
            raise ValueError(
                f"Kokoro speaker_id {speaker_id} out of range for {_VOICES_BIN} "
                f"(file has about {size // STYLE_BYTES} speakers)"
            )
        with open(_VOICES_BIN, "rb") as f:
            f.seek(offset)
            data = f.read(STYLE_DIM * 4)
        return np.frombuffer(data, dtype=np.float32).reshape(1, STYLE_DIM).copy()

    def _run_ort(self, input_ids: np.ndarray, style: np.ndarray, speed: np.ndarray) -> np.ndarray:
        return self._ort_sess.run(
            None,
            {self._token_input_name: input_ids, "style": style, "speed": speed},
        )[0]

    def _run_engine(self, input_ids: np.ndarray, style: np.ndarray, speed: np.ndarray) -> np.ndarray:
        pool = self._pool
        ctx = self._ctx
        assert pool is not None and ctx is not None and self._engine is not None

        def bind_input(name: str, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr)
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            self._set_or_validate_input_shape(ctx, name, arr)

        bind_input(self._token_input_name, input_ids)
        bind_input("style", style.astype(np.float32, copy=False))
        bind_input("speed", speed.astype(np.float32, copy=False))

        output_name = self._output_tensor_name()
        out_shape = tuple(int(d) for d in ctx.get_tensor_shape(output_name))
        if any(d < 0 for d in out_shape):
            pool.free_all()
            raise RuntimeError(
                "Kokoro TRT engine produced a dynamic output shape that the "
                "direct full-engine backend cannot allocate. Use "
                "KOKORO_TRT_RUNTIME=hybrid with the TensorRT prefix engine."
            )
        output = np.empty(out_shape, dtype=np.float32)
        d_out = pool.allocate(output.nbytes)
        ctx.set_tensor_address(output_name, d_out)
        ok = ctx.execute_async_v3(pool.stream_handle())
        if not ok:
            pool.free_all()
            raise RuntimeError("Kokoro TRT execute_async_v3 returned False")
        pool.synchronize()
        pool.copy_dtoh(d_out, output)
        pool.free_all()
        return output

    def _run_hybrid(self, input_ids: np.ndarray, style: np.ndarray, speed: np.ndarray) -> np.ndarray:
        assert self._suffix_sess is not None
        prefix_outputs = self._run_trt_engine(
            {"tokens": input_ids, "style": style.astype(np.float32, copy=False), "speed": speed.astype(np.float32, copy=False)}
        )
        suffix_input_names = {item.name for item in self._suffix_sess.get_inputs()}
        feeds = {}
        for name, arr in {"tokens": input_ids, "style": style, "speed": speed}.items():
            if name in suffix_input_names:
                feeds[name] = arr
        for name, arr in prefix_outputs.items():
            feeds[name] = arr
        return self._suffix_sess.run(None, feeds)[0]

    def _run_split_generator(self, input_ids: np.ndarray, style: np.ndarray, speed: np.ndarray) -> np.ndarray:
        assert self._split_length_sess is not None and self._split_istft_sess is not None

        stage: dict[str, np.ndarray] = {
            "tokens": input_ids,
            "style": style.astype(np.float32, copy=False),
            "speed": speed.astype(np.float32, copy=False),
        }
        stage.update(self._run_named_trt_engine("encoder", stage))
        stage.update(_run_cpu_onnx(self._split_length_sess, stage))
        frame_t = int(stage["/encoder/MatMul_1_output_0"].shape[2])
        bucket_engines, bucket_ctxs = self._select_split_bucket(frame_t)

        stage.update(self._run_split_bucket_engine(bucket_engines, bucket_ctxs, "decoder", {
            "/encoder/MatMul_1_output_0": stage["/encoder/MatMul_1_output_0"],
            "/decoder/decoder/F0_conv/Conv_output_0": stage["/decoder/decoder/F0_conv/Conv_output_0"],
            "/decoder/decoder/N_conv/Conv_output_0": stage["/decoder/decoder/N_conv/Conv_output_0"],
            "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
            "style": stage["style"],
        }))

        if "source" in bucket_engines:
            stage.update(self._run_split_bucket_engine(bucket_engines, bucket_ctxs, "source", {
                "/decoder/decoder/Unsqueeze_output_0": stage["/decoder/decoder/Unsqueeze_output_0"],
            }))
        else:
            assert self._split_source_sess is not None
            stage.update(_run_cpu_onnx(self._split_source_sess, stage))

        gen = self._run_split_bucket_engine(bucket_engines, bucket_ctxs, "generator", {
            "/decoder/decoder/decode.3/Div_4_output_0": stage["/decoder/decoder/decode.3/Div_4_output_0"],
            "/decoder/decoder/generator/Concat_3_output_0": stage["/decoder/decoder/generator/Concat_3_output_0"],
            "style": stage["style"],
        })
        return _run_cpu_onnx(self._split_istft_sess, gen)["audio"]

    def _select_split_bucket(self, frame_t: int):
        if frame_t <= 256:
            return self._split_engines, self._split_ctxs
        if frame_t <= 512 and self._split_long_engines:
            return self._split_long_engines, self._split_long_ctxs
        raise ValueError(
            f"Kokoro split-generator frame length {frame_t} is outside available TRT buckets "
            f"(base<=256, long<=512 loaded={bool(self._split_long_engines)})"
        )

    def _run_named_trt_engine(self, name: str, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        engine = self._split_engines[name]
        ctx = self._split_ctxs[name]
        return self._run_trt_context(engine, ctx, inputs)

    def _run_split_bucket_engine(
        self,
        engines: dict[str, object],
        ctxs: dict[str, object],
        name: str,
        inputs: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        return self._run_trt_context(engines[name], ctxs[name], inputs)

    def _run_trt_engine(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pool = self._pool
        ctx = self._ctx
        assert pool is not None and ctx is not None and self._engine is not None
        return self._run_trt_context(self._engine, ctx, inputs)

    def _run_trt_context(self, engine, ctx, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        pool = self._pool
        assert pool is not None

        def bind_input(name: str, arr: np.ndarray) -> None:
            arr = np.ascontiguousarray(arr)
            self._validate_engine_input_shape(engine, name, arr)
            ptr = pool.allocate(arr.nbytes)
            pool.copy_htod(arr, ptr)
            ctx.set_tensor_address(name, ptr)
            self._set_or_validate_input_shape(ctx, name, arr)

        for name, arr in inputs.items():
            bind_input(name, arr)

        outputs: dict[str, np.ndarray] = {}
        output_ptrs: list[tuple[str, int, np.ndarray]] = []
        import tensorrt as trt

        for i in range(engine.num_io_tensors):
            name = engine.get_tensor_name(i)
            if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
                continue
            shape = tuple(int(d) for d in ctx.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                pool.free_all()
                raise RuntimeError(f"Kokoro hybrid prefix output has dynamic shape: {name} {shape}")
            dtype = _trt_dtype_to_np(engine.get_tensor_dtype(name))
            out = np.empty(shape, dtype=dtype)
            ptr = pool.allocate(out.nbytes)
            ctx.set_tensor_address(name, ptr)
            output_ptrs.append((name, ptr, out))

        ok = ctx.execute_async_v3(pool.stream_handle())
        if not ok:
            pool.free_all()
            raise RuntimeError("Kokoro hybrid prefix TRT execute_async_v3 returned False")
        pool.synchronize()
        for name, ptr, out in output_ptrs:
            pool.copy_dtoh(ptr, out)
            outputs[name] = out
        pool.free_all()
        return outputs

    def _validate_engine_input_shape(self, engine, name: str, arr: np.ndarray) -> None:
        shape = tuple(int(d) for d in engine.get_tensor_shape(name))
        if not shape or any(dim < 0 for dim in shape):
            return
        if shape != tuple(arr.shape):
            raise ValueError(f"{name} shape {tuple(arr.shape)} does not match fixed TRT shape {shape}")

    def _set_or_validate_input_shape(self, ctx, name: str, arr: np.ndarray) -> None:
        try:
            ok = ctx.set_input_shape(name, tuple(arr.shape))
        except Exception:
            return
        if ok is False:
            raise ValueError(f"{name} shape {tuple(arr.shape)} is outside the TRT optimization profile")

    def _output_tensor_name(self) -> str:
        import tensorrt as trt

        names = [self._engine.get_tensor_name(i) for i in range(self._engine.num_io_tensors)]
        outputs = [
            name for name in names
            if self._engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
        ]
        if not outputs:
            raise RuntimeError("Kokoro TRT engine has no output tensors")
        return outputs[0]


def _trt_dtype_to_np(dtype):
    import tensorrt as trt

    if dtype == trt.float32:
        return np.float32
    if dtype == trt.float16:
        return np.float16
    if dtype == trt.int32:
        return np.int32
    if dtype == trt.int64:
        return np.int64
    if dtype == trt.bool:
        return np.bool_
    raise TypeError(f"Unsupported TensorRT dtype: {dtype}")


def _run_cpu_onnx(sess, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    input_names = {item.name for item in sess.get_inputs()}
    output_names = [item.name for item in sess.get_outputs()]
    actual = {name: value for name, value in feeds.items() if name in input_names}
    return dict(zip(output_names, sess.run(output_names, actual)))
