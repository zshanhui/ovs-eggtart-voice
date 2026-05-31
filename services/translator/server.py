"""FastAPI server for NLLB-200 translation via CTranslate2."""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import ctranslate2
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TranslateRequest(BaseModel):
    """Request payload for /translate endpoint."""
    text: str
    src_lang: str = "zho_Hans"
    tgt_lang: str = "eng_Latn"


class TranslateResponse(BaseModel):
    """Response payload for /translate endpoint."""
    translation: str
    src_lang: str
    tgt_lang: str
    model: str = "nllb-200-distilled-600M"


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    device: str


# Global translator instance
_translator: ctranslate2.Translator | None = None
_tokenizer = None

# Worker pool for offloading the blocking CT2 / SentencePiece calls off the
# asyncio event loop. ctranslate2.Translator is internally thread-safe — it
# serialises on its own backing device / engine — so multiple threads can
# submit translate_batch calls concurrently without external locking. Sizing:
# matches the typical small fan-in for InterpreterMode (1 voice user) + a
# couple of headroom slots for subtitle dashboards or admin probes. Tune via
# TRANSLATOR_WORKERS env var.
_executor: ThreadPoolExecutor | None = None


def _load_translator() -> tuple[ctranslate2.Translator, object]:
    """Load NLLB-200 model and tokenizer."""
    global _translator, _tokenizer

    model_path = os.getenv(
        "TRANSLATOR_MODEL_PATH",
        "/models/nllb-200-distilled-600m-ct2-int8"
    )
    device = os.getenv("TRANSLATOR_DEVICE", "cuda")
    device_index = int(os.getenv("TRANSLATOR_DEVICE_INDEX", "0"))

    logger.info(
        "Loading translator from %s (device=%s:%d)",
        model_path, device, device_index
    )

    try:
        import sentencepiece
        tokenizer = sentencepiece.SentencePieceProcessor()
        tokenizer.Load(os.path.join(model_path, "sentencepiece.bpe.model"))
        logger.info("Loaded SentencePiece tokenizer")
    except Exception as e:
        logger.error("Failed to load tokenizer: %s", e)
        raise

    try:
        translator = ctranslate2.Translator(
            model_path,
            device=device,
            device_index=device_index,
        )
        logger.info("Translator model loaded successfully")
    except Exception as e:
        logger.error("Failed to load translator model: %s", e)
        raise

    return translator, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _translator, _tokenizer, _executor
    try:
        _translator, _tokenizer = _load_translator()
        workers = int(os.getenv("TRANSLATOR_WORKERS", "4"))
        _executor = ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="ct2-worker"
        )
        logger.info("Translator service started (workers=%d)", workers)
    except Exception as e:
        logger.error("Failed to start translator service: %s", e)
        raise
    yield
    logger.info("Translator service shutting down")
    if _executor is not None:
        _executor.shutdown(wait=False, cancel_futures=True)


app = FastAPI(
    title="NLLB-200 Translator Service",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if _translator is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")

    device = os.getenv("TRANSLATOR_DEVICE", "cuda")
    return HealthResponse(
        status="ok",
        model="nllb-200-distilled-600M",
        device=device,
    )


def _translate_sync(text: str, src_lang: str, tgt_lang: str) -> str:
    """Synchronous translate — runs on the worker thread pool.

    NLLB tokenization: source token pieces, then ``</s>``, then the FLORES
    src_lang code. ``ctranslate2.Translator.translate_batch`` expects
    token-piece strings (not int IDs), so use ``EncodeAsPieces`` /
    ``DecodePieces`` — not ``EncodeAsIds`` / ``DecodeIds``.
    """
    tokens = _tokenizer.EncodeAsPieces(text) + ["</s>", src_lang]
    results = _translator.translate_batch(
        [tokens],
        target_prefix=[[tgt_lang]],
        max_decoding_length=256,
    )
    # Skip the language token at position 0
    translated_tokens = results[0].hypotheses[0][1:]
    return _tokenizer.DecodePieces(translated_tokens)


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate text from src_lang to tgt_lang.

    The CT2 call is offloaded to ``_executor`` so the asyncio event loop
    stays free for other in-flight requests. CT2's ``translate_batch`` is
    internally thread-safe; concurrent submissions serialise on the
    backing device but the Python side returns control to the loop while
    they wait.
    """
    if _translator is None or _tokenizer is None or _executor is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        loop = asyncio.get_running_loop()
        translated = await loop.run_in_executor(
            _executor,
            _translate_sync,
            text, request.src_lang, request.tgt_lang,
        )

        logger.debug(
            "Translated (%s→%s): %r → %r",
            request.src_lang, request.tgt_lang, text, translated
        )

        return TranslateResponse(
            translation=translated,
            src_lang=request.src_lang,
            tgt_lang=request.tgt_lang,
        )
    except Exception as e:
        logger.error("Translation failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Translation failed: {e}") from e


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9001,
        log_level="info",
    )
