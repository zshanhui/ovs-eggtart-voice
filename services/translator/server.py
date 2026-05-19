"""FastAPI server for NLLB-200 translation via CTranslate2."""
from __future__ import annotations

import logging
import os
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
            device_index=[device_index] if device == "cuda" else None,
        )
        logger.info("Translator model loaded successfully")
    except Exception as e:
        logger.error("Failed to load translator model: %s", e)
        raise

    return translator, tokenizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global _translator, _tokenizer
    try:
        _translator, _tokenizer = _load_translator()
        logger.info("Translator service started")
    except Exception as e:
        logger.error("Failed to start translator service: %s", e)
        raise
    yield
    logger.info("Translator service shutting down")


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


@app.post("/translate", response_model=TranslateResponse)
async def translate(request: TranslateRequest):
    """Translate text from src_lang to tgt_lang."""
    if _translator is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Translator not initialized")

    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        # NLLB format: prefix source language code to input text
        source_prefix = request.src_lang
        input_text = f"{source_prefix} {text}"

        # Tokenize
        tokens = _tokenizer.EncodeAsIds(input_text)

        # Translate
        results = _translator.translate_batch(
            [tokens],
            target_prefix=[[request.tgt_lang]],
            max_decoding_length=256,
        )

        # Decode (skip the language token at position 0)
        translated_tokens = results[0].hypotheses[0][1:]
        translated = _tokenizer.DecodeIds(translated_tokens)

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
