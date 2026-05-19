"""Text translation backends."""
from .base import TranslatorBackend
from .http import CTranslate2Translator
from .noop import NoopTranslator

__all__ = ["TranslatorBackend", "NoopTranslator", "CTranslate2Translator"]
