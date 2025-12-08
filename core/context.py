from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from core.config import AppConfig
from modules.face_search.service import FaceSearchService
from modules.rag.temp_chat import TemporaryRAGChatService
from modules.rag.text_corrector import TextCorrectionRAG
from modules.translation.deepl_api import DeepLAPI
from modules.translation.nllb_inference import NLLBInference
from modules.whisper.whisper_factory import WhisperFactory


@dataclass
class AppContext:
    config: AppConfig
    logger: any
    _whisper: Optional[object] = field(default=None, init=False, repr=False)
    _nllb: Optional[NLLBInference] = field(default=None, init=False, repr=False)
    _deepl: Optional[DeepLAPI] = field(default=None, init=False, repr=False)
    _text_corrector: Optional[TextCorrectionRAG] = field(default=None, init=False, repr=False)
    _rag_chat: Optional[TemporaryRAGChatService] = field(default=None, init=False, repr=False)
    _face_search: Optional[FaceSearchService] = field(default=None, init=False, repr=False)
    _text_corrector_initialized: bool = field(default=False, init=False, repr=False)

    @property
    def whisper(self):
        if self._whisper is None:
            cfg = self.config
            self._whisper = WhisperFactory.create_whisper_inference(
                whisper_type=cfg.whisper_type,
                whisper_model_dir=cfg.whisper_model_dir,
                faster_whisper_model_dir=cfg.faster_whisper_model_dir,
                insanely_fast_whisper_model_dir=cfg.insanely_fast_whisper_model_dir,
                uvr_model_dir=cfg.uvr_model_dir,
                output_dir=cfg.output_dir,
            )
        return self._whisper

    @property
    def nllb(self) -> NLLBInference:
        if self._nllb is None:
            cfg = self.config
            self._nllb = NLLBInference(
                model_dir=cfg.nllb_model_dir,
                output_dir=os.path.join(cfg.output_dir, "translations"),
            )
        return self._nllb

    @property
    def deepl_api(self) -> DeepLAPI:
        if self._deepl is None:
            cfg = self.config
            self._deepl = DeepLAPI(output_dir=os.path.join(cfg.output_dir, "translations"))
        return self._deepl

    @property
    def text_corrector(self) -> Optional[TextCorrectionRAG]:
        if not self._text_corrector_initialized:
            cfg = self.config
            try:
                self._text_corrector = TextCorrectionRAG(
                    knowledge_dir=cfg.rag_kb_dir,
                    persist_dir=cfg.rag_store_dir,
                    embedding_model=cfg.rag_embedding_model,
                )
                self.whisper.register_text_corrector(self._text_corrector)
            except Exception as exc:
                self.logger.warning(f"初始化 RAG 文本纠错失败，将禁用该功能：{exc}")
                self._text_corrector = None
            finally:
                self._text_corrector_initialized = True
        return self._text_corrector

    @property
    def rag_chat_service(self) -> TemporaryRAGChatService:
        if self._rag_chat is None:
            self._rag_chat = TemporaryRAGChatService(embedding_model=self.config.rag_embedding_model)
        return self._rag_chat

    @property
    def face_search_service(self) -> FaceSearchService:
        if self._face_search is None:
            db_dir = os.path.join(self.config.output_dir, "face_db")
            os.makedirs(db_dir, exist_ok=True)
            self._face_search = FaceSearchService(db_dir=db_dir)
        return self._face_search


def create_app_context(config: AppConfig, logger) -> AppContext:
    return AppContext(config=config, logger=logger)


__all__ = ["AppContext", "create_app_context"]

