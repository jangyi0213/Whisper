import argparse
import os
from dataclasses import dataclass
from typing import Optional

from modules.utils.cli_manager import str2bool
from modules.utils.paths import (
    DIARIZATION_MODELS_DIR,
    FASTER_WHISPER_MODELS_DIR,
    INSANELY_FAST_WHISPER_MODELS_DIR,
    KNOWLEDGE_BASE_DIR,
    NLLB_MODELS_DIR,
    OUTPUT_DIR,
    RAG_STORE_DIR,
    UVR_MODELS_DIR,
    WHISPER_MODELS_DIR,
)
from modules.whisper.data_classes import WhisperImpl


PROJECT_ROOT = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DEFAULT_AUTH_DB_PATH = os.path.join(PROJECT_ROOT, "auth.db")
DEFAULT_ADMIN_USERNAME = "admin"
DEFAULT_ADMIN_PASSWORD = "88888888"


def _default_max_workers() -> int:
    cpu_count = os.cpu_count() or 2
    return max(2, cpu_count // 2)


@dataclass
class AppConfig:
    whisper_type: str
    share: bool
    server_name: str
    server_port: Optional[int]
    root_path: Optional[str]
    username: Optional[str]
    password: Optional[str]
    theme: Optional[str]
    colab: bool
    api_open: bool
    allowed_paths: Optional[str]
    inbrowser: bool
    ssl_verify: bool
    auth_db_path: str
    default_admin_username: str
    default_admin_password: str
    ssl_keyfile: Optional[str]
    ssl_keyfile_password: Optional[str]
    ssl_certfile: Optional[str]
    whisper_model_dir: str
    faster_whisper_model_dir: str
    insanely_fast_whisper_model_dir: str
    diarization_model_dir: str
    nllb_model_dir: str
    uvr_model_dir: str
    output_dir: str
    rag_kb_dir: str
    rag_store_dir: str
    rag_embedding_model: str
    max_background_workers: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--whisper_type",
        type=str,
        default=WhisperImpl.FASTER_WHISPER.value,
        choices=[item.value for item in WhisperImpl],
        help="A type of the whisper implementation (Github repo name)",
    )
    parser.add_argument("--share", type=str2bool, default=False, nargs="?", const=True, help="Gradio share value")
    parser.add_argument("--server_name", type=str, default="0.0.0.0", help="Gradio server host")
    parser.add_argument("--server_port", type=int, default=None, help="Gradio server port")
    parser.add_argument("--root_path", type=str, default=None, help="Gradio root path")
    parser.add_argument("--username", type=str, default=None, help="Gradio authentication username")
    parser.add_argument("--password", type=str, default=None, help="Gradio authentication password")
    parser.add_argument("--theme", type=str, default=None, help="Gradio Blocks theme")
    parser.add_argument("--colab", type=str2bool, default=False, nargs="?", const=True, help="Is colab user or not")
    parser.add_argument("--api_open", type=str2bool, default=False, nargs="?", const=True, help="Enable api or not in Gradio")
    parser.add_argument("--allowed_paths", type=str, default=None, help="Gradio allowed paths")
    parser.add_argument("--inbrowser", type=str2bool, default=True, nargs="?", const=True, help="Whether to automatically start Gradio app or not")
    parser.add_argument("--ssl_verify", type=str2bool, default=True, nargs="?", const=True, help="Whether to verify SSL or not")
    parser.add_argument(
        "--auth_db_path",
        type=str,
        default=DEFAULT_AUTH_DB_PATH,
        help="Path to the SQLite database used for UI login",
    )
    parser.add_argument(
        "--default_admin_username",
        type=str,
        default=DEFAULT_ADMIN_USERNAME,
        help="Bootstrap admin username if database is empty",
    )
    parser.add_argument(
        "--default_admin_password",
        type=str,
        default=DEFAULT_ADMIN_PASSWORD,
        help="Bootstrap admin password if database is empty",
    )
    parser.add_argument("--ssl_keyfile", type=str, default=None, help="SSL Key file location")
    parser.add_argument("--ssl_keyfile_password", type=str, default=None, help="SSL Key file password")
    parser.add_argument("--ssl_certfile", type=str, default=None, help="SSL cert file location")
    parser.add_argument("--whisper_model_dir", type=str, default=WHISPER_MODELS_DIR, help="Directory path of the whisper model")
    parser.add_argument(
        "--faster_whisper_model_dir",
        type=str,
        default=FASTER_WHISPER_MODELS_DIR,
        help="Directory path of the faster-whisper model",
    )
    parser.add_argument(
        "--insanely_fast_whisper_model_dir",
        type=str,
        default=INSANELY_FAST_WHISPER_MODELS_DIR,
        help="Directory path of the insanely-fast-whisper model",
    )
    parser.add_argument(
        "--diarization_model_dir",
        type=str,
        default=DIARIZATION_MODELS_DIR,
        help="Directory path of the diarization model",
    )
    parser.add_argument(
        "--nllb_model_dir",
        type=str,
        default=NLLB_MODELS_DIR,
        help="Directory path of the Facebook NLLB model",
    )
    parser.add_argument("--uvr_model_dir", type=str, default=UVR_MODELS_DIR, help="Directory path of the UVR model")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory path of the outputs")
    parser.add_argument(
        "--rag_kb_dir",
        type=str,
        default=KNOWLEDGE_BASE_DIR,
        help="Directory path of txt knowledge base used for RAG correction",
    )
    parser.add_argument(
        "--rag_store_dir",
        type=str,
        default=RAG_STORE_DIR,
        help="Persistent directory for RAG vector store",
    )
    parser.add_argument(
        "--rag_embedding_model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="SentenceTransformer model name for RAG correction",
    )
    parser.add_argument(
        "--max_background_workers",
        type=int,
        default=_default_max_workers(),
        help="Number of worker threads to use for background jobs",
    )
    return parser


def parse_app_config(argv: Optional[list[str]] = None) -> AppConfig:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return AppConfig(**vars(args))


__all__ = [
    "AppConfig",
    "build_arg_parser",
    "parse_app_config",
    "DEFAULT_AUTH_DB_PATH",
    "DEFAULT_ADMIN_USERNAME",
    "DEFAULT_ADMIN_PASSWORD",
]

