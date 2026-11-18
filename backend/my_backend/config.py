"""Centralized configuration for the YOLO+SAM ML backend."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All backend configuration, loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model paths ---
    yolo_checkpoint: str = "../v5.1.pt"
    sam_checkpoint: str = "../sam_vit_b_01ec64.pth"
    sam_model_type: str = "vit_b"

    # --- Inference ---
    yolo_confidence_threshold: float = 0.25
    polygon_simplify_epsilon: float = 0.001
    job_cleanup_keep_recent: int = 5

    # --- Server ---
    ml_backend_host: str = "0.0.0.0"
    ml_backend_port: int = 9090
    log_level: str = "INFO"
    debug_requests: bool = False

    # --- Label Studio label mapping ---
    image_key: str = "image"
    output_from_name: str = "label"
    output_to_name: str = "image"

    @field_validator("yolo_checkpoint", "sam_checkpoint")
    @classmethod
    def resolve_path(cls, v: str) -> str:
        """Resolve relative paths against the project directory."""
        p = Path(v)
        if not p.is_absolute():
            base = Path(__file__).resolve().parent
            p = base / p
        return str(p.resolve())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached settings singleton."""
    return Settings()
