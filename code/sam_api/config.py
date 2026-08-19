from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SAM3_", env_file=".env", extra="ignore")

    model_path: Path = Path("models/sam3.pt")
    data_dir: Path = Path("data")
    device: str = "cuda:0"
    # Environment variables arrive as strings. Keep this as int (rather than
    # Literal[16, 32]) so pydantic-settings can coerce "16" before validation.
    quantize: int = 16
    lifecycle: Literal["on_demand", "resident", "per_job"] = "on_demand"
    idle_unload_seconds: int = Field(default=600, ge=0)
    max_file_bytes: int = Field(default=50 * 1024 * 1024, gt=0)
    max_job_bytes: int = Field(default=20 * 1024 * 1024 * 1024, gt=0)
    upload_chunk_bytes: int = Field(default=1024 * 1024, gt=0)
    api_key: str | None = None

    @field_validator("quantize")
    @classmethod
    def validate_quantize(cls, value: int) -> int:
        if value not in {16, 32}:
            raise ValueError("quantize must be 16 or 32")
        return value

    @property
    def upload_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def result_dir(self) -> Path:
        return self.data_dir / "results"
