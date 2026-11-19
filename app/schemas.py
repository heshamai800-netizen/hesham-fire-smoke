from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field, field_validator

class CameraConfig(BaseModel):
    id: str | None = None
    name: str = Field(min_length=1, max_length=80)
    source_type: Literal['rtsp','usb']
    source: str = Field(min_length=1, max_length=2048)
    enabled: bool = True

    fire_enabled: bool = True
    smoke_enabled: bool = True
    fire_confidence: float = Field(default=.12, ge=.01, le=.99)
    smoke_confidence: float = Field(default=.20, ge=.05, le=.95)
    fire_interval_sec: float = Field(default=.30, ge=.10, le=10)

    sample_fps: float = Field(default=8, ge=3, le=20)
    smoke_interval_sec: float = Field(default=.30, ge=.18, le=10)
    analysis_width: int = Field(default=800, ge=480, le=1600)
    proposal_sensitivity: float = Field(default=1.45, ge=.55, le=2.5)
    max_regions: int = Field(default=5, ge=1, le=8)

    fire_confirm_cycles: int = Field(default=2, ge=1, le=20)
    fire_confirm_sec: float = Field(default=.20, ge=0, le=30)
    smoke_confirm_cycles: int = Field(default=2, ge=1, le=20)
    smoke_confirm_sec: float = Field(default=.30, ge=0, le=30)
    clear_cycles: int = Field(default=3, ge=1, le=30)
    event_cooldown_sec: float = Field(default=30, ge=0, le=86400)
    reconnect_sec: float = Field(default=2, ge=.5, le=60)
    incense_mode: bool = True

    @field_validator('name','source')
    @classmethod
    def strip_text(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError('value cannot be blank')
        return v
