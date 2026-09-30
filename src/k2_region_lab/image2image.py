"""Independent, serializable settings for the Krea2Edit instruction pipeline."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from k2_region_lab.sampling import validate_sampler, validate_scheduler


class Image2ImageSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    prompt: str = Field(default="", max_length=32768)
    source_name: str = Field(default="", max_length=191)
    identity_lora_name: str = Field(default="", max_length=191)
    width: int = Field(default=1024, ge=256, le=2048)
    height: int = Field(default=1024, ge=256, le=2048)
    steps: int = Field(default=8, ge=1, le=100)
    seed: int = Field(default=0, ge=0, le=2147483647)
    cfg: float = Field(default=1.0, ge=1.0, le=10.0)
    sampler: str = "euler"
    scheduler: str = "simple"
    lora_strength: float = Field(default=1.0, gt=0.0, le=2.0)
    grounding_px: int = Field(default=768, ge=384, le=1536)
    ref_boost: float = Field(default=1.0, ge=0.0, le=10.0)
    fit_mode: Literal["fit", "crop"] = "fit"

    @model_validator(mode="after")
    def validate_geometry(self):
        if self.width % 32 or self.height % 32:
            raise ValueError("image2image dimensions must be divisible by 32")
        if self.width * self.height > 2 * 1024 * 1024:
            raise ValueError("image2image output must not exceed 2 megapixels")
        validate_sampler(self.sampler)
        validate_scheduler(self.scheduler)
        return self
