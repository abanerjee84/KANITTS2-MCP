from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_REFERENCE_BYTES = 32 * 1024 * 1024


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    text: str = Field(min_length=1, max_length=12000)
    voice_id: str | None = Field(default=None, pattern=r"^[0-9a-f]{32}$")
    language_tag: Literal[
        "en_us", "en_nyork", "en_oakl", "en_glasg", "en_bost", "en_scou"
    ] = "en_us"
    temperature: float = Field(default=1.0, ge=0.1, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)
    max_new_tokens: int = Field(default=2500, ge=100, le=3000)
    seed: int | None = Field(default=None, ge=-1, le=2**32 - 1)
    client_request_id: str | None = Field(default=None, min_length=1, max_length=128)

    @field_validator("text")
    @classmethod
    def nonempty(cls, value):
        if not value.strip():
            raise ValueError("Text must contain speech")
        if any(marker in value for marker in ("<start_of_", "<end_of_", "<pad>")):
            raise ValueError(
                "Input must be plain speech text without model control tokens"
            )
        return value.strip()
