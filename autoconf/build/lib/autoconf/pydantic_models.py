from pydantic import BaseModel, Field
from typing import Literal, Optional

class JobConfig(BaseModel):
    model_name: str
    method: str
    number_gpus: int = Field(..., ge=1, description="Number of GPUs used")
    gpu_model: str
    tokens_per_sample: int = Field(..., ge=1, description="Max sequence length")
    batch_size: int = Field(..., ge=1)
    is_valid: int = Field(default=0, description="True if job was successful. It is not used for prediction purposes")

    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt-neo-2.7B",
                "method": "LoRA",
                "number_gpus": 4,
                "gpu_model": "A100",
                "tokens_per_sample": 2048,
                "batch_size": 32,
                "is_valid": True
            }
        }
