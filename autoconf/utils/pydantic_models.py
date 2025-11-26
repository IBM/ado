from typing import Optional
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from autoconf.utils.config_mapper import map_valid_model_name


class JobConfig(BaseModel):
    model_name: str
    method: str
    gpu_model: str
    tokens_per_sample: int = Field(..., ge=1, description="Max sequence length")
    batch_size: int = Field(..., ge=1)
    is_valid: Optional[int] = Field(
        default=None,
        description="Ground truth. True if job was successful. It is not used for prediction purposes",
    )
    number_gpus: Optional[int] = Field(
        default=None, ge=1, description="Number of GPUs used"
    )

    #TODO(srikumarv): the below is using validator for the mapping but
    #it needs debugging. Turning off for now
    @field_validator('model_name', mode='wrap')
    @classmethod
    def map_to_valid_model(cls, model_name: str, info: ValidationInfo) -> str:
        """ Map the model name in the input to a valid model name if possible"""
        return map_valid_model_name(model_name=model_name)
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "gpt-neo-2.7B",
                "method": "lora",
                "number_gpus": 4,
                "gpu_model": "NVIDIA-A100-SXM4-80GB",
                "tokens_per_sample": 2048,
                "batch_size": 32,
            }
        }
