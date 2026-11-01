from typing import Literal

from pydantic import BaseModel, Field, model_validator


class DatasetRequest(BaseModel):
    name: str = Field(default="我的数据集", max_length=100)
    text: str = Field(min_length=1, max_length=32 * 1024 * 1024)
    validation_fraction: float = Field(default=0.1, ge=0.05, le=0.3)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class TokenizerRequest(BaseModel):
    dataset_id: str
    vocab_size: int = Field(default=4096, ge=258, le=16384)


class Architecture(BaseModel):
    dim: int = Field(default=128, ge=32, le=512)
    layers: int = Field(default=8, ge=1, le=16)
    heads: int = Field(default=4, ge=1, le=16)
    hidden_dim: int = Field(default=320, ge=32, le=2048)
    context_length: int = Field(default=128, ge=16, le=1024)


class RunRequest(BaseModel):
    name: str = Field(default="新训练实验", max_length=100)
    tokenizer_id: str
    architecture: Architecture = Field(default_factory=Architecture)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    max_steps: int = Field(default=500, ge=1, le=1_000_000)
    batch_size: int = Field(default=4, ge=1, le=64)
    gradient_accumulation: int = Field(default=4, ge=1, le=64)
    learning_rate: float = Field(default=0.0003, gt=0, le=0.01)
    weight_decay: float = Field(default=0.1, ge=0, le=1)
    warmup_steps: int = Field(default=50, ge=0, le=10000)
    eval_interval: int = Field(default=25, ge=1, le=10000)
    checkpoint_interval: int = Field(default=50, ge=1, le=10000)
    eval_batches: int = Field(default=4, ge=1, le=32)
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class ResumeRequest(BaseModel):
    additional_steps: int = Field(default=100, ge=1, le=1_000_000)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"


class GenerateRequest(BaseModel):
    run_id: str
    checkpoint: Literal["latest", "best"] = "best"
    prompt: str = Field(default="从前有一只小猫", max_length=10000)
    max_new_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_k: int = Field(default=40, ge=0, le=1000)
    top_p: float = Field(default=0.95, gt=0, le=1)
    repetition_penalty: float = Field(default=1.1, ge=1, le=2)
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


class CompletionRequest(BaseModel):
    model: str
    prompt: str = Field(max_length=10000)
    max_tokens: int = Field(default=100, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    stream: bool = False

    @model_validator(mode="after")
    def no_stream(self):
        if self.stream:
            raise ValueError("第一版只支持非流式 completions，请使用 stream=false。")
        return self
