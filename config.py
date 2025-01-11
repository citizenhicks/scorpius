from typing import NamedTuple
from enum import Enum
from dataclasses import dataclass

@dataclass
class ModelConfig:
    """Configuration class for model parameters."""
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    norm_eps: float
    rope_theta: float
    use_scaled_rope: bool
    max_seq_len: int
    model_size: str

# Define configurations for different models and model sizes
MODEL_CONFIGS = {
    "3B": ModelConfig(
        dim=3072,
        n_layers=28,
        n_heads=24,
        n_kv_heads=8,
        vocab_size=128256,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        max_seq_len=8192,
        model_size="3B"
    ),
    "1B": ModelConfig(
        dim=2048,
        n_layers=16,
        n_heads=32,
        n_kv_heads=8,
        vocab_size=128256,
        norm_eps=1e-05,
        rope_theta=500000.0,
        use_scaled_rope=True,
        max_seq_len=4096,
        model_size="1B"
    )
}

MODEL_IDS = {
    "1B": "meta-llama/Llama-3.2-1B-Instruct",
    "3B": "meta-llama/Llama-3.2-3B-Instruct"

}

MODEL_PATHS = {
    "1B": "data/1B",
    "3B": "data/3B"
}

class ModelParams(NamedTuple):
  n_layers: int
  n_local_heads: int
  n_local_kv_heads: int
  head_dim: int
  max_seq_len: int
  rope_theta: float
  use_scaled_rope: bool

def get_model_params(config: ModelConfig) -> ModelParams:
    """Create ModelParams from config."""
    return ModelParams(
        n_layers=config.n_layers,
        n_local_heads=config.n_heads,
        n_local_kv_heads=config.n_kv_heads,
        head_dim=config.dim // config.n_heads,
        max_seq_len=config.max_seq_len,
        rope_theta=config.rope_theta,
        use_scaled_rope=config.use_scaled_rope
    )