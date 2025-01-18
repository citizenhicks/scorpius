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

class SamplerConfig:
    def __init__(self, model_size: str = "1B"):
        """
        Initialize SamplerConfig with specified model size.

        Args:
            model_size: One of "1B", or "3B"
        """
        self.model_size = model_size  # Store model_size as instance variable

        if self.model_size == "1B":
            """
            Configuration for the sampling strategy, including threshold values for various metrics
            and adaptive sampling parameters.
            """
            self.temperature = 0.666
            self.top_p = 0.90
            self.top_k = 27
            self.min_p = 0.03

            self.high_logits_entropy_threshold = 2.1
            self.low_logits_varentropy_threshold = 0.05
            self.low_attention_entropy_threshold = 11.915
            self.low_attention_varentropy_threshold = 0.001

        elif self.model_size == "3B":
            
            self.temperature = 0.666
            self.top_p = 0.90
            self.top_k = 27
            self.min_p = 0.03

            self.high_logits_entropy_threshold = 2.1
            self.low_logits_varentropy_threshold = 0.05
            self.low_attention_entropy_threshold = 11.915
            self.low_attention_varentropy_threshold = 0.001
            self.adaptive_score_interaction_strength_coefficient = 0.6

        else:
            raise ValueError(f"Invalid model size: {model_size}. Choose from: 1B, 3B")
