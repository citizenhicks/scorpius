import torch
import torch.nn.functional as F
from typing import Tuple, Dict

from config import SamplerConfig


LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
DEFAULT_MASK_VALUE = -1e9

def _calculate_attention_varentropy(attention_scores: torch.Tensor, current_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of attention probabilities with causal masking."""
    seq_length = attention_scores.shape[-1]
    device = attention_scores.device
    mask = torch.arange(seq_length, device=device) >= current_pos
    mask = mask.reshape(1, 1, 1, -1)
    attention_scores = torch.where(mask, torch.tensor(DEFAULT_MASK_VALUE, device=attention_scores.device), attention_scores)
    
    attention_probs = F.softmax(attention_scores, dim=-1)
    attention_probs_clamped = torch.clamp(attention_probs, 1e-10, 1.0)
    entropy = -torch.sum(attention_probs * torch.log2(attention_probs_clamped), dim=-1)
    varentropy = torch.sum(attention_probs * (torch.log2(attention_probs_clamped) + entropy.unsqueeze(-1))**2, dim=-1)
    
    return entropy, varentropy, attention_probs
    
def _calculate_varentropy_logsoftmax(logits: torch.Tensor, axis: int = -1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculate the entropy and varentropy of the probability distribution using logsoftmax."""
    log_probs = F.log_softmax(logits, dim=axis)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=axis) / LN_2  # Convert to base-2
    varentropy = torch.sum(probs * (log_probs / LN_2 + entropy.unsqueeze(-1))**2, dim=axis)
    return entropy, varentropy

def _calculate_metrics(logits: torch.Tensor, attention_scores: torch.Tensor, current_pos: int) -> Dict[str, torch.Tensor]:
    """
    Calculate various metrics from logits and attention scores.
    """
    logits_entropy, logits_varentropy = _calculate_varentropy_logsoftmax(logits)
    attn_entropy, attn_varentropy, attention_probs = _calculate_attention_varentropy(attention_scores, current_pos)
    mean_attention = torch.mean(attention_probs, dim=1)
    agreement = torch.mean(torch.abs(attention_probs - mean_attention.unsqueeze(1)), dim=(1, 2))
    interaction_strength = torch.mean(torch.abs(attention_scores), dim=(1, 2, 3))

    return {
        "logits_entropy": torch.mean(logits_entropy),
        "logits_varentropy": torch.mean(logits_varentropy),
        "attn_entropy": torch.mean(attn_entropy),
        "attn_varentropy": torch.mean(attn_varentropy),
        "agreement": torch.mean(agreement),
        "interaction_strength": interaction_strength
    }


def _hicksford_rotation(logits, entropy, varentropy, attnentropy, attnvarentropy, magnitude=False, rotation_factor_k = 0.4):


    tau = torch.sin(entropy ** varentropy) * torch.cos(attnentropy ** attnvarentropy)
    tau = torch.clip(tau, -0.7, 0.7)

    complex_tau = 1j * tau
    
    angle = torch.exp(torch.pi/2 * rotation_factor_k * complex_tau)

    # Perform the rotation
    rotated_logits = logits * angle

    if magnitude:
        return torch.abs(rotated_logits), tau
    else:
        return torch.real(rotated_logits), tau

def _multinomial_sample_one(probs_sort: torch.Tensor, generator = torch.Generator(device='mps')) -> torch.Tensor:
    """Samples one token from a multinomial distribution with sorted probabilities."""
    q = torch.rand(probs_sort.shape, generator=generator, device=probs_sort.device)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(torch.int32)


def _sample_top_k_top_p(
    bsz: torch.Tensor,
    logits: torch.Tensor,
    cfg: SamplerConfig,
    device: torch.device = torch.device('mps'),
    generator = torch.Generator(device='mps')
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    #sampling
    logit = logits[:, -1]
    probs = F.softmax(logit / cfg.temperature, dim=-1)

    # Apply min_p sampling
    if cfg.min_p > 0.0:
        p_max = torch.max(probs, dim=-1, keepdim=True).values
        indices_to_remove = probs < (cfg.min_p * p_max)
        logit = torch.where(indices_to_remove, torch.full_like(logit, float('-inf')), logit)
        probs = F.softmax(logit, dim=-1)

    # Apply top-k sampling
    top_k_probs, top_k_indices = torch.topk(probs, k=min(50, probs.shape[-1]))
    probs_sort = torch.flip(top_k_probs, dims=[-1])
    probs_idx = torch.flip(top_k_indices, dims=[-1])
    probs_sum = torch.cumsum(probs_sort, dim=-1)

    # Apply top-p sampling
    mask = torch.where(probs_sum - probs_sort > cfg.top_p, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device))
    probs_sort = probs_sort * (1 - mask)
    probs_sort = probs_sort / torch.sum(probs_sort, dim=-1, keepdim=True)
    next_token = _multinomial_sample_one(probs_sort, generator)
    next_token_g = torch.gather(probs_idx, -1, next_token.reshape(bsz, 1).to(torch.int64))

    return next_token_g

def sample(
    gen_tokens: torch.Tensor,
    logits: torch.Tensor,
    attention_scores: torch.Tensor,
    cfg: SamplerConfig,
    current_pos: int,
    device: torch.device = torch.device('mps')
    
) -> Tuple[torch.Tensor, Tuple[int, int, int], str]:
    """
    Main sampling function that selects the next token based on metrics and configuration.
    Returns the sampled token and the associated color formatting.
    """
    metrics = _calculate_metrics(logits, attention_scores, current_pos)
    ent, vent = metrics["logits_entropy"], metrics["logits_varentropy"]
    attn_ent, attn_vent = metrics["attn_entropy"], metrics["attn_varentropy"]
    magnitude = False

    if (ent > cfg.high_logits_entropy_threshold and
          vent < cfg.low_logits_varentropy_threshold and
          attn_ent < cfg.low_attention_entropy_threshold and
          attn_vent < cfg.low_attention_varentropy_threshold):
          
        magnitude = False
    
    #rotate the logits before sampling
    _, angle = _hicksford_rotation(logits, ent, vent, attn_ent, attn_vent, magnitude=magnitude)
    metrics["angle"] = angle
    metrics["magnitude"] = magnitude

    tokens = []
    generator = torch.Generator(device=device).manual_seed(1337)
    for _ in range(5):
        next_token_candidate = _sample_top_k_top_p(gen_tokens.shape[0], logits, cfg, device, generator)
        tokens.append(next_token_candidate)

    idx = torch.randint(0,4,(1,))
    return tokens[0], metrics