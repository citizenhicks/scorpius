import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Adjust the import paths below to match your code layout
from config import ModelParams, get_model_params
from weights import LayerWeights,load_weights
from kvcache import KVCache
from metrics import AttnStats
from tokenizer import Tokenizer

# Use a large negative for masked-out positions
DEFAULT_MASK_VALUE = -0.7 * float(torch.finfo(torch.float32).max)

class Model(nn.Module):
    """
    Transformer model class that:
      - Applies token embeddings
      - Computes multi-layer self-attention & feed-forward
      - Returns final logits, updated key/value cache, final attention scores,
        and attention statistics (entropies, varentropies, etc.).
    """

    def __init__(
        self,
        model_id: str,
        params: ModelParams,
        device: torch.device,
        tokeniser:Tokenizer
    ):
        """
        Args:
            model_id (str): 
            params (ModelParams): Contains hyperparams (n_layers, n_heads, etc.).
            device: 

        """
        super().__init__()
        self.model_id = model_id
        self.params = get_model_params(params)
        self.device = device
        self.weights = load_weights(model_id=self.model_id, device= self.device)
        self.batch_size = 1
        self.kvcache = self.initialize_kvcache()
        self.attn_mask = None
        self.tokeniser=tokeniser

    
    def getModelId():
        return self.model_id
    
    def getParams():
        return self.params

    def getDevice():
        return self.device

    def getKVCache():
        return self.kvcache

    def tokenize(self, prompt: str, chat_template: bool = False) -> torch.Tensor:
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant. Think step by step and show your reasoning before you answer.",
            },
            {"role": "user", "content": prompt},
        ]
        if chat_template:
            prompt = self._generate_chat_prompt(messages)
        tokens = self.tokeniser.encode(prompt, bos=False, eos=False, allowed_special="all")
        return torch.tensor([tokens], dtype=torch.long).to(self.device)

    def detokenize(self, next_token) -> torch.Tensor:
        return self.tokeniser.decode([next_token.item()])

    def initialize_kvcache(self) -> KVCache:

        return KVCache.new(
            layers = self.params.n_layers,
            bsz = self.batch_size,
            max_seq_len =self.params.max_seq_len,
            kv_heads = self.params.n_local_kv_heads,
            head_dim = self.params.head_dim,
            device = self.device
        )

    def build_attn_mask(self, seq_len: int, cur_pos: int) -> torch.Tensor:
        mask = None
        if seq_len > 1:
            mask = torch.full((seq_len, seq_len), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seq_len, cur_pos)), mask]).to(torch.float32).to(self.device)
        self.attn_mask = mask
        return 

    def precompute_freqs_cis(self) -> torch.Tensor:
        head_dim = self.params.head_dim
        max_seq_len = self.params.max_seq_len
        rope_theta = self.params.rope_theta
        use_scaled_rope = self.params.use_scaled_rope

        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, device=self.device).bfloat16() [: (head_dim // 2)] / head_dim))
        t = torch.arange(max_seq_len, device=self.device).bfloat16().unsqueeze(1)
        freqs = inv_freq.unsqueeze(0)
        freqs = t * freqs 
        return torch.exp(1j * freqs)


    def forward(
        self,
        tokens: torch.Tensor,
        cur_pos: int,
        freqs_cis: torch.Tensor
    ):
        """
        Runs the forward pass over the entire Transformer.

        Args:
            tokens (torch.Tensor):
                Input token IDs, shape (batch_size, seq_len).
            cur_pos (int):
                Current sequence position for KVCache updates.
            freqs_cis (torch.Tensor):
                Precomputed rotary embeddings, shape (seq_len, dim//2).
            kvcache (KVCache):
                Key/value cache for reusing past attention states.
            attn_mask (torch.Tensor, optional):
                Attention mask, e.g. for prefix blocking. Defaults to None.

        Returns:
            Tuple[logits, kvcache, final_scores, attn_stats]:
                - logits (torch.Tensor): (batch_size, seq_len, vocab_size).
                - kvcache (KVCache): The updated key/value cache.
                - final_scores (torch.Tensor): The final layer's raw attention
                  scores (pre-softmax), shape (bsz, n_heads, seq_len, total_k_len).
                - attn_stats (AttnStats): Aggregated attention entropy/varentropy.
        """
        # 1) Token embeddings
        h = self.weights.tok_embeddings[tokens]  # (bsz, seq_len, dim)

        # Prepare an AttnStats object to track cross-layer attention metrics
        attn_stats = AttnStats.new(
            bsz=tokens.shape[0],
            n_layers=self.params.n_layers,
            n_heads=self.params.n_local_heads,
            device=self.device
        )

        final_scores = None

        # 2) Pass through each Transformer layer
        for i in range(self.params.n_layers):
            layer_w = self.weights.layer_weights[i]

            # -- Self-Attention Sub-Layer --
            norm_x = self._rms_norm(h, layer_w.attention_norm)
            h_attn, scores = self._attention(
                norm_x, layer_w, cur_pos, i, freqs_cis
            )
            # Update attention stats using the new token's distribution
            attn_stats = attn_stats.update(scores[:, :, -1, :], i)

            # Residual connection
            h = h + h_attn
            final_scores = scores  # Keep track of the last layer's raw scores

            # -- Feed-Forward Sub-Layer --
            ff_norm_x = self._rms_norm(h, layer_w.ffn_norm)
            h = h + self._feed_forward(ff_norm_x, layer_w)

        # 3) Final RMS norm + linear projection => logits
        norm_h = self._rms_norm(h, self.weights.norm)
        logits = F.linear(norm_h, self.weights.output)

        return logits, final_scores, attn_stats

    # --------------------------------------------------------------------------
    # Internal helper methods
    # --------------------------------------------------------------------------
    def _rms_norm(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        eps: float = 1e-5
    ) -> torch.Tensor:
        """
        RMS LayerNorm: normalizes each token vector by its RMS, then scales by 'w'.
        """
        # x shape: (bsz, seq_len, dim)
        return w * (x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + eps))

    def _apply_rotary_emb(
        self,
        xq: torch.Tensor,
        xk: torch.Tensor,
        freqs_cis: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        """
        Applies rotary embeddings to Q and K.  
        For each 2D slice, treat them as complex numbers and multiply by e^{iθ}.

        xq, xk shapes: (bsz, seq_len, n_heads, head_dim)
        freqs_cis shape: (seq_len, head_dim//2)

        Returns the modified (xq, xk).
        """
        # Reshape so final dimension has 2 => (real, imag)
        reshape_xq = xq.float().reshape(*xq.shape[:-1], -1, 2)
        reshape_xk = xk.float().reshape(*xk.shape[:-1], -1, 2)

        # Convert to complex
        xq_c = torch.complex(reshape_xq[..., 0], reshape_xq[..., 1])
        xk_c = torch.complex(reshape_xk[..., 0], reshape_xk[..., 1])

        # Broadcast freqs_cis up to match batch & heads
        freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2)  # shape => (1, seq_len, 1, head_dim//2)
        xq_out = xq_c * freqs_cis
        xk_out = xk_c * freqs_cis

        # Reshape back to real & imag
        xq_out = torch.stack((xq_out.real, xq_out.imag), dim=-1).reshape(*xq_out.shape[:-1], -1)
        xk_out = torch.stack((xk_out.real, xk_out.imag), dim=-1).reshape(*xk_out.shape[:-1], -1)

        return xq_out.to(xq.dtype), xk_out.to(xk.dtype)

    def _attention(
        self,
        x: torch.Tensor,
        layer_w: LayerWeights,
        cur_pos: int,
        layer_idx: int,
        freqs_cis: torch.Tensor,
    ):
        """
        Single transformer layer self-attention.

        Args:
            x (torch.Tensor): Input features, shape (bsz, seq_len, dim).
            layer_w (LayerWeights): Weights for this layer.
            cur_pos (int): Current pos in the sequence, used for caching.
            layer_idx (int): Which layer we're on, used for indexing cache.
            freqs_cis (torch.Tensor): Rotary embedding data.
            kvcache (KVCache): Cache of keys & values.
            attn_mask (torch.Tensor, optional): If provided, used for prefix blocking.

        Returns:
            (out, kvcache, pre_scores):
                - out (torch.Tensor): Updated representation after attention.
                - kvcache (KVCache): Updated key/value cache.
                - pre_scores (torch.Tensor): Raw attention scores (before softmax).
        """
        bsz, seqlen, _ = x.shape
        mp = self.params

        # number of heads and ratio between n_local_heads and n_local_kv_heads
        n_rep = mp.n_local_heads // mp.n_local_kv_heads

        # Project Q, K, V
        xq = F.linear(x, layer_w.wq).reshape(bsz, seqlen, mp.n_local_heads, mp.head_dim)
        xk = F.linear(x, layer_w.wk).reshape(bsz, seqlen, mp.n_local_kv_heads, mp.head_dim)
        xv = F.linear(x, layer_w.wv).reshape(bsz, seqlen, mp.n_local_kv_heads, mp.head_dim)

        # Apply rotary embeddings
        xq, xk = self._apply_rotary_emb(xq, xk, freqs_cis)

        # Update or retrieve from the KV cache
        keys, values = self.kvcache.update(xk, xv, layer_idx, cur_pos, n_rep)

        # Prepare shapes for matmul
        # xq => (bsz, n_heads, seq_len, head_dim)
        xq = xq.permute(0, 2, 1, 3)
        # keys => (bsz, n_heads, head_dim, cache_len + seq_len)
        keys = keys.permute(0, 2, 3, 1)
        # values => (bsz, n_heads, cache_len + seq_len, head_dim)
        values = values.permute(0, 2, 1, 3)

        # Calculate raw attention scores: QK^T / sqrt(d)
        scores = torch.matmul(xq, keys.to(xq.dtype))
        pre_scores = scores / math.sqrt(mp.head_dim)  # shape => (bsz, n_heads, seq_len, total_k_len)

        # Always perform attention computations in float32
        scores = pre_scores.float()

        # Optionally apply an attention mask (e.g. causal or prefix).
        if cur_pos == 0:
            scores = scores + self.attn_mask

        # Where scores are 0.0, replace with a large negative to exclude them
        mask = torch.where(scores != 0.0, scores, DEFAULT_MASK_VALUE)
        padded_logits = torch.where(
            (mask >= DEFAULT_MASK_VALUE * 0.5),
            scores,
            DEFAULT_MASK_VALUE
        )

        # Softmax along the last dimension
        probs = F.softmax(padded_logits, dim=-1)

        # Multiply by the values
        out = torch.matmul(probs.to(torch.float32), values.to(torch.float32))

        # Reshape back to (bsz, seq_len, dim)
        out = out.transpose(1, 2).reshape(bsz, seqlen, -1).to(x.dtype)
        # Final linear projection
        out = F.linear(out, layer_w.wo)

        return out, pre_scores

    def _feed_forward(self, x: torch.Tensor, layer_w: LayerWeights) -> torch.Tensor:
        """
        Standard feed-forward block with a SiLU activation & gating projection.
        w1, w3 => gating + up-projection, w2 => down-projection
        """
        return F.linear(
            F.silu(F.linear(x, layer_w.w1)) * F.linear(x, layer_w.w3),
            layer_w.w2
        )