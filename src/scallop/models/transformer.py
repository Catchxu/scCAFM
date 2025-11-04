import math
import torch
import torch.nn as nn
from math import ceil

from .component import RMSNorm, SwiGLU


# -------------------- Efficient FWHT --------------------
def _next_power_of_two(n: int) -> int:
    """Return next power of two >= n (for padding)."""
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def fwht_torch(x: torch.Tensor) -> torch.Tensor:
    """
    Fast Walsh-Hadamard transform implemented in PyTorch using tensor ops.

    Args:
        x: Tensor with last dim size P (P must be power of two)
           shape (..., P)

    Returns:
        y: transformed tensor, same shape as x

    Notes:
        - This implementation uses O(P log P) ops and is vectorized across leading dims.
        - Works on CPU and GPU (runs on device of `x`).
    """
    orig_shape = x.shape
    P = orig_shape[-1]
    if P & (P - 1) != 0:
        raise ValueError("fwht_torch requires last dimension size to be a power of two")

    y = x
    h = 1
    # iterative butterfly: each loop doubles h
    while h < P:
        # reshape so that we can do pairwise combine
        # new shape (..., P/(2h), 2h)
        new_shape = (*y.shape[:-1], -1, 2 * h)
        y = y.reshape(new_shape)

        # split left/right
        left = y[..., :h]
        right = y[..., h:2*h]

        # compute sums and diffs
        sum_ = left + right
        diff = left - right

        # concat back [sum_, diff]
        y = torch.cat([sum_, diff], dim=-1)
        # restore last dim P
        y = y.reshape(*orig_shape[:-1], P)
        h *= 2
    return y


# -------------------- Rotary Position Encoding (RoPE) --------------------
def apply_rope(q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply Rotary Position Encoding (RoPE) to Q and K, but SKIP the first token
    (assumed to be a special condition token) so it does not receive RoPE.

    Args:
        q: (C, H, L, D)
        k: (C, H, L, D)

    Returns:
        q_rot, k_rot: (C, H, L, D)
    """
    C, H, L, D = q.shape
    if L <= 1:
        # nothing to do (only condition token present)
        return q, k

    device = q.device
    dtype = q.dtype

    # Work on tokens 1..L-1 (these are the real sequence tokens)
    q_rot = q.clone()
    k_rot = k.clone()

    # slice out the subsequence to apply RoPE to: shape (C,H,L-1,D)
    q_sub = q[..., 1:, :].to(dtype)
    k_sub = k[..., 1:, :].to(dtype)

    # split even/odd dims
    q1, q2 = q_sub[..., ::2], q_sub[..., 1::2]  # (..., D/2)
    k1, k2 = k_sub[..., ::2], k_sub[..., 1::2]

    # frequencies: shape (D/2,)
    freqs = 10000 ** (-torch.arange(0, D, 2, device=device, dtype=dtype) / D)

    # positions for the subsequence start at 0 .. L-2 (so token at index 1 gets pos 0)
    pos = torch.arange(0, L - 1, device=device, dtype=dtype).unsqueeze(1)  # (L-1, 1)
    angle = pos * freqs  # (L-1, D/2)

    # expand to (1,1,L-1,D/2) to broadcast over C,H
    cos = angle.cos()[None, None, :, :]  # (1,1,L-1,D/2)
    sin = angle.sin()[None, None, :, :]  # (1,1,L-1,D/2)

    # apply rotation to subsequence
    q_sub_even = q1 * cos - q2 * sin
    q_sub_odd  = q1 * sin + q2 * cos
    k_sub_even = k1 * cos - k2 * sin
    k_sub_odd  = k1 * sin + k2 * cos

    # interleave even/odd back to (..., D)
    q_sub_rot = torch.empty_like(q_sub)
    q_sub_rot[..., ::2] = q_sub_even
    q_sub_rot[..., 1::2] = q_sub_odd

    k_sub_rot = torch.empty_like(k_sub)
    k_sub_rot[..., ::2] = k_sub_even
    k_sub_rot[..., 1::2] = k_sub_odd

    # assign rotated subsequence back, leaving index 0 unchanged
    q_rot[..., 1:, :] = q_sub_rot
    k_rot[..., 1:, :] = k_sub_rot

    return q_rot, k_rot


# -------------------- FastAttention with Gaussian / SORF features --------------------
class FastAttention(nn.Module):
    """
    Multi-head Performer (FAVOR+) style attention supporting Gaussian random features 
    or Structured Orthogonal Random Features (SORF).
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        nb_features: int = 256,
        feature_type: str = 'sorf',
        redraw_features: bool = False,
        eps: float = 1e-6,
        use_causal: bool = False,
        use_rope: bool = False,
    ):
        """
        Args:
            embed_dim: total embedding dim (E)
            num_heads: number of heads (H)
            nb_features: number of random features (for SORF)
            feature_type: 'gaussian' or 'sorf'
            redraw_features: if True, resample random features each forward
            eps: numerical eps for denom
            use_causal: if True, causal-mask is used
            use_rope: if True, rotary position encoding (RoPE) is used
        """
        super().__init__()
        assert embed_dim % num_heads == 0, 'embed_dim must be divisible by num_heads'
        assert feature_type in ('gaussian', 'sorf'), 'feature_type must be "gaussian" or "sorf"'

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.nb_features = nb_features
        self.feature_type = feature_type
        self.redraw_features = redraw_features
        self.eps = eps
        self.use_causal = use_causal
        self.use_rope = use_rope

        # QKV and output projections
        self.to_qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim, bias=False)

        # Gaussian projection buffer (H, m, D)
        if self.feature_type == 'gaussian':
            proj = torch.randn(self.num_heads, self.nb_features, self.head_dim)
            self.register_buffer('proj_matrix', proj)

        # SORF buffers and configuration
        if self.feature_type == 'sorf':
            # pad D -> P (power of two)
            P = _next_power_of_two(self.head_dim)
            self._sorf_P = P
            # repeats per head required to reach nb_features
            self._sorf_repeats = ceil(self.nb_features / P)
            # pre-sample rademacher signs and gaussian scales
            # signs shape: (H, repeats, 3, P) values in {-1, +1}
            signs = (torch.randint(0, 2, (self.num_heads, self._sorf_repeats, 3, P), dtype=torch.float32) * 2 - 1)
            # scales: gaussian scaling per projected coordinate (H, repeats, P)
            scales = torch.randn(self.num_heads, self._sorf_repeats, P, dtype=torch.float32)
            self.register_buffer('sorf_signs', signs)
            self.register_buffer('sorf_scales', scales)

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (C, L, E) -> (C, H, L, D)
        """
        C, L, E = x.shape
        H = self.num_heads
        D = self.head_dim
        return x.view(C, L, H, D).permute(0, 2, 1, 3).contiguous()  # (C, H, L, D)

    def _feature_map_gaussian(self, x: torch.Tensor, proj_matrix: torch.Tensor) -> torch.Tensor:
        """
        Gaussian random feature map.

        Args:
            x: (C, H, L, D)
            proj_matrix: (H, m, D)
            returns phi: (C, H, L, m)
        """
        # squared norm term for numerical stability
        x_norm_sq = (x * x).sum(dim=-1, keepdim=True) / 2.0  # (C,H,L,1)
        # projections: einsum 'chld,hmd->chlm'
        projected = torch.einsum('chld,hmd->chlm', x, proj_matrix)
        phi = torch.exp(projected - x_norm_sq) / math.sqrt(self.nb_features)
        return phi

    def _feature_map_sorf(self, x: torch.Tensor, signs: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """
        SORF feature map using vectorized FWHT with proper scaling and numerical stability.

        Args:
            x: (C, H, L, D) tensor, input per-head embeddings
            signs: (H, repeats, 3, P), pre-sampled Rademacher signs
            scales: (H, repeats, P), pre-sampled Gaussian scales

        Returns:
            phi: (C, H, L, m) feature map tensor, where m = nb_features
        """
        C, H, L, D = x.shape
        P = self._sorf_P
        repeats = self._sorf_repeats
        device = x.device
        dtype = x.dtype

        # pad last dimension to P if needed
        if D < P:
            pad = torch.zeros((C, H, L, P - D), device=device, dtype=dtype)
            x_p = torch.cat([x, pad], dim=-1)  # (C,H,L,P)
        else:
            x_p = x  # (C,H,L,P)

        blocks = []
        for r in range(repeats):
            # pick signs and scales for this repeat
            s1 = signs[:, r, 0, :].to(device)  # (H,P)
            s2 = signs[:, r, 1, :].to(device)
            s3 = signs[:, r, 2, :].to(device)
            g  = scales[:, r, :].to(device) / math.sqrt(P)  # scale properly

            # broadcast to (C,H,L,P)
            s1_b = s1.unsqueeze(0).unsqueeze(2)
            s2_b = s2.unsqueeze(0).unsqueeze(2)
            s3_b = s3.unsqueeze(0).unsqueeze(2)
            g_b  = g.unsqueeze(0).unsqueeze(2)

            # three-stage FWHT
            t1 = fwht_torch(x_p * s1_b) / math.sqrt(P)
            t2 = fwht_torch(t1 * s2_b) / math.sqrt(P)
            t3 = fwht_torch(t2 * s3_b) / math.sqrt(P)

            # scale
            block = t3 * g_b
            blocks.append(block)

        # concatenate repeats along last dim
        phi = torch.cat(blocks, dim=-1)  # (C,H,L,P*repeats)
        phi = phi[:, :, :, :self.nb_features]  # crop to nb_features if needed

        # compute per-token norm term
        x_norm_sq = (x * x).sum(-1, keepdim=True) / 2.0  # (C,H,L,1)

        # positive feature map for softmax kernel
        phi = torch.exp(phi - x_norm_sq) / math.sqrt(self.nb_features)

        return phi  # (C,H,L,nb_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, L, E)

        Returns:
            out: (C, L, E)
        """
        if x.dim() != 3:
            raise ValueError("x must be shape (C, L, E)")

        C, L, E = x.shape

        # qkv -> (C, L, 3E)
        qkv = self.to_qkv(x)
        q, k, v = qkv.split(self.embed_dim, dim=-1)
        q = self._reshape_to_heads(q)  # (C,H,L,D)
        k = self._reshape_to_heads(k)
        v = self._reshape_to_heads(v)

        # use RoPE or not
        if self.use_rope:
            q, k = apply_rope(q, k)

        # choose feature map type
        if self.feature_type == 'gaussian':
            if self.redraw_features:
                proj = torch.randn(self.num_heads, self.nb_features, self.head_dim, device=x.device)
            else:
                proj = self.proj_matrix.to(x.device)
            q_phi = self._feature_map_gaussian(q, proj)
            k_phi = self._feature_map_gaussian(k, proj)
        else:  # sorf
            if self.redraw_features:
                P = self._sorf_P
                repeats = self._sorf_repeats
                signs = (torch.randint(0, 2, (self.num_heads, repeats, 3, P), device=x.device, dtype=torch.float32) * 2 - 1)
                scales = torch.randn(self.num_heads, repeats, P, device=x.device, dtype=torch.float32)
            else:
                signs = self.sorf_signs.to(x.device)
                scales = self.sorf_scales.to(x.device)
            q_phi = self._feature_map_sorf(q, signs, scales)
            k_phi = self._feature_map_sorf(k, signs, scales)

        # use causal-mask or not
        if self.use_causal:
            # prefix sum along sequence length L
            kv = k_phi.unsqueeze(-1) * v.unsqueeze(-2)  # (C,H,L,m,D)
            S = kv.cumsum(dim=2)  # cumulative sum over L
            sum_k = k_phi.cumsum(dim=2)  # (C,H,L,m)
            numer = torch.einsum('chlm,chlmd->chld', q_phi, S)
            denom = torch.einsum('chlm,chlm->chl', q_phi, sum_k)
        else:
            S = torch.einsum('chlm,chld->chmd', k_phi, v)
            sum_k = k_phi.sum(dim=2)
            numer = torch.einsum('chlm,chmd->chld', q_phi, S)
            denom = torch.einsum('chlm,chm->chl', q_phi, sum_k)

        denom = denom.clamp(min=self.eps)
        out_heads = numer / denom.unsqueeze(-1)  # (C,H,L,D)

        out = out_heads.permute(0, 2, 1, 3).contiguous().view(C, L, E)
        out = self.to_out(out)
        return out


# -------------------- Transformer based on FastAttention --------------------
class SwiGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = RMSNorm(embed_dim)
        self.swiglu = SwiGLU(embed_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm(x)
        out = self.swiglu(x_norm)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        nb_features: int = 256,
        feature_type: str = 'sorf',
        redraw_features: bool = False,
        eps: float = 1e-6,
        use_causal: bool = False,
        use_rope: bool = False,
        ff_hidden_mult: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attn = FastAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            nb_features=nb_features,
            feature_type=feature_type,
            redraw_features=redraw_features,
            eps=eps,
            use_causal=use_causal,
            use_rope=use_rope
        )
        self.norm = RMSNorm(embed_dim)
        self.ff = SwiGLUFFN(embed_dim, ff_hidden_mult * embed_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm(x))
        out = x + self.ff(x)
        return out


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        use_causal: bool = False,
        use_rope: bool = False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads, 
                use_causal=use_causal,
                use_rope=use_rope,
                **kwargs
            )
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x




if __name__ == '__main__':
    # Configuration
    embed_dim = 16
    seq_len = 10
    batch_size = 2
    num_layers = 2
    num_heads = 4

    # Random input tensor (batch_size, seq_len, embed_dim)
    x = torch.randn(batch_size, seq_len, embed_dim)

    print("=== Test 1: Transformer without causal mask and RoPE ===")
    model1 = Transformer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_causal=False,
        use_rope=False
    )
    out1 = model1(x)
    print("Output shape:", out1.shape)
    print("Output tensor:\n", out1)

    print("\n=== Test 2: Transformer with causal mask and RoPE ===")
    model2 = Transformer(
        embed_dim=embed_dim,
        num_layers=num_layers,
        num_heads=num_heads,
        use_causal=True,
        use_rope=True
    )
    out2 = model2(x)
    print("Output shape:", out2.shape)
    print("Output tensor:\n", out2)
