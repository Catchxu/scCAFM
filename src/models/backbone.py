from __future__ import annotations

import torch
import torch.nn as nn

from .attention import FlashMHA


class SwiGLUMLP(nn.Module):
    """
    SwiGLU feed-forward network.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.gate_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(embed_dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = torch.nn.functional.silu(self.gate_proj(x))
        value = self.up_proj(x)
        out = gate * value
        out = self.down_proj(out)
        out = self.dropout(out)
        return out


class TransformerBlock(nn.Module):
    """
    Pre-norm Transformer block with FlashAttention and SwiGLU MLP.

    Shape convention:
    - `L`: full sequence length
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_hidden_dim: int,
        mlp_dropout: float = 0.1,
        attention_backend: str = "fa4",
        use_rotary: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        norm_eps: float = 1e-6,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.RMSNorm(embed_dim, eps=norm_eps)
        self.attn = FlashMHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            attention_backend=attention_backend,
            use_rotary=use_rotary,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            rotary_base=rotary_base,
            rotary_interleaved=rotary_interleaved,
        )
        self.mlp_norm = nn.RMSNorm(embed_dim, eps=norm_eps)
        self.mlp = SwiGLUMLP(
            embed_dim=embed_dim,
            hidden_dim=mlp_hidden_dim,
            dropout=mlp_dropout,
            bias=mlp_bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.attn_norm(x),
            key_padding_mask=key_padding_mask,
            causal=causal,
        )
        x = x + self.mlp(self.mlp_norm(x))

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x


class TransformerBackbone(nn.Module):
    """
    Transformer backbone built from FlashAttention blocks.

    Shape convention:
    - `L`: full sequence length
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_hidden_dim: int | None = None,
        mlp_dropout: float = 0.1,
        attention_backend: str = "fa4",
        use_rotary: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        mlp_bias: bool = True,
        norm_eps: float = 1e-6,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError(f"`num_layers` must be positive, got {num_layers}.")

        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.mlp_hidden_dim = mlp_hidden_dim or (4 * embed_dim)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    mlp_dropout=mlp_dropout,
                    attention_backend=attention_backend,
                    use_rotary=use_rotary,
                    qkv_bias=qkv_bias,
                    out_bias=out_bias,
                    mlp_bias=mlp_bias,
                    norm_eps=norm_eps,
                    rotary_base=rotary_base,
                    rotary_interleaved=rotary_interleaved,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.RMSNorm(embed_dim, eps=norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"`x` must have shape (B, L, D), got {tuple(x.shape)}.")
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                f"Expected input last dim {self.embed_dim}, got {x.shape[-1]}."
            )

        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask, causal=causal)

        x = self.final_norm(x)

        if key_padding_mask is not None:
            x = x.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return x
