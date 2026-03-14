import torch
import torch.nn as nn

from .attention import MaskedMHA
from .utils import RMSNorm, SwiGLU


class TransformerLayer(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ffn_hidden_dim=None,
        use_rotary=False,
        dropout=0.0,
        res_scale=1.0,
    ):
        super().__init__()

        self.res_scale = res_scale

        self.attn = MaskedMHA(
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_rotary=use_rotary,
            attn_dropout=dropout
        )

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)

        if ffn_hidden_dim is None:
            ffn_hidden_dim = int(8 * embed_dim / 3)

        self.ffn = nn.Sequential(
            SwiGLU(embed_dim, ffn_hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_dim, embed_dim),
        )

    def forward(self, x, key_padding_mask=None, causal=False):

        h = self.norm1(x)
        h = self.attn(h, key_padding_mask=key_padding_mask, causal=causal)
        x = x + self.res_scale * h

        h = self.norm2(x)
        h = self.ffn(h)
        x = x + self.res_scale * h

        return x


class TomoEncoder(nn.Module):
    """
    Stack multiple TransformerLayers.
    
    Input:
        x: (C, L, E)
        key_padding_mask: (C, L)
    
    Output:
        x: (C, L, E)
    """
    def __init__(
        self,
        num_layers,
        embed_dim,
        num_heads,
        ffn_hidden_dim=None,
        use_rotary=False,
        **kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_hidden_dim=ffn_hidden_dim,
                use_rotary=use_rotary
            )
            for _ in range(num_layers)
        ])

        self.norm_out = RMSNorm(embed_dim)

    def forward(self, x, key_padding_mask=None, causal=False):
        for layer in self.layers:
            x = layer(
                x,
                key_padding_mask=key_padding_mask,
                causal=causal
            )

        # optional final norm
        x = self.norm_out(x)
        return x




if __name__ == "__main__":
    C, L, E = 2, 128, 256

    x = torch.randn(C, L, E)
    pad = torch.zeros(C, L, dtype=torch.bool)   # no padding

    encoder = TomoEncoder(
        num_layers=4,
        embed_dim=E,
        num_heads=8,
        ffn_hidden_dim=4*E,
        use_rotary=True
    )

    y = encoder(x, key_padding_mask=pad)
    print("Output:", y.shape)
