from __future__ import annotations

import warnings

import pandas as pd
import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Optional

from ..assets import load_vocab_json
from .backbone import TransformerBackbone
from .embedding import ScEmbedding
from .router import GeneRouter


@dataclass(slots=True)
class FactorState:
    """
    Lightweight container for the two factor-assignment tensors.

    Shape convention:
    - `G`: gene-token length
    - `M`: number of latent factors

    Tensor shapes:
    - `u`: (C, G, M)
    - `v`: (C, G, M)
    """

    u: torch.Tensor
    v: torch.Tensor


class SFM(nn.Module):
    """
    Structure foundation model for cell-specific GRN inference.

    Shape convention:
    - `G`: gene-token length
    - `L`: full sequence length, where `L = G + 1`
    - `M`: number of latent factors
    """

    def __init__(
        self,
        token_dict: pd.DataFrame,
        cond_vocab_size: int = 4096,
        embed_dim: int = 512,
        expr_hidden_dim: int = 256,
        expr_dropout: float = 0.1,
        expr_value_scale: float = 1.0,
        mod_hidden_dim: int = 256,
        mod_dropout: float = 0.1,
        cond_dropout: float = 0.1,
        context_hidden_dim: int = 256,
        context_dropout: float = 0.1,
        embedding_dropout: float = 0.1,
        gene_embedding_ckpt: Optional[str] = None,
        freeze_loaded_gene_embeddings: bool = False,
        num_layers: int = 4,
        num_heads: int = 8,
        backbone_mlp_hidden_dim: Optional[int] = None,
        attn_dropout: float = 0.1,
        backbone_mlp_dropout: float = 0.1,
        use_rotary: bool = False,
        qkv_bias: bool = True,
        out_bias: bool = True,
        backbone_mlp_bias: bool = True,
        norm_eps: float = 1e-6,
        rotary_base: float = 10000.0,
        rotary_interleaved: bool = False,
        num_factors: int = 256,
        router_hidden_dim: int = 128,
        router_dropout: float = 0.1,
        topk: Optional[int] = 32,
        router_temperature: float = 1.0,
        beta_momentum: float = 1.0,
    ) -> None:
        super().__init__()

        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        if num_factors <= 0:
            raise ValueError(f"`num_factors` must be positive, got {num_factors}.")

        self.token_dict = token_dict
        self.embed_dim = embed_dim
        self.num_factors = num_factors

        self.embedding = ScEmbedding(
            token_dict=token_dict,
            cond_vocab_size=cond_vocab_size,
            embed_dim=embed_dim,
            expr_hidden_dim=expr_hidden_dim,
            expr_dropout=expr_dropout,
            expr_value_scale=expr_value_scale,
            mod_hidden_dim=mod_hidden_dim,
            mod_dropout=mod_dropout,
            cond_dropout=cond_dropout,
            context_hidden_dim=context_hidden_dim,
            context_dropout=context_dropout,
            embedding_dropout=embedding_dropout,
            gene_embedding_ckpt=gene_embedding_ckpt,
            freeze_loaded_gene_embeddings=freeze_loaded_gene_embeddings,
        )
        self.backbone = TransformerBackbone(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_hidden_dim=backbone_mlp_hidden_dim,
            attn_dropout=attn_dropout,
            mlp_dropout=backbone_mlp_dropout,
            use_rotary=use_rotary,
            qkv_bias=qkv_bias,
            out_bias=out_bias,
            mlp_bias=backbone_mlp_bias,
            norm_eps=norm_eps,
            rotary_base=rotary_base,
            rotary_interleaved=rotary_interleaved,
        )
        self.tf_router = GeneRouter(
            embed_dim=embed_dim,
            num_factors=num_factors,
            hidden_dim=router_hidden_dim,
            dropout=router_dropout,
            topk=topk,
            temperature=router_temperature,
            beta_momentum=beta_momentum,
        )
        self.tg_router = GeneRouter(
            embed_dim=embed_dim,
            num_factors=num_factors,
            hidden_dim=router_hidden_dim,
            dropout=router_dropout,
            topk=topk,
            temperature=router_temperature,
            beta_momentum=beta_momentum,
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _validate_token_batch(tokens: dict[str, torch.Tensor | None]) -> None:
        required_keys = {"input_ids", "expression_values", "condition_ids", "non_tf_mask"}
        missing = [key for key in required_keys if key not in tokens]
        if missing:
            raise KeyError(f"Missing required token entries: {missing}.")

    def forward(
        self,
        tokens: dict[str, torch.Tensor | None],
        return_factors: bool = True,
        compute_grn: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor | None, FactorState]:
        if not compute_grn and not return_factors:
            warnings.warn(
                "`SFM.forward()` was called with both `compute_grn=False` and "
                "`return_factors=False`, so it will return `None`.",
                stacklevel=2,
            )

        self._validate_token_batch(tokens)

        input_ids = tokens["input_ids"]
        expression_values = tokens["expression_values"]
        condition_ids = tokens["condition_ids"]
        non_tf_mask = tokens["non_tf_mask"]
        padding_mask = tokens.get("padding_mask")

        if not torch.is_tensor(input_ids):
            raise TypeError("`tokens['input_ids']` must be a torch.Tensor.")
        if not torch.is_tensor(expression_values):
            raise TypeError("`tokens['expression_values']` must be a torch.Tensor.")
        if not torch.is_tensor(condition_ids):
            raise TypeError("`tokens['condition_ids']` must be a torch.Tensor.")
        if not torch.is_tensor(non_tf_mask):
            raise TypeError("`tokens['non_tf_mask']` must be a torch.Tensor.")
        if padding_mask is not None and not torch.is_tensor(padding_mask):
            raise TypeError("`tokens['padding_mask']` must be a torch.Tensor or None.")

        if non_tf_mask.dtype != torch.bool:
            non_tf_mask = non_tf_mask.to(torch.bool)
        if padding_mask is not None and padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(torch.bool)

        embedding_output = self.embedding(
            input_ids=input_ids,
            expression_values=expression_values,
            condition_ids=condition_ids,
            padding_mask=padding_mask,
            non_tf_mask=non_tf_mask,
        )
        hidden_states = self.backbone(
            embedding_output.embeddings,
            key_padding_mask=embedding_output.key_padding_mask,
            causal=False,
        )

        u = self.tf_router(
            hidden_states,
            non_tf_mask=non_tf_mask,
            padding_mask=padding_mask,
        )
        v = self.tg_router(
            hidden_states,
            non_tf_mask=None,
            padding_mask=padding_mask,
        )

        if u.shape != v.shape:
            raise ValueError(
                "Expected `u` and `v` to share shape (C, G, M), got "
                f"{tuple(u.shape)} and {tuple(v.shape)}."
            )

        if compute_grn:
            grn = torch.einsum("cim,cjm->cij", u, v)
        else:
            grn = None

        if return_factors:
            return grn, FactorState(u=u, v=v)
        else:
            return grn




if __name__ == "__main__":
    from pathlib import Path

    import numpy as np
    from anndata import AnnData

    from ..data.tokenizer import ScTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run the SFM smoke test.")

    root_dir = Path(__file__).resolve().parents[2]
    device = torch.device("cuda")
    token_dict = load_vocab_json(root_dir / "assets" / "vocab.json")
    human_tfs = pd.read_csv(root_dir / "assets" / "human_tfs.csv")
    mouse_tfs = pd.read_csv(root_dir / "assets" / "mouse_tfs.csv")

    obs = pd.DataFrame(
        {
            "platform": ["10X", "smart-seq"],
            "species": ["human", "mouse"],
            "tissue": ["lung", "brain"],
            "disease": ["healthy", "tumor"],
        },
        index=["cell_0", "cell_1"],
    )
    var_names = pd.Index(["AATF", "TSPAN6", "TNMD"])
    adata = AnnData(
        X=np.array([[1.0, 2.0, 0.0], [3.0, 0.0, 4.0]], dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=var_names),
    )

    tokenizer = ScTokenizer(
        token_dict=token_dict,
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        platform_key="platform",
        species_key="species",
        tissue_key="tissue",
        disease_key="disease",
        max_length=6,
    )
    tokens = tokenizer(adata)

    model = SFM(
        token_dict=token_dict,
        cond_vocab_size=128,
        embed_dim=64,
        num_layers=2,
        num_heads=8,
        num_factors=16,
        topk=4,
    ).to(device)
    model.eval()

    batch = {
        "input_ids": tokens.input_ids.to(device),
        "expression_values": tokens.expression_values.to(device),
        "condition_ids": tokens.condition_ids.to(device),
        "padding_mask": None if tokens.padding_mask is None else tokens.padding_mask.to(device),
        "non_tf_mask": tokens.non_tf_mask.to(device),
    }

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        grn, factors = model(
            batch,
            return_factors=True,
            compute_grn=True,
        )

    u_row_nonzero = factors.u.ne(0).sum(dim=-1)
    v_row_nonzero = factors.v.ne(0).sum(dim=-1)
    u_active_rows = factors.u.ne(0).any(dim=-1)
    v_active_rows = factors.v.ne(0).any(dim=-1)

    print("grn_shape:", None if grn is None else tuple(grn.shape))
    print("u_shape:", tuple(factors.u.shape))
    print("v_shape:", tuple(factors.v.shape))
    print(
        "shared_factor_shapes:",
        factors.u.shape == factors.v.shape,
    )
    print("u_nonzero_per_gene:")
    print(u_row_nonzero)
    print("v_nonzero_per_gene:")
    print(v_row_nonzero)
    print(
        "u_topk_sparse:",
        bool((u_row_nonzero[u_active_rows] == model.tf_router.topk).all()) if u_active_rows.any() else True,
    )
    print(
        "v_topk_sparse:",
        bool((v_row_nonzero[v_active_rows] == model.tg_router.topk).all()) if v_active_rows.any() else True,
    )
    print(
        "u_zero_on_inactive_rows:",
        bool((factors.u[~u_active_rows] == 0).all()),
    )
    print(
        "v_zero_on_inactive_rows:",
        bool((factors.v[~v_active_rows] == 0).all()),
    )
