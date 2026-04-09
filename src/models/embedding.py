from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

from ..assets import load_vocab_json, load_vocab_tensor_file


@dataclass
class ScEmbeddingOutput:
    """
    Embedding outputs for the full model sequence.

    Shape convention:
    - `G`: gene-token length
    - `L`: full sequence length, where `L = G + 1`
    """

    embeddings: torch.Tensor
    padding_mask: Optional[torch.BoolTensor]
    key_padding_mask: Optional[torch.BoolTensor]


class ExpressionValueEmbedding(nn.Module):
    """
    Continuous expression embedding with a dedicated zero-expression vector.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        value_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if value_scale <= 0:
            raise ValueError(f"`value_scale` must be positive, got {value_scale}.")

        self.embed_dim = embed_dim
        self.value_scale = float(value_scale)
        self.zero_embedding = nn.Parameter(torch.zeros(embed_dim))
        self.encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        if expression_values.ndim != 2:
            raise ValueError(
                f"`expression_values` must have shape (C, G), got {tuple(expression_values.shape)}."
            )

        values = expression_values.to(torch.float32).unsqueeze(-1) / self.value_scale
        out = self.encoder(values)
        zero_mask = expression_values == 0
        if zero_mask.any():
            out = out.clone()
            out[zero_mask] = self.zero_embedding.to(out.dtype)
        return out


class ExpressionFeatureModulator(nn.Module):
    """
    Generate feature-wise scale and shift from expression values.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2 * embed_dim),
        )

    def forward(self, expression_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if expression_values.ndim != 2:
            raise ValueError(
                f"`expression_values` must have shape (C, G), got {tuple(expression_values.shape)}."
            )

        mod = self.proj(expression_values.to(torch.float32).unsqueeze(-1))
        scale, shift = mod.chunk(2, dim=-1)
        scale = torch.sigmoid(scale)
        return scale, shift


class ConditionEncoder(nn.Module):
    """
    Encode four condition tokens into a prefix token and a global context bias.
    """

    def __init__(
        self,
        cond_vocab_size: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_embedding = nn.Embedding(cond_vocab_size, embed_dim)
        self.prefix_proj = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        self.context_proj = nn.Sequential(
            nn.Linear(4 * embed_dim, 2 * embed_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(self, condition_ids: torch.LongTensor) -> tuple[torch.Tensor, torch.Tensor]:
        if condition_ids.ndim != 2 or condition_ids.shape[1] != 4:
            raise ValueError(
                f"`condition_ids` must have shape (C, 4), got {tuple(condition_ids.shape)}."
            )

        cond_emb = self.cond_embedding(condition_ids)
        flat = cond_emb.reshape(condition_ids.shape[0], -1)
        prefix_token = self.prefix_proj(flat).unsqueeze(1)
        context_bias = self.context_proj(flat).unsqueeze(1)
        return prefix_token, context_bias


class CellContextEncoder(nn.Module):
    """
    Pool gene-token information into a learned cell context token.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.pool_proj = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(
        self,
        token_embeddings: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if token_embeddings.ndim != 3:
            raise ValueError(
                f"`token_embeddings` must have shape (C, G, D), got {tuple(token_embeddings.shape)}."
            )

        if padding_mask is None:
            pooled = token_embeddings.mean(dim=1)
        else:
            valid_mask = (~padding_mask).unsqueeze(-1).to(token_embeddings.dtype)
            pooled = (token_embeddings * valid_mask).sum(dim=1)
            pooled = pooled / valid_mask.sum(dim=1).clamp_min(1.0)

        return self.pool_proj(pooled).unsqueeze(1)


class ScEmbedding(nn.Module):
    """
    Unified embedding module for single-cell tokens.

    Shape convention:
    - `G`: gene-token length
    - `L`: full sequence length, where `L = G + 1`

    Inputs:
    - `input_ids`: (C, G)
    - `expression_values`: (C, G)
    - `condition_ids`: (C, 4)
    - `padding_mask`: optional (C, G), True where padded
    - `non_tf_mask`: (C, G), True for non-TF and False for TF

    Outputs:
    - `embeddings`: (C, L, D)
    - `padding_mask`: optional (C, G), gene-token padding mask
    - `key_padding_mask`: optional (C, L), attention padding mask including prefix
    """

    REQUIRED_TOKEN_COLUMNS = ("token_index", "gene_id")

    def __init__(
        self,
        token_dict: pd.DataFrame,
        cond_vocab_size: int = 4096,
        embed_dim: int = 256,
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
    ) -> None:
        super().__init__()
        self._validate_token_dict(token_dict)

        self.embed_dim = embed_dim
        self.pad_index = self._lookup_pad_index(token_dict)
        self.gene_vocab_size = int(token_dict["token_index"].max()) + 1

        self.gene_embedding = nn.Embedding(
            self.gene_vocab_size,
            embed_dim,
            padding_idx=self.pad_index,
        )
        self.expr_embedding = ExpressionValueEmbedding(
            embed_dim=embed_dim,
            hidden_dim=expr_hidden_dim,
            dropout=expr_dropout,
            value_scale=expr_value_scale,
        )
        self.expr_modulator = ExpressionFeatureModulator(
            embed_dim=embed_dim,
            hidden_dim=mod_hidden_dim,
            dropout=mod_dropout,
        )
        self.condition_encoder = ConditionEncoder(
            cond_vocab_size=cond_vocab_size,
            embed_dim=embed_dim,
            dropout=cond_dropout,
        )
        self.cell_context_encoder = CellContextEncoder(
            embed_dim=embed_dim,
            hidden_dim=context_hidden_dim,
            dropout=context_dropout,
        )

        self.tf_type_embedding = nn.Embedding(2, embed_dim)
        self.position_embedding = nn.Embedding(8192, embed_dim)
        self.prefix_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.gene_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.final_norm = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(embedding_dropout)

        self._loaded_gene_embedding_mask: Optional[torch.BoolTensor] = None
        self.loaded_gene_embedding_ckpt: Optional[str] = None
        self.loaded_gene_embedding_count = 0
        self.loaded_gene_embedding_total = 0

        if gene_embedding_ckpt is not None:
            self._load_gene_embeddings_from_ckpt(
                token_dict=token_dict,
                ckpt_path=gene_embedding_ckpt,
                freeze_loaded=freeze_loaded_gene_embeddings,
            )

    @classmethod
    def _validate_token_dict(cls, token_dict: pd.DataFrame) -> None:
        if not isinstance(token_dict, pd.DataFrame):
            raise TypeError("`token_dict` must be a pandas DataFrame.")

        missing = [column for column in cls.REQUIRED_TOKEN_COLUMNS if column not in token_dict.columns]
        if missing:
            raise ValueError(
                f"`token_dict` is missing required columns: {missing}. "
                f"Expected columns: {cls.REQUIRED_TOKEN_COLUMNS}."
            )

        if token_dict.empty:
            raise ValueError("`token_dict` must not be empty.")

        if token_dict["token_index"].isna().any():
            raise ValueError("`token_dict['token_index']` contains missing values.")

    @staticmethod
    def _lookup_pad_index(token_dict: pd.DataFrame) -> int:
        pad_rows = token_dict[token_dict["gene_id"] == "<pad>"]
        if len(pad_rows) == 0:
            raise ValueError("`token_dict` must contain the '<pad>' token in `gene_id`.")
        return int(pad_rows["token_index"].iloc[0])

    @staticmethod
    def _build_position_ids(batch_size: int, seq_len: int, device: torch.device) -> torch.LongTensor:
        return torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)

    @staticmethod
    def _build_key_padding_mask(
        padding_mask: Optional[torch.BoolTensor],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.BoolTensor]:
        if padding_mask is None:
            return None

        prefix_mask = torch.zeros((batch_size, 1), dtype=torch.bool, device=device)
        return torch.cat([prefix_mask, padding_mask], dim=1)

    def _validate_inputs(
        self,
        input_ids: torch.LongTensor,
        expression_values: torch.Tensor,
        condition_ids: torch.LongTensor,
        padding_mask: Optional[torch.BoolTensor],
        non_tf_mask: torch.BoolTensor,
    ) -> None:
        if input_ids.ndim != 2:
            raise ValueError(f"`input_ids` must have shape (C, G), got {tuple(input_ids.shape)}.")
        if expression_values.shape != input_ids.shape:
            raise ValueError(
                "`expression_values` must match `input_ids` shape, "
                f"got {tuple(expression_values.shape)} vs {tuple(input_ids.shape)}."
            )
        if non_tf_mask.shape != input_ids.shape:
            raise ValueError(
                f"`non_tf_mask` must match `input_ids` shape, got {tuple(non_tf_mask.shape)}."
            )
        if condition_ids.ndim != 2 or condition_ids.shape[0] != input_ids.shape[0] or condition_ids.shape[1] != 4:
            raise ValueError(
                f"`condition_ids` must have shape ({input_ids.shape[0]}, 4), got {tuple(condition_ids.shape)}."
            )
        if padding_mask is not None and padding_mask.shape != input_ids.shape:
            raise ValueError(
                f"`padding_mask` must match `input_ids` shape, got {tuple(padding_mask.shape)}."
            )

    def _load_gene_embeddings_from_ckpt(
        self,
        token_dict: pd.DataFrame,
        ckpt_path: str,
        freeze_loaded: bool = False,
    ) -> None:
        ckpt = Path(ckpt_path).expanduser().resolve()
        if not ckpt.exists():
            raise FileNotFoundError(f"Gene embedding checkpoint not found: {ckpt}")

        symbol_to_index: dict[str, int] = {}
        freezable_indices: set[int] = set()
        if "gene_symbol" in token_dict.columns:
            for _, row in token_dict.iterrows():
                token_index = int(row["token_index"])
                symbol_raw = row["gene_symbol"]
                symbol = "" if pd.isna(symbol_raw) else str(symbol_raw).strip().upper()
                if symbol:
                    symbol_to_index[symbol] = token_index
                    freezable_indices.add(token_index)

        copied = 0
        mask = torch.zeros(self.gene_vocab_size, dtype=torch.bool)
        if ckpt.suffix.lower() == ".safetensors":
            embeddings = load_vocab_tensor_file(ckpt)
            if embeddings.shape != (self.gene_vocab_size, self.embed_dim):
                raise ValueError(
                    f"Asset vocab embedding shape mismatch in {ckpt}: "
                    f"{tuple(embeddings.shape)} vs ({self.gene_vocab_size}, {self.embed_dim})."
                )

            emb_cpu = embeddings.to(torch.float32).cpu()
            with torch.no_grad():
                for token_index in range(self.gene_vocab_size):
                    self.gene_embedding.weight.data[token_index] = emb_cpu[token_index].to(
                        self.gene_embedding.weight.dtype
                    )
                    if token_index in freezable_indices:
                        mask[token_index] = True
            copied = self.gene_vocab_size
            loaded_total = self.gene_vocab_size
        else:
            payload = torch.load(str(ckpt), map_location="cpu")
            if not isinstance(payload, dict):
                raise ValueError(f"Invalid checkpoint format at {ckpt}: expected dict payload.")
            if "embeddings" not in payload:
                raise ValueError(
                    f"Invalid checkpoint format at {ckpt}: expected key 'embeddings'."
                )

            gene_symbols = payload.get("gene_symbols")
            token_indices = payload.get("token_indices")
            embeddings = payload["embeddings"]
            if not torch.is_tensor(embeddings):
                raise ValueError(f"Invalid `embeddings` type in {ckpt}: expected torch.Tensor.")
            if embeddings.ndim != 2:
                raise ValueError(
                    f"Invalid embedding shape in {ckpt}: expected 2D tensor, got {tuple(embeddings.shape)}."
                )
            if embeddings.shape[1] != self.embed_dim:
                raise ValueError(
                    f"Checkpoint embed dim mismatch in {ckpt}: "
                    f"{embeddings.shape[1]} vs model embed_dim={self.embed_dim}."
                )

            use_direct_token_indices = token_indices is not None
            if use_direct_token_indices:
                if not isinstance(token_indices, list):
                    raise ValueError(f"Invalid `token_indices` type in {ckpt}: expected list.")
                if len(token_indices) != embeddings.shape[0]:
                    raise ValueError(
                        f"Checkpoint row mismatch in {ckpt}: "
                        f"len(token_indices)={len(token_indices)} vs embeddings.shape[0]={embeddings.shape[0]}."
                    )
            else:
                if not isinstance(gene_symbols, list):
                    raise ValueError(
                        f"Invalid checkpoint format at {ckpt}: expected list `gene_symbols` when "
                        "`token_indices` is not provided."
                    )
                if len(gene_symbols) != embeddings.shape[0]:
                    raise ValueError(
                        f"Checkpoint row mismatch in {ckpt}: "
                        f"len(gene_symbols)={len(gene_symbols)} vs embeddings.shape[0]={embeddings.shape[0]}."
                    )

            emb_cpu = embeddings.to(torch.float32).cpu()
            with torch.no_grad():
                if use_direct_token_indices:
                    for row_idx, token_index in enumerate(token_indices):
                        token_index = int(token_index)
                        if not 0 <= token_index < self.gene_vocab_size:
                            raise ValueError(
                                f"Checkpoint token index out of range in {ckpt}: {token_index}."
                            )
                        self.gene_embedding.weight.data[token_index] = emb_cpu[row_idx].to(
                            self.gene_embedding.weight.dtype
                        )
                        if token_index in freezable_indices:
                            mask[token_index] = True
                        copied += 1
                else:
                    for row_idx, symbol in enumerate(gene_symbols):
                        token_index = symbol_to_index.get(str(symbol).strip().upper())
                        if token_index is None:
                            continue
                        self.gene_embedding.weight.data[token_index] = emb_cpu[row_idx].to(
                            self.gene_embedding.weight.dtype
                        )
                        if token_index in freezable_indices:
                            mask[token_index] = True
                        copied += 1
            loaded_total = int(embeddings.shape[0])

        self.loaded_gene_embedding_ckpt = str(ckpt)
        self.loaded_gene_embedding_total = loaded_total
        self.loaded_gene_embedding_count = copied
        self._loaded_gene_embedding_mask = mask

        if freeze_loaded and copied > 0:
            self._register_loaded_embedding_grad_hook()

    def _register_loaded_embedding_grad_hook(self) -> None:
        if self._loaded_gene_embedding_mask is None:
            return

        def _mask_loaded_rows(grad: torch.Tensor) -> torch.Tensor:
            masked_grad = grad.clone()
            mask = self._loaded_gene_embedding_mask.to(device=masked_grad.device)
            masked_grad[mask] = 0
            return masked_grad

        self.gene_embedding.weight.register_hook(_mask_loaded_rows)

    def forward(
        self,
        input_ids: torch.LongTensor,
        expression_values: torch.Tensor,
        condition_ids: torch.LongTensor,
        padding_mask: Optional[torch.BoolTensor] = None,
        non_tf_mask: Optional[torch.BoolTensor] = None,
    ) -> ScEmbeddingOutput:
        self._validate_inputs(
            input_ids=input_ids,
            expression_values=expression_values,
            condition_ids=condition_ids,
            padding_mask=padding_mask,
            non_tf_mask=non_tf_mask if non_tf_mask is not None else torch.ones_like(input_ids, dtype=torch.bool),
        )

        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if condition_ids.max().item() >= self.condition_encoder.cond_embedding.num_embeddings:
            raise ValueError(
                "Found `condition_ids` outside the configured `cond_vocab_size`. "
                f"Max value: {condition_ids.max().item()}, "
                f"vocab size: {self.condition_encoder.cond_embedding.num_embeddings}."
            )

        if non_tf_mask is None:
            non_tf_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if padding_mask is not None and padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(torch.bool)
        if non_tf_mask.dtype != torch.bool:
            non_tf_mask = non_tf_mask.to(torch.bool)

        gene_emb = self.gene_embedding(input_ids)
        expr_emb = self.expr_embedding(expression_values).to(gene_emb.dtype)
        expr_scale, expr_shift = self.expr_modulator(expression_values)
        expr_scale = expr_scale.to(gene_emb.dtype)
        expr_shift = expr_shift.to(gene_emb.dtype)

        tf_type_ids = non_tf_mask.long()
        tf_type_emb = self.tf_type_embedding(tf_type_ids).to(gene_emb.dtype)
        token_emb = gene_emb * (1.0 + expr_scale) + expr_shift + expr_emb + tf_type_emb

        if padding_mask is not None:
            token_emb = token_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        prefix_token, context_bias = self.condition_encoder(condition_ids)
        prefix_token = prefix_token.to(gene_emb.dtype)
        context_bias = context_bias.to(gene_emb.dtype)
        token_emb = token_emb + context_bias

        cell_token = prefix_token + self.cell_context_encoder(
            token_emb,
            padding_mask=padding_mask,
        ).to(gene_emb.dtype)

        full_embeddings = torch.cat([cell_token, token_emb], dim=1)

        position_ids = self._build_position_ids(batch_size, seq_len + 1, device=device)
        position_emb = self.position_embedding(position_ids).to(full_embeddings.dtype)
        full_embeddings = full_embeddings + position_emb
        full_embeddings[:, :1] = full_embeddings[:, :1] + self.prefix_type_embedding.to(full_embeddings.dtype)
        full_embeddings[:, 1:] = full_embeddings[:, 1:] + self.gene_type_embedding.to(full_embeddings.dtype)

        full_embeddings = self.final_norm(full_embeddings)
        full_embeddings = self.dropout(full_embeddings)

        key_padding_mask = self._build_key_padding_mask(
            padding_mask=padding_mask,
            batch_size=batch_size,
            device=device,
        )
        if key_padding_mask is not None:
            full_embeddings = full_embeddings.masked_fill(key_padding_mask.unsqueeze(-1), 0.0)

        return ScEmbeddingOutput(
            embeddings=full_embeddings,
            padding_mask=padding_mask,
            key_padding_mask=key_padding_mask,
        )




if __name__ == "__main__":
    from pathlib import Path

    import numpy as np

    from anndata import AnnData

    from ..assets import resolve_model_assets
    from ..data.tokenizer import ScTokenizer

    root_dir = Path(__file__).resolve().parents[2]
    assets = resolve_model_assets(root_dir / "assets")
    token_dict = load_vocab_json(assets.vocab)
    human_tfs = pd.read_csv(assets.human_tfs)
    mouse_tfs = pd.read_csv(assets.mouse_tfs)

    obs = pd.DataFrame(
        {
            "platform": ["10X", "smart-seq"],
            "species": ["human", "mouse"],
            "tissue": ["lung", "brain"],
            "disease": ["healthy", "tumor"],
        },
        index=["cell_0", "cell_1"],
    )
    var_names = pd.Index(["AATF", "TSPAN6"])
    adata = AnnData(
        X=np.array([[1.0, 2.0], [3.0, 0.0]], dtype=np.float32),
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
        max_length=5,
    )
    tokens = tokenizer(adata)

    model = ScEmbedding(
        token_dict=token_dict,
        cond_vocab_size=128,
        embed_dim=128,
    )
    output = model(
        input_ids=tokens.input_ids,
        expression_values=tokens.expression_values,
        condition_ids=tokens.condition_ids,
        padding_mask=tokens.padding_mask,
        non_tf_mask=tokens.non_tf_mask,
    )

    print("embeddings_shape:", tuple(output.embeddings.shape))
    print("key_padding_mask_shape:", None if output.key_padding_mask is None else tuple(output.key_padding_mask.shape))
    print("dtype:", output.embeddings.dtype)
