from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from typing import Optional

from ..assets import load_vocab_json, load_vocab_tensor_file
from .initialization import (
    EMBEDDING_INIT_STD,
    init_embedding,
    init_linear_xavier,
    init_parameter_normal,
    zero_parameter,
)


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


class ExpressionEmbedding(nn.Module):
    """
    Prototype-based soft-bin expression embedding with a dedicated zero-expression bin.
    """

    def __init__(
        self,
        embed_dim: int,
        num_bins: int = 32,
        hidden_dim: int = 256,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        if num_bins < 2:
            raise ValueError(f"`num_bins` must be at least 2, got {num_bins}.")
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")
        if tau <= 0:
            raise ValueError(f"`tau` must be positive, got {tau}.")

        self.embed_dim = int(embed_dim)
        self.num_bins = int(num_bins)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(tau)
        self.bin_embeddings = nn.Parameter(torch.empty(num_bins, embed_dim))
        self.bin_prototypes = nn.Parameter(torch.empty(num_bins - 1, hidden_dim))
        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_parameter_normal(self.bin_embeddings, std=EMBEDDING_INIT_STD)
        init_parameter_normal(self.bin_prototypes, std=EMBEDDING_INIT_STD)
        for module in self.value_encoder:
            if isinstance(module, nn.Linear):
                init_linear_xavier(module)

    def forward(self, expression_values: torch.Tensor) -> torch.Tensor:
        if expression_values.ndim != 2:
            raise ValueError(
                f"`expression_values` must have shape (C, G), got {tuple(expression_values.shape)}."
            )

        batch_size, seq_len = expression_values.shape
        zero_mask = expression_values == 0
        nonzero_mask = ~zero_mask
        out = torch.zeros(
            batch_size,
            seq_len,
            self.embed_dim,
            device=expression_values.device,
            dtype=self.bin_embeddings.dtype,
        )
        if nonzero_mask.any():
            values = expression_values[nonzero_mask].to(self.bin_embeddings.dtype).unsqueeze(-1)
            encoded = self.value_encoder(values)
            dist2 = (encoded.unsqueeze(1) - self.bin_prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
            probs = F.softmax(-dist2 / self.tau, dim=-1)
            out[nonzero_mask] = torch.matmul(probs, self.bin_embeddings[1:]).to(out.dtype)
        if zero_mask.any():
            out[zero_mask] = self.bin_embeddings[0].to(out.dtype)
        return out


class BatchEmbedding(nn.Module):
    """
    Prototype-based soft-bin cell-level batch embedding estimated with attention pooling.
    """

    def __init__(
        self,
        embed_dim: int,
        num_bins: int = 32,
        hidden_dim: int = 128,
        tau: float = 1.0,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError(f"`embed_dim` must be positive, got {embed_dim}.")
        if num_bins < 1:
            raise ValueError(f"`num_bins` must be positive, got {num_bins}.")
        if hidden_dim <= 0:
            raise ValueError(f"`hidden_dim` must be positive, got {hidden_dim}.")
        if tau <= 0:
            raise ValueError(f"`tau` must be positive, got {tau}.")

        self.embed_dim = int(embed_dim)
        self.num_bins = int(num_bins)
        self.hidden_dim = int(hidden_dim)
        self.tau = float(tau)
        self.bin_embeddings = nn.Parameter(torch.empty(num_bins, embed_dim))
        self.bin_prototypes = nn.Parameter(torch.empty(num_bins, hidden_dim))
        self.value_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.RMSNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention_score = nn.Linear(hidden_dim, 1)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_parameter_normal(self.bin_embeddings, std=EMBEDDING_INIT_STD)
        init_parameter_normal(self.bin_prototypes, std=EMBEDDING_INIT_STD)
        for module in self.value_encoder:
            if isinstance(module, nn.Linear):
                init_linear_xavier(module)
        init_linear_xavier(self.attention_score)

    def forward(
        self,
        expression_values: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        if expression_values.ndim != 2:
            raise ValueError(
                f"`expression_values` must have shape (C, G), got {tuple(expression_values.shape)}."
            )
        if padding_mask is not None:
            if padding_mask.shape != expression_values.shape:
                raise ValueError(
                    "`padding_mask` must match `expression_values` shape, "
                    f"got {tuple(padding_mask.shape)} vs {tuple(expression_values.shape)}."
                )
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.to(torch.bool)

        token_latents = self.value_encoder(expression_values.to(self.bin_embeddings.dtype).unsqueeze(-1))
        attention_logits = self.attention_score(token_latents).squeeze(-1)
        if padding_mask is not None:
            all_padded = padding_mask.all(dim=1, keepdim=True)
            attention_logits = attention_logits.masked_fill(padding_mask, torch.finfo(attention_logits.dtype).min)
            attention_logits = attention_logits.masked_fill(all_padded, 0.0)
        attention_weights = F.softmax(attention_logits, dim=1)
        if padding_mask is not None:
            attention_weights = attention_weights.masked_fill(padding_mask, 0.0)
            attention_weights = attention_weights / attention_weights.sum(dim=1, keepdim=True).clamp_min(1e-6)

        cell_latent = torch.sum(token_latents * attention_weights.unsqueeze(-1), dim=1)
        dist2 = (cell_latent.unsqueeze(1) - self.bin_prototypes.unsqueeze(0)).pow(2).sum(dim=-1)
        probs = F.softmax(-dist2 / self.tau, dim=-1)
        return torch.matmul(probs, self.bin_embeddings)


class ConditionEmbedding(nn.Module):
    """
    Encode four condition tokens into a prefix token.
    """

    def __init__(
        self,
        cond_vocab_size: int,
        embed_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.cond_embedding = nn.Embedding(cond_vocab_size, embed_dim)
        self.cond_position_embedding = nn.Embedding(4, embed_dim)
        self.cond_encoder = nn.Sequential(
            nn.Linear(4 * embed_dim, embed_dim),
            nn.SiLU(),
            nn.RMSNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_embedding(self.cond_embedding, std=EMBEDDING_INIT_STD)
        init_embedding(self.cond_position_embedding, std=EMBEDDING_INIT_STD)
        for module in self.cond_encoder:
            if isinstance(module, nn.Linear):
                init_linear_xavier(module)

    def forward(self, condition_ids: torch.LongTensor) -> torch.Tensor:
        if condition_ids.ndim != 2 or condition_ids.shape[1] != 4:
            raise ValueError(
                f"`condition_ids` must have shape (C, 4), got {tuple(condition_ids.shape)}."
            )

        position_ids = torch.arange(4, device=condition_ids.device, dtype=torch.long)
        cond_emb = self.cond_embedding(condition_ids)
        cond_emb = cond_emb + self.cond_position_embedding(position_ids).unsqueeze(0)
        flat = cond_emb.reshape(condition_ids.shape[0], -1)
        return self.cond_encoder(flat).unsqueeze(1)


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
        cond_vocab_size: int = 256,
        embed_dim: int = 768,
        expr_num_bins: int = 32,
        expr_hidden_dim: int = 128,
        expr_tau: float = 1.0,
        batch_num_bins: int = 128,
        batch_hidden_dim: int = 128,
        batch_tau: float = 1.0,
        cond_dropout: float = 0.1,
        out_dropout: float = 0.1,
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
        self.expr_embedding = ExpressionEmbedding(
            embed_dim=embed_dim,
            num_bins=expr_num_bins,
            hidden_dim=expr_hidden_dim,
            tau=expr_tau,
        )
        self.batch_embedding = BatchEmbedding(
            embed_dim=embed_dim,
            num_bins=batch_num_bins,
            hidden_dim=batch_hidden_dim,
            tau=batch_tau,
        )
        self.condition_embedding = ConditionEmbedding(
            cond_vocab_size=cond_vocab_size,
            embed_dim=embed_dim,
            dropout=cond_dropout,
        )

        self.tf_type_embedding = nn.Embedding(2, embed_dim)
        self.prefix_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.gene_type_embedding = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.final_norm = nn.RMSNorm(embed_dim)
        self.dropout = nn.Dropout(out_dropout)

        self._loaded_gene_embedding_mask: Optional[torch.BoolTensor] = None
        self.loaded_gene_embedding_ckpt: Optional[str] = None
        self.loaded_gene_embedding_count = 0
        self.loaded_gene_embedding_total = 0
        self.reset_parameters()

        if gene_embedding_ckpt is not None:
            self._load_gene_embeddings_from_ckpt(
                token_dict=token_dict,
                ckpt_path=gene_embedding_ckpt,
                freeze_loaded=freeze_loaded_gene_embeddings,
            )

    def reset_parameters(self) -> None:
        init_embedding(
            self.gene_embedding,
            std=EMBEDDING_INIT_STD,
            zero_padding_idx=True,
        )
        init_embedding(self.tf_type_embedding, std=EMBEDDING_INIT_STD)
        zero_parameter(self.prefix_type_embedding)
        zero_parameter(self.gene_type_embedding)
        self.final_norm.reset_parameters()

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

        if condition_ids.max().item() >= self.condition_embedding.cond_embedding.num_embeddings:
            raise ValueError(
                "Found `condition_ids` outside the configured `cond_vocab_size`. "
                f"Max value: {condition_ids.max().item()}, "
                f"vocab size: {self.condition_embedding.cond_embedding.num_embeddings}."
            )

        if non_tf_mask is None:
            non_tf_mask = torch.ones_like(input_ids, dtype=torch.bool)
        if padding_mask is not None and padding_mask.dtype != torch.bool:
            padding_mask = padding_mask.to(torch.bool)
        if non_tf_mask.dtype != torch.bool:
            non_tf_mask = non_tf_mask.to(torch.bool)

        gene_emb = self.gene_embedding(input_ids)
        expr_emb = self.expr_embedding(expression_values).to(gene_emb.dtype)

        tf_type_ids = non_tf_mask.long()
        tf_type_emb = self.tf_type_embedding(tf_type_ids).to(gene_emb.dtype)
        token_emb = gene_emb + expr_emb + tf_type_emb

        if padding_mask is not None:
            token_emb = token_emb.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        prefix_token = self.condition_embedding(condition_ids)
        prefix_token = prefix_token.to(gene_emb.dtype)
        batch_emb = self.batch_embedding(expression_values, padding_mask=padding_mask).unsqueeze(1)
        cell_token = prefix_token + batch_emb.to(gene_emb.dtype)

        full_embeddings = torch.cat([cell_token, token_emb], dim=1)

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
