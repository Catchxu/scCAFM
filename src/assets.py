from __future__ import annotations

import json

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from safetensors.torch import load_file, save_file


SFM_CONFIG_NAME = "sfm_config.json"
SFM_MODEL_NAME = "sfm_model.safetensors"
VOCAB_NAME = "vocab.json"
VOCAB_TENSORS_NAME = "vocab.safetensors"
MODELS_DIR_NAME = "models"
COND_DICT_NAME = "cond_dict.csv"
HUMAN_TFS_NAME = "human_tfs.csv"
MOUSE_TFS_NAME = "mouse_tfs.csv"
OMNIPATH_NAME = "OmniPath.csv"


@dataclass(slots=True)
class ModelAssets:
    model_source: str
    local_dir: Path
    model_dir: Path
    sfm_config: Path
    sfm_model: Path
    vocab: Path
    vocab_tensors: Path
    cond_dict: Path
    human_tfs: Path
    mouse_tfs: Path
    omnipath: Path


def _require_directory(path: Path, source: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"`model_source` does not exist: {source}")
    if not path.is_dir():
        raise NotADirectoryError(f"`model_source` must resolve to a directory: {source}")
    return path.resolve()


def _snapshot_download_model_repo(repo_id: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError(
            "Resolving a remote Hugging Face `model_source` requires `huggingface_hub`."
        ) from exc

    snapshot_path = snapshot_download(repo_id=repo_id, repo_type="model")
    return Path(snapshot_path).resolve()


def _resolve_model_file(local_dir: Path, filename: str) -> Path:
    model_subdir_path = local_dir / MODELS_DIR_NAME / filename
    if model_subdir_path.exists():
        return model_subdir_path
    return local_dir / filename


def _resolve_model_dir(local_dir: Path) -> Path:
    model_subdir = local_dir / MODELS_DIR_NAME
    if model_subdir.exists() and model_subdir.is_dir():
        return model_subdir
    return local_dir


def resolve_model_assets(
    model_source: str | Path,
    *,
    require_model_weights: bool = False,
) -> ModelAssets:
    model_source_str = str(model_source)
    local_candidate = Path(model_source_str).expanduser()
    if local_candidate.exists():
        local_dir = _require_directory(local_candidate, model_source_str)
    else:
        local_dir = _snapshot_download_model_repo(model_source_str)

    assets = ModelAssets(
        model_source=model_source_str,
        local_dir=local_dir,
        model_dir=_resolve_model_dir(local_dir),
        sfm_config=_resolve_model_file(local_dir, SFM_CONFIG_NAME),
        sfm_model=_resolve_model_file(local_dir, SFM_MODEL_NAME),
        vocab=_resolve_model_file(local_dir, VOCAB_NAME),
        vocab_tensors=_resolve_model_file(local_dir, VOCAB_TENSORS_NAME),
        cond_dict=local_dir / COND_DICT_NAME,
        human_tfs=local_dir / HUMAN_TFS_NAME,
        mouse_tfs=local_dir / MOUSE_TFS_NAME,
        omnipath=local_dir / OMNIPATH_NAME,
    )

    required_paths = {
        "sfm_config": assets.sfm_config,
        "vocab": assets.vocab,
        "vocab_tensors": assets.vocab_tensors,
        "cond_dict": assets.cond_dict,
        "human_tfs": assets.human_tfs,
        "mouse_tfs": assets.mouse_tfs,
    }
    if require_model_weights:
        required_paths["sfm_model"] = assets.sfm_model

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(f"{name}={required_paths[name]}" for name in missing)
        raise FileNotFoundError(
            f"`model_source` is missing required asset file(s): {missing_text}"
        )

    return assets


def load_json(path: str | Path) -> Any:
    resolved = Path(path).expanduser().resolve()
    with resolved.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)
        handle.write("\n")


def load_sfm_config(path: str | Path) -> dict[str, Any]:
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"`{path}` must contain a JSON object.")
    if "sfm" not in payload or "vgae" not in payload:
        raise ValueError(f"`{path}` must contain both 'sfm' and 'vgae' sections.")
    return payload


def save_sfm_config(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(path, payload)


def load_vocab_json(path: str | Path) -> pd.DataFrame:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"`{path}` must contain a JSON array of token records.")

    token_dict = pd.DataFrame(payload)
    required_columns = {"token_index", "gene_symbol", "gene_id"}
    missing = required_columns.difference(token_dict.columns)
    if missing:
        raise ValueError(
            f"`{path}` is missing required vocab columns: {sorted(missing)}."
        )

    token_dict = token_dict.copy()
    token_dict["token_index"] = token_dict["token_index"].astype(int)
    token_dict = token_dict.sort_values("token_index").reset_index(drop=True)
    return token_dict


def save_vocab_json(path: str | Path, token_dict: pd.DataFrame) -> None:
    required_columns = ["token_index", "gene_symbol", "gene_id"]
    missing = [column for column in required_columns if column not in token_dict.columns]
    if missing:
        raise ValueError(
            f"`token_dict` is missing required columns for vocab export: {missing}."
        )

    export_df = token_dict[required_columns].copy()
    export_df = export_df.sort_values("token_index").reset_index(drop=True)
    records = export_df.where(pd.notna(export_df), None).to_dict(orient="records")
    save_json(path, records)


def load_model_state_dict(path: str | Path) -> dict[str, torch.Tensor]:
    resolved = Path(path).expanduser().resolve()
    state_dict = load_file(str(resolved), device="cpu")
    return {key: value.detach().cpu() for key, value in state_dict.items()}


def save_model_state_dict(
    path: str | Path,
    state_dict: dict[str, torch.Tensor],
) -> Path:
    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    tensor_state = {}
    for key, value in state_dict.items():
        if not torch.is_tensor(value):
            raise TypeError(f"State dict entry {key!r} is not a tensor.")
        tensor_state[key] = value.detach().cpu().float() if value.is_floating_point() else value.detach().cpu()

    save_file(tensor_state, str(resolved))
    return resolved


def save_vocab_tensor_file(
    path: str | Path,
    embeddings: torch.Tensor,
) -> Path:
    if embeddings.ndim != 2:
        raise ValueError(
            f"`embeddings` must be 2D for vocab tensor export, got {tuple(embeddings.shape)}."
        )

    resolved = Path(path).expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    save_file({"embeddings": embeddings.detach().cpu().float()}, str(resolved))
    return resolved


def load_vocab_tensor_file(path: str | Path) -> torch.Tensor:
    resolved = Path(path).expanduser().resolve()
    payload = load_file(str(resolved), device="cpu")
    if "embeddings" not in payload:
        raise ValueError(f"`{resolved}` is missing required tensor key 'embeddings'.")
    embeddings = payload["embeddings"].detach().cpu()
    if embeddings.ndim != 2:
        raise ValueError(
            f"`{resolved}` must contain a 2D 'embeddings' tensor, got {tuple(embeddings.shape)}."
        )
    return embeddings


def apply_model_assets_to_runtime_config(
    config: dict[str, Any],
    assets: ModelAssets,
    *,
    require_model_weights: bool = False,
) -> dict[str, Any]:
    resolved = deepcopy(config)

    data_cfg = resolved.setdefault("data", {})
    data_cfg["token_dict_path"] = str(assets.vocab)
    data_cfg["cond_dict_path"] = str(assets.cond_dict)
    data_cfg["human_tfs_path"] = str(assets.human_tfs)
    data_cfg["mouse_tfs_path"] = str(assets.mouse_tfs)

    loss_cfg = resolved.setdefault("loss", {})
    prior_cfg = loss_cfg.get("prior")
    if isinstance(prior_cfg, dict) and prior_cfg.get("enabled", False):
        prior_cfg["prior_grn_path"] = str(assets.omnipath)

    evaluator_cfg = resolved.get("evaluator")
    if isinstance(evaluator_cfg, dict):
        evaluator_cfg["checkpoint_path"] = str(assets.sfm_model)

    if require_model_weights and not assets.sfm_model.exists():
        raise FileNotFoundError(f"Model weights not found: {assets.sfm_model}")

    return resolved
