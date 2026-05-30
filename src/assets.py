from __future__ import annotations

import json
import shutil

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from safetensors.torch import load_file, save_file


SFM_CONFIG_NAME = "sfm_config.json"
SFM_MODEL_NAME = "sfm_model.safetensors"
EFM_CONFIG_NAME = "efm_config.json"
EFM_MODEL_NAME = "efm_model.safetensors"
VOCAB_NAME = "vocab.json"
VOCAB_TENSORS_NAME = "vocab.safetensors"
RELEASE_MANIFEST_NAME = "release.json"
RESOURCES_DIR_NAME = "resources"
TOKENIZER_DIR_NAME = "tokenizer"
MODELS_DIR_NAME = "models"
SFM_DIR_NAME = "sfm"
EFM_DIR_NAME = "efm"
COND_DICT_NAME = "cond_dict.json"
HUMAN_TFS_NAME = "human_tfs.csv"
MOUSE_TFS_NAME = "mouse_tfs.csv"
OMNIPATH_NAME = "OmniPath.csv"
HOMOLOGOUS_NAME = "homologous.csv"


@dataclass(slots=True)
class ModelAssets:
    model_source: str
    local_dir: Path
    model_dir: Path
    resources_dir: Path
    tokenizer_dir: Path
    sfm_dir: Path
    efm_dir: Path
    release_manifest: Path
    sfm_config: Path
    sfm_model: Path
    efm_config: Path
    efm_model: Path
    vocab: Path
    vocab_tensors: Path
    cond_dict: Path
    human_tfs: Path
    mouse_tfs: Path
    omnipath: Path
    homologous: Path


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


def _first_existing_path(*candidates: Path) -> Path:
    if not candidates:
        raise ValueError("At least one candidate path is required.")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _resolve_release_assets(local_dir: Path, model_source: str) -> ModelAssets:
    resources_dir = local_dir / RESOURCES_DIR_NAME
    tokenizer_dir = local_dir / TOKENIZER_DIR_NAME
    model_dir = local_dir / MODELS_DIR_NAME
    sfm_dir = model_dir / SFM_DIR_NAME
    efm_dir = model_dir / EFM_DIR_NAME
    legacy_model_dir = local_dir / MODELS_DIR_NAME

    return ModelAssets(
        model_source=model_source,
        local_dir=local_dir,
        model_dir=model_dir,
        resources_dir=resources_dir,
        tokenizer_dir=tokenizer_dir,
        sfm_dir=sfm_dir,
        efm_dir=efm_dir,
        release_manifest=local_dir / RELEASE_MANIFEST_NAME,
        sfm_config=_first_existing_path(
            sfm_dir / SFM_CONFIG_NAME,
            legacy_model_dir / SFM_CONFIG_NAME,
            local_dir / SFM_CONFIG_NAME,
        ),
        sfm_model=_first_existing_path(
            sfm_dir / SFM_MODEL_NAME,
            legacy_model_dir / SFM_MODEL_NAME,
            local_dir / SFM_MODEL_NAME,
        ),
        efm_config=_first_existing_path(
            efm_dir / EFM_CONFIG_NAME,
            legacy_model_dir / EFM_CONFIG_NAME,
            local_dir / EFM_CONFIG_NAME,
        ),
        efm_model=_first_existing_path(
            efm_dir / EFM_MODEL_NAME,
            legacy_model_dir / EFM_MODEL_NAME,
            local_dir / EFM_MODEL_NAME,
        ),
        vocab=_first_existing_path(
            tokenizer_dir / VOCAB_NAME,
            legacy_model_dir / VOCAB_NAME,
            local_dir / VOCAB_NAME,
        ),
        vocab_tensors=_first_existing_path(
            tokenizer_dir / VOCAB_TENSORS_NAME,
            legacy_model_dir / VOCAB_TENSORS_NAME,
            local_dir / VOCAB_TENSORS_NAME,
        ),
        cond_dict=_first_existing_path(
            tokenizer_dir / COND_DICT_NAME,
            legacy_model_dir / COND_DICT_NAME,
            local_dir / COND_DICT_NAME,
        ),
        human_tfs=_first_existing_path(
            resources_dir / HUMAN_TFS_NAME,
            local_dir / HUMAN_TFS_NAME,
        ),
        mouse_tfs=_first_existing_path(
            resources_dir / MOUSE_TFS_NAME,
            local_dir / MOUSE_TFS_NAME,
        ),
        omnipath=_first_existing_path(
            resources_dir / OMNIPATH_NAME,
            local_dir / OMNIPATH_NAME,
        ),
        homologous=_first_existing_path(
            resources_dir / HOMOLOGOUS_NAME,
            local_dir / HOMOLOGOUS_NAME,
        ),
    )


def _structured_release_assets(local_dir: Path, model_source: str) -> ModelAssets:
    resources_dir = local_dir / RESOURCES_DIR_NAME
    tokenizer_dir = local_dir / TOKENIZER_DIR_NAME
    model_dir = local_dir / MODELS_DIR_NAME
    sfm_dir = model_dir / SFM_DIR_NAME
    efm_dir = model_dir / EFM_DIR_NAME
    return ModelAssets(
        model_source=model_source,
        local_dir=local_dir,
        model_dir=model_dir,
        resources_dir=resources_dir,
        tokenizer_dir=tokenizer_dir,
        sfm_dir=sfm_dir,
        efm_dir=efm_dir,
        release_manifest=local_dir / RELEASE_MANIFEST_NAME,
        sfm_config=sfm_dir / SFM_CONFIG_NAME,
        sfm_model=sfm_dir / SFM_MODEL_NAME,
        efm_config=efm_dir / EFM_CONFIG_NAME,
        efm_model=efm_dir / EFM_MODEL_NAME,
        vocab=tokenizer_dir / VOCAB_NAME,
        vocab_tensors=tokenizer_dir / VOCAB_TENSORS_NAME,
        cond_dict=tokenizer_dir / COND_DICT_NAME,
        human_tfs=resources_dir / HUMAN_TFS_NAME,
        mouse_tfs=resources_dir / MOUSE_TFS_NAME,
        omnipath=resources_dir / OMNIPATH_NAME,
        homologous=resources_dir / HOMOLOGOUS_NAME,
    )


def resolve_model_assets(
    model_source: str | Path,
    *,
    require_model_weights: bool = False,
    require_efm_weights: bool = False,
    require_cond_dict: bool = True,
) -> ModelAssets:
    model_source_str = str(model_source)
    local_candidate = Path(model_source_str).expanduser()
    if local_candidate.exists():
        local_dir = _require_directory(local_candidate, model_source_str)
    else:
        local_dir = _snapshot_download_model_repo(model_source_str)

    assets = _resolve_release_assets(local_dir, model_source_str)

    required_paths = {
        "sfm_config": assets.sfm_config,
        "vocab": assets.vocab,
        "vocab_tensors": assets.vocab_tensors,
        "human_tfs": assets.human_tfs,
        "mouse_tfs": assets.mouse_tfs,
    }
    if require_cond_dict:
        required_paths["cond_dict"] = assets.cond_dict
    if require_model_weights:
        required_paths["sfm_model"] = assets.sfm_model
    if require_efm_weights:
        required_paths["efm_config"] = assets.efm_config
        required_paths["efm_model"] = assets.efm_model

    missing = [name for name, path in required_paths.items() if not path.exists()]
    if missing:
        missing_text = ", ".join(f"{name}={required_paths[name]}" for name in missing)
        raise FileNotFoundError(
            f"`model_source` is missing required asset file(s): {missing_text}"
        )

    return assets


def _copy_file_if_needed(source: Path, destination: Path, *, overwrite: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        return
    shutil.copy2(source, destination)


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _copy_file_if_present(source: Path, destination: Path, *, overwrite: bool) -> None:
    if source.exists():
        _copy_file_if_needed(source, destination, overwrite=overwrite)


def write_release_manifest(assets: ModelAssets) -> None:
    model_entries: dict[str, Any] = {
        "sfm": {
            "config": f"{MODELS_DIR_NAME}/{SFM_DIR_NAME}/{SFM_CONFIG_NAME}",
            "weights": f"{MODELS_DIR_NAME}/{SFM_DIR_NAME}/{SFM_MODEL_NAME}",
        }
    }
    if assets.efm_config.exists() or assets.efm_model.exists():
        model_entries["efm"] = {
            "config": f"{MODELS_DIR_NAME}/{EFM_DIR_NAME}/{EFM_CONFIG_NAME}",
            "weights": f"{MODELS_DIR_NAME}/{EFM_DIR_NAME}/{EFM_MODEL_NAME}",
        }

    payload = {
        "schema_version": 1,
        "resources": {
            "human_tfs": f"{RESOURCES_DIR_NAME}/{HUMAN_TFS_NAME}",
            "mouse_tfs": f"{RESOURCES_DIR_NAME}/{MOUSE_TFS_NAME}",
            "omnipath": f"{RESOURCES_DIR_NAME}/{OMNIPATH_NAME}",
            "homologous": f"{RESOURCES_DIR_NAME}/{HOMOLOGOUS_NAME}",
        },
        "tokenizer": {
            "vocab": f"{TOKENIZER_DIR_NAME}/{VOCAB_NAME}",
            "vocab_tensors": f"{TOKENIZER_DIR_NAME}/{VOCAB_TENSORS_NAME}",
            "condition_vocab": f"{TOKENIZER_DIR_NAME}/{COND_DICT_NAME}",
        },
        "models": model_entries,
    }
    save_json(assets.release_manifest, payload)


def materialize_model_package(
    source_assets: ModelAssets,
    target_dir: str | Path,
    *,
    include_model_weights: bool = True,
    include_efm_weights: bool = False,
    include_cond_dict: bool = True,
    overwrite: bool = False,
) -> ModelAssets:
    target_path = Path(target_dir).expanduser().resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    target_assets = _structured_release_assets(target_path, str(target_path))

    _copy_file_if_needed(source_assets.sfm_config, target_assets.sfm_config, overwrite=overwrite)
    _copy_file_if_needed(source_assets.vocab, target_assets.vocab, overwrite=overwrite)
    _copy_file_if_needed(
        source_assets.vocab_tensors,
        target_assets.vocab_tensors,
        overwrite=overwrite,
    )
    _copy_file_if_present(source_assets.human_tfs, target_assets.human_tfs, overwrite=overwrite)
    _copy_file_if_present(source_assets.mouse_tfs, target_assets.mouse_tfs, overwrite=overwrite)
    _copy_file_if_present(source_assets.omnipath, target_assets.omnipath, overwrite=overwrite)
    _copy_file_if_present(source_assets.homologous, target_assets.homologous, overwrite=overwrite)

    if include_model_weights:
        if source_assets.sfm_model.exists():
            _copy_file_if_needed(
                source_assets.sfm_model,
                target_assets.sfm_model,
                overwrite=overwrite,
            )
        elif overwrite and target_assets.sfm_model.exists():
            _safe_unlink(target_assets.sfm_model)

    if include_cond_dict:
        if source_assets.cond_dict.exists():
            _copy_file_if_needed(
                source_assets.cond_dict,
                target_assets.cond_dict,
                overwrite=overwrite,
            )
    elif overwrite and target_assets.cond_dict.exists():
        _safe_unlink(target_assets.cond_dict)

    if include_efm_weights:
        if source_assets.efm_config.exists():
            _copy_file_if_needed(source_assets.efm_config, target_assets.efm_config, overwrite=overwrite)
        if source_assets.efm_model.exists():
            _copy_file_if_needed(source_assets.efm_model, target_assets.efm_model, overwrite=overwrite)
    elif overwrite:
        _safe_unlink(target_assets.efm_config)
        _safe_unlink(target_assets.efm_model)

    write_release_manifest(target_assets)
    return target_assets


def resolve_sfm_checkpoint_path(path: str | Path) -> Path:
    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        return candidate

    resolved = _first_existing_path(
        candidate / MODELS_DIR_NAME / SFM_DIR_NAME / SFM_MODEL_NAME,
        candidate / SFM_MODEL_NAME,
        candidate / MODELS_DIR_NAME / SFM_MODEL_NAME,
    )
    return resolved


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


def load_table_json(path: str | Path) -> pd.DataFrame:
    payload = load_json(path)
    if not isinstance(payload, list):
        raise ValueError(f"`{path}` must contain a JSON array of records.")

    return pd.DataFrame(payload)


def save_table_json(path: str | Path, table: pd.DataFrame) -> None:
    records = table.where(pd.notna(table), None).to_dict(orient="records")
    save_json(path, records)


def load_vocab_json(path: str | Path) -> pd.DataFrame:
    token_dict = load_table_json(path)
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
    save_table_json(path, export_df)


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
        if evaluator_cfg.get("checkpoint_path") is None:
            evaluator_cfg["checkpoint_path"] = str(assets.sfm_model)

    if require_model_weights and not assets.sfm_model.exists():
        raise FileNotFoundError(f"Model weights not found: {assets.sfm_model}")

    return resolved
