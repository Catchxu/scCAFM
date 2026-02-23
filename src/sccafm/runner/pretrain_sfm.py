import argparse
from pathlib import Path

from sccafm.load import load_cfg, load_resources
from sccafm.builder import build_model, build_loss, build_tokenizer
from sccafm.trainer import sfm_trainer


def _as_float(v, key):
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError as e:
            raise ValueError(f"`{key}` must be a float, got: {v}") from e
    raise ValueError(f"`{key}` must be a float, got type: {type(v).__name__}")


def _as_int(v, key):
    if isinstance(v, bool):
        raise ValueError(f"`{key}` must be an int, got bool: {v}")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as e:
            raise ValueError(f"`{key}` must be an int, got: {v}") from e
    raise ValueError(f"`{key}` must be an int, got type: {type(v).__name__}")


def _as_bool(v, key):
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"true", "1", "yes", "y"}:
            return True
        if s in {"false", "0", "no", "n"}:
            return False
    raise ValueError(f"`{key}` must be a bool, got: {v}")


def _normalize_pretrain_cfg(train_cfg, loss_cfg):
    train_cfg["learning_rate"] = _as_float(train_cfg.get("learning_rate", 1e-5), "train.learning_rate")
    train_cfg["weight_decay"] = _as_float(train_cfg.get("weight_decay", 1e-2), "train.weight_decay")
    train_cfg["epochs_per_file"] = _as_int(train_cfg.get("epochs_per_file", 1), "train.epochs_per_file")
    train_cfg["batch_size"] = _as_int(train_cfg.get("batch_size", 32), "train.batch_size")
    train_cfg["resume"] = _as_bool(train_cfg.get("resume", True), "train.resume")

    loss_kwargs = loss_cfg.get("kwargs", {})
    if "alpha" in loss_kwargs:
        loss_kwargs["alpha"] = _as_float(loss_kwargs["alpha"], "loss.kwargs.alpha")
    if "rho" in loss_kwargs:
        loss_kwargs["rho"] = _as_float(loss_kwargs["rho"], "loss.kwargs.rho")
    if "rho_max" in loss_kwargs:
        loss_kwargs["rho_max"] = _as_float(loss_kwargs["rho_max"], "loss.kwargs.rho_max")
    if "update_period" in loss_kwargs:
        loss_kwargs["update_period"] = _as_int(loss_kwargs["update_period"], "loss.kwargs.update_period")


def _expand_adata_entry(entry: str):
    p = Path(entry).expanduser()
    if p.is_dir():
        files = sorted(str(f.resolve()) for f in p.rglob("*.h5ad"))
        if not files:
            raise FileNotFoundError(f"No .h5ad files found in directory: {p}")
        return files
    if p.is_file():
        return [str(p.resolve())]
    raise FileNotFoundError(f"`datasets.adata_files` path not found: {entry}")


def _resolve_adata_files(raw):
    if isinstance(raw, str):
        return _expand_adata_entry(raw)
    if isinstance(raw, list):
        out = []
        for item in raw:
            if not isinstance(item, str):
                raise ValueError(
                    "`datasets.adata_files` list items must be string paths."
                )
            out.extend(_expand_adata_entry(item))
        return out
    raise ValueError(
        "`datasets.adata_files` must be a string path or a list of string paths."
    )


def main():
    parser = argparse.ArgumentParser(description="Pretrain scCAFM-SFM model")
    parser.add_argument("--config", type=str, default="meta.yaml", help="Meta config YAML path")
    parser.add_argument("--override", nargs="*", default=[], help="Optional overrides: key=value")
    args = parser.parse_args()

    # Load the configs
    meta_cfg = load_cfg(args.config)

    model_cfg = load_cfg(meta_cfg["model_config"])
    tokenizer_cfg = load_cfg(meta_cfg["tokenizer_config"])
    pretrain_cfg = load_cfg(meta_cfg["pretrain_sfm_config"])

    data_cfg = pretrain_cfg["datasets"]
    loss_cfg = pretrain_cfg["loss"]
    train_cfg = pretrain_cfg["train"]

    for item in args.override:
        key, value = item.split("=")
        keys = key.split(".")
    
        d = train_cfg
        for k in keys[:-1]:
            d = d[k]

        old_val = d[keys[-1]]
        if isinstance(old_val, bool):
            value = value.lower() == "true"
        elif isinstance(old_val, int):
            value = int(value)
        elif isinstance(old_val, float):
            value = float(value)
        d[keys[-1]] = value

    _normalize_pretrain_cfg(train_cfg, loss_cfg)

    adata_files = _resolve_adata_files(data_cfg["adata_files"])
    token_dict = load_resources(data_cfg["token_dict"])
    human_tfs = load_resources(data_cfg["human_tfs"])
    mouse_tfs = load_resources(data_cfg["mouse_tfs"])
    true_grn_df = load_resources(data_cfg["true_grn"])

    model = build_model(model_cfg["SFM"], token_dict=token_dict)
    tokenizer = build_tokenizer(tokenizer_cfg["Tome"], token_dict=token_dict)
    loss = build_loss(loss_cfg, tome_tokenizer=tokenizer, true_grn_df=true_grn_df)

    sfm_trainer(
        model=model,
        adata_files=adata_files,
        tokenizer=tokenizer,
        criterion=loss,
        human_tfs=human_tfs,
        mouse_tfs=mouse_tfs,
        **train_cfg
    )




if __name__ == "__main__":
    main()
