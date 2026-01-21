import argparse

from sccafm.load import load_cfg, load_resources
from sccafm.builder import build_model, build_loss, build_tokenizer
from sccafm.trainer import sfm_trainer


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
    
    token_dict = load_resources(data_cfg["token_dict"])
    adata_files = data_cfg["adata_files"]

    model = build_model(model_cfg, token_dict=token_dict)
    tokenizer = build_tokenizer(tokenizer_cfg, token_dict=token_dict)
    loss = build_loss(loss_cfg)

    sfm_trainer(
        model=model,
        adata_files=adata_files,
        tokenizer=tokenizer,
        criterion=loss,
        **train_cfg
    )




if __name__ == "__main__":
    main()