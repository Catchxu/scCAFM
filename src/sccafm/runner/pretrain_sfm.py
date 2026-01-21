import argparse

from sccafm.load import load_resources, load_cfg
from sccafm.builder import build_model, build_loss, build_tokenizer

def main():
    parser = argparse.ArgumentParser(description="Pretrain scCAFM-SFM model")
    parser.add_argument("--config", type=str, required=True, help="Meta config YAML path")
    parser.add_argument("--override", nargs="*", default=[], help="Optional overrides: key=value")
    args = parser.parse_args()

    # Load the configs
    meta_cfg = load_cfg(args.config)

    model_cfg = load_cfg(meta_cfg["model_config"])
    tokenizer_cfg = load_cfg(meta_cfg["tokenizer_config"])
    pretrain_cfg = load_cfg(meta_cfg["pretrain_sfm_config"])

    data_cfg = pretrain_cfg["datasets"]
    loss_cfg = pretrain_cfg["loss"]

    for item in args.override:
        key, value = item.split("=")
        keys = key.split(".")
    
        d = pretrain_cfg
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
    
    model = build_model(model_cfg)
    tokenizer = build_tokenizer(tokenizer_cfg)
    loss = build_loss(loss_cfg)