import argparse
import logging
import os
from pathlib import Path

import pandas as pd
import torch
from modelscope import AutoModel, AutoTokenizer


DEFAULT_MODEL_NAME = "/data1021/xukaichen/data/LLM/PubMedBERT"
DEFAULT_INPUT_CSV = "resources/gene_descriptions.csv"
DEFAULT_OUTPUT_CKPT = "checkpoints/gene_embeddings.pt"


def _setup_logger(
    logger_name: str,
    log_dir: str,
    log_name: str,
    log_overwrite: bool = True,
):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh = logging.FileHandler(log_path, mode="w" if log_overwrite else "a")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build gene-text embeddings from gene_descriptions.csv.")
    parser.add_argument(
        "--input-csv",
        type=str,
        default=DEFAULT_INPUT_CSV,
        help="Path to input gene descriptions CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_CKPT,
        help="Output checkpoint path (.pt).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Local model path or model id.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum token length for description text.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to use. Set negative value to force CPU.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs.",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="gene_embedding.log",
        help="Log filename.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Log progress every N batches.",
    )
    parser.add_argument(
        "--log-overwrite",
        action="store_true",
        help="Overwrite logs instead of appending.",
    )
    return parser.parse_args()


def load_gene_descriptions(input_csv: Path) -> tuple[list[str], list[str]]:
    df = pd.read_csv(input_csv)
    required_columns = {"gene_symbol", "description"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {input_csv}: {sorted(missing)}")

    df = df[["gene_symbol", "description"]].copy()
    df["gene_symbol"] = df["gene_symbol"].astype(str).str.strip()
    df["description"] = df["description"].fillna("").astype(str).str.strip()
    df = df[df["gene_symbol"] != ""]
    df = df.drop_duplicates(subset=["gene_symbol"], keep="last")
    df.loc[df["description"] == "", "description"] = "Not well characterized"
    return df["gene_symbol"].tolist(), df["description"].tolist()


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).expanduser().resolve()
    output_ckpt = Path(args.output).expanduser().resolve()
    output_ckpt.parent.mkdir(parents=True, exist_ok=True)

    logger = _setup_logger(
        logger_name="sccafm.gene_embedding",
        log_dir=args.log_dir,
        log_name=args.log_name,
        log_overwrite=args.log_overwrite,
    )
    logger.info("Loading gene descriptions from %s", input_csv)
    genes, descriptions = load_gene_descriptions(input_csv)
    total = len(genes)
    logger.info("Loaded %s unique genes", total)
    if total == 0:
        raise ValueError("No valid genes found in input CSV.")

    use_cuda = args.gpu_index >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu_index)
        device = torch.device(f"cuda:{args.gpu_index}")
        logger.info("Using device %s", device)
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    embedding_chunks = []
    num_batches = (total + args.batch_size - 1) // args.batch_size
    for batch_idx, start in enumerate(range(0, total, args.batch_size), start=1):
        end = min(start + args.batch_size, total)
        batch_text = descriptions[start:end]
        enc = tokenizer(
            batch_text,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            emb = mean_pooling(outputs.last_hidden_state, enc["attention_mask"])

        embedding_chunks.append(emb.detach().cpu().to(torch.float32))
        if args.log_interval > 0 and (batch_idx % args.log_interval == 0 or batch_idx == num_batches):
            logger.info("Embedding progress: batch %s/%s", batch_idx, num_batches)

    embeddings = torch.cat(embedding_chunks, dim=0)
    payload = {
        "gene_symbols": genes,
        "embeddings": embeddings,
        "source_csv": str(input_csv),
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "pooling": "mean",
    }
    torch.save(payload, output_ckpt)
    logger.info("Saved checkpoint to %s", output_ckpt)
    logger.info("Embedding shape: %s", tuple(embeddings.shape))


if __name__ == "__main__":
    main()
