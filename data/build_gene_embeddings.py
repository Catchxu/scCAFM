import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer

from src.assets import load_vocab_json, save_vocab_tensor_file


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "/data1021/xukaichen/data/LLM/pubmedbert-base-embeddings"
DEFAULT_TOKEN_DICT_JSON = ROOT_DIR / "assets" / "models" / "vocab.json"
DEFAULT_INPUT_CSV = ROOT_DIR / "checkpoints" / "gene_go_terms.csv"
DEFAULT_OUTPUT_CKPT = ROOT_DIR / "checkpoints" / "models" / "vocab.safetensors"
DEFAULT_BATCH_SIZE = 1024
UNKNOWN_TERM_TEXT = "GO term: unknown. Domain: unknown."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--token-dict", type=Path, default=DEFAULT_TOKEN_DICT_JSON)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-ckpt", type=Path, default=DEFAULT_OUTPUT_CKPT)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--gpus",
        default="0,1,2,3",
        help="Comma-separated CUDA device ids, for example: 0 or 0,1,2,3",
    )
    return parser.parse_args()


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    expanded_mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * expanded_mask, dim=1) / torch.clamp(
        expanded_mask.sum(dim=1), min=1e-9
    )


def parse_gpu_ids(gpus_arg):
    gpu_ids = [int(part.strip()) for part in gpus_arg.split(",") if part.strip()]
    if not gpu_ids:
        raise ValueError("`--gpus` must specify at least one CUDA device id.")
    return gpu_ids


def load_gene_tokens(path):
    token_path = Path(path).expanduser().resolve()
    if token_path.suffix.lower() == ".json":
        token_df = load_vocab_json(token_path)
    else:
        token_df = pd.read_csv(token_path)
    token_df = token_df.dropna(subset=["gene_symbol", "gene_id"]).copy()
    token_df["gene_symbol"] = token_df["gene_symbol"].astype(str).str.strip()
    token_df["gene_id"] = token_df["gene_id"].astype(str).str.strip()
    token_df = token_df[token_df["gene_id"].str.startswith("ENSG")].copy()
    token_df["token_index"] = token_df["token_index"].astype(int)
    return token_df.sort_values("token_index").reset_index(drop=True)


def load_gene_go_terms(path):
    if not path.exists():
        return pd.DataFrame(columns=["gene_id", "go_term_name", "go_domain"])

    term_df = pd.read_csv(path)
    expected_columns = {"gene_id", "go_term_name", "go_domain"}
    missing_columns = expected_columns - set(term_df.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing_columns)}")

    term_df = term_df.dropna(subset=["gene_id"]).copy()
    term_df["gene_id"] = term_df["gene_id"].astype(str).str.strip()
    return term_df


def build_gene_term_texts(token_df, term_df):
    valid_term_df = term_df.dropna(subset=["go_term_name", "go_domain"]).copy()
    valid_term_df["text"] = (
        "GO term: "
        + valid_term_df["go_term_name"].astype(str).str.strip()
        + ". Domain: "
        + valid_term_df["go_domain"].astype(str).str.strip()
        + "."
    )

    texts_by_gene = (
        valid_term_df.groupby("gene_id")["text"]
        .apply(lambda values: list(dict.fromkeys(values.tolist())))
        .to_dict()
    )

    records = []
    for row in token_df.itertuples(index=False):
        texts = texts_by_gene.get(row.gene_id)
        if not texts:
            texts = [UNKNOWN_TERM_TEXT]
        records.append(
            {
                "token_index": row.token_index,
                "gene_symbol": row.gene_symbol,
                "gene_id": row.gene_id,
                "texts": texts,
            }
        )

    return pd.DataFrame(records)


def collect_unique_texts(gene_text_df):
    unique_texts = []
    seen = set()

    for texts in gene_text_df["texts"]:
        for text in texts:
            if text not in seen:
                seen.add(text)
                unique_texts.append(text)

    return unique_texts


def encode_texts(texts, tokenizer, model, batch_size, device):
    embeddings = []

    for start in range(0, len(texts), batch_size):
        batch_texts = texts[start:start + batch_size]
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        batch_embeddings = mean_pooling(outputs, inputs["attention_mask"]).cpu()
        embeddings.append(batch_embeddings)

    return torch.cat(embeddings, dim=0)


def build_gene_embeddings(gene_text_df, text_to_embedding):
    gene_symbols = []
    gene_embeddings = []

    for row in gene_text_df.itertuples(index=False):
        term_embeddings = torch.stack([text_to_embedding[text] for text in row.texts], dim=0)
        gene_symbols.append(row.gene_symbol)
        gene_embeddings.append(term_embeddings.mean(dim=0))

    return gene_symbols, torch.stack(gene_embeddings, dim=0)


def load_model(model_name, gpu_ids):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script, but no GPU is available.")
    if max(gpu_ids) >= torch.cuda.device_count():
        raise RuntimeError(
            f"Requested GPU ids {gpu_ids}, but only {torch.cuda.device_count()} CUDA devices are available."
        )

    device = torch.device(f"cuda:{gpu_ids[0]}")
    model = AutoModel.from_pretrained(model_name).to(device)
    if len(gpu_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids)
    model.eval()
    return model, device


def main():
    args = parse_args()
    gpu_ids = parse_gpu_ids(args.gpus)

    token_df = load_gene_tokens(args.token_dict)
    term_df = load_gene_go_terms(args.input_csv)
    gene_text_df = build_gene_term_texts(token_df, term_df)
    unique_texts = collect_unique_texts(gene_text_df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model, device = load_model(args.model_name, gpu_ids)
    text_embeddings = encode_texts(unique_texts, tokenizer, model, args.batch_size, device)
    text_to_embedding = {
        text: embedding for text, embedding in zip(unique_texts, text_embeddings)
    }
    gene_symbols, embeddings = build_gene_embeddings(gene_text_df, text_to_embedding)

    args.output_ckpt.parent.mkdir(parents=True, exist_ok=True)
    if args.output_ckpt.suffix.lower() == ".safetensors":
        save_vocab_tensor_file(args.output_ckpt, embeddings)
    else:
        torch.save(
            {
                "gene_symbols": gene_symbols,
                "embeddings": embeddings,
            },
            args.output_ckpt,
        )

    print(
        f"Saved {len(gene_symbols)} gene embeddings to {args.output_ckpt} "
        f"using GPUs {gpu_ids}"
    )




if __name__ == "__main__":
    main()
