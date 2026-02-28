import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


DEFAULT_GPU_INDICES = [0, 1, 2, 3]


def build_gene_prompt(gene_symbol: str) -> str:
    prompt = f"""
        [ROLE]
        You are a biomedical expert.

        [OBJECTIVE]
        Provide structured, factual, and concise gene descriptions.

        [INPUT]
        Gene symbol: {gene_symbol}

        [OUTPUT FORMAT]
        1. Gene Overview
        - Full gene name
        - Synonyms (if any)
        - Organism (assume Homo sapiens unless otherwise specified)
        - Chromosomal location

        2. Gene Function
        - Molecular function
        - Biological processes involved
        - Cellular role

        3. Protein Information
        - Protein name
        - Protein function
        - Key domains or structural features

        4. Expression Profile
        - Tissue-specific expression (high/low/ubiquitous)
        - Developmental or condition-specific expression (if known)

        5. Pathways and Interactions
        - Major biological pathways
        - Known interacting genes or proteins

        6. Clinical Relevance
        - Associated diseases or disorders
        - Known mutations and their effects
        - Clinical significance (if applicable)

        7. Additional Notes
        - Evolutionary conservation (if known)
        - Any notable research findings or unique characteristics

        [CONSTRAINTS]
        - Follow the requested section format strictly
        - Avoid speculation or unsupported claims
        - Use clear and professional scientific language
        - Be concise but informative
        - If information is unknown, state "Not well characterized"
    """.strip()
    return prompt


def ask_ollama(prompt: str, server_name: str, gpu_index: int, model_name: str, timeout: int = 180) -> str:
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "options": {"seed": 42},
    }
    port = f"{server_name}{gpu_index}"
    response = requests.post(
        f"http://localhost:{port}/api/generate",
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    if "response" not in data:
        raise ValueError(f"Missing 'response' field in Ollama payload from port {port}")
    return data["response"].strip()


def load_gene_symbols(token_dict_path: Path) -> list[str]:
    df = pd.read_csv(token_dict_path)
    if "gene_symbol" not in df.columns:
        raise ValueError(f"`gene_symbol` column not found in {token_dict_path}")

    symbols = (
        df["gene_symbol"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    symbols = [s for s in symbols.tolist() if s]
    # Preserve order while removing duplicates.
    return list(dict.fromkeys(symbols))


def load_completed_symbols(output_path: Path) -> set[str]:
    if not output_path.exists():
        return set()
    df = pd.read_csv(output_path)
    if "gene_symbol" not in df.columns:
        return set()
    return set(df["gene_symbol"].dropna().astype(str).tolist())


def shard_items(items: list[str], num_shards: int) -> list[list[str]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)
    return shards


def fetch_gene_descriptions_for_gpu(
    gene_symbols: list[str],
    gpu_index: int,
    server_name: str,
    model_name: str,
    timeout: int,
    output_path: Path,
    save_lock: threading.Lock,
) -> list[dict]:
    rows = []
    for gene_symbol in gene_symbols:
        row = {
            "gene_symbol": gene_symbol,
            "gpu_index": gpu_index,
            "status": "ok",
            "description": "",
            "error": "",
        }
        try:
            prompt = build_gene_prompt(gene_symbol)
            row["description"] = ask_ollama(
                prompt=prompt,
                server_name=server_name,
                gpu_index=gpu_index,
                model_name=model_name,
                timeout=timeout,
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)
        with save_lock:
            save_results(output_path, [row])
    return rows


def save_results(output_path: Path, rows: list[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(rows)
    if output_path.exists():
        old_df = pd.read_csv(output_path)
        merged = pd.concat([old_df, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["gene_symbol"], keep="last")
    else:
        merged = new_df
    merged.to_csv(output_path, index=False)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate gene descriptions for all token_dict gene symbols.")
    parser.add_argument(
        "--token-dict",
        type=str,
        default="resources/token_dict.csv",
        help="Path to token_dict.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="resources/gene_descriptions.csv",
        help="Path to output CSV",
    )
    parser.add_argument(
        "--server-name",
        type=str,
        default="102",
        help="Port prefix; final port is <server-name><gpu-index>, e.g. 1020",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="gemma3:27b",
        help="Ollama model name",
    )
    parser.add_argument(
        "--gpu-indices",
        type=int,
        nargs="+",
        default=DEFAULT_GPU_INDICES,
        help="GPU indices / Ollama port suffixes to query in parallel",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of genes to process",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Ignore existing output file and regenerate everything",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token_dict_path = Path(args.token_dict).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    gene_symbols = load_gene_symbols(token_dict_path)
    if args.limit is not None:
        gene_symbols = gene_symbols[: max(0, args.limit)]

    completed = set() if args.overwrite else load_completed_symbols(output_path)
    pending_symbols = [g for g in gene_symbols if g not in completed]

    print(f"Loaded {len(gene_symbols)} gene symbols from {token_dict_path}")
    print(f"Existing completed symbols: {len(completed)}")
    print(f"Pending symbols: {len(pending_symbols)}")

    if not pending_symbols:
        print("No pending gene symbols. Nothing to do.")
        return

    gpu_indices = list(dict.fromkeys(args.gpu_indices))
    shards = shard_items(pending_symbols, len(gpu_indices))
    save_lock = threading.Lock()

    all_rows = []
    with ThreadPoolExecutor(max_workers=len(gpu_indices)) as executor:
        future_to_gpu = {}
        for gpu_index, shard in zip(gpu_indices, shards):
            if not shard:
                continue
            future = executor.submit(
                fetch_gene_descriptions_for_gpu,
                gene_symbols=shard,
                gpu_index=gpu_index,
                server_name=args.server_name,
                model_name=args.model_name,
                timeout=args.timeout,
                output_path=output_path,
                save_lock=save_lock,
            )
            future_to_gpu[future] = gpu_index

        for future in as_completed(future_to_gpu):
            gpu_index = future_to_gpu[future]
            rows = future.result()
            all_rows.extend(rows)
            ok_count = sum(1 for row in rows if row["status"] == "ok")
            err_count = len(rows) - ok_count
            print(f"GPU {gpu_index}: completed {len(rows)} genes (ok={ok_count}, error={err_count})")

    total_ok = sum(1 for row in all_rows if row["status"] == "ok")
    total_err = len(all_rows) - total_ok
    print(f"Saved {len(all_rows)} rows to {output_path} (ok={total_ok}, error={total_err})")


if __name__ == "__main__":
    main()
