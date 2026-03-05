import argparse
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_NAME = "/data1021/xukaichen/data/LLM/Llama-3.1-8B-Instruct"
DEFAULT_GPU_INDICES = [0, 1, 2, 3]
SYSTEM_PROMPT = "You are a biomedical expert."


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


def _make_worker_log_name(base_name: str, gpu_index: int) -> str:
    p = Path(base_name)
    if p.suffix:
        return f"{p.stem}.gpu{gpu_index}{p.suffix}"
    return f"{base_name}.gpu{gpu_index}.log"


def build_gene_prompt(gene_symbol: str) -> str:
    prompt = f"""
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


def ask_model(
    prompt: str,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    max_new_tokens: int,
    target_device: torch.device,
) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(target_device)
    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response.strip()


def load_gene_symbols(token_dict_path: Path) -> list[str]:
    df = pd.read_csv(token_dict_path)
    if "gene_symbol" not in df.columns:
        raise ValueError(f"`gene_symbol` column not found in {token_dict_path}")

    symbols = df["gene_symbol"].dropna().astype(str).str.strip()
    symbols = [s for s in symbols.tolist() if s]
    return list(dict.fromkeys(symbols))


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


def shard_items(items: list[str], num_shards: int) -> list[list[str]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    shards = [[] for _ in range(num_shards)]
    for i, item in enumerate(items):
        shards[i % num_shards].append(item)
    return shards


def process_gene_shard(
    gene_symbols: list[str],
    gpu_index: int,
    model_name: str,
    max_new_tokens: int,
    log_interval: int,
    log_dir: str,
    log_name: str,
    log_overwrite: bool,
) -> list[dict]:
    worker_logger = _setup_logger(
        logger_name=f"sccafm.gene_text.gpu{gpu_index}",
        log_dir=log_dir,
        log_name=_make_worker_log_name(log_name, gpu_index),
        log_overwrite=log_overwrite,
    )

    use_cuda = gpu_index >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(gpu_index)
        target_device = torch.device(f"cuda:{gpu_index}")
        device_map = {"": f"cuda:{gpu_index}"}
        dtype = torch.bfloat16
    else:
        target_device = torch.device("cpu")
        device_map = "cpu"
        dtype = torch.float32

    worker_logger.info("GPU %s starting; assigned genes=%s", gpu_index, len(gene_symbols))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    rows = []
    total = len(gene_symbols)
    for i, gene_symbol in enumerate(gene_symbols, start=1):
        row = {
            "gene_symbol": gene_symbol,
            "gpu_index": gpu_index,
            "status": "ok",
            "description": "",
            "error": "",
        }
        try:
            prompt = build_gene_prompt(gene_symbol)
            row["description"] = ask_model(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
                target_device=target_device,
            )
        except Exception as exc:
            row["status"] = "error"
            row["error"] = str(exc)
        rows.append(row)

        if log_interval > 0 and (i % log_interval == 0 or i == total):
            worker_logger.info("GPU %s progress: %s/%s genes processed", gpu_index, i, total)

    return rows


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
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Local model path or model id",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of newly generated tokens",
    )
    parser.add_argument(
        "--gpu-indices",
        type=int,
        nargs="+",
        default=DEFAULT_GPU_INDICES,
        help="One process per GPU index (example: --gpu-indices 0 1 2 3). Default: 0 1 2 3.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory for logs",
    )
    parser.add_argument(
        "--log-name",
        type=str,
        default="gene_text.log",
        help="Main log filename",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=20,
        help="Log progress every N genes",
    )
    parser.add_argument(
        "--log-overwrite",
        action="store_true",
        help="Overwrite logs instead of appending",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    token_dict_path = Path(args.token_dict).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    logger = _setup_logger(
        logger_name="sccafm.gene_text",
        log_dir=args.log_dir,
        log_name=args.log_name,
        log_overwrite=args.log_overwrite,
    )

    gene_symbols = load_gene_symbols(token_dict_path)
    logger.info("Loaded %s unique gene symbols from %s", len(gene_symbols), token_dict_path)
    logger.info("Regenerating descriptions for all genes.")

    if not gene_symbols:
        logger.info("No gene symbols found. Nothing to do.")
        return

    if output_path.exists():
        output_path.unlink()
        logger.info("Removed existing output: %s", output_path)

    gpu_indices = list(dict.fromkeys(args.gpu_indices))
    shards = shard_items(gene_symbols, len(gpu_indices))
    logger.info("Using GPUs: %s", gpu_indices)

    rows = []
    processed_total = 0
    with ProcessPoolExecutor(
        max_workers=len(gpu_indices),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        future_to_gpu = {}
        for gpu_index, shard in zip(gpu_indices, shards):
            if not shard:
                continue
            future = executor.submit(
                process_gene_shard,
                gene_symbols=shard,
                gpu_index=gpu_index,
                model_name=args.model_name,
                max_new_tokens=args.max_new_tokens,
                log_interval=args.log_interval,
                log_dir=args.log_dir,
                log_name=args.log_name,
                log_overwrite=args.log_overwrite,
            )
            future_to_gpu[future] = gpu_index

        for future in as_completed(future_to_gpu):
            gpu_index = future_to_gpu[future]
            shard_rows = future.result()
            rows.extend(shard_rows)
            save_results(output_path, shard_rows)
            processed_total += len(shard_rows)
            ok_count = sum(1 for row in shard_rows if row["status"] == "ok")
            err_count = len(shard_rows) - ok_count
            logger.info(
                "GPU %s completed %s genes (ok=%s, error=%s). Global progress: %s/%s",
                gpu_index,
                len(shard_rows),
                ok_count,
                err_count,
                processed_total,
                len(gene_symbols),
            )

    total_ok = sum(1 for row in rows if row["status"] == "ok")
    total_err = len(rows) - total_ok
    logger.info("Saved %s rows to %s (ok=%s, error=%s)", len(rows), output_path, total_ok, total_err)


if __name__ == "__main__":
    main()
