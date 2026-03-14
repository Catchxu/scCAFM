import torch


def _format_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def summarize_model_size(model: torch.nn.Module) -> dict:
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_buffers = sum(b.numel() for b in model.buffers())

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    total_bytes = param_bytes + buffer_bytes

    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "total_buffers": int(total_buffers),
        "param_bytes": int(param_bytes),
        "buffer_bytes": int(buffer_bytes),
        "total_bytes": int(total_bytes),
        "param_size": _format_bytes(param_bytes),
        "buffer_size": _format_bytes(buffer_bytes),
        "total_size": _format_bytes(total_bytes),
    }


def print_model_size(model: torch.nn.Module, prefix: str = "Model size") -> dict:
    stats = summarize_model_size(model)
    print(
        f"{prefix} | "
        f"params={stats['total_params']:,} "
        f"trainable={stats['trainable_params']:,}"
    )
    return stats
