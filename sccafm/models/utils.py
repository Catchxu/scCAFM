import torch


def reparameterize(mu: torch.Tensor, sigma: torch.Tensor):
    if mu.shape != sigma.shape:
        raise ValueError(
            f"mu and sigma should have the same shape, got {mu.shape} and {sigma.shape}!"
        )
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


def expand_grn(grn: torch.Tensor, binary_tf: torch.Tensor, binary_tg: torch.Tensor):
    C, _, TG = grn.shape
    device = grn.device
    dtype = grn.dtype

    grn_full = torch.zeros((C, TG, TG), device=device, dtype=dtype)

    # Since all rows are the same, take the first row as reference
    tf_pos = binary_tf[0].nonzero(as_tuple=True)[0]
    tg_pos = binary_tg[0].nonzero(as_tuple=True)[0]

    # Assign grn[c] into grn_full[c] for each sample
    for c in range(C):
        grn_full[c][tf_pos[:, None], tg_pos] = grn[c]

    return grn_full