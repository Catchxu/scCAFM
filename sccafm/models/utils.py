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


def expand_u(u: torch.Tensor, binary_tf: torch.Tensor):
    """
    Expands u from [C, TF, M] to [C, TG, M] based on the TF indices.
    
    Args:
        u: Factor matrix for TFs of shape (C, TF, M)
        binary_tf: Binary gate/mask for TFs of shape (C, TG, 1)
    """
    C, _, M = u.shape
    TG = binary_tf.shape[1] # The total number of genes in the sequence
    device = u.device
    dtype = u.dtype

    # Initialize the full matrix with zeros
    u_full = torch.zeros((C, TG, M), device=device, dtype=dtype)

    # Get the indices where binary_tf is 1
    # Assuming the TF selection is consistent across cells in the batch
    tf_pos = binary_tf[0].squeeze().nonzero(as_tuple=True)[0]

    # Assign u[c] into u_full[c] for each sample
    for c in range(C):
        # We slice u_full at the TF positions and assign the dense factors
        u_full[c, tf_pos, :] = u[c]

    return u_full