"""GEARS expression decoder for contextual EFM gene states.

Adapted from https://github.com/snap-stanford/GEARS.

MIT License

Copyright (c) 2022 Yusuf Roohani, Kexin Huang, Jure Leskovec

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _MLP(nn.Module):
    def __init__(self, sizes: list[int]) -> None:
        super().__init__()
        layers = []
        for input_size, output_size in zip(sizes[:-1], sizes[1:]):
            layers.extend([
                nn.Linear(input_size, output_size),
                nn.BatchNorm1d(output_size),
                nn.ReLU(),
            ])
        self.network = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class GEARSPerturbationHead(nn.Module):
    """Decode per-cell gene states with the reference GEARS forward logic.

    The co-expression branch is inactive for this task. Projected external
    gene vectors are L2-normalized, and control rows are excluded from the
    perturbation-fusion BatchNorm. Contextual EFM states preserve their cell axis;
    supplying repeated gene vectors recovers the reference fixed-vector case.
    """

    def __init__(
        self, *, num_genes: int, num_perturbations: int,
        context_dim: int, hidden_size: int,
        ontology_edges: torch.Tensor, ontology_weights: torch.Tensor,
        use_gene_context: bool,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import SGConv

        self.num_genes = num_genes
        self.num_perturbations = num_perturbations
        self.hidden_size = hidden_size
        self.use_gene_context = use_gene_context
        self.register_buffer("ontology_edges", ontology_edges.long())
        self.register_buffer("ontology_weights", ontology_weights.float())
        if use_gene_context:
            self.context_projection = nn.Linear(context_dim, hidden_size)
        else:
            self.gene_embedding = nn.Embedding(num_genes, hidden_size, max_norm=1.0)
        self.perturbation_embedding = nn.Embedding(
            num_perturbations, hidden_size, max_norm=1.0
        )
        self.perturbation_graph = SGConv(hidden_size, hidden_size, K=1)
        self.base_norm = nn.BatchNorm1d(hidden_size)
        self.base_transform = _MLP([hidden_size, hidden_size, hidden_size])
        self.perturbation_fusion = _MLP([hidden_size, hidden_size, hidden_size])
        self.perturbed_norm = nn.BatchNorm1d(hidden_size)
        self.recovery = _MLP([hidden_size, 2 * hidden_size, hidden_size])
        self.gene_weight = nn.Parameter(torch.rand(num_genes, hidden_size, 1))
        self.gene_bias = nn.Parameter(torch.rand(num_genes, 1))
        nn.init.xavier_normal_(self.gene_weight)
        nn.init.xavier_normal_(self.gene_bias)
        self.cross_gene = _MLP([num_genes, hidden_size, hidden_size])
        self.output_weight = nn.Parameter(torch.rand(1, num_genes, hidden_size + 1))
        self.output_bias = nn.Parameter(torch.rand(1, num_genes))
        nn.init.xavier_normal_(self.output_weight)
        nn.init.xavier_normal_(self.output_bias)

    def forward(
        self, control_expression: torch.Tensor,
        perturbation_mask: torch.Tensor,
        gene_context: torch.Tensor | None,
    ) -> torch.Tensor:
        batch_size = len(control_expression)
        gene_ids = torch.arange(self.num_genes, device=control_expression.device)
        if self.use_gene_context:
            base = F.normalize(self.context_projection(gene_context), p=2, dim=-1)
        else:
            base = self.gene_embedding(gene_ids).unsqueeze(0).expand(batch_size, -1, -1)
        base = self.base_norm(base.reshape(-1, self.hidden_size)).relu()
        base = self.base_transform(base)
        base = base.reshape(batch_size, self.num_genes, self.hidden_size)

        perturbation_ids = torch.arange(
            self.num_perturbations, device=control_expression.device
        )
        perturbations = self.perturbation_graph(
            self.perturbation_embedding(perturbation_ids),
            self.ontology_edges, self.ontology_weights,
        )
        rows = perturbation_mask.any(1).nonzero(as_tuple=True)[0]
        if len(rows):
            summed = perturbation_mask[rows] @ perturbations
            # The reference duplicates a singleton perturbed row for BatchNorm.
            if len(rows) == 1:
                fused = self.perturbation_fusion(summed.repeat(2, 1))[:1]
            else:
                fused = self.perturbation_fusion(summed)
            base = base.index_add(
                0, rows, fused.unsqueeze(1).expand(-1, self.num_genes, -1)
            )
        base = self.perturbed_norm(base.reshape(-1, self.hidden_size)).relu()
        recovered = self.recovery(base)
        recovered = recovered.reshape(batch_size, self.num_genes, self.hidden_size)
        first = (recovered.unsqueeze(-1) * self.gene_weight).sum(2) + self.gene_bias
        cross = self.cross_gene(first.squeeze(-1)).unsqueeze(1).expand(
            -1, self.num_genes, -1
        )
        delta = (torch.cat([first, cross], dim=-1) * self.output_weight).sum(-1)
        return control_expression + (delta + self.output_bias)
