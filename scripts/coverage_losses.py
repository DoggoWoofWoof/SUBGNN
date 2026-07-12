"""Coverage-focused retrieval losses for Jigsaw training."""

import math

import torch
import torch.nn.functional as F


def _aggregate_positive_terms(
    per_positive_terms,
    positive_mask,
    mode="mean",
    cvar_fraction=0.25,
    smoothmax_temperature=0.1,
):
    """Aggregate variable-count positive terms one query at a time."""
    aggregated = []
    for row_terms, row_mask in zip(per_positive_terms, positive_mask):
        values = row_terms[row_mask]
        if values.numel() == 0:
            continue
        if mode == "mean":
            aggregated.append(values.mean())
        elif mode == "cvar":
            count = max(1, math.ceil(values.numel() * float(cvar_fraction)))
            aggregated.append(torch.topk(values, count).values.mean())
        elif mode == "smoothmax":
            temperature = max(1e-6, float(smoothmax_temperature))
            aggregated.append(
                temperature * torch.logsumexp(values / temperature, dim=0)
                - temperature * math.log(values.numel())
            )
        else:
            raise ValueError(
                f"Unknown positive aggregation {mode!r}; use mean, cvar, or smoothmax"
            )
    if not aggregated:
        return per_positive_terms.new_tensor(0.0)
    return torch.stack(aggregated).mean()


def partition_coverage_loss(
    zq,
    partition_embeddings,
    query_partition_ids,
    temperature=0.05,
    target_topk=0,
    topk_bucket_size=10,
    topk_weight=0.0,
    topk_margin=0.0,
    positive_aggregation="mean",
    cvar_fraction=0.25,
    smoothmax_temperature=0.1,
    live_partition_indices=None,
    live_partition_embeddings=None,
):
    """
    Multi-label coverage loss aligned with FullCov@K.

    `cvar` and `smoothmax` emphasize the weakest required partitions. Optional
    live embeddings replace stale cached rows and allow gradients to reach the
    positive partition encoder as well as the query encoder.
    """
    if live_partition_indices is not None and live_partition_embeddings is not None:
        partition_embeddings = partition_embeddings.clone()
        partition_embeddings[live_partition_indices] = live_partition_embeddings

    batch_size = zq.shape[0]
    num_partitions = partition_embeddings.shape[0]
    logits = torch.matmul(zq, partition_embeddings.T) / temperature

    positive_mask = torch.zeros(
        batch_size, num_partitions, dtype=torch.bool, device=zq.device
    )
    for row, partition_ids in enumerate(query_partition_ids):
        for partition_id in partition_ids:
            if 0 <= partition_id < num_partitions:
                positive_mask[row, partition_id] = True

    valid = positive_mask.any(dim=1)
    if not valid.any():
        return zq.new_tensor(0.0)
    logits = logits[valid]
    positive_mask = positive_mask[valid]

    log_probs = F.log_softmax(logits, dim=1)
    per_positive_ce = -log_probs

    negative_logits = logits.masked_fill(positive_mask, float("-inf"))
    hardest_negative = negative_logits.max(dim=1).values
    has_negative = torch.isfinite(hardest_negative)
    per_positive_margin = F.softplus(hardest_negative.unsqueeze(1) - logits)
    per_positive_margin = torch.where(
        has_negative.unsqueeze(1),
        per_positive_margin,
        torch.zeros_like(per_positive_margin),
    )
    loss = _aggregate_positive_terms(
        per_positive_ce + 0.25 * per_positive_margin,
        positive_mask,
        mode=positive_aggregation,
        cvar_fraction=cvar_fraction,
        smoothmax_temperature=smoothmax_temperature,
    )

    if target_topk > 0 and topk_weight > 0.0:
        barrier_rows = []
        barrier_masks = []
        for row_logits, row_positive_mask in zip(logits, positive_mask):
            positive_count = int(row_positive_mask.sum().item())
            if positive_count == 0:
                continue
            bucket_size = max(1, int(topk_bucket_size))
            effective_topk = (
                int(target_topk)
                if positive_count <= target_topk
                else ((positive_count + bucket_size - 1) // bucket_size)
                * bucket_size
            )
            negative_row = row_logits[~row_positive_mask]
            if negative_row.numel() == 0:
                continue
            kth_negative = min(
                max(1, effective_topk - positive_count + 1),
                negative_row.numel(),
            )
            threshold = torch.topk(negative_row, kth_negative).values[-1]
            barrier_rows.append(
                F.softplus(threshold + topk_margin - row_logits)
            )
            barrier_masks.append(row_positive_mask)
        if barrier_rows:
            barrier_loss = _aggregate_positive_terms(
                torch.stack(barrier_rows),
                torch.stack(barrier_masks),
                mode=positive_aggregation,
                cvar_fraction=cvar_fraction,
                smoothmax_temperature=smoothmax_temperature,
            )
            loss = loss + topk_weight * barrier_loss

    return loss
