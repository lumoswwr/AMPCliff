#!/usr/bin/env python3
"""Corrected minibatch pairwise/listwise ranking surrogates for STS-B.

All losses operate on raw cosine scores with a FIXED, shared temperature.
They never use MSE or a trainable score scale. The list is the random STS-B
minibatch, NOT a query/document group. LambdaRank here is *batch-NDCG style*,
not a claim of standard query-based LambdaRank.

Methods:
- ranknet: logistic pairwise RankNet
- hinge_fixed: pairwise hinge, margin=1 in logit units (0.05 cosine)
- hinge_label: pairwise hinge, margin in [0.25,1] scaled by label gap
- lambda_ndcg: pairwise RankNet weighted by minibatch delta-NDCG
- listnet_kl: top-one ListNet KL with normalized-label softmax target
- grouped_listmle: grouped Plackett-Luce surrogate using *tie groups* rather
  than imposing arbitrary orders inside equal-label groups. Equals ListMLE
  when all label values differ, but is not exact marginal Plackett-Luce over
  all tie-consistent permutations.
"""
from __future__ import annotations

import math
import torch
import torch.nn.functional as F

METHODS = (
    "ranknet",
    "hinge_fixed",
    "hinge_label",
    "lambda_ndcg",
    "listnet_kl",
    "grouped_listmle",
)

PAIRWISE = frozenset(("ranknet", "hinge_fixed", "hinge_label", "lambda_ndcg"))


def _pairs(labels):
    n = labels.numel()
    indices = torch.triu_indices(n, n, 1, device=labels.device)
    i, j = indices[0], indices[1]
    delta = labels[i] - labels[j]
    selected = delta.abs() > 1e-8
    return i[selected], j[selected], delta[selected], int(selected.sum().item()), int(i.numel())


def compute_ltr_loss(method, cosine, labels, temperature=0.05, label_temperature=0.20):
    """Return (differentiable scalar, eligible comparisons, possible comparisons).

    Labels are STS-B 0..5, cosine in [-1, 1]. All losses use the identical
    positive, fixed score temperature. KL target uses labels/5. No MSE gradient.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}")
    if cosine.ndim != 1 or labels.ndim != 1 or cosine.shape != labels.shape:
        raise ValueError("Scores and labels must be matching 1D vectors")
    if temperature <= 0 or label_temperature <= 0:
        raise ValueError("Temperatures must be positive")
    logits = cosine / temperature
    n = labels.numel()
    possible = n * (n - 1) // 2

    if method in PAIRWISE:
        i, j, delta, eligible, _ = _pairs(labels)
        if not eligible:
            return logits.sum() * 0.0, 0, possible
        signed = torch.sign(delta) * (logits[i] - logits[j])
        if method == "ranknet":
            objective = F.softplus(-signed).mean()
        elif method == "hinge_fixed":
            objective = F.relu(1.0 - signed).mean()
        elif method == "hinge_label":
            # margin spans [0.25,1.0] *temperature in raw-cos units,
            # less punitive than the old raw-score margin of 1.0.
            margin = 0.25 + 0.75 * (delta.abs() / 5.0)
            objective = F.relu(margin - signed).mean()
        else:
            # Approximate LambdaRank: rank positions are NON-differentiable
            # weights; weights derive from batch-local NDCG.
            with torch.no_grad():
                order = torch.argsort(logits.detach(), descending=True, stable=True)
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(n, device=labels.device)
                discounts = 1.0 / torch.log2(ranks.to(logits.dtype) + 2.0)
                gains = torch.pow(2.0, labels / 5.0) - 1.0
                ideal_gains = torch.sort(gains, descending=True).values
                ideal_discount = 1.0 / torch.log2(
                    torch.arange(n, device=labels.device, dtype=logits.dtype) + 2.0
                )
                idcg = (ideal_gains * ideal_discount).sum().clamp_min(1e-12)
                weight = (
                    (gains[i] - gains[j]).abs()
                    * (discounts[i] - discounts[j]).abs()
                    / idcg
                )
                weight_sum = weight.sum()
            if float(weight_sum.detach()) <= 1e-12:
                return logits.sum() * 0.0, 0, possible
            objective = (weight * F.softplus(-signed)).sum() / weight_sum
        return objective, eligible, possible

    if n < 2:
        return logits.sum() * 0.0, 0, possible

    if method == "listnet_kl":
        target_logits = (labels / 5.0) / label_temperature
        qlog = F.log_softmax(target_logits, dim=0)
        q = qlog.exp()
        loss = torch.sum(q * (qlog - F.log_softmax(logits, dim=0)))
        return loss, 0, possible

    # ListMLE grouped (ties are unordered relevance groups). Use one
    # Plackett-Luce choice among group logits at each relevance tier.
    # Group logits are log-MEAN-exp (not logsumexp), avoiding a mechanical
    # advantage for a group merely because it has more tied examples.
    levels = torch.unique(labels.detach(), sorted=True).flip(0)
    if levels.numel() < 2:
        return logits.sum() * 0.0, 0, possible
    group_logits = []
    for level in levels:
        members = logits[labels == level]
        group_logits.append(torch.logsumexp(members, dim=0) - math.log(members.numel()))
    group_logits = torch.stack(group_logits)
    losses = []
    for k in range(group_logits.numel() - 1):
        losses.append(torch.logsumexp(group_logits[k:], dim=0) - group_logits[k])
    return torch.stack(losses).mean(), 0, possible


def self_test():
    """Fail fast: orientation, ties, finite losses, finite/nonzero gradients."""
    labels = torch.tensor([0.0, 0.0, 1.0, 3.0, 5.0], dtype=torch.float32)
    # Moderately separated correct ordering: keeps hinge violations active,
    # so every objective should have a nonzero gradient in this smoke test.
    good = torch.tensor([0.0, 0.0, 0.01, 0.02, 0.03], dtype=torch.float32)
    reverse = torch.tensor([0.03, 0.03, 0.02, 0.01, 0.0], dtype=torch.float32)
    for method in METHODS:
        pred = good.clone().requires_grad_(True)
        lg, selected, possible = compute_ltr_loss(method, pred, labels)
        lb, _, _ = compute_ltr_loss(method, reverse, labels)
        if not torch.isfinite(lg).item() or not torch.isfinite(lb).item():
            raise AssertionError(f"nonfinite objective {method}")
        if not float(lg.detach()) < float(lb.detach()):
            raise AssertionError(f"wrong ranking direction {method}: {lg} >= {lb}")
        if selected < 0 or possible != 10:
            raise AssertionError(f"wrong pair count {method}: {selected}/{possible}")
        lg.backward()
        if pred.grad is None or not torch.isfinite(pred.grad).all().item():
            raise AssertionError(f"gradient invalid {method}")
        if pred.grad.detach().abs().sum().item() <= 0.0:
            raise AssertionError(f"zero gradient {method}")
        # Swapping two equal-label elements must not change objective.
        swap = torch.tensor([1, 0, 2, 3, 4])
        left = compute_ltr_loss(method, good, labels)[0].detach()
        right = compute_ltr_loss(method, good[swap], labels[swap])[0].detach()
        if not torch.allclose(left, right, atol=1e-5):
            raise AssertionError(f"tie order dependence {method}")

    tied = torch.ones_like(labels)
    for method in PAIRWISE | {"grouped_listmle"}:
        z = good.clone().requires_grad_(True)
        loss, count, _ = compute_ltr_loss(method, z, tied)
        if count or abs(float(loss.detach())) > 1e-8:
            raise AssertionError(f"all-tied labels must not impose an ordering: {method}")
        loss.backward()
        if z.grad is None or not torch.isfinite(z.grad).all().item():
            raise AssertionError(f"all-tied loss gradient invalid: {method}")
    print("[loss-self-test] 6 classical pairwise/listwise losses PASS: "
          "orientation, tied labels, gradients, finiteness", flush=True)


if __name__ == "__main__":
    self_test()
