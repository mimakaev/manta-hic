"""Purity contracts of the loss/metric ops: callers may reuse the same tensors for loss AND metrics, so none of
these functions may mutate its arguments (create_expected_matrix used to zero the caller's weight markers, and
hic_hierarchical_loss used to mutate pred/target through reshape views)."""

import pytest
import torch

from manta_hic.ops.hic_ops import coarsegrained_hic_corrs, create_expected_matrix, hic_hierarchical_loss

B, C, N = 2, 3, 32  # N % 16 == 0 (4 hierarchical levels)


def _data(seed=0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.poisson(torch.full((B, C, N, N), 3.0), generator=g)
    raw = (raw + raw.transpose(-1, -2)) / 2
    weight = torch.rand(B, C, N, generator=g) + 0.5
    weight[:, :, 5:8] = 0  # bad bins
    exp = torch.rand(B, C, N * 5 // 4, generator=g) + 0.5
    return raw, weight, exp


def test_create_expected_matrix_is_pure():
    raw, weight, exp = _data()
    args = [raw.clone(), weight.clone(), exp.clone()]
    target, expmat = create_expected_matrix(raw, weight, exp)
    for a, b in zip(args, (raw, weight, exp)):
        assert torch.equal(a, b)
    assert (weight == 0).any()  # the bad-bin markers survived
    assert target is not raw  # returned snippet is a copy...
    assert torch.equal(target[expmat != 0], raw[expmat != 0])  # ...changed only where expected == 0
    assert (target[expmat == 0] == 0).all()


def test_loss_is_pure_and_differentiable():
    raw, weight, exp = _data()
    target, expmat = create_expected_matrix(raw, weight, exp)
    pred = (torch.rand(B, C, N, N) + 0.1).requires_grad_()
    saved = [pred.detach().clone(), target.clone(), expmat.clone()]
    loss = hic_hierarchical_loss(pred, target, expmat)
    loss.backward()
    assert pred.grad is not None and torch.isfinite(loss)
    for a, b in zip(saved, (pred.detach(), target, expmat)):
        assert torch.equal(a, b), "hic_hierarchical_loss mutated an input"


def test_corrs_pure_and_deterministic_across_calls():
    raw, weight, exp = _data()
    pred = torch.rand(B, C, N, N) + 0.1
    saved = [pred.clone(), raw.clone(), weight.clone(), exp.clone()]
    r1 = coarsegrained_hic_corrs(pred, raw, weight, exp, also_divide_by_mean=True)
    r2 = coarsegrained_hic_corrs(pred, raw, weight, exp, also_divide_by_mean=True)  # same tensors, second call
    for a, b in zip(saved, (pred, raw, weight, exp)):
        assert torch.equal(a, b), "coarsegrained_hic_corrs mutated an input"
    for x, y in zip(r1, r2):
        assert torch.equal(x, y), "corr changed between identical calls (leftover in-place mutation)"


def test_loss_rejects_nothing_but_still_needs_div16():
    raw, weight, exp = _data()
    target, expmat = create_expected_matrix(raw, weight, exp)
    pred = torch.rand(B, C, N, N) + 0.1
    # N=32 works; a non-divisible-by-16 N fails in reshape -- documented contract, enforced by the trainer CLI
    with pytest.raises(RuntimeError):
        hic_hierarchical_loss(pred[..., :24, :24], target[..., :24, :24], expmat[..., :24, :24])
