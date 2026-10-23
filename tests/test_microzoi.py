import math

import pytest
import torch

from manta_hic.nn.layers import apply_rotary_emb, precompute_rope
from manta_hic.nn.microzoi import Microzoi, borzoi_loss
from manta_hic.nn.train_microzoi import lr_at

SMALL = dict(
    tower_mults=[8, 9, 10, 11, 12, 14, 16, 16],
    base_channels=32,
    n_heads=4,
    transf_layers=2,
    seq_length=2**18,  # smallest power of 2 whose crop still covers Borzoi's 196,608 bp target window
    last_channels_mult=8,
    output_channels_human=6,
    output_channels_mouse=4,
)


@pytest.mark.parametrize("checkpoint_first", [True, False])
@pytest.mark.parametrize("qk_norm", [False, True])
@pytest.mark.parametrize("num_gn_checkpoints", [0, 2])
def test_every_parameter_trains(checkpoint_first, qk_norm, num_gn_checkpoints):
    """Every parameter of the active head path gets a nonzero gradient. Guards against the reentrant-checkpoint
    bug that left the first convolution of the first-generation MicroZoi trunks at its random initialization."""
    torch.manual_seed(0)
    model = Microzoi(
        **SMALL, checkpoint_first=checkpoint_first, qk_norm=qk_norm, num_gn_checkpoints=num_gn_checkpoints
    ).train()
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (1, SMALL["seq_length"])), 4).float().permute(0, 2, 1)
    out = model(x, genome="hg38")
    loss = borzoi_loss(out, torch.rand_like(out) * 5)
    loss.backward()
    dead = [n for n, p in model.named_parameters() if "mouse" not in n and (p.grad is None or not p.grad.any())]
    assert dead == []


def _rotary_complex(xq, xk, theta=10000.0):
    """The original (llama) complex-number rotary embedding, as used to train the first-generation models."""
    d, n = xq.shape[-1], xq.shape[1]
    freqs = 1.0 / (theta ** (torch.arange(0, d, 2)[: (d // 2)].float() / d))
    freqs_cis = torch.polar(torch.ones(n, d // 2), torch.outer(torch.arange(n), freqs).float())[None, :, None]
    rot = lambda x: torch.view_as_real(torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) * freqs_cis)
    return rot(xq).flatten(3), rot(xk).flatten(3)


def test_rotary_matches_complex_version():
    torch.manual_seed(0)
    xq, xk = torch.randn(2, 300, 4, 64), torch.randn(2, 300, 4, 64)
    q1, k1 = apply_rotary_emb(xq, xk, precompute_rope(64, 300))
    q0, k0 = _rotary_complex(xq, xk)
    torch.testing.assert_close(q1, q0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(k1, k0, atol=1e-5, rtol=1e-5)


def test_rotary_rejects_low_precision_table():
    x = torch.randn(1, 8, 2, 16)
    with pytest.raises(TypeError):
        apply_rotary_emb(x, x, precompute_rope(16, 8).bfloat16())


def test_borzoi_loss_total_weight():
    torch.manual_seed(0)
    out, targ = torch.rand(2, 3, 50) + 0.1, torch.rand(2, 3, 50) * 4
    loss = lambda w: borzoi_loss(out, targ, total_weight=w)
    assert loss(0.5) - loss(0.0) == pytest.approx(0.5 * (loss(1.0) - loss(0.0)))  # linear in the weight
    assert loss(0.5) != pytest.approx(loss(0.2))


def test_lr_schedule():
    kw = dict(total_steps=1000, max_lr=1.0, warmup_steps=100)
    assert lr_at(0, **kw, schedule="cosine", min_lr_frac=0.0) == pytest.approx(0.01)
    assert lr_at(99, **kw, schedule="cosine", min_lr_frac=0.0) == pytest.approx(1.0)
    assert lr_at(550, **kw, schedule="cosine", min_lr_frac=0.0) == pytest.approx(0.5)
    assert lr_at(1000, **kw, schedule="cosine", min_lr_frac=0.1) == pytest.approx(0.1)
    assert lr_at(999, **kw, schedule="constant", min_lr_frac=0.0) == 1.0
    assert math.isclose(lr_at(325, **kw, schedule="cosine", min_lr_frac=0.0), 0.5 * (1 + math.cos(math.pi / 4)))
