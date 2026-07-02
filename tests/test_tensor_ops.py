import numpy as np
import pytest

from manta_hic.ops.tensor_ops import round_mantissa


def test_round_mantissa_identity_and_dtype():
    x = np.array([1.5, -0.3, 100.0, 0.0, 3.14], dtype=np.float16)
    # keep_bits >= 10 keeps all mantissa bits -> unchanged copy (but a distinct array)
    y = round_mantissa(x, 10)
    np.testing.assert_array_equal(y, x)
    assert y is not x
    assert round_mantissa(x, 12).dtype == np.float16
    with pytest.raises(ValueError):
        round_mantissa(x.astype(np.float32), 4)


def test_round_mantissa_clears_low_bits_and_keeps_sign():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(50000) * rng.uniform(0.01, 50, 50000)).astype(np.float16)
    for keep in (8, 4, 2, 0):
        y = round_mantissa(x, keep)
        drop = 10 - keep
        # the low `drop` mantissa bits must be zero after rounding
        assert (y.view(np.uint16) & np.uint16((1 << drop) - 1)).sum() == 0
        # sign is preserved (0 has no sign to flip); rounding is to-nearest so error is bounded by one ULP
        nz = x != 0
        assert np.array_equal(np.signbit(y[nz]), np.signbit(x[nz]))
        rel = np.abs((y[nz].astype(np.float32) - x[nz].astype(np.float32)) / x[nz].astype(np.float32))
        assert rel.max() < 2.0 ** -(keep + 1) + 1e-6  # <= half the lowest kept bit


def test_round_mantissa_is_high_fidelity_at_4_bits():
    # The cache default (4 kept bits) must stay tightly correlated with full precision.
    rng = np.random.default_rng(1)
    x = (rng.standard_normal(200000) * rng.uniform(0.01, 30, 200000)).astype(np.float16)
    y = round_mantissa(x, 4)
    corr = np.corrcoef(x.astype(np.float32), y.astype(np.float32))[0, 1]
    assert corr > 0.9999
