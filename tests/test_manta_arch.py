"""final_channels auto-expansion: one preset must work for any output_channels (no per-model override)."""
import math

import pytest
import torch

from manta_hic.nn.manta import MANTA_PRESETS, Manta, manta_from_preset, save_manta_checkpoint


def _final_channels(model):
    # join_conv maps (direct+tower) -> final_channels, so its conv out_channels IS final_channels
    return model.join_conv.conv.out_channels


def _expected(floor, oc):
    return max(floor, -(-2 * oc // 8) * 8)  # ceil(2*oc/8)*8, floored at the preset value


@pytest.mark.parametrize("oc", [1, 2, 4, 8, 9, 13, 16, 20])
def test_final_channels_autoexpand(oc):
    floor = MANTA_PRESETS["opt1M"]["final_channels"]  # 16
    m = manta_from_preset("opt1M", n_bins=256, bins_pad=64, output_channels=oc, tower_height=1)
    fc = _final_channels(m)
    assert fc == _expected(floor, oc)
    assert fc >= 2 * oc and fc % 8 == 0
    if oc <= floor // 2:  # low-channel models are untouched (exact preset)
        assert fc == floor


def test_high_channel_builds_and_roundtrips(tmp_path):
    # oc=13 (bonev) used to raise "Final channels must be at least 2 times the output channels"
    m = manta_from_preset("opt1M", n_bins=256, bins_pad=64, output_channels=13, tower_height=1)
    assert _final_channels(m) == 32
    x = torch.randn(1, 1032, 2 ** (1 + 1) * (256 + 2 * 64))
    with torch.no_grad():
        assert m(x).shape == (1, 13, 256, 256)
    p = tmp_path / "m.pth"
    save_manta_checkpoint(m, str(p), model_params=MANTA_PRESETS["opt1M"])
    ck = torch.load(p, map_location="cpu", weights_only=False)
    m2 = manta_from_preset("opt1M", n_bins=256, bins_pad=64,
                           output_channels=ck["config"]["output_channels"], tower_height=1)
    m2.load_state_dict(ck["state_dict"], strict=True)  # deterministic expansion -> shapes match on reload


def test_explicit_final_channels_is_floor_not_cap():
    # an explicitly large final_channels is preserved; expansion only ever raises it
    m = Manta(n_bins=256, bins_pad=64, output_channels=2, tower_height=1, final_channels=48)
    assert _final_channels(m) == 48


def test_init_rejects_odd_n_bins_and_zero_pad():
    with pytest.raises(ValueError, match="even"):
        Manta(n_bins=255, bins_pad=64, output_channels=2, tower_height=1)
    with pytest.raises(ValueError, match="bins_pad"):
        Manta(n_bins=256, bins_pad=0, output_channels=2, tower_height=1)


def test_checkpoint_records_effective_final_channels(tmp_path):
    # oc=13 expands the preset's 16 -> 32; the saved config must say 32, not the configured floor
    m = manta_from_preset("opt1M", n_bins=256, bins_pad=64, output_channels=13, tower_height=1)
    p = tmp_path / "m.pth"
    save_manta_checkpoint(m, str(p), model_params=MANTA_PRESETS["opt1M"])
    ck = torch.load(p, map_location="cpu", weights_only=False)
    assert ck["config"]["model_params"]["final_channels"] == _final_channels(m) == 32
    # feeding the recorded (effective) value back is a no-op: same shapes, strict reload works
    m2 = Manta(n_bins=256, bins_pad=64, output_channels=13, tower_height=1,
               **{**MANTA_PRESETS["opt1M"], **{"final_channels": 32}})
    m2.load_state_dict(ck["state_dict"], strict=True)
