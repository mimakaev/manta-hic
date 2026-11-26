"""MicroZoi checkpoint versioning: self-describing checkpoints, checked strict loading of old raw state dicts, the
cache round trip, and the MicroZoi-version guard between Manta heads and caches. CPU, tiny models."""

import io
import json

import h5py
import numpy as np
import pytest
import torch

from manta_hic.nn.fetchers import create_microzoi_model_from_cache
from manta_hic.nn.inference import MantaInference
from manta_hic.nn.manta import MANTA_PRESETS, manta_from_preset, save_manta_checkpoint
from manta_hic.nn.microzoi import (
    MICROZOI_ARCH_KEYS,
    Microzoi,
    load_microzoi,
    microzoi_checkpoint,
    resolve_microzoi_checkpoint,
    save_microzoi,
)

SMALL = dict(
    base_channels=32,
    n_heads=4,
    transf_layers=2,
    seq_length=2**18,
    last_channels_mult=8,
    output_channels_human=6,
    output_channels_mouse=4,
)


def _model(qk_norm, **kw):
    torch.manual_seed(0)
    m = Microzoi(**SMALL, qk_norm=qk_norm, **kw).eval()
    if qk_norm:  # move the QK-norm gains off their init, so a model that silently skipped them would differ
        with torch.no_grad():
            for name, p in m.named_parameters():
                if "_norm" in name and "mha_tower" in name:
                    p.mul_(1.5)
    return m


def _same_output(a, b):
    torch.manual_seed(1)
    x = torch.nn.functional.one_hot(torch.randint(0, 4, (1, SMALL["seq_length"])), 4).permute(0, 2, 1).float()
    with torch.no_grad():
        return torch.equal(a.trunk(x), b.trunk(x))


@pytest.mark.parametrize("qk_norm,version", [(False, 1), (True, 2)])
def test_checkpoint_is_self_describing(tmp_path, qk_norm, version):
    m = _model(qk_norm)
    save_microzoi(m, tmp_path / "m.pth")
    ck = torch.load(tmp_path / "m.pth", weights_only=True)
    assert ck["version"] == m.version == version
    assert set(ck["config"]) == set(MICROZOI_ARCH_KEYS) and ck["config"]["n_heads"] == 4
    m2 = load_microzoi(tmp_path / "m.pth")  # no params needed: the checkpoint carries its architecture
    assert m2.version == version and _same_output(m, m2)


def test_checkpoint_rejects_conflicting_params(tmp_path):
    save_microzoi(_model(True), tmp_path / "m.pth")
    for bad in ({"qk_norm": False}, {"n_heads": 8}, {"seq_length": 2**19}):  # n_heads/seq_length load "fine"
        with pytest.raises(ValueError, match="disagree"):
            load_microzoi(tmp_path / "m.pth", SMALL | {"qk_norm": True} | bad)
    # non-architecture params (dropouts, checkpointing) may differ
    load_microzoi(tmp_path / "m.pth", SMALL | {"qk_norm": True, "attn_dropout": 0.0, "checkpoint_first": False})


def test_checkpoint_rejects_inconsistent_version():
    ck = microzoi_checkpoint(_model(True))
    ck["version"] = 1
    with pytest.raises(ValueError, match="version"):
        resolve_microzoi_checkpoint(ck)


@pytest.mark.parametrize("weights_qk,params_qk", [(True, False), (False, True)])
def test_raw_state_dict_version_must_match_params(weights_qk, params_qk):
    sd = _model(weights_qk).state_dict()
    with pytest.raises(ValueError, match="QK-norm"):
        load_microzoi(sd, SMALL | {"qk_norm": params_qk})


def test_raw_state_dict_loads_strictly_with_matching_params():
    m = _model(True)
    m2 = load_microzoi(m.state_dict(), SMALL | {"qk_norm": True})
    assert _same_output(m, m2)
    with pytest.raises(RuntimeError):  # any other shape mismatch is still caught by the strict load
        load_microzoi(m.state_dict(), SMALL | {"qk_norm": True, "last_channels_mult": 16})


def test_mha_load_drops_heads():
    m = _model(False)
    m2 = load_microzoi(microzoi_checkpoint(m), return_type="mha")
    assert m2.return_type == "mha" and not hasattr(m2, "final_conv_human") and _same_output(m, m2)


def _write_cache(path, blob_obj, model_params, version=None):
    with io.BytesIO() as buf:
        torch.save(blob_obj, buf)
        blob = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    with h5py.File(path, "w") as f:
        f.attrs["model_params"] = json.dumps({"model": model_params})
        if version is not None:
            f.attrs["microzoi_version"] = version
        f.create_dataset("model_blob", data=blob)


def test_cache_blob_round_trip(tmp_path):
    m = Microzoi(**SMALL, qk_norm=True, return_type="mha").eval()
    # new caches: self-describing blob
    _write_cache(tmp_path / "new.h5", microzoi_checkpoint(m), SMALL | {"qk_norm": True}, version=2)
    assert _same_output(m, create_microzoi_model_from_cache(tmp_path / "new.h5", device="cpu"))
    # old caches: raw state dict + model_params attr
    m1 = Microzoi(**SMALL, return_type="mha").eval()
    _write_cache(tmp_path / "old.h5", m1.state_dict(), SMALL)
    assert _same_output(m1, create_microzoi_model_from_cache(tmp_path / "old.h5", device="cpu"))
    # a cache whose recorded params disagree with its weights is refused, not silently loaded
    _write_cache(tmp_path / "bad.h5", m.state_dict(), SMALL)
    with pytest.raises(ValueError):
        create_microzoi_model_from_cache(tmp_path / "bad.h5", device="cpu")


@pytest.mark.parametrize("head_version,cache_version,ok", [(None, 1, True), (None, 2, False), (2, 2, True),
                                                           (2, 1, False), (1, None, True)])
def test_manta_rejects_cache_of_other_microzoi_version(tmp_path, head_version, cache_version, ok):
    m = manta_from_preset("opt1M", n_bins=256, bins_pad=64, output_channels=2, tower_height=1)
    save_manta_checkpoint(m, tmp_path / "h.pth", model_params=MANTA_PRESETS["opt1M"], microzoi_version=head_version)

    class _Fetcher:  # a cache fetcher of the given version (None: a fetcher that reads no cache)
        pass

    fetcher = _Fetcher()
    if cache_version is not None:
        fetcher.microzoi_version = cache_version
    if ok:
        assert MantaInference(tmp_path / "h.pth", fetcher, device="cpu").microzoi_version == (head_version or 1)
    else:
        with pytest.raises(ValueError, match="MicroZoi version"):
            MantaInference(tmp_path / "h.pth", fetcher, device="cpu")
