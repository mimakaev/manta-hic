"""Single-knob attribution: start from a base preset, change ONE arch knob, measure fwd+bwd step-time delta.
Tells us the marginal training cost of each lever (fwd+bwd, the real target) at a given n_bins."""
import argparse, torch
from manta_hic.nn.manta import MANTA_PRESETS
from bench_arch import bench, build, TOWER_H, BINS_PAD


def run(base_name, n_bins, oc, iters, dev):
    base = dict(MANTA_PRESETS[base_name])
    knobs = {
        "tower_2d_height 9->5": dict(tower_2d_height=5),
        "tower_2d_height 9->3": dict(tower_2d_height=3),
        "tower_2d_channels 24->16": dict(tower_2d_channels=16),
        "tower_2d_width 5->3": dict(tower_2d_width=3),
        "transformer_layers 4->2": dict(transformer_layers=2),
        "channels_1d 192->128": dict(channels_1d=128),
        "channels_1d 192->96": dict(channels_1d=96),
        "tower_2d_input_ch 48->32": dict(tower_2d_input_channels=32),
    }
    m = build(base_name, n_bins, oc)
    r0 = bench(m, n_bins, oc=oc, iters=iters, device=dev); del m; torch.cuda.empty_cache()
    print(f"BASE {base_name} n_bins={n_bins}: {r0['step_ms']:.1f}ms  {r0['params_m']:.3f}M  {r0['peak_gb']:.2f}GB")
    for label, ov in knobs.items():
        p = dict(base); p.update(ov)
        try:
            m = build(p, n_bins, oc)
            r = bench(m, n_bins, oc=oc, iters=iters, device=dev)
            d = r0["step_ms"] - r["step_ms"]
            print(f"  {label:28s} -> {r['step_ms']:7.1f}ms  (saves {d:6.1f}ms, {100*d/r0['step_ms']:4.1f}%)  "
                  f"{r['params_m']:.3f}M")
        except torch.OutOfMemoryError:
            print(f"  {label:28s} -> OOM")
        finally:
            del m; torch.cuda.empty_cache(); torch.cuda.synchronize(dev)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="medium")
    ap.add_argument("--n-bins", default=768, type=int)
    ap.add_argument("--oc", default=4, type=int)
    ap.add_argument("--iters", default=12, type=int)
    args = ap.parse_args()
    from bench_arch import pick_device
    run(args.base, args.n_bins, args.oc, args.iters, pick_device())
