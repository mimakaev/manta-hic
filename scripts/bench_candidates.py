"""Benchmark candidate 'small' presets vs medium. Design principle from attribution: the 1D backbone +
transformer are ~free (<2% each), all compute is the 2D tower (height/width/channels). So keep the front-end
rich and only trim the 2D tower. Reports step time, speedup vs medium, params, at each n_bins."""
import torch
from bench_arch import bench, build, pick_device

MED = dict(channels_1d=192, transformer_layers=4, tower_2d_height=9, tower_2d_channels=24,
           direct_2d_channels=8, tower_2d_input_channels=48, direct_2d_input_channels=16, final_channels=16,
           tower_2d_width=5)

# All candidates KEEP channels_1d=192 + transformer_layers=4 (nearly free) and the asymmetric direct split.
def cand(**ov):
    p = dict(MED); p.update(ov); return p

CANDS = {
    "medium":       MED,
    # placeholder (cuts free stuff): channels_1d=128, tl=2, h=3, c=16, w=5
    "placeholder":  dict(channels_1d=128, transformer_layers=2, tower_2d_height=3, tower_2d_channels=16,
                         direct_2d_channels=16, tower_2d_input_channels=32, direct_2d_input_channels=24,
                         final_channels=16, tower_2d_width=5),
    # keep rich front-end + rich tower channels, cut geometry (height 9->5, width 5->3)
    "S_geom":       cand(tower_2d_height=5, tower_2d_width=3, tower_2d_channels=24),
    # a touch deeper, lean channels
    "S_deep_lean":  cand(tower_2d_height=6, tower_2d_width=3, tower_2d_channels=16),
    # balanced middle (rich channels, moderate depth)
    "S_mid":        cand(tower_2d_height=4, tower_2d_width=3, tower_2d_channels=24),
    # aggressive shallow lean
    "S_aggr":       cand(tower_2d_height=3, tower_2d_width=3, tower_2d_channels=16),
    # shallow but keep rich channels (channels matter for cell-type signal)
    "S_shallow_rich": cand(tower_2d_height=3, tower_2d_width=3, tower_2d_channels=24),
}

if __name__ == "__main__":
    dev = pick_device()
    for nb in (512, 768):
        print(f"\n=== n_bins={nb}  (2048bp, bs=8, oc=4, bf16, {dev}) ===")
        base = None
        results = {}
        for name, p in CANDS.items():
            m = None
            try:
                m = build(p, nb, 4)
                r = bench(m, nb, oc=4, iters=12, device=dev)
                results[name] = r
                if name == "medium":
                    base = r["step_ms"]
                sp = base / r["step_ms"] if base else 1.0
                print(f"  {name:16s} {r['params_m']:6.3f}M  step={r['step_ms']:7.1f}ms  "
                      f"speedup={sp:4.2f}x  peak={r['peak_gb']:5.2f}GB")
            except torch.OutOfMemoryError:
                print(f"  {name:16s} OOM")
            finally:
                del m; torch.cuda.empty_cache(); torch.cuda.synchronize(dev)

