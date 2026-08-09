"""
Compute benchmark for Manta architectures: forward+backward step time (the training-relevant cost),
param count, and peak activation memory. bf16 autocast, batch 8, output_channels=4, 2048bp (tower_height=2)
-- the laptop-training target. Reports median step time over N iters after warmup, on whichever CUDA device
is freest. The GPU here is a 4090 (shared w/ polymer sims ~half load), so ABSOLUTE times are ~2x a free 4090;
RATIOS between archs are contention-robust and are the primary output.
"""
import argparse, time, json
import torch
from manta_hic.nn.manta import Manta, manta_from_preset, MANTA_PRESETS

TOWER_H = 2          # 2048bp
INPUT_CH = 1024 + 8
BINS_PAD = 64


def build(preset_or_params, n_bins, oc):
    if isinstance(preset_or_params, str):
        return manta_from_preset(preset_or_params, n_bins=n_bins, bins_pad=BINS_PAD,
                                 output_channels=oc, tower_height=TOWER_H)
    return Manta(n_bins=n_bins, bins_pad=BINS_PAD, output_channels=oc, tower_height=TOWER_H,
                 **preset_or_params)


def bench(model, n_bins, *, batch=8, oc=4, dtype=torch.bfloat16, iters=12, warmup=4, device="cuda:0"):
    model = model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    L = 2 ** (TOWER_H + 1) * (n_bins + 2 * BINS_PAD)
    x = torch.randn(batch, INPUT_CH, L, device=device)
    tgt = torch.randn(batch, oc, n_bins, n_bins, device=device).abs()
    torch.cuda.reset_peak_memory_stats(device)

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=dtype):
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, tgt)
        loss.backward()
        opt.step()
        return loss

    for _ in range(warmup):
        step()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize(device)
    dt = (time.perf_counter() - t0) / iters
    peak = torch.cuda.max_memory_allocated(device) / 1e9
    nparam = sum(p.numel() for p in model.parameters())
    return dict(step_ms=dt * 1e3, peak_gb=peak, params_m=nparam / 1e6)


def pick_device():
    # choose the GPU with lowest utilization*memory proxy: just least memory used
    best, bestmem = 0, 1e18
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        used = total - free
        if used < bestmem:
            best, bestmem = i, used
    return f"cuda:{best}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="medium")
    ap.add_argument("--n-bins", default="512", type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--oc", default=4, type=int)
    ap.add_argument("--batch", default=8, type=int)
    ap.add_argument("--iters", default=12, type=int)
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    dev = pick_device()
    print(f"device={dev}  batch={args.batch}  oc={args.oc}  tower_h={TOWER_H}(2048bp)  bf16")
    rows = []
    for preset in args.presets.split(","):
        for nb in args.n_bins:
            try:
                m = build(preset, nb, args.oc)
                r = bench(m, nb, batch=args.batch, oc=args.oc, iters=args.iters, device=dev)
                r.update(preset=preset, n_bins=nb)
                rows.append(r)
                print(f"  {preset:10s} n_bins={nb:5d}  {r['params_m']:6.3f}M  "
                      f"step={r['step_ms']:8.1f}ms  peak={r['peak_gb']:5.2f}GB")
            except torch.OutOfMemoryError:
                print(f"  {preset:10s} n_bins={nb:5d}  OOM (skipped)")
            finally:
                del m
                torch.cuda.empty_cache()
                torch.cuda.synchronize(dev)
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)
        print("wrote", args.json)
