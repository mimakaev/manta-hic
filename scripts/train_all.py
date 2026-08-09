#!/usr/bin/env python
"""Campaign scheduler: co-train medium Manta models for the whole banded collection, per (genome, resolution).

- One `manta_hic train manta` invocation per (genome, res), co-training every dataset present at that res
  (shared MicroZoi fetch = less IO). Epochs auto = library RES_EPOCHS. val-fold 7 / test-fold 0 (-> _70 cache).
- 2 GPU slots run 2 jobs at once. Coarse->fine (lightest runs first). mm10 jobs wait for the mm10_70 cache copy.
- Restart-safe: a dataset is 'done' when its <name>.pth history reaches RES_EPOCHS[res]; finished datasets are
  dropped from a job's model list, and a fully-done (genome,res) is skipped.
Outputs: /workspace/manta_trained_2026/<genome>_<res>/<dataset>.pth  (+ train.log)
"""
import os, sys, time, subprocess, glob
import torch

BV2 = "/mnt/rw/2024_intact_hic/banded_v2"
MAX = "/mnt/ro/max/2026-manta-banded-inputs"
SSD = "/workspace/data_ssd"
OUTROOT = "/workspace/manta_trained_2026"
PRESET, BATCH, VALF, TESTF = "medium", "4", "7", "0"
RES_LIST = [256, 512, 1024, 2048, 4096, 8192, 16384]      # fine -> coarse: fewest epochs + smallest 1D
# input first (coarse res has tower_height up to 5 => 64x longer 1D input AND up to 160 epochs = far heavier).
RES_EPOCHS = {256: 10, 512: 20, 1024: 30, 2048: 50, 4096: 80, 8192: 120, 16384: 160}

# dataset -> preferred dir. banded_v2 (new grouping) wins; external/mm10 from the max collection.
HG38_V2 = ["HCT116_architectural", "HCT116_transcription", "T2_all", "cardiovascular", "celllines_1",
           "celllines_2", "combined-everything", "digestive", "immune_Tcell", "immune_innate_B",
           "nervous", "organs_misc", "reproductive"]
HG38_EXT = ["krietenstein", "hansen_microc_combined", "4dn-diff", "sarcoma-flyone", "hic_2009"]
MM10 = ["bonev", "bonev-merged", "hsieh", "masahiro", "masahiro-10B", "masahiro-2B", "masahiro-500m",
        "masahiro-5B", "masahiro2"]
GENOMES = {"hg38": HG38_V2 + HG38_EXT, "mm10": MM10}
CACHE = {"hg38": f"{SSD}/microzoi_cache_hg38_70.h5", "mm10": f"{SSD}/microzoi_cache_mm10_70.h5"}


def path_for(ds, res):
    for d in (BV2, MAX):
        p = f"{d}/{ds}_{res}.bhic.h5"
        if os.path.exists(p):
            return p
    return None


def done(outdir, ds, res):
    p = f"{outdir}/{ds}.pth"
    if not os.path.exists(p):
        return False
    try:
        ck = torch.load(p, map_location="cpu", weights_only=False)
        return len(ck["config"].get("history", [])) >= RES_EPOCHS[res]
    except Exception:
        return False


def build_jobs():
    jobs = []
    for res in RES_LIST:
        for genome, dss in GENOMES.items():
            outdir = f"{OUTROOT}/{genome}_{res}"
            members = [(ds, path_for(ds, res)) for ds in dss]
            members = [(ds, p) for ds, p in members if p]
            if members:
                jobs.append(dict(genome=genome, res=res, outdir=outdir, members=members))
    return jobs


def remaining(job):
    return [(ds, p) for ds, p in job["members"] if not done(job["outdir"], ds, job["res"])]


def launch(job, device):
    os.makedirs(job["outdir"], exist_ok=True)
    rem = remaining(job)
    # plain --preset: Manta auto-expands final_channels to fit each model's output_channels, so no per-model
    # override is needed (medium works for 1..16+ channels alike).
    cmd = ["manta_hic", "train", "manta"]
    for ds, p in rem:
        cmd += ["-m", f"{ds}={p}"]
    cmd += ["-c", CACHE[job["genome"]], "-o", job["outdir"], "-g", job["genome"],
            "--preset", PRESET, "--batch-size", BATCH, "--val-fold", VALF, "--test-fold", TESTF, "-d", device]
    log = open(f"{job['outdir']}/train.log", "a")
    log.write(f"\n===== launch {job['genome']}_{job['res']} on {device}: {len(rem)} models, "
              f"{RES_EPOCHS[job['res']]} epochs =====\n{' '.join(cmd)}\n")
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)


def cache_ready(job):
    return os.path.exists(CACHE[job["genome"]])


def main():
    jobs = build_jobs()
    print(f"[sched] {len(jobs)} (genome,res) jobs. Plan (coarse->fine):", flush=True)
    for j in jobs:
        r = remaining(j)
        print(f"  {j['genome']}_{j['res']}: {len(j['members'])} datasets ({len(r)} to train), "
              f"{RES_EPOCHS[j['res']]} ep -> {j['outdir']}", flush=True)

    pending = [j for j in jobs if remaining(j)]
    devices = ["cuda:0", "cuda:1"]
    running = {}  # device -> (proc, job)
    while pending or running:
        # assign free devices
        for dev in devices:
            if dev in running or not pending:
                continue
            # pick next pending job whose cache is ready; else leave dev idle this tick
            idx = next((i for i, j in enumerate(pending) if cache_ready(j)), None)
            if idx is None:
                continue
            job = pending.pop(idx)
            if not remaining(job):
                continue
            proc = launch(job, dev)
            running[dev] = (proc, job)
            print(f"[sched] START {job['genome']}_{job['res']} on {dev} "
                  f"({len(remaining(job))} models)", flush=True)
        # poll
        time.sleep(30)
        for dev in list(running):
            proc, job = running[dev]
            if proc.poll() is not None:
                rc = proc.returncode
                left = remaining(job)
                status = "OK" if rc == 0 and not left else f"rc={rc}, {len(left)} unfinished"
                print(f"[sched] DONE  {job['genome']}_{job['res']} on {dev}: {status}", flush=True)
                del running[dev]
                if rc != 0 and left:          # re-queue once if it died with work left
                    if not job.get("_retried"):
                        job["_retried"] = True
                        pending.append(job)
                        print(f"[sched] REQUEUE {job['genome']}_{job['res']} (retry)", flush=True)
    print("[sched] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
