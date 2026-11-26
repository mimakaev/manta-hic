# TODO

- **Mutation-recompute tile placement is asymmetric** (`nn/specs.py`, `tile_pattern`): the greedy
  tiler starts the recompute tile at the coverage window's lower boundary, so at zero shift a
  mutation sits 131 kb from the tile start but 262 kb from its end (tile output 393 kb, coverage
  = mutation ± 131 kb). Harmless for the 2026 CCRE sweep (the 65 kb soft-causality margin holds on
  both sides after the ±65 kb shift; documented in the paper methods), but unusual — future sweeps
  should center the tile on the coverage window (start = coverage_mid − tile_size/2) so the seam
  margins are symmetric. Any change alters recompute geometry and therefore sweep reproducibility;
  do not apply to reruns meant to match the 2026 sweep.

- **Legacy-checkpoint evaluations before commit 84589b5 are suspect**: the tower-branch distance
  matrix was mis-scaled for `legacy=True` models (fixed in 84589b5). Re-evaluate anything scored
  from 2024-era checkpoints with the refactored library before that commit.

- **Rename `create_expected_matrix`** (`ops/hic_ops.py`): it returns the distance expectation
  multiplied by the per-bin balancing biases (w_i·w_j), i.e. the expected *raw* count matrix, not
  the expected. Callers that want the plain distance expectation (e.g. to display log contact
  probability from O/E) must build the Toeplitz from `exp` directly. (Max, 2026-09-11.)

- **MicroZoi v2 is merged but experimental — do not train it before the paper is out.**
  `configs/microzoi_v2.json` is a proposed recipe, not a trained model. It is the same `Microzoi`
  class with opt-in flags (`qk_norm=True`, `ff_dropout` separate from `attn_dropout`) plus a new
  training loop (AdamW, warmup + cosine, grad clipping, bf16, `torch.compile`, EMA). Every
  released model and cache in the paper is first-generation MicroZoi, and the library defaults
  still build that model, so existing checkpoints load strictly and give the same outputs (checked
  on `microzoi_all.pth` and a Manta head: activations match `main` to ~4e-7 relative, heads to
  ~2e-4, which is fp32 rounding from the real-arithmetic RoPE, far below bf16 precision).
  Before a v2 run:
  - Versions are tracked: `microzoi_version` = 1 (no QK-norm) or 2 (QK-norm). New MicroZoi
    checkpoints, including the weights stored in caches, carry their version and full
    architecture; `load_microzoi` loads them, and the old raw state dicts, strictly and refuses
    params that disagree. Caches record the version as an attr, Manta checkpoints record the
    version of the cache they were trained on, and `MantaInference` refuses a head/cache
    mismatch. Anything without a version counts as v1. Fold (12/34/56/70/all) is not checked:
    a head can still be paired with a cache of another fold.
  - `checkpoint_first=True` (now the default) lets `conv0` train; first-generation training did
    not (reentrant-checkpoint bug). Exactly reproducing v1 training needs `checkpoint_first=False`
    plus the old caller-side reentrant checkpoint.
  - A v2 MicroZoi needs its own activation caches and newly trained Manta heads; nothing
    downstream exists for it yet.
  - The recipe has not been validated end to end (no full run, no comparison against v1).
