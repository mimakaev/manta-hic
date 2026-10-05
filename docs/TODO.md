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
