# TODO

- **Mutation-recompute tile placement is asymmetric** (`nn/specs.py`, `tile_pattern`): the greedy
  tiler starts the recompute tile at the coverage window's lower boundary, so at zero shift a
  mutation sits 131 kb from the tile start but 262 kb from its end (tile output 393 kb, coverage
  = mutation ± 131 kb). Harmless for the 2026 CCRE sweep (the 65 kb soft-causality margin holds on
  both sides after the ±65 kb shift; documented in the paper methods), but unusual — future sweeps
  should center the tile on the coverage window (start = coverage_mid − tile_size/2) so the seam
  margins are symmetric. Any change alters recompute geometry and therefore sweep reproducibility;
  do not apply to reruns meant to match the 2026 sweep.
