# obsolete/

Retired code kept for reference/reproducibility. **Not** part of the installed package (excluded by
`pyproject.toml`'s `packages.find`), not on the CLI, not tested.

## `mutate_manta.py`

The original mutation-screen driver (`manta_hic mutate manta-mutate-file`). For each region it recomputed
MicroZoi from scratch over the whole Manta window for 4 replicates (random crop / tile-offset / bp-shift per
rep) and ran every Manta head, then aggregated WT-vs-mutant log-ratios. **This is the tool the current
publication sweep was produced with** — keep it readable so those results stay reproducible.

Superseded by the spec-based inference stack (`nn/specs.py` + `fetch_activations_batch`; see
`docs/INFERENCE_SPEC_PLAN.md`), which reuses cached activations instead of recomputing from scratch. A
rewrite is planned (the old path likely under-randomized). It still imports live package symbols
(`Manta`, `fetch_tile_microzoi_activations`, `make_quiescent_seq`), so `python obsolete/mutate_manta.py`
continues to run.
