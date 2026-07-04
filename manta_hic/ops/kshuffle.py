"""
Pure-Python k-let-preserving sequence shuffle -- a drop-in replacement for the ``ushuffle`` C extension,
which is unmaintained and does not build on Python 3.12+ (it includes the removed ``longintrepr.h``).

Given a sequence, produce a random rearrangement that preserves the exact count of every *k-let* (contiguous
k-mer): ``k=1`` is a plain symbol permutation, ``k=2`` preserves dinucleotide content, ``k=4`` preserves
tetranucleotide content, and so on. This is the Euler-path algorithm of Altschul & Erikson (1985),
generalized to arbitrary ``k`` -- the same method the uShuffle tool (Jiang et al. 2008, BMC Bioinformatics)
implements in C.

Why not a plain permutation? Local composition (di/tetranucleotide frequency) alone can read as
"promoter-like"/"regulatory" to a sequence model, so a k-let-preserving null is the right control for "same
local composition, scrambled arrangement".

The sequences here are small (a regulatory element, or the 4 kb quiescent tile) and this runs only on the
mutation/inference paths -- never the training hot loop -- so pure Python is comfortably fast enough.

Algorithm
---------
Read the length-``L`` sequence as a walk on a multigraph whose vertices are the ``(k-1)``-mers and whose
edges are the k-lets: edge ``i`` goes from vertex ``s[i:i+k-1]`` to ``s[i+1:i+k]`` and carries the new symbol
``s[i+k-1]``. Any walk that uses every edge exactly once (an Eulerian path from the first to the last
``(k-1)``-mer) spells a sequence with identical k-let counts. To draw one (Altschul-Erikson):

1. For every vertex except the terminal one, pick an outgoing edge to traverse *last*, such that these
   "last" edges form a tree into the terminal vertex. A uniform such tree is drawn directly with Wilson's
   loop-erased random walk (no rejection -- naive rejection sampling blows up as the vertex count grows,
   e.g. ~64 tries for tetranucleotide shuffles).
2. At each vertex, randomly order the remaining outgoing edges and append its "last" edge at the end.
3. Walk from the first vertex, always taking the next unused edge and emitting its symbol.

The result is the first ``k-1`` symbols followed by the emitted symbols: same length, same first and last
``(k-1)``-mer, same k-let counts. Sampling is uniform over exactly that set -- verified de novo against
brute-force enumeration and against the reference uShuffle C tool (``test_kshuffle``), including the
Eulerian-*circuit* case where the first and last ``(k-1)``-mers coincide.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

# Module-default RNG, used when a caller does not pass its own. ``set_seed`` reseeds it; prefer passing an
# explicit ``rng`` to :func:`k_let_shuffle` for reproducibility.
_default_rng = np.random.default_rng()


def set_seed(seed: int | None = None) -> None:
    """Reseed the module-default RNG (parity with the old ``ushuffle.set_seed``). ``None`` = fresh entropy."""
    global _default_rng
    _default_rng = np.random.default_rng(None if seed is None else int(seed))


def _last_edges_toward(out, sources, terminal, rng) -> dict:
    """Draw a uniform arborescence into ``terminal`` via Wilson's algorithm; return ``{vertex: edge_index}``
    giving each source's "last" (toward-root) outgoing edge.

    Wilson's loop-erased random walk: from each not-yet-connected vertex, take a random-successor walk
    (uniform over its outgoing edges, weighted by multiplicity -- exactly the weighting Altschul-Erikson
    needs) until it hits the growing tree, overwriting each vertex's chosen edge on revisits (so loops erase
    themselves); then splice the loop-erased path into the tree. Terminal is reachable from every vertex (the
    original sequence is an Eulerian path ending there), so every walk terminates. O(expected edges), no
    rejection.
    """
    in_tree = {terminal}
    next_idx: dict = {}
    for start in sources:
        v = start
        while v not in in_tree:  # random-successor walk, recording (and overwriting) the edge taken
            j = int(rng.integers(len(out[v])))
            next_idx[v] = j
            v = out[v][j][0]
        v = start
        while v not in in_tree:  # retrace the loop-erased path, attaching it to the tree
            in_tree.add(v)
            v = out[v][next_idx[v]][0]
    return next_idx


def k_let_shuffle(seq: bytes | str, k: int, rng: np.random.Generator | None = None) -> bytes:
    """
    Shuffle ``seq`` preserving exact k-let (k-mer) counts; returns ``bytes`` (regardless of input type).

    ``k=1`` is a plain permutation. If ``len(seq) <= k`` only one arrangement preserves the k-lets, so the
    input is returned unchanged. Pass ``rng`` (a ``numpy.random.Generator``) for reproducibility; otherwise
    the module-default RNG is used.
    """
    rng = _default_rng if rng is None else rng
    data = seq.encode() if isinstance(seq, str) else bytes(seq)
    k = int(k)
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    L = len(data)

    if k == 1:
        arr = np.frombuffer(data, dtype=np.uint8).copy()
        rng.shuffle(arr)
        return arr.tobytes()
    if L <= k:
        return data

    # Vertices are the (k-1)-mers at positions 0..L-k+1; edge i: verts[i] -> verts[i+1], symbol data[i+k-1].
    n_edge = L - k + 1
    verts = [data[i : i + k - 1] for i in range(L - k + 2)]
    terminal = verts[-1]
    out: dict[bytes, list[tuple[bytes, int]]] = defaultdict(list)  # vertex -> [(dest_vertex, symbol_byte)]
    for i in range(n_edge):
        out[verts[i]].append((verts[i + 1], data[i + k - 1]))

    sources = [v for v in out if v != terminal]  # vertices that need a "last" edge chosen
    last_idx = _last_edges_toward(out, sources, terminal, rng)

    ordered: dict[bytes, list[tuple[bytes, int]]] = {}  # per vertex: edges in traversal order
    for v, edges in out.items():
        if v == terminal:  # root: no "last" constraint, order all edges freely
            ordered[v] = [edges[j] for j in rng.permutation(len(edges))]
        else:
            j = last_idx[v]
            rest = edges[:j] + edges[j + 1 :]
            ordered[v] = [rest[t] for t in rng.permutation(len(rest))] + [edges[j]]

    ptr: dict[bytes, int] = defaultdict(int)
    cur = verts[0]
    result = bytearray(verts[0])  # the first (k-1) symbols are fixed by the start vertex
    for _ in range(n_edge):
        dest, sym = ordered[cur][ptr[cur]]
        ptr[cur] += 1
        result.append(sym)
        cur = dest
    return bytes(result)
