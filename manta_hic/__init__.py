"""manta-hic: predict Hi-C contact maps from DNA sequence."""

import os

# Blosc (de)compression thread count for the HDF5 Blosc filter used by the banded Hi-C stores and the
# MicroZoi activation caches. The filter reads this env var; 4 threads make blosc-zstd reads ~30% faster
# than the old single-threaded zstd on the activation cache (and speed up writes too). Blosc's threads run
# in C, outside the GIL, so this does not contend with Python work. `setdefault` lets an explicit
# environment override win.
os.environ.setdefault("BLOSC_NTHREADS", "4")
