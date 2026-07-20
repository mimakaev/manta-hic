"""manta-hic: predict Hi-C contact maps from DNA sequence."""

import os

# Blosc (de)compression thread count for the HDF5 Blosc filter used by the banded Hi-C stores and the
# MicroZoi activation caches. The filter reads this env var; blosc's threads run in C, outside the GIL, so
# this does not contend with Python work. Measured on a blosc-zstd+bitshuffle cache (warm): 1 thread
# 960 MB/s, 4 threads 1516 MB/s, 8 threads 1685 MB/s. Going 4->8 is +11% speed for only +8% CPU (~2 cores
# actually engaged, not 8 -- fast CPUs saturate a chunk quickly), so 8 never over-subscribes here and gives
# more headroom on slower CPUs / under load where the extra threads do get used. `setdefault` lets an
# explicit environment override win.
os.environ.setdefault("BLOSC_NTHREADS", "8")
