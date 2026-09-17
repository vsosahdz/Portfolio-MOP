"""Thread pinning, applied before any numerical library is imported.

**Why this exists.** XGBoost and PyTorch each link their own OpenMP runtime. On this
platform, importing XGBoost first and then fitting a TabPFN model segfaults the interpreter:

```
   xgboost then tabpfn   ->  SIGSEGV
   tabpfn then xgboost   ->  fine
   either order, threads pinned to one  ->  fine
```

The failure is **order-dependent**, which makes it far worse than a deterministic crash. The
grid iterates screeners in configuration order, so whether a run survives would depend on
which screener happened to come first -- and a multi-hour grid would die partway through with
no Python traceback, only exit code 139.

Pinning every numerical runtime to a single thread avoids the conflict in both directions.
The cost is small here: the measured per-cell optimiser time is a fraction of a second and
the screeners are seconds, so the grid is not thread-bound. It also makes the ablation's
wall-clock measurements more comparable across levels, which that analysis depends on.

**This module must be imported before numpy, torch or xgboost.** ``bmvport/__init__`` does so
first, so importing anything from the package is enough. A caller that imports torch before
bmvport is outside that guarantee, which is why the manifest records the thread settings that
were actually in force rather than the ones this module intended.
"""

from __future__ import annotations

import os

__all__ = ["THREAD_VARIABLES", "pin_threads", "active_thread_settings"]

THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


def pin_threads(threads: int = 1, *, override: bool = False) -> dict[str, str]:
    """Pin numerical runtimes to ``threads``.

    An existing setting is respected unless ``override`` is given: a caller who deliberately
    configured threading should not have it silently changed, and the manifest records what
    was in force either way.

    Returns the settings now active for the tracked variables.
    """
    for variable in THREAD_VARIABLES:
        if override or variable not in os.environ:
            os.environ[variable] = str(threads)
    return active_thread_settings()


def active_thread_settings() -> dict[str, str]:
    """The tracked thread settings currently in the environment."""
    return {v: os.environ[v] for v in THREAD_VARIABLES if v in os.environ}
