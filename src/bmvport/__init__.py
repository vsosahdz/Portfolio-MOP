"""Multi-objective portfolio selection with ML stock screening on the Mexican Stock Exchange.

Replication package for Sosa Hernández, Rodríguez Rodríguez and Monroy Borja. The pipeline
stages are independent: each reads the previous stage's stored artefacts, so a defect found
late does not require re-running everything.
"""

# Imported first, before any numerical library: XGBoost and PyTorch each link their own
# OpenMP runtime and their combination segfaults this interpreter in one import order.
# See bmvport._threading for the measurement.
from . import _threading as _threading

_threading.pin_threads()

__version__ = "0.1.0"
