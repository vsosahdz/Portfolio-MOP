"""Prove the cache-only claim instead of asserting it.

The data availability statement says the study reproduces with the network unavailable. That
is a claim a reader can test, so it should be one the package tests too: a stage that quietly
reaches for a provider would not show up in any other check, and would fail for whoever tried
to replicate rather than for us.

DNS resolution is what gets blocked, not the socket class. Replacing ``socket.socket`` breaks
``ssl.SSLSocket``, which subclasses it, and the resulting failure looks like a network error
when it is an error in the harness.
"""
import socket
import ssl  # noqa: F401  -- imported before the block so its class hierarchy is intact
import sys
import time
import warnings

sys.path.insert(0, "src")
warnings.filterwarnings("ignore")


class NetworkBlocked(RuntimeError):
    pass


def _deny(*args, **kwargs):
    raise NetworkBlocked("network access attempted")


socket.getaddrinfo = _deny
socket.create_connection = _deny
socket.socket.connect = _deny
socket.socket.connect_ex = _deny

try:
    socket.getaddrinfo("example.com", 443)
except NetworkBlocked:
    print("block verified: getaddrinfo raises")
else:
    sys.exit("the block is not in effect; this check would prove nothing")

from bmvport.backtest.results import build_results_table  # noqa: E402
from bmvport.config import load_config  # noqa: E402
from bmvport.grid import load_grid_inputs, run_grid  # noqa: E402

config = load_config("configs/smoke.yaml")
started = time.time()
rows, failures = run_grid(config, load_grid_inputs(config), progress=False)
frame = build_results_table(rows)
print(f"smoke grid with no network: {len(frame):,} rows, {len(failures)} failures, "
      f"{time.time() - started:.1f}s")
print(f"  months={frame.month.nunique()}  "
      f"screeners={sorted(frame.screener.dropna().unique())}")
if failures or frame.empty:
    sys.exit("the offline claim does not hold")
print("the cache-only claim holds")
