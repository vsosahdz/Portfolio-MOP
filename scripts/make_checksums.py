"""Record what was retrieved and when, so a reader can verify an independent download.

The cached series are redistributed to support replication, but they carry their providers'
terms rather than this package's licence. A reader who prefers to obtain them directly can
check their own download against these digests instead of trusting the copy here.

Writes data/checksums.json.
"""
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

MANIFEST = json.loads(Path("results/manifest.json").read_text(encoding="utf-8"))
records = {}
for path in sorted(Path("data").rglob("*")):
    if not path.is_file() or path.name == "checksums.json":
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    records[str(path.as_posix())] = {"sha256": digest, "bytes": path.stat().st_size}

Path("data/checksums.json").write_text(json.dumps({
    "data_vintage": MANIFEST["data_vintage"],
    "config_fingerprint": MANIFEST["config_fingerprint"],
    "note": ("Digests of the cached dataset of record. The equity series come from a free "
             "provider and the rate and exchange series from Banco de México's SIE API; both "
             "are redistributed here only to support replication."),
    "files": records,
}, indent=2), encoding="utf-8")
print(f"{len(records)} files, {sum(r['bytes'] for r in records.values()) / 1e6:.1f} MB")
