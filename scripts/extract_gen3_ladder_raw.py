"""
Extract gen3 battles from the HF metamon-raw-replays v6 parquet shards
(shards 41-45 of 46 contain all gen3* formatids) into per-battle raw JSONs
for the v6.1 reparse. Skips smogtours-* ids (already downloaded separately
via the Showdown replay API).
"""

import io
import json
import os

import re

import httpx
import pyarrow.parquet as pq

OUT = "/home/eddie/metamon_cache/raw-replays-ladder"
V6 = "https://huggingface.co/datasets/jakegrigsby/metamon-raw-replays/resolve/v6/data/train-%05d-of-00046.parquet"
GEN3 = {"gen3ou", "gen3ubers", "gen3uu", "gen3nu"}
COLS = ["id", "format", "players", "log", "uploadtime", "formatid", "rating"]
# Keep only true Showdown ladder battles: "gen3ou-2021773630" etc.
# This drops ancient side-server imports ("pokemononline-gen3ou-16",
# "china-gen3ou-*", "rom-gen3ou-*", ...) and junk formats.
SHOWDOWN_ID = re.compile(r"^gen3(ou|uu|nu|ubers)-\d+$")

os.makedirs(OUT, exist_ok=True)
counts = {t: 0 for t in GEN3}
skipped_smogtours = 0
skipped_ancient = 0

for i in range(41, 46):
    url = V6 % i
    print(f"downloading shard {i}...", flush=True)
    r = httpx.get(url, follow_redirects=True, timeout=900)
    r.raise_for_status()
    pf = pq.ParquetFile(io.BytesIO(r.content))
    for batch in pf.iter_batches(batch_size=4000, columns=COLS):
        d = batch.to_pydict()
        for j in range(len(d["id"])):
            fmt = d["formatid"][j]
            rid = d["id"][j]
            if fmt not in GEN3:
                continue
            if rid.startswith("smogtours-"):
                skipped_smogtours += 1
                continue
            if not SHOWDOWN_ID.match(rid):
                skipped_ancient += 1
                continue
            row = {c: d[c][j] for c in COLS}
            ddir = os.path.join(OUT, fmt)
            os.makedirs(ddir, exist_ok=True)
            path = os.path.join(ddir, f"{rid}.json")
            if not os.path.exists(path):
                with open(path, "w") as f:
                    json.dump(row, f)
            counts[fmt] += 1
    print(
        f"shard {i} done. counts: {counts}, smogtours skipped: {skipped_smogtours}, ancient skipped: {skipped_ancient}",
        flush=True,
    )

print(
    "EXTRACT DONE",
    counts,
    "smogtours skipped:",
    skipped_smogtours,
    "ancient skipped:",
    skipped_ancient,
    flush=True,
)
