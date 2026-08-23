"""
Download raw smogtours replays from the Pokemon Showdown replay API.

Reads id_list.json (one Showdown replay id per battle, e.g.
"smogtours-gen3ou-326073") and fetches https://replay.pokemonshowdown.com/<id>.json
for each, writing <outdir>/<formatid>/<id>.json. Resumable: existing files are
skipped. Failed ids are retried with backoff; permanently failing ids go to
failed.json.
"""
import concurrent.futures as cf
import json
import os
import random
import sys
import threading
import time

import httpx

OUTDIR = "/home/eddie/metamon_cache/raw-replays-smogtours"
API = "https://replay.pokemonshowdown.com/{rid}.json"
WORKERS = 8
MAX_ATTEMPTS = 6

_tls = threading.local()
lock = threading.Lock()
counters = {"done": 0, "fail": 0, "skip": 0}


def client() -> httpx.Client:
    if not hasattr(_tls, "client"):
        _tls.client = httpx.Client(
            timeout=60.0,
            headers={"User-Agent": "metamon-research/1.0 (replay archival)"},
            follow_redirects=True,
        )
    return _tls.client


def dest_path(rid: str, formatid: str) -> str:
    d = os.path.join(OUTDIR, formatid)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{rid}.json")


def existing_path(rid: str) -> str | None:
    # find existing download under any format dir
    for d in os.listdir(OUTDIR):
        p = os.path.join(OUTDIR, d, f"{rid}.json")
        if os.path.isdir(os.path.join(OUTDIR, d)) and os.path.exists(p):
            return p
    return None


def fetch(rid: str) -> None:
    if existing_path(rid):
        with lock:
            counters["skip"] += 1
        return
    delay = 1.0
    for attempt in range(MAX_ATTEMPTS):
        try:
            r = client().get(API.format(rid=rid))
            if r.status_code == 200:
                data = r.json()
                log = data.get("log") or ""
                if not log.strip():
                    raise ValueError("empty log")
                fmt = data.get("formatid") or "unknown"
                tmp = dest_path(rid, fmt) + ".tmp"
                with open(tmp, "w") as f:
                    json.dump(data, f)
                os.rename(tmp, dest_path(rid, fmt))
                with lock:
                    counters["done"] += 1
                return
            elif r.status_code in (429, 500, 502, 503, 504):
                time.sleep(delay + random.random())
                delay *= 2
                continue
            else:
                # 404 etc: battle deleted/private -> permanent failure
                raise ValueError(f"http {r.status_code}")
        except Exception:
            if attempt == MAX_ATTEMPTS - 1:
                with lock:
                    counters["fail"] += 1
                    with open(os.path.join(OUTDIR, "failed.jsonl"), "a") as f:
                        f.write(json.dumps({"id": rid}) + "\n")
                return
            time.sleep(delay + random.random())
            delay *= 2


def main():
    with open(os.path.join(OUTDIR, "id_list.json")) as f:
        ids = json.load(f)
    print(f"total ids: {len(ids)}", flush=True)
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(fetch, rid): rid for rid in ids}
        n = 0
        for _ in cf.as_completed(futs):
            n += 1
            if n % 500 == 0:
                el = time.time() - t0
                rate = n / el
                eta = (len(ids) - n) / rate if rate > 0 else 0
                print(
                    f"[{n}/{len(ids)}] done={counters['done']} skip={counters['skip']} "
                    f"fail={counters['fail']} rate={rate:.1f}/s eta={eta/60:.0f}min",
                    flush=True,
                )
    print("FINISHED", counters, flush=True)


if __name__ == "__main__":
    main()
