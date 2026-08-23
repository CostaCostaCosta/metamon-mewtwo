#!/usr/bin/env python
"""Split a parsed-replay format directory into quality buckets via hardlinks.

Creates sibling directories next to the source format dir (zero disk cost via
hardlinks), e.g. for ``.../parsed-replays/gen3ou``:

    parsed-replays/
        gen3ou/                        <- source (untouched)
        gen3ou_smogtours/gen3ou/       <- tournament replays (filename prefix)
        gen3ou_ladder_gte1500/gen3ou/  <- rated ladder >= 1500
        gen3ou_ladder_lt1500/gen3ou/   <- rated ladder < 1500
        gen3ou_ladder_unrated/gen3ou/  <- ladder without a rating

Each split dir contains a nested ``<format>/`` subdir so that
``MetamonDataset(dset_root=<split_dir>, formats=[<format>])`` auto-detects a
flat directory (``dset_root/<format>``), while the format check inside
``MetamonDataset._filter_filename`` matches the battle-id token in the
filename (the parent-dir rename is irrelevant).

Bucket rules (see docs/gen3_regi_plan.md SS2):
  * smogtours  -> filename battle-id token contains "smogtours". IMPORTANT:
    these files carry the rating token "Unrated", so they MUST be bucketed by
    filename prefix, NOT by the rating filter.
  * ladder gte1500 / lt1500 -> rating token parses to an int, split at 1500.
  * ladder unrated -> rating token does not parse to an int.

Idempotent: the split dirs are removed and rebuilt on every run.

Usage:
    uv run python -m metamon.data.split_gen3_replays <format_dir> [<format_dir> ...]

Example:
    UV ... METAMON_CACHE_DIR=/home/eddie/metamon_cache python -m \
        metamon.data.split_gen3_replays \
        /home/eddie/metamon_cache/parsed-replays/gen3ou \
        /home/eddie/metamon_cache/parsed-replays/gen3uu \
        /home/eddie/metamon_cache/parsed-replays/gen3nu \
        /home/eddie/metamon_cache/parsed-replays/gen3ubers
"""

import argparse
import os
import shutil
import sys

BUCKETS = ("smogtours", "ladder_gte1500", "ladder_lt1500", "ladder_unrated")
RATING_THRESHOLD = 1500


def parse_name_parts(filename: str) -> tuple[str, str]:
    """Return (battle_id, rating_token) using the same anchor rule as
    ``MetamonDataset._filter_filename``: parts[0] is the battle id,
    parts[1] is the rating token."""
    name = filename[:-9] if filename.endswith(".json.lz4") else filename[:-5]
    parts = name.split("_")
    if len(parts) < 7 or "vs" not in parts[2:-2]:
        raise ValueError(f"unparseable filename: {filename}")
    return parts[0], parts[1]


def bucket_for(filename: str) -> str:
    battle_id, rating_token = parse_name_parts(filename)
    # smogtours first: tournament replays are Unrated and must not fall into
    # the ladder_lt1500 bucket via the rating filter.
    if "smogtours" in battle_id.lower():
        return "smogtours"
    try:
        rating = int(rating_token)
    except ValueError:
        return "ladder_unrated"
    return "ladder_gte1500" if rating >= RATING_THRESHOLD else "ladder_lt1500"


def iter_files(format_dir: str):
    """Yield (abs_path, rel_subdir) for every file under format_dir."""
    for root, dirs, files in os.walk(format_dir):
        dirs.sort()
        rel = os.path.relpath(root, format_dir)
        for name in sorted(files):
            yield os.path.join(root, name), ("" if rel == "." else rel)


def scan_format_dir(format_dir: str) -> tuple[dict, int, int]:
    """Bucket-every-file scan: returns (counts, total, errors) without linking."""
    counts = {b: 0 for b in BUCKETS}
    errors = 0
    total = 0
    for abs_path, _rel in iter_files(format_dir):
        total += 1
        try:
            bucket = bucket_for(os.path.basename(abs_path))
        except ValueError as e:
            errors += 1
            if errors <= 5:
                print(f"  WARN {e}")
            continue
        counts[bucket] += 1
    return counts, total, errors


def split_format_dir(format_dir: str, dry_run: bool = False) -> dict:
    if not os.path.isdir(format_dir):
        print(f"SKIP {format_dir}: not a directory")
        return {}
    fmt = os.path.basename(os.path.normpath(format_dir))
    parent = os.path.dirname(os.path.normpath(format_dir))

    split_dirs = {b: os.path.join(parent, f"{fmt}_{b}", fmt) for b in BUCKETS}
    counts, total, errors = scan_format_dir(format_dir)

    if not dry_run:
        for d in split_dirs.values():
            shutil.rmtree(os.path.dirname(d), ignore_errors=True)
        for d in split_dirs.values():
            os.makedirs(d, exist_ok=True)

        placed = {b: 0 for b in BUCKETS}
        for abs_path, rel in iter_files(format_dir):
            try:
                bucket = bucket_for(os.path.basename(abs_path))
            except ValueError:
                continue
            target = os.path.join(split_dirs[bucket], rel, os.path.basename(abs_path))
            os.makedirs(os.path.dirname(target), exist_ok=True)
            os.link(abs_path, target)
            placed[bucket] += 1
        counts = placed
        counts["_total"] = total
        counts["_errors"] = errors

    print(
        f"{fmt}: {total:,} files -> " + ", ".join(f"{b}={counts[b]:,}" for b in BUCKETS)
    )
    if errors:
        print(f"  {errors} unparseable files skipped")
    return counts


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("format_dirs", nargs="+", help="parsed-replay format dirs to split")
    p.add_argument("--dry-run", action="store_true", help="only print bucket counts")
    args = p.parse_args(argv)

    grand_total = 0
    grand_linked = 0
    for d in args.format_dirs:
        c = split_format_dir(d, dry_run=args.dry_run)
        if c:
            grand_total += c.get("_total", 0)
            grand_linked += sum(c.get(b, 0) for b in BUCKETS)

    if not args.dry_run:
        print(
            f"\nTOTAL source files: {grand_total:,}; hardlinks placed: {grand_linked:,}"
        )
        if grand_total != grand_linked:
            print(
                f"WARNING: mismatch! linked {grand_linked:,} != source {grand_total:,}"
            )
            return 1
        print("OK: every source file hardlinked into exactly one bucket.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
