"""
Regenerate the Gen 3 static ID tables in this directory.

Sources:
- species/moves: metamon/backend/showdown_dex/static/{pokemon/gen3pokedex,
  moves/gen3moves}.json, filtered to num in [1,386] / [1,354] (these files also
  carry later-gen/CAP entries, so the num filter is load-bearing).
- abilities: pokeemerald-expansion include/constants/abilities.h enum (1-76),
  cross-checked against the vendored Showdown dex
  (server/pokemon-showdown/dist). CANONICAL ID = the expansion enum; the lone
  divergence is lightningrod (Showdown num 32, expansion enum 31).
- items: /home/eddie/repos/poke-plastic-ox/plastic_ox/agent/
  gen3_items_expansion_enum.json (gen3-legal held items -> expansion ITEM_*
  enum), produced by the ROM-side agent from items.h x Showdown gen<=3.

Run: `uv run python metamon/rom_native_obs/gen3_static/build_gen3_tables.py`
from the repo root. Idempotent; overwrites the four json files.
"""
import json, os, re

HERE = os.path.dirname(os.path.abspath(__file__))
METAMON_ROOT = os.path.dirname(os.path.dirname(HERE))  # -> metamon/ package dir
SROOT = os.path.join(METAMON_ROOT, "backend", "showdown_dex", "static")
POX_ITEMS = "/home/eddie/repos/poke-plastic-ox/plastic_ox/agent/gen3_items_expansion_enum.json"
POX_ABILITIES_H = "/home/eddie/repos/poke-plastic-ox/include/constants/abilities.h"


def _clean(n: str) -> str:
    return (n.lower().replace(" ", "").replace("-", "").replace(".", "")
            .replace("'", "").replace(":", "").replace("é", "e"))


def build_species():
    dex = json.load(open(os.path.join(SROOT, "pokemon", "gen3pokedex.json")))
    out = {}
    for key, val in dex.items():
        num = val.get("num", 0)
        if 1 <= num <= 386:
            out.setdefault(_clean(val.get("name", key)), num)
            out.setdefault(_clean(key), num)
            if "baseSpecies" in val:
                out.setdefault(_clean(val["baseSpecies"]), num)
    return out


def build_moves():
    mvs = json.load(open(os.path.join(SROOT, "moves", "gen3moves.json")))
    out = {}
    for key, val in mvs.items():
        num = val.get("num", 0)
        if 1 <= num <= 354:
            out.setdefault(_clean(val.get("name", key)), num)
            out.setdefault(_clean(key), num)
    return out


def build_abilities():
    src = open(POX_ABILITIES_H).read()
    out = {}
    for name, num in re.findall(r"ABILITY_([A-Z0-9_]+)\s*=\s*(\d+)", src):
        n = int(num)
        if 1 <= n <= 76:
            out[_clean(name.replace("_", " "))] = n  # keep _ so _clean normalizes ABILITY_LIGHTNING_ROD -> lightningrod
    return out


def build_items():
    return json.load(open(POX_ITEMS))


if __name__ == "__main__":
    tables = {
        "gen3species.json": build_species(),
        "gen3moves.json": build_moves(),
        "gen3abilities.json": build_abilities(),
        "gen3items.json": build_items(),
    }
    for fname, tbl in tables.items():
        tbl = dict(sorted(tbl.items(), key=lambda kv: kv[1]))
        with open(os.path.join(HERE, fname), "w") as f:
            json.dump(tbl, f, indent=1)
        print(f"{fname}: {len(tbl)} entries (max id {max(tbl.values())})")
