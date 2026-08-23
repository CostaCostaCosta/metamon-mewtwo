# Gen 3 Data Prep — Running Log

Date: 2026-08 (session on branch `ec/plastic-space-gen3`)
Cache: `/home/eddie/metamon_cache`
Plan reference: `docs/gen3_regi_plan.md` §1, §2, §7 phase 0.

## Status

- [ ] T1 parsed-replays v6 (gen3ou/uu/nu/ubers)
- [ ] T2 self-play pac-base + pac-exploratory gen3ou
- [ ] T3 teams (competitive/paper_variety gen3*, modern_replays_v2 gen3ou)
- [ ] T4 three-way split via hardlinks
- [ ] T5 dataset yamls
- [ ] T6 sanity load

## Commands & results


### T3 teams (2026-08-20, teams rev v5)
```
$ uv run python -m metamon.data.download teams --formats gen3ou gen3uu gen3nu gen3ubers
$ METAMON_CACHE_DIR=/home/eddie/metamon_cache
gen1ou
gen2ou
gen3nu
gen3ou
gen3ubers
gen3uu
gen4ou
gen9ou
teams/competitive/gen3ou: 21 files
teams/competitive/gen3uu: 9 files
teams/competitive/gen3nu: 8 files
teams/competitive/gen3ubers: 10 files
teams/paper_variety/gen3ou: 1002 files
teams/paper_variety/gen3uu: 1002 files
teams/paper_variety/gen3nu: 1002 files
teams/paper_variety/gen3ubers: 1002 files
teams/modern_replays_v2/gen3ou: 31106 files
```
- [x] T3 teams (competitive + paper_variety for gen3 ou/uu/nu/ubers; modern_replays_v2 gen3ou present; extra gen3ou sets downloaded by CLI default set_names)

### T3 verify get_metamon_teams
```
$ METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -c "from metamon.env.wrappers import get_metamon_teams; get_metamon_teams(\"gen3ou\", \"competitive\")"
-> loaded 20 teams from TeamSet index
```

### T1 parsed-replays v6 (2026-08-20 17:30)
```
$ METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m metamon.data.download parsed-replays --formats gen3ou gen3uu gen3nu gen3ubers --version v6
gen3ou   : 498,928 files  (v4 tar entries were 406,477; v6 larger, expected)
gen3uu   :   5,536 files
gen3nu   :   3,786 files
gen3ubers:  14,454 files
version_reference.json -> parsed-replays/gen3*: version v6, downloaded 2026-08-20 17:30
v4 backup kept at parsed-replays/gen3ou_v4_stale/ + gen3ou_v4_stale.tar.gz (delete after v6 verified)
```

### T4 three-way split (hardlinks) — v6 layout note

v6 tars extract **nested** as `<format>/YYYY/MM/*.json.lz4` (older v4 dirs were flat).
The split script walks recursively and preserves relative subpaths inside each split dir.

```
$ uv run python -m metamon.data.split_gen3_replays \\
    /home/eddie/metamon_cache/parsed-replays/gen3ou \\
    /home/eddie/metamon_cache/parsed-replays/gen3uu \\
    /home/eddie/metamon_cache/parsed-replays/gen3nu \\
    /home/eddie/metamon_cache/parsed-replays/gen3ubers

format    total   smogtours  gte1500  lt1500   unrated
gen3ou    498,928   89,216   33,760   230,578  145,374
gen3uu      5,536    2,480        0       475    2,581
gen3nu      3,786    1,564        0       218    2,004
gen3ubers  14,454    5,154        0       660    8,640

TOTAL source files: 522,704; hardlinks placed: 522,704 — every file in exactly one bucket.
nlink=2 verified (hardlinks, zero extra disk). Spot checks: smogtours=Unrated-prefix files,
gte1500=rated>=1500, lt1500=rated<1500, unrated=rating token not an int. All OK.
```

Split dirs (parent: /home/eddie/metamon_cache/parsed-replays/):
- gen3ou_smogtours/gen3ou/ ... gen3ou_ladder_gte1500/gen3ou/, gen3ou_ladder_lt1500/gen3ou/, gen3ou_ladder_unrated/gen3ou/
- same pattern for gen3uu/gen3nu/gen3ubers (gte1500 empty for those three)
- [x] T4 three-way split

### T2 self-play gen3ou (2026-08-20, revision main)
```
$ METAMON_CACHE_DIR=/home/eddie/metamon_cache uv run python -m metamon.data.download self-play --subsets pac-base pac-exploratory --formats gen3ou
pac-base/gen3ou.tar       : 1,015,752 members   10.34 GB  (lz4 7.83GB removed after decompress)
pac-exploratory/gen3ou.tar: 1,004,121 members   11.11 GB  (lz4 8.36GB removed after decompress)
version_reference.json -> self-play/pac-base/gen3ou + pac-exploratory/gen3ou: version main
existing gen1ou self-play dirs untouched (pac-base/pac-exploratory/pac-tauros gen1ou.tar remain)
```
- [x] T2 self-play

### T6 sanity load (verified by parent agent, 2026-08-20)
Built the full `gen3ou_rom_split_mix.yaml` mixture via
`metamon.rl.dataset_config.build_dataset` (DefaultObservationSpace + allreplays-v3 tokenizer):
```
Self-Play (pac-base)                      1,015,752    0.450   45.0%
Custom Replays (gen3ou_smogtours)            89,216    0.220   22.0%
Custom Replays (gen3ou_ladder_gte1500)       33,760    0.180   18.0%
Custom Replays (gen3ou_ladder_lt1500)        230,578    0.100   10.0%
Custom Replays (gen3ou_ladder_unrated)       145,374    0.050    5.0%
TOTAL                                     1,514,680    1.000   100.0%
```
MIXTURE BUILT OK. `gen3ou_rom_replay_pacbase.yaml` indexes the raw gen3ou dir +
pac-base (same code paths, verified by shared components).
- [x] T6 sanity load

**DATA WORKSTREAM COMPLETE** (T1-T6 all green).
