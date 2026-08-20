# Gen 3 ROM-Distillation Program — `ec/regi`

Status: planning / pick-up doc. Branch created from `ec/student-rby` @ `f545e5b5`.

## Goal

Train a **~15M-parameter Tauros-architecture** (grouped_v2 lineage) proof-of-concept
policy for **Gen 3** play, then distill it into a ROM-native student model deployable
inside a **pokeemerald-expansion** ROM hack.

The ROM hack is being built as a faithful representation of the Showdown gen 3 battle
simulator:

- ROM base: **pokeemerald-expansion**
- Mechanics: **original gen 3 rules across the board** — physical/special split is
  **type-based** (not per-move), no Fairy type, no gen4+ moves/abilities/items in vocab.
- Deployment target: in-game battle AI (GBA-class constraints), which is why the
  student uses the fixed-width integer ROM-native schema
  (`metamon/rom_native_obs/`, `ROM_NATIVE_OBSERVATION.md`).

Tier coverage: **OU is the crux of the game** and the optimization target, but the ROM
will include **UU, NU, Ubers, and LC**, so the data/obs design must not be OU-ignorant.

---

## 1. Offline data inventory (verified 2026-07)

### Local cache (`/home/eddie/metamon_cache`, parsed-replays **v4**)

| Format | Total files | Smogtours | Ladder |
|---|---|---|---|
| gen1ou | 175,570 | 50,704 | 124,866 |
| gen2ou | 16,182 | 5,454 | 10,728 |
| **gen3ou** | **199,669** | **38,976** | **160,693** |
| gen1uu | 5,836 | 3,196 | 2,640 |
| gen1nu | 4,630 | 2,602 | 2,028 |
| gen1ubers | 7,160 | 2,720 | 4,440 |

gen3ou ladder elo breakdown (parsed from filename rating token):

| Bucket | Files |
|---|---|
| rated < 1500 | 82,029 |
| rated >= 1500 | **11,418** |
| Unrated | 67,246 |

⚠️ The local gen3ou copy is **v4 and possibly slightly stale/partial**: the retained
`gen3ou.tar.gz` (2.1 GB) contains 406,477 entries vs 199,669 on disk, and
`version_reference.json` never logged it. **Re-pull v6 from HF before any real run**
(`gen3ou.tar.gz` @ v6 = 2.69 GB, so v6 is meaningfully larger).

### Hugging Face (`jakegrigsby/metamon-parsed-replays` @ v6)

| File | Size | Notes |
|---|---|---|
| gen3ou.tar.gz | 2.69 GB | primary dataset |
| gen3uu.tar.gz | ~0.03 GB | thin |
| gen3nu.tar.gz | ~0.02 GB | thin |
| gen3ubers.tar.gz | ~0.08 GB | thin |
| **gen3lc** | — | **does not exist** (no LC anywhere in parsed replays, raw replays, or teams) |

Self-play (`SELF_PLAY_SUBSET_FORMATS` in `metamon/data/download.py`):
- `pac-base` and `pac-exploratory`: published for gen1/2/3/4/9ou — **gen3ou available
  on HF but not downloaded locally** (local self-play cache is gen1ou-only: 55G + 18G + 60G).
- **`pac-tauros` is gen1ou-only.** There is no tauros-lineage gen3 self-play data;
  gen3 self-play must be generated (see §5).

Teams (`jakegrigsby/metamon-teams`):
- `competitive/`: gen3ou, gen3uu, gen3nu, gen3ubers (gen3ou already local, v4)
- `paper_variety/`: same four gen3 tiers
- `modern_replays_v2/`: gen3ou only
- **No LC team sets anywhere.**

### LC gap

gen3lc exists on Showdown but has no presence in any metamon dataset. Options:
1. Scrape gen3lc replays with `metamon/data/raw_replay_util.py` + parse (moderate effort,
   replay volume on the ladder is decent).
2. Synthesize LC from gen3ou teams de-leveled to 5 (mechanics differ — LC has its own
   banlist/meta; this is a last resort).
3. Defer LC; design obs space so LC is "just another format tag" and backfill later.

**Open question:** how much do we care about LC at PoC time? Recommendation: defer to
post-OU-PoC, but keep the obs space tier-agnostic (§3).

---

## 2. Three-way offline data split

Split every gen3 tier's parsed replays into:

1. **smogtours** — filename prefix `*-smogtours-<format>-*`. Tournament play, highest
   quality, but small (~20% of gen3ou). Note: smogtours files are `Unrated`, and
   `MetamonDataset` treats unrated as elo 1000, so **they must be split by filename
   prefix, not by the rating filter** — otherwise they silently land in the <1500 bucket.
2. **ladder < 1500** — rated ladder games below 1500 (unrated policy TBD: either a 4th
   bucket, folded into <1500, or dropped; see open questions).
3. **ladder >= 1500** — only **11.4K files for gen3ou**. This is thin for IL on its own;
   expect to use it as an eval/validation split or an upweighted mixture component,
   not a standalone training set.

Implementation: physical split into sibling directories (hardlinks — zero disk cost),
e.g. `parsed-replays/gen3ou_smogtours/`, `gen3ou_ladder_lt1500/`, `gen3ou_ladder_gte1500/`.
The existing `MetamonDataset` format check matches on the battle-id token inside the
filename, so renamed parent dirs are fine; `dset_path` just points at the split dir.
Mixture weights then live in the dataset YAML, same pattern as
`metamon/rl/configs/datasets/*.yaml`.

Motivation for the split: curriculum + quality control. Smogtours/>=1500 for the
quality signal; <1500 as volume filler; also enables "how much does low-elo data hurt"
ablations, which directly informs how much gen3 self-play we need to generate.

---

## 3. Observation space plan

### 3a. Paired-down gen3 space (from the universal space)

Universal `GroupedObservationSpace` + `DefaultObservationSpace-v1` tokenizer (2,541
tokens) covers gens 1–9. A gen3-scoped space cuts the vocab to:

- species: 386 (gen3 national dex)
- moves: 354 (gen3 move index range, minus expansion additions)
- abilities: 77 (gen3 set)
- items: gen3-legal only
- types: 17 (no Fairy; keep the ROM type-enum ordering incl. the `TYPE_BIRD` gap —
  see `ROM_NATIVE_OBSERVATION.md` "Known Mismatches")
- **action space: `MinimalActionSpace` (9 actions)** — gen3 has no tera, so the
  `DefaultActionSpace` tera slots (9–12) are dead weight. 9 actions is also exactly
  the GBA-deployable action set.
- drop weather/field vocab entries that can't occur in gen3 (none — gen3 has
  perma-weather; keep), drop tera conditions entirely.

### 3b. "Base stats + ability" reduction (proposal from Eddie)

Reduce the per-Pokémon representation to **species identity → (base stats, ability)**
rather than richer per-instance features, so a species has the same embedding
regardless of tier. Rationale: the same species appears across OU/UU/NU/Ubers/LC, so a
tier-agnostic representation lets multi-tier data share statistical strength, and it
matches what the ROM can cheaply materialize (species table lookups).

Open design questions:
- Does this *replace* instance features (current HP, status, boosts, revealed moves) or
  just the *static* features? Boosts/status/HP are clearly load-bearing; the proposal
  is presumably about static identity features (species/types/base stats/ability) vs.
  learned per-species embeddings.
- Gen3 has no team preview and no revealed ability until it triggers — ability must be
  masked until observed (the rom_native schema already has per-slot masks; extend the
  same discipline here).
- Risk: over-compression hurts exactly where OU is decided (speed-tie math, damage
  rolls on EVd sets). Mitigation: keep computed-stat numerics for the active mons,
  compress only bench/unknown mons.

### 3c. ROM-native translation (later phase)

`metamon/rom_native_obs/mappings.py` is currently gen1-only (reads `gen1pokedex.json`,
`gen1moves.json`). Gen3 versions of the static dex files already ship in-repo
(`metamon/backend/showdown_dex/static/{pokemon,moves,typechart}/gen3*.json`), so
generating gen3 mapping tables is mechanical. The non-mechanical part is
information-visibility tracking (revealed moves, HP-known, ability-revealed) and
porting the C-encoder comparison tests in `rom_native_obs/tests/`.

---

## 4. Architecture

~15M params, Tauros family = `grouped_v2` tstep encoder + AMAGO trajectory encoder.
Reference points in-repo:

- `V2AGroupedV2Tauros35M` (~35M, `grouped_v2_35m.gin`) — scaled-down TaurosV0
- `TaurosV0` (~62M, `grouped_v2_50m.gin`)
- grouped_belief teacher: 15,986,886 params — proves the 15M budget trains fine and
  distills fine (`rom_native_obs/train_student.py`)

A `grouped_v2_15m.gin` gen3 variant: shrink tstep/traj encoder widths to hit 15M
*after* the vocab reduction (smaller embedding tables buy headroom for the trunk).

---

## 5. Self-play gap

The tauros recipe leans on self-play (`online_selfplay.yaml`: 40% pac-tauros /
35% pac-base / 20% pac-exploratory / 5% replays). For gen3:

- pac-base/pac-exploratory gen3ou exist on HF — download as the offline self-play floor.
- There is no pac-tauros gen3. Plan: IL seed (offline replays + HF self-play) → online
  RL on gen3ou ladder/self-play à la `metamon/rl/online_rl.py` → harvest the online
  buffer as the gen3 "tauros-style" pile.
- Team supply for online play: `competitive/gen3ou` is local; Eddie is collecting
  gen3ou sample teams in parallel with experiment 1.

---

## 6. Experiment 1 (agreed first move): gen1ou transfer test of the pared-down space

**Question:** does scoping the observation space + tokenizer to a single gen (the same
*class* of change we plan for gen3) hurt, help, or leave unchanged a 15M tauros model?

**Design:**

- Arm A (control): 15M grouped_v2 tauros, universal `GroupedObservationSpace`,
  `DefaultObservationSpace-v1` tokenizer, gen1ou-only data mix.
- Arm B (treatment): identical arch params + identical data, gen1-scoped obs space
  (151 species / 165 moves vocab, no items/abilities — gen1 has none in battle anyway,
  `MinimalActionSpace`-equivalent 9 actions). Closest existing thing:
  `Gen1OpponentMoveObservationSpace` (interface.py L1372) — likely needs a grouped
  variant rather than the text-space one.
- Eval: GXE vs. the standard baseline ladder + head-to-head A vs B. Success =
  Arm B within ~2–3 points of Arm A → green light for gen3 scoping. If Arm B wins
  (plausible — less vocab dilution, dead-token removal), even better.

**Why gen1 first:**

1. gen1ou has the deepest local data (175K replays + 133G of self-play), so the
   experiment is not data-starved and the obs-space effect is isolated.
2. gen1 vocab is the smallest, so the paired-down space is the *easiest* version of
   the change — a failure here would be a strong negative signal for gen3.
3. The full pipeline (grouped space → tokenizer → tar datasets → gin configs) already
   runs for gen1, so only the space/tokenizer is new code.

**Transfer caveats (honest):**

- gen1 has no abilities/items, so the "base stats + ability" reduction (§3b) is only
  half-exercised. A follow-up mini-arm with the static-feature reduction can be bolted
  on cheaply once Arm A/B plumbing exists.
- gen1ou is data-rich; gen3ou (and especially gen3 >= 1500) is not. If pared-down helps
  in the data-poor regime (likely — fewer wasted params), the gen1 test will *understate*
  the gen3 benefit.
- gen2ou is the actual data-poor canary if we want a second data point before gen3
  (16K local replays).

---

## 7. Phases & complexity

| # | Phase | Estimate | Notes |
|---|---|---|---|
| 0 | Data prep: re-pull parsed-replays **v6** gen3 ou/uu/nu/ubers; pac-base/pac-exploratory gen3ou; three-way split script (hardlinks); dataset YAMLs | 1–2 days | fix the stale v4 extraction first |
| 1 | **Exp 1: gen1ou pared-space A/B @ 15M** | ~1 week incl. run | first deliverable |
| 2 | gen3 IL PoC @ 15M on split offline mix (OU focus, multi-tier aware) | 3–5 days + GPU run | depends on 0; can start on universal space |
| 3 | gen3 online self-play bootstrap (IL seed → `online_rl.py`, fresh FIFO) | infra exists; wall-clock = days–weeks of sim | needs team sets from Eddie |
| 4 | gen3-scoped obs space + tokenizer (port of Exp 1 arm B) | 2–4 days | informed by Exp 1 result |
| 5 | ROM-native gen3 schema: mappings, encoder, visibility masks, C-encoder comparison tests | 1–2 weeks | abilities/items visibility is the new surface area |
| 6 | Translate training data → ROM space; distill 15M teacher → student | 1–2 weeks | validate against `rom_native_obs/tests/` patterns |

Critical path: 0 → 2 → 3 (data/self-play) in parallel with 1 → 4 (obs space) → 5 → 6.

## 8. Risks / open questions

1. gen3ou >= 1500 is only 11.4K files — high-elo IL alone won't carry OU performance;
   self-play generation is on the critical path, not optional polish.
2. Unrated ladder games (67K in gen3ou): keep as their own bucket, fold into <1500,
   or drop? (Showdown unrated ≠ low quality, but it's unrated for a reason.)
3. LC: no data anywhere — scrape, synthesize, or defer?
4. pac-base/pac-exploratory gen3ou quality is unknown relative to pac-tauros gen1;
   expect a weaker offline floor for gen3 than gen1 had.
5. The ROM's type-based phys/spec must be mirrored exactly in the obs/training data —
   Showdown's gen3 mod already implements this, so *Showdown gen3 data is correct by
   construction*; the risk is only in vocab hygiene (no expansion-only moves/abilities).
6. Every "Known Mismatch" in ROM_NATIVE_OBSERVATION.md (type enum ordering, stat-stage
   ordering, forced-switch edge cases, omniscient C encoder) is a train/deploy skew
   class that gets *worse* in gen3 (abilities, items, more volatiles). Extend the
   comparison tests before translating data, not after.
