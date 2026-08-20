# ROM-Native Observation Representation

## Overview

This document describes a compact, structured battle-state representation designed to:
1. Replace Metamon's text/token observation pipeline with fixed-width integer IDs
2. Be reproducible from pokeemerald-expansion battle engine state
3. Support training of small distilled policies for eventual GBA deployment

## Current Metamon State Pipeline

### UniversalState (metamon/interface.py)

The canonical structured battle state in Metamon is `UniversalState`, a dataclass containing:
- `player_active_pokemon: UniversalPokemon` — full info (species, HP%, types, item, ability, level, status, effect, moves, base stats, stat boosts, computed stats)
- `opponent_active_pokemon: UniversalPokemon` — same structure but with limited visibility (moves often empty, item often "unknownitem")
- `available_switches: List[UniversalPokemon]` — bench Pokémon (full info for player side)
- `player_prev_move / opponent_prev_move: UniversalMove`
- `opponents_remaining: int` — count of non-fainted opponent Pokémon
- `player_conditions / opponent_conditions: str` — side conditions (e.g., "reflect", "lightscreen")
- `weather: str`, `battle_field: str` — field effects
- `forced_switch: bool`, `battle_won: bool`, `battle_lost: bool`
- `can_tera: bool`, `opponent_teampreview: List[str]`

### Observation Spaces (metamon/interface.py)

Several observation spaces convert `UniversalState` to model inputs:

1. **DefaultObservationSpace**: Produces `{"text": string, "numbers": float[48]}`. Text contains ~87 whitespace-separated words (species names, move names, type names, etc.). Used by the original paper.

2. **Gen1OpponentMoveObservationSpace**: Gen1 specialist — removes items/abilities/tera/weather, adds PP warnings, sleep/freeze memory, revealed opponent species, and opponent revealed moves. Produces `{"text": string, "numbers": float[54]}`.

3. **GroupedObservationSpace**: Groups by entity — separate arrays for each Pokémon and a misc array. Uses text tokens for names/types/moves.

4. **TokenizedObservationSpace**: Wraps any base space, tokenizing text features into integer arrays via `PokemonTokenizer`.

### Where Categorical Values Become Text/Token IDs

All observation spaces convert categorical values to strings:
- Pokémon species → `pokemon.name` (e.g., "gengar")
- Move names → `clean_name(move.name)` (e.g., "thunderbolt")
- Types → `pokemon.types` (e.g., "ghost poison")
- Status → `clean_name(status.name)` (e.g., "slp")
- Weather → `clean_name(weather.name)` (e.g., "raindance")
- Side conditions → `clean_name(condition.name)`

These strings are then either:
1. Joined into a single text string and tokenized via `PokemonTokenizer` (which maps each unique word to an integer ID)
2. Or used directly as text input to a transformer-based encoder

The tokenizer vocabulary is built from the training dataset and contains ~2,541 tokens for the Gen1 specialist model.

### Current Model Architecture

The "superkazam" (Alakazam) architecture used by the strongest models:
- **TstepEncoder**: `MetamonPerceiverTstepEncoder` — Perceiver-based encoder with 168-dim, 10 layers, 8 heads, 8 latent tokens
- **TrajEncoder**: `TformerTrajEncoder` — Transformer with 900-dim, 10 layers, 12 heads, 3600 FF
- **Actor**: `MetamonMaskedResidualActor` — 500 feature dim, 800 FF, 3 residual blocks
- **Critic**: `NCriticsTwoHot` — 700 hidden, 3 layers, 6 critics, 96 bins
- **Total**: ~15-20M parameters (AMAGO agent)

The "grouped_belief" variant (the local checkpoint used as teacher):
- **TstepEncoder**: `MetamonGroupedTstepEncoderV2` — 64-dim Pokémon encoder, 48-dim global, 84-dim fusion
- **TrajEncoder**: `TformerTrajEncoder` — 400-dim, 3 layers, 8 heads
- **Total**: 15,986,886 parameters

### Action Encoding

`MinimalActionSpace` (9 actions):
- Actions 0-3: Move slots (sorted alphabetically by move name via `consistent_move_order`)
- Actions 4-8: Switch slots (sorted alphabetically by species name via `consistent_pokemon_order`)

`DefaultActionSpace` (13 actions): adds tera variants at indices 9-12.

## poke-plastic-ox Battle State Locations

### Core Structures

| Concept | Location | Type |
|---------|----------|------|
| Active battler mons | `gBattleMons[MAX_BATTLERS_COUNT]` | `struct BattlePokemon` |
| Party data | `gParties[MAX_BATTLE_TRAINERS][PARTY_SIZE]` | `struct Pokemon` |
| Party→battler mapping | `gBattlerPartyIndexes[MAX_BATTLERS_COUNT]` | `u8` |
| Weather | `gBattleWeather` | `u16` (bitfield) |
| Side conditions | `gSideStatuses[NUM_BATTLE_SIDES]` | `u32` (bitfield) |
| Field effects | `gFieldStatuses` | `u32` |
| Battle struct | `gBattleStruct` | `struct BattleStruct *` |
| Last moves | `gLastMoves[MAX_BATTLERS_COUNT]` | `u16` |

### struct BattlePokemon (include/pokemon.h:338)

```c
struct BattlePokemon {
    enum Species species;      // National Dex ID
    u16 hp, maxHP;             // current/max HP
    u16 attack, defense, speed, spAttack, spDefense;
    enum Move moves[4];        // move IDs
    u8 pp[4];                  // current PP per move
    s8 statStages[8];          // 0-12 (6=neutral), order: atk,def,spe,spa,spd,acc,eva
    enum Type types[3];        // type IDs
    enum Ability ability;
    enum Item item;
    u8 level;
    u32 status1;               // bitfield: sleep(0-2), poison(3), burn(4), freeze(5), para(6), toxic(7)
    struct Volatiles volatiles; // confusion, flinch, leech seed, etc.
};
```

### ID Mappings (Gen1)

| Concept | poke-plastic-ox | Metamon/Showdown | Match? |
|---------|-----------------|-------------------|--------|
| Species | `SPECIES_BULBASAUR=1` ... `SPECIES_MEW=151` | `num` field in pokedex JSON | ✓ Exact (National Dex) |
| Moves | `MOVE_POUND=1` ... `MOVES_COUNT_GEN1` | `num` field in moves JSON | ✓ Exact (by number) |
| Types | `TYPE_NORMAL=1, TYPE_FIGHTING=2, ...` | String names mapped to enum | ⚠ Reorder needed (BUG/GHOST/STEEL differ) |
| Status | `STATUS1_SLEEP`, `STATUS1_POISON`, etc. | String names ("slp", "psn") | Mapped via lookup table |
| Weather | `B_WEATHER_RAIN`, `B_WEATHER_SUN`, etc. | String names ("raindance", "sunnyday") | Mapped via lookup table |

### Information Visibility

The ROM has full omniscient access. The observation encoder must restrict to player-visible info:
- **Player Pokémon**: Full info (species, HP, moves, PP, stats, boosts, status)
- **Opponent active**: Species, types, HP (from HP bar), status, stat boosts, revealed moves only
- **Opponent bench**: Only species of previously-seen opponents; HP and moves unknown
- **Hidden**: Opponent's unrevealed moves, exact stats, IVs/EVs, RNG state, full party

## Canonical Schema (metamon/rom_native_obs/schema.py)

### Design Principles
- Fixed-width integer categorical IDs (no strings)
- Normalized float numerical values (0.0-1.0 or sentinel -2.0)
- Explicit masking for unknown/unrevealed information
- Fixed-size: 13 Pokémon slots × 9 categorical + 31 numerical + 4 mask features each
- Deterministic and documented

### Global Features (6 categorical + 3 numerical)

| Field | Type | Range | Description |
|-------|------|-------|-------------|
| weather | u8/int | 0-7 | Weather enum |
| field_effect | u8/int | 0-7 | Field effect enum |
| player_side_cond | u8/int | 0-7 | Player side condition |
| opponent_side_cond | u8/int | 0-7 | Opponent side condition |
| player_prev_move | u16/int | 0-165 | Last move used by player |
| opponent_prev_move | u16/int | 0-165 | Last move used by opponent |
| turn_norm | float | 0-1 | turn / 200 (clipped) |
| opponents_remaining | float | 0-1 | remaining / 6 |
| forced_switch | float | 0/1 | Forced switch flag |

### Per-Pokémon Features (13 slots)

Slot ordering: 0=player active, 1-5=switches, 6=opponent active, 7-12=revealed opponents

**Categorical (9 per slot):** species, type_1, type_2, status, effect, move_1_id, move_2_id, move_3_id, move_4_id

**Move categorical (4 per slot):** move_1_category, move_2_category, move_3_category, move_4_category

**Move type (4 per slot):** move_1_type, move_2_type, move_3_type, move_4_type

**Numerical (31 per slot):** hp_fraction, level_norm, 6 base_stats (normalized /255), 7 stat boosts (normalized /6), 4×(bp/200, acc, pri/5, pp_ratio)

**Mask (4 per slot):** valid, fainted, moves_revealed, hp_known

### Legal Action Mask (9 values)

Actions 0-3: move slots (sorted alphabetically), 4-8: switch slots (sorted alphabetically)

### Tensor Shapes

```
global_cat:     (6,)              int32
global_num:     (3,)              float32
pokemon_cat:    (13, 9)           int32
pokemon_move_cat: (13, 4)         int32
pokemon_move_type: (13, 4)        int32
pokemon_num:    (13, 31)          float32
pokemon_mask:   (13, 4)           int32
legal_action_mask: (9,)           int32

Flat: categorical (227,), numerical (406,), masks (61,)
```

## Mapping Decisions

1. **Species IDs**: Use National Dex numbers directly (1-151 for Gen1). Verified identical across both systems.
2. **Move IDs**: Use Showdown move numbers (1-165 for Gen1). Verified identical to ROM move enum values.
3. **Type IDs**: Use ROM Type enum values. Python side maps Showdown type names to these IDs. Note: ROM has TYPE_BIRD=7 (unused Gen1 type) which shifts BUG/GHOST/STEEL vs Showdown.
4. **Status encoding**: Single enum (0=none, 1=sleep, 2=poison, 3=burn, 4=freeze, 5=paralysis, 6=toxic, 7=faint). ROM's status1 bitfield is decoded to this enum.
5. **Stat stages**: ROM uses 0-12 (6=neutral). Python uses boost/6.0 (-1.0 to 1.0). C version stores raw 0-12.
6. **HP fraction**: Python uses 0.0-1.0. C uses 0-255 (hp*255/maxHP).
7. **Move ordering**: Sorted alphabetically by move name (matching Metamon's `consistent_move_order`).
8. **Switch ordering**: Sorted alphabetically by species name (matching Metamon's `consistent_pokemon_order`).
9. **Revealed opponents**: Tracked across timesteps (same as Gen1OpponentMoveObservationSpace).

## Information-Visibility Decisions

| Information | Player可见 | Opponent可见 | Notes |
|-------------|-----------|-------------|-------|
| Species | ✓ Full | ✓ Active + revealed | |
| HP fraction | ✓ Full | ✓ Active only | Bench HP unknown |
| Moves | ✓ Full | ✓ Revealed only | Tracked over time |
| Move PP | ✓ Full | ✗ | Not visible to opponent |
| Base stats | ✓ From dex | ✓ From dex | Public knowledge |
| Stat boosts | ✓ Active | ✓ Active | Visible on active mons |
| Status | ✓ Full | ✓ Active | |
| Item | ✓ Full | ✗ Unknown | Gen1: no items in battle |
| Ability | ✓ Full | ✗ Unknown | Gen1: no abilities |
| Weather | ✓ | ✓ | |
| Side conditions | ✓ | ✓ | |

## Known Mismatches

1. **Type enum ordering**: ROM has TYPE_BIRD=7 between TYPE_ROCK and TYPE_BUG, while the canonical schema follows the ROM. The Python encoder maps Showdown names to ROM enum values.
2. **Move name normalization**: Showdown uses concatenated names (e.g., "acidarmor"), ROM uses underscored names (e.g., "MOVE_ACID_ARMOR"). Both map to the same integer ID via the Showdown `num` field.
3. **Stat stage ordering**: ROM orders statStages as [atk, def, spe, spa, spd, acc, eva]. Canonical schema orders as [atk, spa, def, spd, spe, acc, eva]. The C encoder reorders.
4. **Base stat ordering**: ROM SpeciesInfo has [baseHP, baseAttack, baseDefense, baseSpeed, baseSpAttack, baseSpDefense]. Canonical schema uses [atk, spa, def, spd, spe, hp]. The C encoder reorders.
5. **C encoder omniscient mode**: The debug C encoder sets `moves_revealed=1` and `hp_known=1` for all valid Pokémon. A production version would need to track visibility per the AI's perspective.
6. **Forced switch**: C encoder uses `hp==0` on the active battler. Metamon uses the Showdown `force_switch` flag which may differ in edge cases.

## Expected GBA Memory Footprint

### RomBattleState struct (C)

```
RomBattleGlobal:  9 bytes (6 u8 + 2 u16 + 1 u8)
RomBattlePokemon: ~40 bytes each (2 u16 + ~38 u8/s8)
13 Pokémon:       ~520 bytes
legal_action_mask: 9 bytes
Total:            ~538 bytes
```

### Neural Model (estimated for 1M parameter student)

```
Weights (int8):   ~1 MB
Embedding tables: ~50 KB (species 152×16 + moves 166×12 + types 20×4 + ...)
Linear layers:    ~950 KB
Activation RAM:   ~10 KB (batch=1)
Total ROM:        ~1 MB
Total RAM:        ~15 KB
```

This fits comfortably within GBA constraints (4MB ROM, 256KB EWRAM, 96KB IWRAM).


## Training Integration (2026-08-20, branch ec/plastic-space-gen1)

The schema is now wired into the main training loop (`metamon/rl/train.py`) for
Experiment 1 of the gen3 program (docs/gen3_regi_plan.md §6a):

- `metamon/interface.py::RomNativeObservationSpace` — registered `ObservationSpace`
  producing these tensors from `UniversalState` (pass-through under
  `TokenizedObservationSpace`; no tokenizer).
- `metamon/rl/metamon_to_amago.py::MetamonRomNativeTstepEncoder` — grouped_v2-lineage
  perceiver tstep encoder consuming the schema.
- `metamon/rl/configs/models/plastic_rom_native_15m.gin` — ~15M Tauros config.

**No changes to the tensor layout were made** — the C encoder stays in sync.

## Files

| File | Description |
|------|-------------|
| `metamon/rom_native_obs/schema.py` | Canonical schema definition |
| `metamon/rom_native_obs/mappings.py` | ID mapping tables (Showdown names ↔ integer IDs) |
| `metamon/rom_native_obs/metamon_encoder.py` | Metamon UniversalState → RomBattleState encoder |
| `metamon/rom_native_obs/student_model.py` | Small student policy models (4 presets: ~500k/1M/2M/4M) |
| `metamon/rom_native_obs/generate_dataset.py` | Dataset generation (pseudo-teacher or AMAGO teacher) |
| `metamon/rom_native_obs/train_student.py` | Training script (KL distillation + behavioral cloning) |
| `metamon/rom_native_obs/compare.py` | Cross-system state comparison utility |
| `metamon/rom_native_obs/tests/test_encoder.py` | Encoder tests (15 tests) |
| `poke-plastic-ox/include/rom_native_obs.h` | C struct definitions |
| `poke-plastic-ox/src/rom_native_obs/rom_native_obs.c` | C encoder implementation |
