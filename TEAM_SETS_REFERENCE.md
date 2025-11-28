# Metamon Team Sets Reference

## Available Team Sets

From the Metamon paper and codebase:

| Team Set | Size | Diversity | Description | Best For |
|----------|------|-----------|-------------|----------|
| **`competitive`** | 10-20 | Low | Forum sample teams, beginner-friendly | Quick eval, sanity checks |
| **`modern_replays`** | ~100s | High | Recent replays, top player approximation | Training, evaluation |
| **`modern_replays_v2`** | ~100s | High | Updated Sept 2025, PokéAgent verified | **PSRO (recommended)** |
| **`paper_replays`** | ~100s | Med-High | Replay set from paper (backwards compat) | Paper reproduction |
| **`paper_variety`** | 1000 | Very High | Procedurally generated, OOD lead-offs | Stress testing, diversity |

## Three Sets from the Paper

### 1. Variety Set → `paper_variety`
> "Procedurally generates 1k intentionally diverse teams per gen/tier and will be used to
> evaluate OOD gameplay and to generate unambiguous self-play data"

- **Size**: 1000 teams per format
- **Generation**: Procedural (sampling from all-time usage stats)
- **Characteristics**: Intentionally diverse, OOD lead-offs
- **Use cases**:
  - OOD evaluation
  - Self-play data generation
  - Stress testing policies
  - Finding edge cases

### 2. Replay Set → `modern_replays_v2` (or `modern_replays`, `paper_replays`)
> "Approximates the choices of top players based on their replays and infers
> unrevealed details"

- **Size**: Hundreds of teams per format
- **Generation**: Predicted from high-ELO replays
- **Characteristics**: Realistic meta representation
- **Use cases**:
  - **Primary training set for PSRO** ✅
  - Realistic meta evaluation
  - Ladder-style play
  - Win-rate benchmarking

**Versions:**
- `modern_replays_v2`: Latest (Sept 2025), verified against PokéAgent rules
- `modern_replays`: Older version, still good
- `paper_replays`: Original paper version (backwards compatible)

### 3. Competitive Set → `competitive`
> "Comprises 10-20 complete 'sample' teams per gen/tier scraped from forum
> discussions; these are generally designed for beginners by experts"

- **Size**: 10-20 teams per format
- **Source**: Smogon forums, sample teams
- **Characteristics**: Expert-designed, beginner-friendly
- **Use cases**:
  - Quick sanity checks
  - Baseline evaluation
  - **Not diverse enough for PSRO** ⚠️

## Recommendations by Use Case

### PSRO Training (Phase 1-3) → `modern_replays_v2`
**Why:**
- ✅ High diversity (~100s of teams vs 10-20)
- ✅ Realistic meta (top player approximation)
- ✅ Verified for Gen 1 OU (PokéAgent Challenge)
- ✅ Recent data (Sept 2025)

**Avoid:** `competitive` (too few teams, limits exploration)

### Self-Play Data Generation → `paper_variety`
**Why:**
- ✅ Maximum diversity (1000 teams)
- ✅ Unambiguous team matchups
- ✅ Forces policies to handle edge cases

**Caution:** OOD lead-offs may not transfer to real meta

### Quick Evaluation → `competitive`
**Why:**
- ✅ Small set (fast)
- ✅ Well-known teams
- ✅ Good sanity check

**Limitation:** Not representative of full meta

### Ladder Simulation → `modern_replays_v2`
**Why:**
- ✅ Most realistic team distribution
- ✅ Approximates what you'd see on ladder
- ✅ Top player strategies

### Stress Testing → `paper_variety`
**Why:**
- ✅ Maximum coverage
- ✅ Tests OOD robustness
- ✅ Finds edge cases

## Gen 1 OU Specific Notes

For Gen 1 OU PSRO training:

**Phase 0** (your interaction matrix):
- Likely used: `competitive` (10-20 teams)
- Result: Good initial matrix, but limited team diversity

**Phase 1** (PSRO):
- **Use: `modern_replays_v2`** ✅
- Reason: Much higher diversity needed for BR exploration
- Expected: BRs discover team-specific exploits

**Phase 2+** (NFSP):
- Continue: `modern_replays_v2`
- Optional: Mix in `paper_variety` for robustness testing

## Team Set Statistics (Gen 1 OU)

Approximate counts (may vary):

```
competitive:         ~15 teams
paper_replays:       ~200 teams
modern_replays:      ~300 teams
modern_replays_v2:   ~400 teams (expanded)
paper_variety:       1000 teams
```

## Switching Team Sets

To use a different team set, just change `--team_set`:

```bash
# Phase 0 with competitive (quick)
python -m metamon.nash.compute_matrix \
    --team_set competitive \
    ...

# Phase 1 with modern_replays_v2 (diverse)
python -m metamon.nash.run_psro \
    --team_set modern_replays_v2 \
    ...

# Stress test with variety (maximum diversity)
python -m metamon.nash.run_psro \
    --team_set paper_variety \
    ...
```

## When to Re-run Phase 0

If you switch from `competitive` → `modern_replays_v2`:

**Option 1**: Keep Phase 0 results (faster)
- Pro: Saves time, interaction matrix still valid
- Con: Matrix computed with fewer teams

**Option 2**: Re-run Phase 0 with `modern_replays_v2` (recommended if time permits)
- Pro: More accurate interaction matrix
- Con: Takes ~2-4 hours

**Recommendation**: Keep Phase 0 as-is for now. The Nash equilibrium
(100% V1_SelfPlay) is still valid. Phase 1 PSRO will use `modern_replays_v2`
for training, which gives you the diversity you need.

## Troubleshooting

### "Team not found" errors
**Cause**: Team set not downloaded yet
**Fix**: First run will auto-download from HuggingFace

### OOM with `paper_variety`
**Cause**: 1000 teams = more team sampling = more memory
**Fix**: Same as usual (reduce batch size, actors)

### Teams seem repetitive
**Symptom**: Seeing same teams over and over
**Likely**: Using `competitive` (only 10-20 teams)
**Fix**: Switch to `modern_replays_v2`

## Summary

For your Phase 1 PSRO:

✅ **Use: `modern_replays_v2`**
- Best diversity while staying realistic
- ~400 Gen 1 OU teams
- Verified for competitive play
- Recent meta (Sept 2025)

This gives your BRs room to explore different team matchups and find
exploitable patterns in the current Nash equilibrium (V1_SelfPlay).
