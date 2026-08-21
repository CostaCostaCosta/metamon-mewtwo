# gen3_rom_online_v0 — run card (goal: gen3 ROM-native training run, v6.1 + online play)

Status: **RUNNING** (launched 2026-08-21 ~08:10, RESUME-clean after a startup race).
Branch: `ec/plastic-space-gen3`. Commits: `2b4f9032` (setup) + follow-ups.

## What this run is
From-scratch 15M ROM-native gen3 model (schema v2 + v6.1 spikes layers), online RL
(self-play vs a league) + expert-heavy v6.1 offline mix. tmux session
`gen3_rom_online_v0` with 3 windows (learner / collector / validator).

- Learner: grad updates, writes `ckpts/latest/policy.pt` each epoch, `--psro_fifo_reweight`.
- Collector: self-play rollouts into the FIFO buffer, `--psro_weighting` (writes
  `buffer/gen3ou/meta_weights.json`).
- Validator: reloads `latest/policy.pt` every 5 epochs, evals vs **SyntheticRLV2** on
  `competitive` gen3ou (500 ts), logs `val/Average Win Rate in gen3ou_vs_SyntheticRLV2-competitive`.

## The 4 requested changes
0. **v6.1 data + spikes layers.** Reparsed gen3 ou/ubers/uu/nu to
   `metamon_cache/parsed-replays-v6.1` and split into quality buckets. Extended the
   ROM-native gen3 schema: `global_num` 3 -> 5 (append player/opponent spikes layers,
   normalized /3). Files: `rom_native_obs/schema.py`, `schema_gen3.py`,
   `metamon_encoder_gen3.py`, `rl/metamon_to_amago.py` (rebuild `global_num_fuse` at
   gen3 width in `MetamonRomNativeGen3TstepEncoder`). Tests: 30/30 pass.
   **Schema change => PoC weights incompatible => trained from scratch.**
1. **Measurement.** `scripts/eval_gen3_250.py`: 250 battles vs 6 heuristics + vs
   SyntheticRLV2, `competitive` gen3ou, `--seed` fixed, JSON out. Run per checkpoint.
2. **Online play.** Pool `rl/configs/opponent_pools/gen3ou_rom_online.yaml`:
   latest policy (self, ckpt -1) x2 + SyntheticRLV2 x3 (temp 0.8/1.2/1.6) + 4 PSRO
   FIFO slots (empty at launch). Online battles on `mrv2_smogtours_hilo/gen3ou`.
   FIFO manager: `~/metamon_runs/monitor_gen3/monitor_gen3_fifo.py` (cron every 30m).
3. **Offline mix** `rl/configs/datasets/gen3ou_rom_split_mix.yaml` (v6.1, 4 tiers,
   expert-heavy, NO pac-* exploratory): OU smogtours .30 / OU >=1500 .26 / OU <1500
   .22 / OU unrated .14 / ubers+uu+nu .096. formats lists all 4 tiers (filename
   format-check requires it).
4. **More steps.** 1000 grad steps/epoch, ~2:25/epoch (GPU-bound at fp32;
   `mixed_precision` is hardcoded "no" in online_rl.py). CPU governor set to
   `performance` (pkexec) to unblock collection. Target: max epochs in wall clock.

## Registrations (rl/pretrained.py)
- `Gen3RomNative15M` — base arch/obs/action/reward donor (PoC ckpt dir).
- `Gen3RomOnlineV0` — this run (`LocalFinetunedModel`, default LATEST_CHECKPOINT).

## Bug fixes along the way
- `MetamonMultiTaskAgent` gin alias (amago `MultiTaskAgent` rename) so legacy
  SyntheticRLV2 gin (`synthetic_multitaskagent.gin`) loads.
- `LocalPretrainedModel` tolerates missing ckpt dir when default == LATEST_CHECKPOINT
  (self-play pool row ckpt=-1 instantiates before first checkpoint exists).
- Restored truncated `rom_native_obs/gen3_static/gen3abilities.json` (git checkout).
- `npm install` in `metamon/env/vectorized` (pokemon-showdown dist for battle_host.js).

## Known risks / TODO
- Collector initial crash was a startup race (launched before first latest/policy.pt);
  resolved by resume. Watch for recurrence if learner restarts and deletes latest/.
- ~~No auto-restart watchdog~~ FIXED: a crash-safe watchdog (process-liveness
  based, verified non-destructive) now restarts dead roles via cron every 4 min.
  Manual restart fallback: `~/metamon_runs/monitor_gen3/start_{learner,collector,validator}.sh`.
- fp32 limits throughput; consider bf16 in a future run (breaks current resume).
- Eval-vs-heuristics path (`pretrained_vs_baselines`) has no per-battle seed plumb;
  reproducibility via fixed 250-battle count + competitive set + agent seeding.

## Paths
- save_dir: ~/metamon_runs/gen3_rom_online_v0 ; ckpts under .../gen3_rom_online_v0/ckpts
- logs: ~/metamon_runs/gen3_rom_online_v0_{learner,collector,validator}.log
- monitor: ~/metamon_runs/monitor_gen3/ (monitor_gen3_fifo.py, start_*.sh, state.json)
- eval: uv run python scripts/eval_gen3_250.py --model Gen3RomOnlineV0 --checkpoint N

## Throughput note (measured 2026-08-21 ~08:30)
- ~2:25 / epoch (1000 grad steps) at fp32, batch 24, GPU-bound (GPU 91-100%).
- CPU governor `performance` set via pkexec -> collection ~19.6k battles/hr, buffer
  crosses the 5k online-mix threshold quickly.
- Projected: ~24 epochs/hr -> ~190 epochs (~190k grad steps) over 8h wall clock.
  bf16 (~1.3x) rejected mid-run: changes dynamics + breaks optimizer/PopArt resume
  for ~15M params. Keep stable fp32; revisit for a future fresh run.

## Validation milestones (2026-08-21)
- End-to-end smoke test PASSED before launch (collection + train step + val envs).
- Run resumed clean; all 3 roles healthy, no errors/NaN/OOM.
- Buffer crossed 5,000 (online-mix threshold) at ~epoch 9; online weight ramps
  0 -> 0.40 over 20 anneal epochs (initial_sampling_weights=[0, offline]).
- Validator logging val WR vs SyntheticRLV2 (competitive); near-0 early as expected
  for a from-scratch policy vs the 200M reference; monitor reads this key.
- PSRO sidecar (buffer/gen3ou/meta_weights.json) written each epoch by collector.

## Monitor fixes + auto-eval (2026-08-21 ~08:55)
- Fixed monitor_gen3_fifo.py: WANDB_LOGS now points at wandb_logs/wandb and
  latest_val_wr() unwraps find_run_files' {mode:(run_dir,wandb_file)} tuples,
  preferring the validator run. Now reads val WR correctly (0.0087 @ epoch 10).
- eval_gen3_250.py now emits a clean {opponent: win_rate} summary.
- Epoch-10 baseline (25-battle smoke): Random 1.0, Gen1BossAI 1.0, GymLeader 0.40,
  Grunt 0.17, PokeEnvHeuristic/EmeraldKaizo 0.0.
- Cron: FIFO checkin every 30m; eval_at_50.sh every 10m fires the full 250-battle
  eval (heuristics + SyntheticRLV2, competitive, seed 0) once policy_epoch_50.pt
  exists -> ~/metamon_runs/eval250_ckpt50_seed0.json.

## Safe watchdog (2026-08-21 ~09:00)
- Replaced the removed buggy watchdog with a crash-safe one
  (~/metamon_runs/monitor_gen3/watchdog.sh, cron every 4 min). Liveness = a live
  python/uv/node process in the role's pane process tree (walk descendants of the
  pane PID). It NEVER kills a running role (verified: reports all 3 ALIVE). Only
  restarts a role whose process is gone, or recreates a missing window/session.
  start_{learner,collector,validator}.sh relaunch each role with
  --resume_training_state (no progress lost).
- Cron stack: watchdog (4m) + FIFO checkin (30m) + epoch-50 eval trigger (10m).
- PSRO-Lite confirmed adapting: weights shifted onto the stronger SRV2 t12/t16
  deployments as the policy improved past the weak ones.

## 250-battle eval @ epoch 20 (task 1 measurement; 2026-08-21 ~09:15)
`eval_gen3_250.py --model Gen3RomOnlineV0 --checkpoint 20 --battles 250 --seed 0`
(competitive gen3ou, fixed seed). Win rates:
- RandomBaseline 0.714 | Gen1BossAI 0.188 | PokeEnvHeuristic 0.071
- Grunt 0.041 | GymLeader 0.038 | EmeraldKaizo 0.000
- **vs SyntheticRLV2 (ckpt48): 0.012**
From-scratch at epoch 20 (~20k grad steps): beats Random reliably, mid-tier
heuristics still ahead. SyntheticRLV2 (200M) is the fixed reference; WR tracks the
validator panel. Composite heuristic score (excl. Random): ~0.084.

## Periodic eval sweep (2026-08-21 ~09:16)
- Replaced the single eval_at_50 cron with `monitor_gen3/eval_sweep.sh` (every 15m):
  runs the 250-battle harness (heuristics + SyntheticRLV2, competitive, seed 0) at
  epochs 50 / 150 / 300 / 500 / 750 / 1000, each once (sentinel
  ~/metamon_runs/eval250_ckpt{N}_seed0.json), never concurrently. Captures the
  task-1 measurement across the whole training arc (the learning curve), not just
  one point. epoch-20 already done (see above).

## Opponent pool change (2026-08-21 ~10:40, user request)
- Pool is now: latest policy (self) x2 + **SyntheticRLV2 x1** (ckpt48, temp 1.0)
  + **Kakuna x3** (ckpt34, temps 0.8/1.2/1.6) + 4 PSRO FIFO slots.
  (Was: SRV2 x3 at varying temps; user asked for 1 SRV2 + 3 Kakuna instead.)
- Kakuna = gen3-capable 143M policy (~63% GXE vs humans on competitive gen3ou),
  battle_backend=metamon, reward matches the run. Verified loads (ckpt 34).
- Collector restarted via monitor_gen3/start_collector.sh (resume) to pick up the
  new pool; confirmed collecting vs Kakuna on mrv2_smogtours_hilo.
- Buffer at restart ~36k trajectories (dset_max_size 50000); PSRO-Lite sidecar
  reweights over the new 6-row pool.
