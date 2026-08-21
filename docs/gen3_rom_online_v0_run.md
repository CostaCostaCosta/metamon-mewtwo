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
- No auto-restart watchdog (a buggy liveness check killed healthy roles; removed).
  The 30-min heartbeat + FIFO cron provide monitoring; restart roles manually via
  `~/metamon_runs/monitor_gen3/start_{learner,collector,validator}.sh` if one dies.
- fp32 limits throughput; consider bf16 in a future run (breaks current resume).
- Eval-vs-heuristics path (`pretrained_vs_baselines`) has no per-battle seed plumb;
  reproducibility via fixed 250-battle count + competitive set + agent seeding.

## Paths
- save_dir: ~/metamon_runs/gen3_rom_online_v0 ; ckpts under .../gen3_rom_online_v0/ckpts
- logs: ~/metamon_runs/gen3_rom_online_v0_{learner,collector,validator}.log
- monitor: ~/metamon_runs/monitor_gen3/ (monitor_gen3_fifo.py, start_*.sh, state.json)
- eval: uv run python scripts/eval_gen3_250.py --model Gen3RomOnlineV0 --checkpoint N
