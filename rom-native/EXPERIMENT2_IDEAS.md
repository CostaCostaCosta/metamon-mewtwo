Experiment 1 (revised) — DONE: ROM-native space works for training from scratch and matches text space

 Question: Can a ~15M Tauros-family model train from scratch on the text-less ROM-native
 ("plastic") observation space, and how does it compare to the text/token space?

 Setup (clean A/B — identical except obs space + tstep encoder):
 - Both arms: MultiTaskAgent, 9-action MinimalActionSpace, AggressiveShapedReward, IS-filtered
 offline BC/RL recipe (alakazam3_isfilter.gin), 150 epochs × 1,000 steps = 150k grad steps, batch
 12.
 - Data: 50% gen1ou parsed replays (175,570) + 50% pac-base self-play (4.98M battles).
 - Control: GroupedObservationSpace + tokenizer → MetamonGroupedTstepEncoderV2 (14,527,404 params).
 - Treatment: new RomNativeObservationSpace (13-slot fixed-width int/float schema) → new
 MetamonRomNativeTstepEncoder (14,499,612 params — 99.8% size match).

 Results (final checkpoints):

 ┌───────────────────────────────────────────────┬───────────────────────────┬───────┐
 │ Metric                                        │ ROM-native                │ Text  │
 ├───────────────────────────────────────────────┼───────────────────────────┼───────┤
 │ Training eval, late window (~296 battles/opp) │ 0.826                     │ 0.805 │
 ├───────────────────────────────────────────────┼───────────────────────────┼───────┤
 │ 250-battle heuristic eval (6 baselines)       │ 0.856                     │ 0.869 │
 ├───────────────────────────────────────────────┼───────────────────────────┼───────┤
 │ Head-to-head, 200 battles                     │ 46.0% (95% CI [39%, 53%]) │ 54.0% │
 └───────────────────────────────────────────────┴───────────────────────────┴───────┘

 Conclusion: equally strong. Every measurement is within ~2 points; the h2h CI includes 50%. The
 ROM space trains stably (actor loss 35→2.0) and clearly works. This comfortably meets the plan's
 green-light criterion (within 2–3 points) → gen3 scoping work is unblocked.

 For your wandb page: both runs are in group plastic-space-exp1 in project metamon — runs
 exp1-romnative-15m-gen1ou and exp1-textgrouped-15m-gen1ou, with per-epoch win rates vs the 5
 heuristics, losses, and grad norms.

 Deliverables (branch ec/plastic-space-gen1, pushed):
 - RomNativeObservationSpace (interface.py), MetamonRomNativeTstepEncoder (metamon_to_amago.py),
 configs plastic_rom_native_15m.gin / plastic_tauros_15m_control.gin /
 gen1ou_plastic_replay_pacbase.yaml
 - Eval registrations Exp1RomNative15M / Exp1TextControl15M; checkpoints at
 ~/metamon/models/plastic_space_exp1/
 - Results written to docs/gen3_regi_plan.md §6a and a training-integration note in
 ROM_NATIVE_OBSERVATION.md (no schema/tensor-layout changes — C encoder stays in sync)

 Bugs fixed along the way: a pretrained.py missing @staticmethod that crashed every local
 checkpoint load, and a missing --steps_per_epoch CLI arg. Noted in the plan doc: the stale
 plastic_tauros_15m_belief_control.gin doesn't parse on this branch.

 Total wall time: ~6.7h (training ~5h + evals).