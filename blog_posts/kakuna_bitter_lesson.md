# Kakuna and The Bitter Lesson: When Bigger Beats Smarter

We spent months building what felt like the future of RL for competitive Pokémon. Dynamic damping to prevent policy collapse. Epistemic uncertainty weighting to prioritize high-quality experiences. PSRO (Policy Space Response Oracles) to systematically construct Nash equilibrium policies. The plan was elegant: start with a diverse population, iteratively train best-response oracles, converge to an unexploitable equilibrium. This was *sophisticated* RL.

Then Kakuna happened.

Kakuna is a 142M parameter transformer trained on 7.8 million self-play battles in Gen 1 OU. No fancy algorithms. No equilibrium-seeking. Just: play against yourself, collect data, train, repeat. Scale the hell out of it. And it worked—better than our carefully-tuned PSRO setup, better than our damping innovations, better than our uncertainty-weighted experience replay.

If you've been in ML long enough, this story feels uncomfortably familiar.

## The Bitter Lesson Strikes Again

Rich Sutton wrote "The Bitter Lesson" in 2019, documenting a pattern that repeats across AI: human ingenuity loses to compute. Handcrafted chess evaluation functions lost to brute-force search. Feature engineering in computer vision lost to end-to-end CNNs trained on ImageNet. AlphaGo's sophisticated tree search lost to AlphaZero's tabula rasa self-play. GPT-3 reminded us that scaling up simple architectures beats architectural cleverness.

Metamon just became another data point in this story. Our Gen 1 Pokémon project started as an RL research platform—we published pretrained models at RLC 2025, built infrastructure for offline RL from human replays, and designed a suite of baselines. When we turned to self-play, we knew policy collapse was a risk. So we built tools: reverse-KL damping, adaptive KL targeting, population-based training. We felt smart.

But Kakuna didn't need to be smart. It needed to be *big* and *hungry*. Train a 142M model, generate millions of battles, filter for quality, retrain. Loop. The simplicity is almost insulting. No equilibrium guarantees, no theoretical elegance. Just scale.

The humbling part? It's not even close. Kakuna's win rate against our best PSRO checkpoint sits comfortably above 60%. Our Phase 1 PSRO iteration spent 500 battles collecting data per oracle, trained for 3 epochs per best-response, carefully balanced the population. Kakuna just... kept eating data.

## What We Actually Built

Here's the uncomfortable truth: the sophisticated RL infrastructure wasn't *wrong*, it was just premature optimization.

Dynamic damping works. It prevents the catastrophic policy collapse we saw in early self-play experiments. Reverse-KL regularization against a reference policy keeps entropy high, prevents mode collapse, stabilizes training. The theory is sound.

But it doesn't matter if you can't iterate fast enough. Our PSRO loop took weeks per iteration. Kakuna's self-play loop—once we optimized it—can generate millions of battles in days. Speed compounds. More data means better models. Better models mean better data. The flywheel spins faster when you're not blocked on oracle convergence guarantees.

The lesson isn't that our tools are useless. It's that they're infrastructure for a *different* regime. Dynamic damping will matter when we hit the limits of naive self-play. Uncertainty weighting will matter when data quality bottlenecks exceed data quantity bottlenecks. PSRO will matter when we need formal guarantees about exploitability.

But right now? Scale first, sophisticate later.

## Adapting the Research Strategy

So what do we do with this lesson? Give up on algorithmic innovation and just rent more GPUs?

Not quite. The playbook shifts:

**1. Engineer for iteration speed.** We rebuilt our self-play infrastructure from scratch. The old system generated ~1,000 battles/day. The new system, using batched inference and parallel subprocesses, targets 10x that. Faster iteration means faster feedback loops. It's not sexy, but neither is waiting three weeks to discover your oracle diverged.

**2. Scale what works, defer what doesn't.** Kakuna's approach is dead simple: self-play with a binary win/loss reward. No shaping, no damping, no tricks. It works because the model is big enough to not collapse under its own self-play. We'll scale this until it stops working, *then* bring in the sophistication.

**3. Save the clever bits for test-time.** This is where I'm most excited. Training-time sophistication didn't win, but what about inference-time sophistication? AlphaGo didn't just train a big policy network—it wrapped it in MCTS at test time. The Ataraxos project showed that even simple tree search over learned value functions can massively boost RL agent performance. Our next experiments focus here: Can we build a test-time search wrapper that turns Kakuna into something even stronger?

**4. Stay humble, stay empirical.** We thought PSRO was the path to superhuman Gen 1 play. The data said otherwise. Good research means killing your darlings when they don't work. Kakuna forced us to confront reality: our bottleneck was scale, not algorithms.

## Scale *And* Sophistication

The Bitter Lesson isn't really bitter if you learn from it. Yes, compute wins. Yes, simple methods scaled beat clever methods optimized. But the history of AI isn't *just* scaling—it's scaling the *right* architectures. Transformers scaled better than LSTMs. Residual networks scaled better than plain convnets. Self-play scaled better than imitation learning.

The sophistication that matters is the kind that *enables* scale. Transformers didn't win because they were clever; they won because their inductive biases allowed them to efficiently use massive data and compute. Our dynamic damping won't beat Kakuna head-to-head, but it might let us push self-play into regimes where naive approaches collapse.

Right now, we're in the "scale what works" phase. Kakuna is our workhorse—142M parameters, binary rewards, pure self-play. But we're not abandoning the tools we built. They're waiting in the wings for when scale alone stops being enough.

And in the meantime? We're building a 10x faster self-play pipeline and designing test-time search experiments. Because the real lesson isn't "sophistication doesn't matter." It's "sophistication matters at the right time."

Sometimes you need a Kakuna to teach you that the right time isn't always now.

---

*Metamon is an open-source project for training RL agents to play competitive Pokémon. All code, pretrained models, and datasets are available at [github.com/EDM-Research/metamon](https://github.com/EDM-Research/metamon). This work was published at the Reinforcement Learning Conference (RLC) 2025.*
