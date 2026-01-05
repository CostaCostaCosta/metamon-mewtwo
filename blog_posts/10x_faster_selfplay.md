# 10x Faster Self-Play: The Batching Inference Journey

I needed to generate millions of Pokémon battles for self-play training. My inference pipeline was painfully slow at 1.9 battles per second, and I was staring down weeks of data collection. So I did what any reasonable person would do: I went on a batching adventure that turned into a debugging odyssey. Spoiler: I got a 10.9x speedup, but not before discovering that my "optimized" code was secretly sabotaging itself.

## The Sequential Baseline: 1.9 Battles/Sec

The original pipeline was straightforward: run 16 parallel environments, but query the policy network separately for each one. Simple, correct, and slow. The transformer spent most of its time waiting around, processing one environment at a time while the GPU sat there twiddling its thumbs. Classic underutilization.

The obvious fix: batch the inference across all 16 environments. Send 16 game states through the network in one shot, get 16 actions back. GPU goes brrrr, everyone's happy. Except this is RL with recurrent models, and nothing is ever that simple.

## The Challenge: RL2 State Management

Here's where things get interesting. I'm using AMAGO, a meta-RL framework where the transformer maintains hidden states across episodes and tracks time indices for positional encoding. Every environment has its own timeline, its own context buffer, its own hidden state. When you batch inference, you need to:

1. **Track per-environment time indices** - Environment 3 might be on timestep 47 while Environment 11 just started at timestep 0
2. **Manage independent hidden states** - Each env's transformer state needs to persist across its own episodes
3. **Handle asynchronous resets** - When Environment 7 finishes, you reset *just* that environment's state, not the others

This is the kind of bookkeeping that makes you question your life choices. But I wanted 10x speedups, so into the weeds I went.

## The Implementation: VectorizedAMAGO

I built a `VectorizedAMAGO` class that wraps the standard AMAGO agent. The key insight: maintain parallel arrays for everything. `batch_time_idxs` tracks each environment's current timestep. `batch_traj_histories` holds each env's context buffer. `batch_tstep_encodings` stores per-env hidden states.

The inference loop becomes:

```python
def batched_forward(self, observations, dones):
    # Update time indices
    self.batch_time_idxs[dones] = 0  # Reset finished envs
    self.batch_time_idxs[~dones] += 1

    # Batch forward pass
    actions, values, new_states = self.agent.batch_forward(
        obs=observations,
        time_idxs=self.batch_time_idxs,
        tstep_encodings=self.batch_tstep_encodings
    )

    # Selective state updates
    self.batch_tstep_encodings[~dones] = new_states[~dones]

    return actions, values
```

Selective masking is critical. When an environment finishes, you reset its state but leave the others untouched. This was the part I thought I had right.

## The Bug: Resetting Everyone When Anyone Finishes

I ran the batched version. It worked! 20.8 battles per second! I was about to declare victory when I noticed something odd in my profiling data. The speedup was exactly 10.9x, but the math suggested it should be closer to 15-16x with 16 environments.

I added debug logging. And there it was, the stupidest bug I've written in months:

```python
# Old (wrong) code
if dones.any():
    self.batch_tstep_encodings[:] = self.agent.init_tstep_encodings
```

See it? When *any* environment finished, I was resetting the hidden states for *all 16 environments*. Every single battle ending would wipe everyone's context, forcing the transformer to start fresh. This was happening ~40 times per 100 battles.

The fix was embarrassingly simple:

```python
# New (correct) code
if dones.any():
    init = self.agent.init_tstep_encodings
    self.batch_tstep_encodings[dones] = init[dones]
```

After this fix, I profiled battles completed vs state resets. Before: 100 battles → 40 full resets → 640 unnecessary state wipes (16 envs × 40 resets). After: 100 battles → 100 targeted resets. The performance impact was real - I was throwing away 20x more hidden state than necessary.

## Mixed Precision: The Free Lunch

RL people love to fear mixed precision training. "What about the action masking?" "What about numerical instability?" "What about the logits?"

I tried bfloat16 anyway. It just... worked? No NaN gradients, no invalid actions, no weird policy collapse. The legal action masking (setting illegal move logits to -inf) survived the lower precision just fine. This isn't magic - bfloat16 handles large negative numbers well, and RL training is surprisingly robust to small numerical errors.

The gains weren't massive (we're inference-bound, not memory-bound), but every bit helps. And it's nice to know that mixed precision isn't the footgun everyone makes it out to be.

## The New Bottleneck: PyKMN Simulation

With batched inference working, I profiled again. Here's where the time goes now:

- **38%** - PyKMN battle simulation (state transitions, damage calculation, move legality)
- **28%** - Neural network inference (batched transformer forward passes)
- **21%** - Observation processing (tokenization, feature extraction)
- **13%** - Everything else (logging, data serialization, overhead)

PyKMN - the Rust-based Pokémon battle simulator - is now the bottleneck. This is actually great news. It means my inference pipeline is finally efficient enough that the actual game simulation matters. I'm limited by how fast I can compute "Thunderbolt vs Zapdos with Light Screen up while paralyzed" rather than how fast I can push tensors around.

There might be more juice to squeeze here (vectorize PyKMN? rewrite observation processing in Rust?), but these are diminishing returns. I'm at 20.8 battles/second. That's 1.8 million battles per day on a single GPU. For reference, my Phase 0 training needed 500 battles per iteration. I can now collect an iteration's worth of data in 24 seconds.

## What I Learned

1. **Batch everything** - 10.9x is real, even with messy RL2 state management
2. **Profile relentlessly** - I would've never caught the state reset bug without detailed logging
3. **Mixed precision is fine** - Stop worrying and embrace bfloat16
4. **Success looks like new bottlenecks** - When PyKMN becomes your problem, your inference pipeline is probably pretty good

The deeper lesson: RL research iteration speed is criminally underrated. Going from 1.9 to 20.8 battles/second isn't just a nice speedup - it's the difference between "run overnight and check in the morning" and "iterate interactively during your afternoon coffee." That changes how you think, what experiments you try, and how quickly you can chase down ideas.

Now if you'll excuse me, I have 1.8 million battles per day to generate. The Nash equilibrium awaits.
