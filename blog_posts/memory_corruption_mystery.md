# A Memory Corruption Mystery: When Shared State Goes Wrong

It's 3am and my self-play training run just crashed. Again. `free(): invalid next size (unsorted)`. The dreaded heap corruption message that every systems programmer knows means "you're in for a bad time."

The pattern was maddeningly consistent: somewhere between 80 and 256 battles, the process would segfault. Not at battle 1, not at battle 10,000, but always in that narrow window. Classic memory corruption behavior—silent damage accumulating until some unlucky allocation triggers the corruption detector.

## The Obvious Suspect

I had recently integrated PyKMN, a C++ library for ultra-fast Pokémon battle simulation, into my reinforcement learning pipeline. When you start getting heap corruption right after adding C++ bindings, the natural instinct is to blame the FFI boundary. Python's reference counting meets C++'s manual memory management is a well-known footgun. My first hypothesis: PyKMN must be holding onto freed memory or double-freeing something.

So I did what any reasonable engineer would do: I wrote a minimal stress test.

```python
# Hammer PyKMN directly, no RL framework overhead
for i in range(4000):
    battle = pykmn.Battle(...)
    while not battle.ended:
        battle.update(random_actions())
    print(f"Battle {i} complete")
```

I let it run. 1000 battles. 2000 battles. 4000 battles at 5000 battles per second. Zero crashes. Not a single segfault. PyKMN was rock solid.

## The Plot Twist

If PyKMN wasn't the problem, what was? I started adding instrumentation everywhere. Memory profiling. Heap snapshots. Verbose logging around every PyKMN call. That's when I noticed something odd: certain Pokémon were appearing in the wrong battles. An opponent's Pokémon from environment 3 would show up in the observation for environment 7.

My training setup used 16 vectorized environments running in parallel. Each environment needed to convert the raw game state into an "observation"—the processed representation that the neural network actually sees. To avoid duplicating code, I had created a shared `ObservationSpace` object that all 16 environments would call into:

```python
# Simplified version of the bug
class ObservationSpace:
    def __init__(self):
        self.seen_moves = set()  # Accumulates across ALL calls!

    def observe(self, battle_state):
        for move in battle_state.moves:
            self.seen_moves.add(move)  # Whoops
        return self._encode(battle_state)

# Meanwhile, in the vectorized environment:
obs_space = ObservationSpace()  # One shared instance
for env_id in range(16):
    obs = obs_space.observe(battles[env_id])  # All using same object
```

The observation space was supposed to be stateless—a pure function converting game state to tensor. But it wasn't. It was *accumulating state* across all 16 environments. Every call to `observe()` would insert new items into internal sets tracking seen moves, species, abilities. Environment 0's Pokémon would leak into environment 1's observation. Environment 15's move history would contaminate environment 0.

## The Compound Bug

This wasn't just causing incorrect observations (bad enough on its own). It was also causing unbounded memory growth. Each battle would add new entries to these sets. After 80 battles across 16 environments, that's 1,280 individual calls to `observe()`. Thousands of string insertions into Python sets. The heap would grow and grow, fragmenting as sets rehashed and reallocated, until finally some allocation would stumble into the corrupted region and trigger the crash.

The segfault wasn't coming from C++ at all. It was pure Python heap corruption from unbounded state accumulation, triggered by my own design mistake.

## The Fix: State-Explicit Everything

The solution turned out to be a strict protocol: *never mutate shared state*. If you need to track something during observation construction, create it locally and pass it as a parameter:

```python
class ObservationSpace:
    def observe(self, battle_state):
        seen_moves = set()  # Local to this call
        for move in battle_state.moves:
            seen_moves.add(move)
        return self._encode(battle_state, seen_moves)
```

Even better: make the observation space truly immutable by moving all temporary state into the encoder functions themselves. If you can't mutate it, you can't accumulate unbounded state.

After this refactor, I ran 10,000+ battles without a single crash. Memory usage flatlined at a healthy constant level. The observations were finally correct—no more Pokémon leaking between environments.

## The Lesson

When you're debugging across language boundaries (Python/C++, JavaScript/Rust, whatever), the first instinct is to blame the foreign code. It's opaque, it's scary, it could be doing anything under the hood.

But before you go down that rabbit hole, **test the library in isolation**. Write the simplest possible harness that exercises it heavily. If it survives 4000 iterations at 5000 ops/sec, the bug is probably on your side.

The other lesson is about state management in parallel systems. Shared mutable state is always dangerous, but it's especially insidious when the mutation is *subtle*—adding to a set here, updating a flag there. These kinds of bugs don't fail fast. They accumulate silently until the corruption is catastrophic.

If I could go back and give myself advice at 3am, it would be this: when you see intermittent crashes after N iterations, check for unbounded accumulation first. Profile the memory. Check for sets or lists that keep growing. Look for "stateless" objects that aren't actually stateless.

And maybe get some sleep before debugging. But who am I kidding—the mystery is too compelling to wait until morning.

---

*P.S. — If you're working with vectorized RL environments, consider making your observation/action spaces frozen dataclasses or pure functions. Future you will thank you.*
