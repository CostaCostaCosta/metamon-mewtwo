# Test-Time Compute Meets Pokémon: Deliberation in the Fog of War

The most interesting development in AI this year wasn't a bigger model—it was OpenAI's o1, which showed that you can get dramatically better performance by letting models "think" at test time. The idea is beautifully simple: instead of generating an answer immediately, spend compute budget on internal deliberation (so-called "reasoning traces"), then reinforce trajectories that lead to correct answers. This is test-time compute scaling, and it's opened up a new dimension beyond just training bigger models.

Shortly after o1's release, the Ataraxos paper explored similar ideas in game-playing agents. Their insight: even with a fixed policy, you can improve performance by deliberating longer at test time—generating multiple candidate actions, simulating outcomes, and choosing based on learned value estimates. They showed meaningful gains in NetHack, a challenging roguelike with partial observability.

This brings me to my current research problem: training human-level Pokémon agents. Pokémon is an imperfect information game—you don't know your opponent's exact team composition, held items, movesets, or stat distributions until they're revealed through play. When o1 came out, my immediate thought was: *can we do this for Pokémon?*

## The Challenge of Hidden State

In perfect information games like chess or Go, test-time search is straightforward. You know the exact board state, so you can explicitly enumerate possible moves and recursively evaluate resulting positions. AlphaGo's MCTS does exactly this: build a search tree, guided by a policy and value network, and pick the move that leads to the best-evaluated outcomes.

But in imperfect information games, the opponent's hidden state creates exponential branching. When I see my opponent's Tauros for the first time in Gen 1 OU, I don't know:
- Which moves it has (Hyper Beam? Earthquake? Blizzard? Body Slam?)
- Whether it's holding leftovers or another item
- Its exact EV spread (though Gen 1 uses DVs, the principle holds)

Each of these hidden variables multiplies the possible game states. If I try to enumerate all possibilities explicitly, the search tree explodes before I've looked even a few moves ahead. Classical game tree search assumes you can evaluate positions efficiently; here, I first need to infer what the true game state *is*.

## Learned Value Functions and Uncertainty

This is where learned models become essential. Instead of trying to maintain a belief distribution over all possible opponent configurations (intractable), we can:

1. **Use learned value functions as heuristics**: Our transformer policies learn to estimate win probability from partial observations. This implicitly marginalizes over likely opponent states based on patterns seen in training.

2. **Sample plausible trajectories**: Generate multiple candidate action sequences (potentially with temperature > 0), simulate forward using the learned policy as an opponent model, and evaluate terminal or intermediate states.

3. **Leverage ensemble uncertainty**: Train an ensemble of value predictors. High disagreement signals states where the model is uncertain—likely because the opponent's hidden information matters most. This tells us *where* deliberation would help.

The Ataraxos approach is conceptually similar: generate candidate actions, roll out imagined futures, aggregate value estimates, and select the action with the best expected outcome. The key difference in imperfect information is that our value functions must implicitly handle hidden state, and our rollouts are approximations over a distribution of possible true game states.

## Engineering Prerequisites

Here's the practical reality: test-time search is only viable if inference is fast. If your policy takes 500ms per forward pass, you can't afford to evaluate 50 candidate action sequences in real-time gameplay (that's 25 seconds per move—unacceptable for Pokémon Showdown's turn timers).

This is why our recent focus has been on engineering: we achieved a ~10x inference speedup by moving to batch-mode evaluation with optimized C++ bindings to the Pokémon battle engine. What was once 300-500ms per action is now 30-50ms. Suddenly, evaluating 20-50 trajectories per move becomes feasible within a 1-2 second decision budget.

Fast inference also enables higher-quality self-play data collection, which improves the policy and value networks that guide search. It's a virtuous cycle: better models make search more effective, and faster search enables more ambitious training.

## What Would This Look Like in Practice?

Imagine our agent faces a critical turn: opponent's Chansey is at 60% HP, and we're deciding between switching to our Tauros or staying in with Zapdos. A naive policy picks one immediately based on win probability from the current observation.

With test-time search:
1. Generate 30 candidate action sequences (e.g., "switch to Tauros, then Hyper Beam, then Earthquake" vs "stay in, Thunderbolt, then switch")
2. Roll out each sequence for 3-5 turns using the policy as an opponent model
3. Evaluate terminal/intermediate states with an ensemble of value heads
4. Weight by ensemble agreement (high confidence) and upside potential
5. Select the action beginning the highest-value trajectory

The ensemble uncertainty is critical: if all value heads agree, trust the prediction. If they disagree, we're in a fog-of-war situation where the opponent's hidden state matters—maybe Chansey has Reflect, or their back-line has a Tauros counter. High uncertainty signals "deliberate more here."

## Looking Forward

We haven't implemented this yet—it's the natural next step now that fast inference is working. The open questions are:

- **How many rollouts?** More is better, but we're bounded by turn timers. Adaptive budgeting (spend more compute in critical positions) seems promising.
- **Opponent modeling**: Using our own policy as the opponent is convenient but biased. An ensemble of opponent policies (or opponent prediction heads) would be more robust.
- **Value head training**: Our current value functions are trained for action selection, not search. We may need to retrain with objectives that explicitly support multi-step lookahead.

The exciting part: this combines two scaling axes. Our largest model, Kakuna (150M parameters), is still training and should give us the strongest base policy yet. If we *also* add test-time search, we're scaling both model capacity and inference-time compute. o1 showed this works for reasoning tasks; Ataraxos showed it works for sequential decision-making in partial observability. Pokémon is the next frontier.

The dream is agents that don't just react based on learned patterns, but genuinely deliberate under uncertainty—agents that can plan ahead even when they can't see the full board. We're not there yet, but the pieces are falling into place.
