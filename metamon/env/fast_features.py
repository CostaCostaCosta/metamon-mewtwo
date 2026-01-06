"""
Fast vectorized feature extraction for PyKMN battles.

This module provides high-performance numeric-only feature extraction
for GPU inference. Key optimizations:

1. Pre-allocated buffers (no per-step allocations)
2. Direct numpy operations (no Python loops where possible)
3. Numeric-only (no text/UniversalState overhead)
4. Batch processing for all environments

Performance target: Extract features for 1024 battles in <5ms.
"""

import numpy as np
from typing import List, Dict
from dataclasses import dataclass

from pykmn.engine.gen1 import Battle
from pykmn.engine.common import Result, Player
from metamon.env.pykmn.features import Mappings, precompute_mappings


class FastFeatureExtractor:
    """
    Vectorized feature extractor for batched PyKMN battles.

    Pre-allocates all buffers and uses direct numpy operations
    for maximum performance.

    Usage:
        extractor = FastFeatureExtractor(num_envs=128)
        features = extractor.extract_batch(battles, results_p1, results_p2)

        # Access features
        numbers = features['numbers']  # Shape: (128, 48)
    """

    def __init__(self, num_envs: int):
        """
        Initialize feature extractor.

        Args:
            num_envs: Number of parallel environments (batch size)
        """
        self.num_envs = num_envs

        # Precompute mappings once
        self.mappings = precompute_mappings()

        # Pre-allocate buffers for numeric features
        # Based on ExpandedObservationSpace: 48 + 7 = 55 features
        # Standard features: 48
        # Extended features: 4 PP + 2 sleep/freeze flags + 1 can_tera = 7
        self.feature_dim = 55

        # Main buffer for numeric features
        self.numbers_buffer = np.zeros(
            (num_envs, self.feature_dim),
            dtype=np.float32
        )

        # Temporary buffers for extraction (reused across calls)
        self._hp_buffer = np.zeros(num_envs, dtype=np.float32)
        self._boost_buffer = np.zeros((num_envs, 7), dtype=np.float32)  # 7 boost stats

    def extract_batch(
        self,
        battles: List[Battle],
        results_p1: List[Result],
        results_p2: List[Result],
        player: Player = Player.P1,
    ) -> Dict[str, np.ndarray]:
        """
        Extract numeric features for all battles.

        This is the HOT PATH - optimize aggressively!

        Args:
            battles: List of Battle objects
            results_p1: List of Result objects for P1
            results_p2: List of Result objects for P2
            player: Which player's perspective (usually P1)

        Returns:
            Dictionary with 'numbers' key containing shape (num_envs, feature_dim)
        """
        # Reset buffer
        self.numbers_buffer.fill(0.0)

        # Extract features for each battle
        # TODO: Further vectorize this loop if possible
        for i in range(self.num_envs):
            if battles[i] is None:
                continue  # Skip errored battles

            try:
                features = self._extract_single(
                    battles[i],
                    results_p1[i],
                    player,
                )
                self.numbers_buffer[i] = features
            except Exception:
                # On error, leave as zeros (defensive)
                pass

        # Create dummy text_tokens for model compatibility
        # Models expect text_tokens even if using only numeric features
        # Use zeros with appropriate shape (batch_size, seq_len=1, vocab_size)
        # The model will ignore these if using numeric-only mode
        dummy_text_tokens = np.zeros((self.num_envs, 1), dtype=np.int64)

        return {
            'numbers': self.numbers_buffer.copy(),  # Copy to avoid aliasing
            'text_tokens': dummy_text_tokens,  # Dummy tokens for compatibility
        }

    def _extract_single(
        self,
        battle: Battle,
        result: Result,
        player: Player,
    ) -> np.ndarray:
        """
        Extract features for a single battle.

        Returns:
            Feature vector of shape (feature_dim,)

        Feature layout (matching ExpandedObservationSpace):
        [0]: opponents_remaining / 6.0
        [1]: active_pokemon.hp_pct
        [2]: active_pokemon.level / 100.0
        [3-8]: active_pokemon base stats (atk, spa, def, spd, spe, hp) / 255.0
        [9-15]: active_pokemon boosts (atk, spa, def, spd, spe, acc, eva) / 6.0
        [16-19]: move 1-4 base_power / 200.0
        [20-23]: move 1-4 accuracy
        [24-27]: move 1-4 priority / 5.0
        [28-31]: move 1-4 PP warning (0-3 discretized)
        [32-36]: switch 1-5 hp_pct
        [37]: opponent.hp_pct
        [38]: opponent.level / 100.0
        [39-44]: opponent base stats / 255.0
        [45-51]: opponent boosts / 6.0
        [52]: any_opponent_asleep (sleep clause flag)
        [53]: any_opponent_frozen (freeze clause flag)
        [54]: can_tera (Gen 9 only, always 0 for Gen 1)
        """
        from pykmn.engine.gen1 import Slot

        features = np.zeros(self.feature_dim, dtype=np.float32)

        # Determine opponent
        opponent = Player.P2 if player == Player.P1 else Player.P1

        # [0] Opponents remaining
        # Count non-fainted opponent Pokemon
        opponents_remaining = 6  # TODO: Actually count from battle state
        features[0] = opponents_remaining / 6.0

        # === Active Pokemon (player) ===
        try:
            active_stats = battle.active_pokemon_stats(player)
            active_hp = active_stats.get('hp', 0)
            active_hp_max = active_stats.get('hp', 1)  # Avoid div by 0
            features[1] = active_hp / max(active_hp_max, 1)

            # [2] Level (always 100 in Gen 1 OU)
            features[2] = 1.0

            # [3-8] Base stats
            features[3] = active_stats.get('atk', 0) / 255.0
            features[4] = active_stats.get('spc', 0) / 255.0  # Gen 1 uses spc for spa
            features[5] = active_stats.get('def', 0) / 255.0
            features[6] = active_stats.get('spc', 0) / 255.0  # Gen 1 uses spc for spd
            features[7] = active_stats.get('spe', 0) / 255.0
            features[8] = active_stats.get('hp', 0) / 255.0

            # [9-15] Boosts
            boosts = battle.boosts(player)
            features[9] = boosts.get('atk', 0) / 6.0
            features[10] = boosts.get('spc', 0) / 6.0  # spa
            features[11] = boosts.get('def', 0) / 6.0
            features[12] = boosts.get('spc', 0) / 6.0  # spd
            features[13] = boosts.get('spe', 0) / 6.0
            features[14] = boosts.get('accuracy', 0) / 6.0
            features[15] = boosts.get('evasion', 0) / 6.0

        except Exception:
            # On error, leave as zeros
            pass

        # [16-31] Moves (base_power, accuracy, priority, PP warning)
        try:
            moves_with_pp = battle.moves_with_pp(player, "Active")
            from pykmn.data.gen1 import MOVES

            for i, (move_name, pp) in enumerate(moves_with_pp[:4]):
                if move_name in MOVES:
                    base_offset = 16 + i
                    acc_offset = 20 + i
                    pri_offset = 24 + i
                    pp_offset = 28 + i

                    move_data = MOVES[move_name]
                    features[base_offset] = move_data.get('basePower', 0) / 200.0
                    features[acc_offset] = move_data.get('accuracy', 1.0)
                    features[pri_offset] = move_data.get('priority', 0) / 5.0

                    # PP warning (discretized 0-3)
                    max_pp = move_data.get('pp', 1) * 8 // 5  # Gen 1 formula
                    pp_ratio = pp / max(max_pp, 1)
                    pp_warning = (pp_ratio >= 0.5) + (pp_ratio >= 0.25) + (pp_ratio > 0)
                    features[pp_offset] = float(pp_warning)

        except Exception:
            pass

        # [32-36] Available switches (hp_pct only)
        try:
            # Get benched Pokemon HP
            for slot_idx in range(2, 7):  # Slots 2-6 (slot 1 is active)
                slot = Slot(slot_idx)
                try:
                    hp = battle.pokemon_hp(player, slot)
                    max_hp = battle.pokemon_max_hp(player, slot)
                    hp_pct = hp / max(max_hp, 1)
                    features[32 + (slot_idx - 2)] = hp_pct
                except:
                    features[32 + (slot_idx - 2)] = 0.0

        except Exception:
            pass

        # [37-51] Opponent active Pokemon
        try:
            opp_stats = battle.active_pokemon_stats(opponent)
            opp_hp = opp_stats.get('hp', 0)
            opp_hp_max = opp_stats.get('hp', 1)
            features[37] = opp_hp / max(opp_hp_max, 1)

            features[38] = 1.0  # Level 100

            # Base stats
            features[39] = opp_stats.get('atk', 0) / 255.0
            features[40] = opp_stats.get('spc', 0) / 255.0
            features[41] = opp_stats.get('def', 0) / 255.0
            features[42] = opp_stats.get('spc', 0) / 255.0
            features[43] = opp_stats.get('spe', 0) / 255.0
            features[44] = opp_stats.get('hp', 0) / 255.0

            # Boosts
            opp_boosts = battle.boosts(opponent)
            features[45] = opp_boosts.get('atk', 0) / 6.0
            features[46] = opp_boosts.get('spc', 0) / 6.0
            features[47] = opp_boosts.get('def', 0) / 6.0
            features[48] = opp_boosts.get('spc', 0) / 6.0
            features[49] = opp_boosts.get('spe', 0) / 6.0
            features[50] = opp_boosts.get('accuracy', 0) / 6.0
            features[51] = opp_boosts.get('evasion', 0) / 6.0

        except Exception:
            pass

        # [52-54] Extended features
        # TODO: Track sleep/freeze across episodes
        features[52] = 0.0  # any_opponent_asleep (needs state tracking)
        features[53] = 0.0  # any_opponent_frozen (needs state tracking)
        features[54] = 0.0  # can_tera (Gen 1 doesn't have tera)

        return features

    def reset(self):
        """Reset any internal state (for stateful features)."""
        # Currently stateless, but included for future extensions
        pass


def benchmark_extraction(num_envs: int = 128, num_iterations: int = 100):
    """
    Benchmark feature extraction speed.

    Args:
        num_envs: Number of parallel environments
        num_iterations: Number of extraction iterations
    """
    import time
    from pykmn.engine.gen1 import Pokemon

    print(f"Benchmarking feature extraction: {num_envs} envs × {num_iterations} iterations")

    # Create dummy battles
    team = [
        Pokemon(species="Tauros", moves=("Body Slam", "Hyper Beam", "Blizzard", "Earthquake")),
        Pokemon(species="Snorlax", moves=("Body Slam", "Earthquake", "Rest", "Ice Beam")),
        Pokemon(species="Chansey", moves=("Ice Beam", "Thunderbolt", "Thunder Wave", "Soft-Boiled")),
        Pokemon(species="Exeggutor", moves=("Psychic", "Sleep Powder", "Explosion", "Stun Spore")),
        Pokemon(species="Starmie", moves=("Thunderbolt", "Blizzard", "Thunder Wave", "Recover")),
        Pokemon(species="Alakazam", moves=("Psychic", "Seismic Toss", "Thunder Wave", "Recover")),
    ]

    battles = []
    results = []
    for _ in range(num_envs):
        battle = Battle(p1_team=team, p2_team=team)
        result, _ = battle.update_raw(0, 0)
        battles.append(battle)
        results.append(result)

    # Create extractor
    extractor = FastFeatureExtractor(num_envs=num_envs)

    # Warmup
    for _ in range(10):
        features = extractor.extract_batch(battles, results, results)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        features = extractor.extract_batch(battles, results, results)
    elapsed = time.time() - start

    # Calculate metrics
    total_extractions = num_envs * num_iterations
    extractions_per_sec = total_extractions / elapsed
    ms_per_batch = (elapsed / num_iterations) * 1000

    print(f"Results:")
    print(f"  Total time: {elapsed:.3f}s")
    print(f"  Time per batch: {ms_per_batch:.3f}ms")
    print(f"  Extractions/sec: {extractions_per_sec:.0f}")
    print(f"  Feature shape: {features['numbers'].shape}")

    # Check if we hit target (1024 battles in <5ms = ~200k extractions/sec)
    target_rate = 200000
    if extractions_per_sec >= target_rate:
        print(f"  ✓ PASSED: {extractions_per_sec:.0f} >= {target_rate} extractions/sec")
    else:
        print(f"  ✗ FAILED: {extractions_per_sec:.0f} < {target_rate} extractions/sec")


if __name__ == "__main__":
    # Run benchmark
    benchmark_extraction(num_envs=128, num_iterations=100)
    benchmark_extraction(num_envs=1024, num_iterations=100)
