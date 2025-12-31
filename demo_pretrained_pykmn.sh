#!/bin/bash
# Demo: Generate self-play data with pretrained SyntheticRLV2 model using pykmn

# Activate environment and set cache
source .venv/bin/activate
export METAMON_CACHE_DIR=/home/eddie/metamon_cache

# Run 10 battles with SyntheticRLV2 playing against itself
echo "Generating 10 self-play battles with SyntheticRLV2..."
python scripts/generate_selfplay_pykmn.py \
    --team_dir ~/metamon_cache/teams/modern_replays_v2 \
    --num_battles 10 \
    --num_envs 2 \
    --save_dir ~/pykmn_demo_output \
    --format gen1ou \
    --model SyntheticRLV2 \
    --checkpoint 48 \
    --device cuda \
    --temperature 1.0 \
    --verbose

echo ""
echo "Demo complete! Check ~/pykmn_demo_output/gen1ou/ for generated trajectories"
