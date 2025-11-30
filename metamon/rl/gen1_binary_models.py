"""
Custom model wrappers for Gen1 specialist finetuning checkpoints.

These models can be used with the evaluation scripts to test different training epochs
against each other and compute ELO ratings.

Includes:
- Gen1BinaryV0 checkpoints: Failed BinaryReward experiment (for reference)
- P0_SYN_V2_GEN1: Phase 0 baseline for Nash equilibrium training
"""

from metamon.rl.pretrained import LocalFinetunedModel, pretrained_model, SyntheticRLV2
from metamon.interface import get_reward_function

# Checkpoint directory - adjust this path if your checkpoints are elsewhere
CHECKPOINT_DIR = "/home/eddie/metamon_checkpoints"


#####################
## Phase 0: Baseline
#####################


@pretrained_model("P0_SYN_V2_GEN1")
class P0_SYN_V2_GEN1(LocalFinetunedModel):
    """
    Phase 0 Gen1 Baseline - Nash equilibrium training foundation.

    This is a Gen1 OU specialist created by fine-tuning SyntheticRLV2 on Gen1 replays
    using DefaultShapedReward. It serves as the initial strong policy π₁ in the Nash
    population before PSRO training begins.

    Training config:
    - Base: SyntheticRLV2 (200M params, multi-gen general model)
    - Format: Gen1 OU only (175k battles)
    - Reward: DefaultShapedReward (learned from BinaryReward failure)
    - Epochs: 3 (~75k steps)
    - Purpose: Baseline for interaction matrix and PSRO
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="P0_SYN_V2_GEN1",
            default_checkpoint=2,  # Epoch 2 is typically best for fine-tuning
            reward_function=get_reward_function("DefaultShapedReward"),
        )


###################################
## Dynamic Damping Self-Play Models
###################################


@pretrained_model("DampedConservative100k_Epoch2")
class DampedConservative100k_Epoch2(LocalFinetunedModel):
    """
    Conservative Damping Self-Play - Epoch 2

    Self-play training with conservative damping parameters on Gen1 OU data.
    Uses adaptive regularization to prevent policy collapse during self-play.

    Training config:
    - Base: SyntheticRLV2
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: DefaultShapedReward
    - Damping: Conservative power-law KL regularization
    - WandB: damped-conservative-100k (run 0jrll78y)
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/gen1_selfplay_damped_con_ckpt",
            model_name="damped-conservative-100k",
            default_checkpoint=2,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("DefaultShapedReward"),
        )


@pretrained_model("DampedConservative100k_Epoch3")
class DampedConservative100k_Epoch3(LocalFinetunedModel):
    """
    Conservative Damping Self-Play - Epoch 3 (Latest)

    Self-play training with conservative damping parameters on Gen1 OU data.
    Uses adaptive regularization to prevent policy collapse during self-play.

    Training config:
    - Base: SyntheticRLV2
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: DefaultShapedReward
    - Damping: Conservative power-law KL regularization
    - WandB: damped-conservative-100k (run 0jrll78y)
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/gen1_selfplay_damped_con_ckpt",
            model_name="damped-conservative-100k",
            default_checkpoint=3,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("DefaultShapedReward"),
        )


@pretrained_model("DampedConservativeBinaryV2_Epoch2")
class DampedConservativeBinaryV2_Epoch2(LocalFinetunedModel):
    """
    Conservative Damping Binary Reward V2 - Epoch 2

    Finetuned from DampedConservative100k using BinaryReward (sparse +/-100 win/loss).
    Tests whether the conservative damping policy can adapt to binary rewards while
    maintaining stability.

    Training config:
    - Base: DampedConservative100k (finetuned from SyntheticRLV2)
    - Format: Gen1 OU
    - Training: vanilla_selfplay_damped_conservative.gin
    - Reward: BinaryReward (sparse)
    - Damping: Conservative power-law KL regularization
    """

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir="/home/eddie/gen1_selfplay_damped_con_binary_v2_ckpt",
            model_name="damped-conservative-100k-binary-v2",
            default_checkpoint=2,
            train_gin_config="vanilla_selfplay_damped_conservative.gin",
            reward_function=get_reward_function("BinaryReward"),
        )


##############################
## Gen1 BinaryReward Experiment
##############################


@pretrained_model("Gen1BinaryV0_Epoch0")
class Gen1BinaryV0_Epoch0(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 0 (75k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=0,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch2")
class Gen1BinaryV0_Epoch2(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 2 (125k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=2,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch4")
class Gen1BinaryV0_Epoch4(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 4 (175k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=4,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch6")
class Gen1BinaryV0_Epoch6(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 6 (225k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=6,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch8")
class Gen1BinaryV0_Epoch8(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 8 (275k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=8,
            reward_function=get_reward_function("BinaryReward"),
        )


@pretrained_model("Gen1BinaryV0_Epoch10")
class Gen1BinaryV0_Epoch10(LocalFinetunedModel):
    """Gen1 BinaryReward Specialist - Epoch 10 (325k steps)"""

    def __init__(self):
        super().__init__(
            base_model=SyntheticRLV2,
            amago_ckpt_dir=CHECKPOINT_DIR,
            model_name="Gen1BinaryRewardV0",
            default_checkpoint=10,
            reward_function=get_reward_function("BinaryReward"),
        )
