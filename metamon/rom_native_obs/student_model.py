"""
Small student policy models consuming the RomBattleState tensor format.

These models read the compact tensor dict produced by
``RomBattleState.to_tensors()`` (see ``metamon/rom_native_obs/schema.py``) and
produce 9 action logits matching Metamon's MinimalActionSpace
(4 move slots + 5 switch slots).

Design (small MLP with learned embeddings, NO transformer):

  1. Categorical embeddings, one ``nn.Embedding`` per ID space, shared across
     all 13 Pokémon slots:
       species (152), moves (166), types (20), status (9), weather (8),
       side conditions (8), field effects (8), effects (8), move category (5).

  2. Per-Pokémon encoding: concatenate embedded species/type/status/effect,
     4 move-id embeddings, 4 move-type embeddings, 4 move-category embeddings,
     the 31 numerical features and the 4 mask features, then a shared MLP
     (``pokemon_hidden`` -> ``pokemon_hidden`` with GELU/ReLU + LayerNorm).

  3. Global encoding: concatenate embedded weather/field/side-conditions/
     previous-moves plus the 3 numerical features, then an MLP.

  4. Combine (configurable):
       - ``"pool"``: global_repr + mean(pokemon_reprs) + max(pokemon_reprs)
         (invalid/padding slots are masked out using the ``valid`` bit)
       - ``"cat"``: [global_repr, pokemon_0_repr, ..., pokemon_12_repr]
     followed by the final MLP producing ``num_actions`` logits.

  5. The legal action mask is applied last (illegal logits set to ``-inf``).

Input shape conventions (leading dims):
  - Per-timestep (``sequence=False``):
      no leading dims            -> logits ``(9,)``
      leading batch dim ``B``    -> logits ``(B, 9)``
  - Sequence (``sequence=True``):
      leading time dim ``T``     -> logits ``(T, 9)``
      leading ``B, T``           -> logits ``(B, T, 9)``

``RomStudentGRUPolicy`` runs the encoded per-step features through a GRU before
the action head, for temporal policies (e.g. recurrent RL or sequence IL).

The module is self-contained: it only imports ``torch`` and the schema module
(no amago / gym / einops / gin dependencies).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .schema import (
    NUM_ACTIONS,
    NUM_POKEMON_SLOTS,
    NUM_MOVES_PER_POKEMON,
    GLOBAL_NUM_LEN,
    POKEMON_NUM_LEN,
    POKEMON_MASK_LEN,
    SPECIES_MAX_GEN1,
    MOVE_MAX_GEN1,
    TYPE_MAX,
    STATUS_MAX,
    WEATHER_MAX,
    SIDE_COND_MAX,
    FIELD_MAX,
    EFFECT_UNKNOWN,
    CATEGORY_UNKNOWN,
)

# ============================================================================
# ID space sizes (0-indexed; value 0 == unknown/none in every space)
# ============================================================================
VOCAB_SPECIES = SPECIES_MAX_GEN1 + 1  # 152
VOCAB_MOVE = MOVE_MAX_GEN1 + 1  # 166
VOCAB_TYPE = TYPE_MAX + 1  # 20
VOCAB_STATUS = STATUS_MAX + 1  # 9
VOCAB_WEATHER = WEATHER_MAX + 1  # 8
VOCAB_SIDE_COND = SIDE_COND_MAX + 1  # 8
VOCAB_FIELD = FIELD_MAX + 1  # 8
VOCAB_EFFECT = EFFECT_UNKNOWN + 1  # 8
VOCAB_MOVE_CAT = CATEGORY_UNKNOWN + 1  # 5

# ============================================================================
# Configuration
# ============================================================================


@dataclass
class StudentConfig:
    """Hyperparameters for :class:`RomStudentEncoder` / policies.

    Direct values can be passed via the ``**kwargs`` of any model constructor;
    ``preset`` (see :func:`preset_config`) fills every field first and explicit
    keyword arguments override individual fields.
    """

    species_emb_dim: int = 32
    move_emb_dim: int = 32
    type_emb_dim: int = 16
    status_emb_dim: int = 16
    weather_emb_dim: int = 16
    side_cond_emb_dim: int = 16
    field_emb_dim: int = 16
    effect_emb_dim: int = 16
    move_cat_emb_dim: int = 16

    pokemon_hidden: int = 320  # hidden width of the shared per-Pokémon MLP
    pokemon_layers: int = 2  # number of Linear layers in the per-Pokémon MLP
    global_hidden: int = 320  # hidden width of the global MLP
    global_layers: int = 2  # number of Linear layers in the global MLP

    combine: str = "pool"  # "pool" (sum of pooled reprs) or "cat" (concat 13 slots)
    feature_dim: Optional[int] = None  # optional projection of the combined feature

    final_hidden: int = 320  # hidden width of the final action-head MLP
    final_layers: int = 2  # number of Linear layers in the head
    activation: str = "gelu"  # "gelu", "relu" or "silu"
    dropout: float = 0.0
    use_layernorm: bool = True
    num_actions: int = NUM_ACTIONS

    def __post_init__(self) -> None:
        if self.combine not in ("pool", "cat"):
            raise ValueError(f"combine must be 'pool' or 'cat', got {self.combine!r}")
        if self.activation not in ("gelu", "relu", "silu"):
            raise ValueError(f"unknown activation {self.activation!r}")
        for name in ("pokemon_layers", "global_layers", "final_layers"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")

    # -- derived feature widths -------------------------------------------
    @property
    def pokemon_input_dim(self) -> int:
        return (
            self.species_emb_dim
            + 2 * self.type_emb_dim
            + self.status_emb_dim
            + self.effect_emb_dim
            + NUM_MOVES_PER_POKEMON * self.move_emb_dim
            + NUM_MOVES_PER_POKEMON * self.type_emb_dim
            + NUM_MOVES_PER_POKEMON * self.move_cat_emb_dim
            + POKEMON_NUM_LEN
            + POKEMON_MASK_LEN
        )

    @property
    def global_input_dim(self) -> int:
        return (
            self.weather_emb_dim
            + self.field_emb_dim
            + 2 * self.side_cond_emb_dim
            + 2 * self.move_emb_dim
            + GLOBAL_NUM_LEN
        )

    @property
    def combined_dim(self) -> int:
        """Dimension of the combine step output before any projection."""
        if self.combine == "cat":
            return self.global_hidden + NUM_POKEMON_SLOTS * self.pokemon_hidden
        return self.global_hidden  # pool: global + mean + max of pokemon reprs

    @property
    def output_feature_dim(self) -> int:
        """Final per-timestep feature dimension fed to the action head."""
        if self.combine == "cat":
            return (
                self.feature_dim if self.feature_dim is not None else self.combined_dim
            )
        if self.feature_dim is not None and self.feature_dim != self.combined_dim:
            return self.feature_dim
        return self.combined_dim


_PRESET_DEFAULTS: Dict[str, Dict[str, object]] = {
    # ~500k parameters
    "tiny": dict(
        species_emb_dim=32,
        move_emb_dim=32,
        type_emb_dim=16,
        status_emb_dim=16,
        weather_emb_dim=16,
        side_cond_emb_dim=16,
        field_emb_dim=16,
        effect_emb_dim=16,
        move_cat_emb_dim=16,
        pokemon_hidden=320,
        pokemon_layers=2,
        global_hidden=320,
        global_layers=2,
        combine="pool",
        feature_dim=None,
        final_hidden=320,
        final_layers=2,
    ),
    # ~1M parameters
    "small": dict(
        species_emb_dim=48,
        move_emb_dim=48,
        type_emb_dim=24,
        status_emb_dim=24,
        weather_emb_dim=24,
        side_cond_emb_dim=24,
        field_emb_dim=24,
        effect_emb_dim=24,
        move_cat_emb_dim=24,
        pokemon_hidden=448,
        pokemon_layers=2,
        global_hidden=448,
        global_layers=2,
        combine="pool",
        feature_dim=None,
        final_hidden=448,
        final_layers=2,
    ),
    # ~2M parameters
    "medium": dict(
        species_emb_dim=48,
        move_emb_dim=48,
        type_emb_dim=24,
        status_emb_dim=24,
        weather_emb_dim=24,
        side_cond_emb_dim=24,
        field_emb_dim=24,
        effect_emb_dim=24,
        move_cat_emb_dim=24,
        pokemon_hidden=256,
        pokemon_layers=2,
        global_hidden=256,
        global_layers=2,
        combine="cat",
        feature_dim=None,
        final_hidden=480,
        final_layers=2,
    ),
    # ~4M parameters
    "large": dict(
        species_emb_dim=48,
        move_emb_dim=48,
        type_emb_dim=24,
        status_emb_dim=24,
        weather_emb_dim=24,
        side_cond_emb_dim=24,
        field_emb_dim=24,
        effect_emb_dim=24,
        move_cat_emb_dim=24,
        pokemon_hidden=384,
        pokemon_layers=2,
        global_hidden=384,
        global_layers=2,
        combine="cat",
        feature_dim=None,
        final_hidden=640,
        final_layers=2,
    ),
}


def preset_config(preset: Optional[str], **overrides) -> StudentConfig:
    """Resolve a preset + explicit overrides into a :class:`StudentConfig`."""
    cfg = dict(_PRESET_DEFAULTS["tiny"])
    if preset is not None:
        if preset not in _PRESET_DEFAULTS:
            raise ValueError(
                f"unknown preset {preset!r}; available: {sorted(_PRESET_DEFAULTS)}"
            )
        cfg.update(_PRESET_DEFAULTS[preset])
    for key, value in overrides.items():
        if value is not None:
            if not hasattr(StudentConfig, key):
                raise TypeError(f"unexpected config argument {key!r}")
            cfg[key] = value
    return StudentConfig(**cfg)


# ============================================================================
# Shared building blocks
# ============================================================================


def _activation(name: str) -> nn.Module:
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"unknown activation {name!r}")


class MLP(nn.Module):
    """Small MLP: ``n_layers`` Linear layers with activations and optional
    LayerNorm/dropout between them.  The last layer is a plain Linear (no
    activation) producing ``out_dim``."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        n_layers: int = 2,
        activation: str = "gelu",
        dropout: float = 0.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        layers: list = []
        cur = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(cur, hidden_dim))
            layers.append(_activation(activation))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            if use_layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            cur = hidden_dim
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CategoricalEmbeddings(nn.Module):
    """Learned embeddings for every categorical ID space in the schema.

    All 13 Pokémon slots share these tables.
    """

    vocab = {
        "species": VOCAB_SPECIES,
        "move": VOCAB_MOVE,
        "type": VOCAB_TYPE,
        "status": VOCAB_STATUS,
        "weather": VOCAB_WEATHER,
        "side_cond": VOCAB_SIDE_COND,
        "field": VOCAB_FIELD,
        "effect": VOCAB_EFFECT,
        "move_cat": VOCAB_MOVE_CAT,
    }

    def __init__(self, config: StudentConfig):
        super().__init__()
        self.config = config
        self.species = nn.Embedding(VOCAB_SPECIES, config.species_emb_dim)
        self.move = nn.Embedding(VOCAB_MOVE, config.move_emb_dim)
        self.type_ = nn.Embedding(VOCAB_TYPE, config.type_emb_dim)
        self.status = nn.Embedding(VOCAB_STATUS, config.status_emb_dim)
        self.weather = nn.Embedding(VOCAB_WEATHER, config.weather_emb_dim)
        self.side_cond = nn.Embedding(VOCAB_SIDE_COND, config.side_cond_emb_dim)
        self.field = nn.Embedding(VOCAB_FIELD, config.field_emb_dim)
        self.effect = nn.Embedding(VOCAB_EFFECT, config.effect_emb_dim)
        self.move_cat = nn.Embedding(VOCAB_MOVE_CAT, config.move_cat_emb_dim)


# ============================================================================
# Encoder
# ============================================================================


class RomStudentEncoder(nn.Module):
    """Turns a RomBattleState tensor dict into per-timestep feature vectors.

    Pure pointwise MLP processing: any number of leading dims (batch/time) is
    flattened internally, so a whole (batch, time) stack can be encoded in one
    call.  Returns ``(features, legal_action_mask, leading_shape)``.
    """

    def __init__(self, preset: Optional[str] = None, **kwargs):
        super().__init__()
        self.config = preset_config(preset, **kwargs)
        cfg = self.config

        self.embeddings = CategoricalEmbeddings(cfg)

        self.pokemon_mlp = MLP(
            in_dim=cfg.pokemon_input_dim,
            out_dim=cfg.pokemon_hidden,
            hidden_dim=cfg.pokemon_hidden,
            n_layers=cfg.pokemon_layers,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        )
        self.global_mlp = MLP(
            in_dim=cfg.global_input_dim,
            out_dim=cfg.global_hidden,
            hidden_dim=cfg.global_hidden,
            n_layers=cfg.global_layers,
            activation=cfg.activation,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
        )

        if cfg.combine == "cat":
            # combine = concat of [global_repr, 13 x pokemon_repr]
            self.poke_proj = nn.Identity()
            self.feature_proj = (
                nn.Identity()
                if cfg.feature_dim is None
                else nn.Linear(cfg.combined_dim, cfg.feature_dim)
            )
        else:
            # combine = global_repr + mean(pokemon_reprs) + max(pokemon_reprs)
            self.poke_proj = (
                nn.Identity()
                if cfg.pokemon_hidden == cfg.global_hidden
                else nn.Linear(cfg.pokemon_hidden, cfg.global_hidden)
            )
            self.feature_proj = (
                nn.Identity()
                if cfg.feature_dim is None or cfg.feature_dim == cfg.global_hidden
                else nn.Linear(cfg.global_hidden, cfg.feature_dim)
            )

        self.feature_dim = cfg.output_feature_dim

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _emb(table: nn.Embedding, ids: torch.Tensor) -> torch.Tensor:
        """Embed categorical IDs with a defensive clamp into the table range.

        The schema guarantees in-range IDs; the clamp is a safety net so a
        single out-of-range ID can never crash a long-running RL loop."""
        return table(ids.clamp(0, table.num_embeddings - 1))

    @staticmethod
    def _as_tensors(tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        out = {}
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            out[key] = value
        return out

    @staticmethod
    def _leading_shape(
        tensors: Dict[str, torch.Tensor], sequence: bool
    ) -> Tuple[int, ...]:
        """Validate tensor shapes and return the shared leading dims."""
        expected = {
            "global_cat": (6,),
            "global_num": (3,),
            "pokemon_cat": (NUM_POKEMON_SLOTS, 9),
            "pokemon_move_cat": (NUM_POKEMON_SLOTS, NUM_MOVES_PER_POKEMON),
            "pokemon_move_type": (NUM_POKEMON_SLOTS, NUM_MOVES_PER_POKEMON),
            "pokemon_num": (NUM_POKEMON_SLOTS, POKEMON_NUM_LEN),
            "pokemon_mask": (NUM_POKEMON_SLOTS, POKEMON_MASK_LEN),
            "legal_action_mask": (NUM_ACTIONS,),
        }
        gcat = tensors["global_cat"]
        leading = gcat.shape[:-1]
        for key, value in tensors.items():
            if key not in expected:
                continue
            exp = expected[key]
            if value.shape[-len(exp) :] != exp:
                raise ValueError(
                    f"tensor {key!r} has trailing dims {tuple(value.shape[-len(exp):])}, "
                    f"expected {exp}"
                )
            if value.shape[: len(leading)] != leading:
                raise ValueError(
                    f"leading dims of {key!r} {tuple(value.shape[:len(leading)])} "
                    f"do not match global_cat {tuple(leading)}"
                )
        # pokemon tensors carry one extra leading dim (the slot dim)
        for key in (
            "pokemon_cat",
            "pokemon_move_cat",
            "pokemon_move_type",
            "pokemon_num",
            "pokemon_mask",
        ):
            if key in tensors:
                value = tensors[key]
                if value.shape[:-1] != leading + (NUM_POKEMON_SLOTS,):
                    raise ValueError(
                        f"pokemon tensor {key!r} shape {tuple(value.shape)} does not "
                        f"match leading dims {tuple(leading)} + slot dim"
                    )
        if sequence and len(leading) == 0:
            raise ValueError(
                "sequence=True requires a leading time dim (T, ...) or (B, T, ...)"
            )
        if sequence and len(leading) > 2:
            raise ValueError(f"too many leading dims {tuple(leading)} for a sequence")
        if not sequence and len(leading) > 1:
            raise ValueError(
                f"too many leading dims {tuple(leading)}; for sequences pass sequence=True"
            )
        return tuple(int(d) for d in leading)

    # -- main encoding -----------------------------------------------------

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        sequence: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        x = self._as_tensors(tensors)
        leading = self._leading_shape(x, sequence)

        if "legal_action_mask" not in x:
            x = dict(x)
            x["legal_action_mask"] = torch.ones(
                leading + (self.config.num_actions,), dtype=torch.int64
            )
        if "pokemon_mask" not in x:
            x = dict(x)
            x["pokemon_mask"] = torch.ones(
                leading + (NUM_POKEMON_SLOTS, POKEMON_MASK_LEN), dtype=torch.int64
            )

        n = 1
        for d in leading:
            n *= d

        # flatten all leading dims -> (N, ...)
        flat = {k: v.reshape(n, *v.shape[len(leading) :]) for k, v in x.items()}
        gcat = flat["global_cat"].long()
        gnum = flat["global_num"].float()
        pcat = flat["pokemon_cat"].long()
        pmove_cat = flat["pokemon_move_cat"].long()
        pmove_type = flat["pokemon_move_type"].long()
        pnum = flat["pokemon_num"].float()
        pmask = flat["pokemon_mask"].long()
        legal = flat["legal_action_mask"]

        emb = self.embeddings

        # ---- per-Pokémon encoding (shared MLP) --------------------------
        move_ids = self._emb(emb.move, pcat[..., 5:9])  # (N, 13, 4, move_emb)
        move_types = self._emb(emb.type_, pmove_type)  # (N, 13, 4, type_emb)
        move_cats = self._emb(emb.move_cat, pmove_cat)  # (N, 13, 4, cat_emb)
        pokemon_feat = torch.cat(
            [
                self._emb(emb.species, pcat[..., 0]),  # species
                self._emb(emb.type_, pcat[..., 1]),  # type_1
                self._emb(emb.type_, pcat[..., 2]),  # type_2
                self._emb(emb.status, pcat[..., 3]),  # status
                self._emb(emb.effect, pcat[..., 4]),  # effect
                move_ids.flatten(-2),  # 4 x move id
                move_types.flatten(-2),  # 4 x move type
                move_cats.flatten(-2),  # 4 x move category
                pnum,  # 31 numerical
                pmask.float(),  # 4 mask bits
            ],
            dim=-1,
        )  # (N, 13, pokemon_input_dim)
        prepr = self.pokemon_mlp(pokemon_feat)  # (N, 13, pokemon_hidden)

        # mask out padding / invalid slots before any pooling or concat
        valid = pmask[..., :1].float()  # (N, 13, 1)
        prepr = prepr * valid

        # ---- global encoding ---------------------------------------------
        global_feat = torch.cat(
            [
                self._emb(emb.weather, gcat[..., 0]),
                self._emb(emb.field, gcat[..., 1]),
                self._emb(emb.side_cond, gcat[..., 2]),
                self._emb(emb.side_cond, gcat[..., 3]),
                self._emb(emb.move, gcat[..., 4]),
                self._emb(emb.move, gcat[..., 5]),
                gnum,
            ],
            dim=-1,
        )  # (N, global_input_dim)
        grepr = self.global_mlp(global_feat)  # (N, global_hidden)

        # ---- combine ------------------------------------------------------
        if self.config.combine == "pool":
            prepr = self.poke_proj(prepr)  # -> (N, 13, global_hidden)
            count = valid.sum(dim=1).clamp(min=1.0)  # (N, 1)
            mean = prepr.sum(dim=1) / count  # (N, global_hidden)
            max_ = prepr.max(dim=1).values  # (N, global_hidden)
            feat = grepr + mean + max_  # (N, global_hidden)
        else:  # "cat"
            feat = torch.cat([grepr, prepr.flatten(1, 2)], dim=-1)  # (N, combined_dim)
        feat = self.feature_proj(feat)  # (N, feature_dim)

        # ---- restore the leading shape ------------------------------------
        features = feat.reshape(*leading, self.feature_dim)
        legal = legal.reshape(*leading, self.config.num_actions)
        return features, legal, leading

    def count_parameters(self, trainable_only: bool = False) -> int:
        params = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        return int(sum(p.numel() for p in params))


# ============================================================================
# Policies
# ============================================================================


class RomStudentPolicy(nn.Module):
    """Pointwise student policy: encoder + MLP head -> 9 action logits.

    Supports per-timestep (single state, optionally batched) and sequence
    inputs (steps processed independently).  For temporal sequence features
    with recurrence use :class:`RomStudentGRUPolicy`.
    """

    def __init__(self, preset: Optional[str] = None, **kwargs):
        super().__init__()
        self.encoder = RomStudentEncoder(preset=preset, **kwargs)
        self.config = self.encoder.config
        self.num_actions = self.config.num_actions
        self.head = MLP(
            in_dim=self.encoder.feature_dim,
            out_dim=self.num_actions,
            hidden_dim=self.config.final_hidden,
            n_layers=self.config.final_layers,
            activation=self.config.activation,
            dropout=self.config.dropout,
            use_layernorm=self.config.use_layernorm,
        )

    def forward(
        self, tensors: Dict[str, torch.Tensor], sequence: bool = False
    ) -> torch.Tensor:
        """Return masked action logits.

        Args:
            tensors: dict of tensors as produced by ``RomBattleState.to_tensors()``.
                Leading dims: none (single state), ``B`` (batch), or, with
                ``sequence=True``, ``T`` or ``B, T``.
            sequence: if True, the leading dims are interpreted as time steps
                (``T`` or ``B, T``).
        Returns:
            logits ``(9,)`` / ``(B, 9)`` / ``(T, 9)`` / ``(B, T, 9)``.
        """
        features, legal, _ = self.encoder(tensors, sequence=sequence)
        logits = self.head(features)
        logits = logits.masked_fill(legal == 0, float("-inf"))
        return logits

    def count_parameters(self, trainable_only: bool = False) -> int:
        params = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        return int(sum(p.numel() for p in params))


class RomStudentGRUPolicy(nn.Module):
    """Recurrent student policy: encoder + GRU + head -> 9 action logits.

    The per-timestep encoded features are fed through a GRU (batch_first) and
    the action head produces logits either for every step or only for the last
    step of each sequence.

    Inputs follow the sequence conventions described in the module docstring:
      - ``sequence=True``: ``(T, ...)`` or ``(B, T, ...)``
      - ``sequence=False``: single step, ``(...)`` or ``(B, ...)``
    """

    def __init__(
        self,
        preset: Optional[str] = None,
        *,
        gru_hidden: Optional[int] = None,
        gru_layers: int = 1,
        return_full_sequence: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.encoder = RomStudentEncoder(preset=preset, **kwargs)
        self.config = self.encoder.config
        self.num_actions = self.config.num_actions
        self.gru_hidden = (
            gru_hidden if gru_hidden is not None else self.encoder.feature_dim
        )
        self.gru_layers = gru_layers
        self.return_full_sequence = return_full_sequence
        self.gru = nn.GRU(
            input_size=self.encoder.feature_dim,
            hidden_size=self.gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
        )
        self.head = nn.Linear(self.gru_hidden, self.num_actions)

    def reset_hidden(self, batch_size: int = 1, device=None) -> torch.Tensor:
        """Return a zeroed initial GRU hidden state ``(gru_layers, B, gru_hidden)``."""
        return torch.zeros(self.gru_layers, batch_size, self.gru_hidden, device=device)

    def forward(
        self,
        tensors: Dict[str, torch.Tensor],
        hidden_state: Optional[torch.Tensor] = None,
        sequence: bool = True,
        return_full_sequence: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, new_hidden_state)``.

        Args:
            tensors: RomBattleState tensor dict (see :class:`RomStudentPolicy`).
            hidden_state: optional ``(gru_layers, B, gru_hidden)`` tensor; if
                None the GRU starts from a zero state.
            sequence: interpret leading dims as time steps (``T``/``B, T``)
                when True, or as a single step (``()``/``B``) when False.
            return_full_sequence: overrides the constructor setting; when True
                logits are returned for every step, otherwise only for the
                last step of each sequence.
        Returns:
            logits and the updated hidden state.
        """
        features, legal, leading = self.encoder(tensors, sequence=sequence)

        if sequence:
            if len(leading) == 2:
                batch, seq_len = leading
            else:
                batch, seq_len = 1, leading[0]
        else:
            batch = leading[0] if len(leading) == 1 else 1
            seq_len = 1

        feat = features.reshape(batch, seq_len, self.encoder.feature_dim)
        legal = legal.reshape(batch, seq_len, self.num_actions)

        gru_out, new_hidden = self.gru(feat, hidden_state)  # (B, T, gru_hidden)
        logits = self.head(gru_out)  # (B, T, 9)

        logits = logits.masked_fill(legal == 0, float("-inf"))

        rfs = (
            self.return_full_sequence
            if return_full_sequence is None
            else return_full_sequence
        )

        # ---- restore the user-facing shape --------------------------------
        if rfs:
            if sequence:
                if len(leading) == 2:
                    out = logits  # (B, T, 9)
                else:
                    out = logits[0]  # (T, 9)
            else:
                if len(leading) == 1:
                    out = logits[:, 0]  # (B, 9)
                else:
                    out = logits[0, 0]  # (9,)
        else:
            last = logits[:, -1]  # (B, 9)
            if len(leading) == 0 or (sequence and len(leading) == 1):
                out = last[0]  # (9,) unbatched
            else:
                out = last  # (B, 9)
        return out, new_hidden

    def count_parameters(self, trainable_only: bool = False) -> int:
        params = [p for p in self.parameters() if p.requires_grad or not trainable_only]
        return int(sum(p.numel() for p in params))


def build_model(
    preset: Optional[str] = "small",
    recurrent: bool = False,
    **kwargs,
):
    """Convenience factory: build a :class:`RomStudentPolicy` or
    :class:`RomStudentGRUPolicy` (default preset: ``"small"``)."""
    cls = RomStudentGRUPolicy if recurrent else RomStudentPolicy
    return cls(preset=preset, **kwargs)
