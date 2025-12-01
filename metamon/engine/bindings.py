"""
Minimal ctypes bindings for pkmn/engine Gen 1 battle simulator.

This module provides a thin wrapper around the libpkmn C library, treating
battle state as opaque and exposing only essential functions.
"""

import ctypes
import os
from pathlib import Path
from typing import List, Tuple

# Path to the engine library
ENGINE_LIB_PATH = Path(__file__).parent.parent.parent.parent / "engine" / "zig-out" / "lib" / "libpkmn-showdown.a"

# Load the library
# Note: For static libraries (.a), we need to link against them during compilation
# For now, we'll create a simple shared library wrapper
_lib = None

def _load_library():
    """Load or compile the engine library for Python access."""
    global _lib
    if _lib is not None:
        return _lib

    # Check if we have a .so version
    so_path = ENGINE_LIB_PATH.parent / "libpkmn-showdown.so"
    if not so_path.exists():
        # Need to create a shared library from the static one
        # For MVP, we'll use cffi or create a simple wrapper
        raise RuntimeError(
            f"Shared library not found at {so_path}. "
            f"Please build with: zig build -Dshowdown -Ddynamic -Doptimize=ReleaseFast"
        )

    _lib = ctypes.CDLL(str(so_path))
    _setup_function_signatures(_lib)
    return _lib


def _setup_function_signatures(lib):
    """Define C function signatures for type safety."""

    # Choice type: uint8_t
    lib.pkmn_choice_init.argtypes = [ctypes.c_uint8, ctypes.c_uint8]
    lib.pkmn_choice_init.restype = ctypes.c_uint8

    lib.pkmn_choice_type.argtypes = [ctypes.c_uint8]
    lib.pkmn_choice_type.restype = ctypes.c_uint8

    lib.pkmn_choice_data.argtypes = [ctypes.c_uint8]
    lib.pkmn_choice_data.restype = ctypes.c_uint8

    # Result functions
    lib.pkmn_result_type.argtypes = [ctypes.c_uint8]
    lib.pkmn_result_type.restype = ctypes.c_uint8

    lib.pkmn_result_p1.argtypes = [ctypes.c_uint8]
    lib.pkmn_result_p1.restype = ctypes.c_uint8

    lib.pkmn_result_p2.argtypes = [ctypes.c_uint8]
    lib.pkmn_result_p2.restype = ctypes.c_uint8

    # Gen 1 battle functions
    # pkmn_result pkmn_gen1_battle_update(battle, c1, c2, options)
    # We treat battle as opaque bytes (ctypes.c_char * 384)
    lib.pkmn_gen1_battle_update.argtypes = [
        ctypes.c_void_p,  # battle pointer
        ctypes.c_uint8,    # choice c1
        ctypes.c_uint8,    # choice c2
        ctypes.c_void_p,   # options (NULL for now)
    ]
    lib.pkmn_gen1_battle_update.restype = ctypes.c_uint8

    # uint8_t pkmn_gen1_battle_choices(battle, player, request, out[], len)
    lib.pkmn_gen1_battle_choices.argtypes = [
        ctypes.c_void_p,  # battle pointer
        ctypes.c_uint8,    # player (0 or 1)
        ctypes.c_uint8,    # request type
        ctypes.c_void_p,   # output array
        ctypes.c_size_t,   # array length
    ]
    lib.pkmn_gen1_battle_choices.restype = ctypes.c_uint8


# Constants from pkmn.h
PKMN_GEN1_BATTLE_SIZE = 384
PKMN_CHOICES_SIZE = 9  # Maximum number of possible choices
PKMN_PLAYER_P1 = 0
PKMN_PLAYER_P2 = 1

# Choice kinds
PKMN_CHOICE_PASS = 0
PKMN_CHOICE_MOVE = 1
PKMN_CHOICE_SWITCH = 2

# Result kinds
PKMN_RESULT_NONE = 0
PKMN_RESULT_WIN = 1
PKMN_RESULT_LOSE = 2
PKMN_RESULT_TIE = 3
PKMN_RESULT_ERROR = 4


class Gen1Battle:
    """
    Minimal wrapper around a Gen 1 battle.

    The battle state is kept as opaque bytes. This class provides
    a simple interface for stepping through battles.
    """

    def __init__(self, battle_bytes: bytes):
        """Initialize from raw battle bytes (384 bytes)."""
        if len(battle_bytes) != PKMN_GEN1_BATTLE_SIZE:
            raise ValueError(f"Battle bytes must be exactly {PKMN_GEN1_BATTLE_SIZE} bytes")

        self._lib = _load_library()
        self._battle = (ctypes.c_uint8 * PKMN_GEN1_BATTLE_SIZE)(*battle_bytes)
        self._done = False
        self._result = None

    def update(self, choice_p1: int, choice_p2: int) -> Tuple[int, int, int]:
        """
        Update the battle with both players' choices.

        Args:
            choice_p1: Player 1's choice (from pkmn_choice_init)
            choice_p2: Player 2's choice

        Returns:
            Tuple of (result_type, request_p1, request_p2)
            - result_type: PKMN_RESULT_* constant
            - request_p1/p2: PKMN_CHOICE_* indicating what type of choice is needed next
        """
        result = self._lib.pkmn_gen1_battle_update(
            ctypes.byref(self._battle),
            choice_p1,
            choice_p2,
            None  # No options for now
        )

        result_type = self._lib.pkmn_result_type(result)
        if result_type != PKMN_RESULT_NONE:
            self._done = True
            self._result = result_type

        request_p1 = self._lib.pkmn_result_p1(result)
        request_p2 = self._lib.pkmn_result_p2(result)

        return result_type, request_p1, request_p2

    def get_legal_choices(self, player: int, request_type: int) -> List[int]:
        """
        Get legal choices for a player.

        Args:
            player: PKMN_PLAYER_P1 or PKMN_PLAYER_P2
            request_type: Type of request (from update result)

        Returns:
            List of legal choice values
        """
        choices_out = (ctypes.c_uint8 * PKMN_CHOICES_SIZE)()
        n_choices = self._lib.pkmn_gen1_battle_choices(
            ctypes.byref(self._battle),
            player,
            request_type,
            choices_out,
            PKMN_CHOICES_SIZE
        )

        return list(choices_out[:n_choices])

    @property
    def done(self) -> bool:
        """Whether the battle has finished."""
        return self._done

    @property
    def result(self) -> int:
        """Battle result (from P1's perspective): WIN, LOSE, TIE, or ERROR."""
        return self._result

    def get_battle_bytes(self) -> bytes:
        """Get current battle state as raw bytes (for debugging)."""
        return bytes(self._battle)


def choice_init(choice_type: int, data: int) -> int:
    """
    Create a choice value.

    Args:
        choice_type: PKMN_CHOICE_MOVE or PKMN_CHOICE_SWITCH
        data: Move slot (1-4) or switch target (2-6)

    Returns:
        Encoded choice value
    """
    lib = _load_library()
    return lib.pkmn_choice_init(choice_type, data)


# Example hardcoded battle from C example
EXAMPLE_BATTLE_BYTES = bytes([
    0x25, 0x01, 0xc4, 0x00, 0xc4, 0x00, 0xbc, 0x00, 0xe4, 0x00, 0x4f, 0x18, 0x0e, 0x30, 0x4b, 0x28,
    0x22, 0x18, 0x25, 0x01, 0x00, 0x01, 0x3a, 0x64, 0x19, 0x01, 0xca, 0x00, 0xb8, 0x00, 0xe4, 0x00,
    0xc6, 0x00, 0x7e, 0x08, 0x53, 0x18, 0xa3, 0x20, 0x44, 0x20, 0x19, 0x01, 0x00, 0x04, 0x88, 0x64,
    0x23, 0x01, 0xc2, 0x00, 0xe4, 0x00, 0xb8, 0x00, 0xc6, 0x00, 0x39, 0x18, 0x3b, 0x08, 0x22, 0x18,
    0x9c, 0x10, 0x23, 0x01, 0x00, 0x07, 0x99, 0x64, 0x11, 0x01, 0xd0, 0x00, 0x9e, 0x00, 0x16, 0x01,
    0xc6, 0x00, 0x55, 0x18, 0x56, 0x20, 0x39, 0x18, 0x45, 0x20, 0x11, 0x01, 0x00, 0x19, 0xbb, 0x64,
    0x07, 0x01, 0xd2, 0x00, 0xa8, 0x00, 0xf2, 0x00, 0x94, 0x00, 0xa2, 0x10, 0x22, 0x18, 0x3b, 0x08,
    0x55, 0x18, 0x07, 0x01, 0x00, 0x13, 0x00, 0x64, 0x1b, 0x01, 0xbc, 0x00, 0xb2, 0x00, 0xd2, 0x00,
    0xa8, 0x00, 0x26, 0x18, 0x62, 0x30, 0x11, 0x38, 0x77, 0x20, 0x1b, 0x01, 0x00, 0x10, 0x20, 0x64,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00, 0x61, 0x01, 0x2a, 0x01, 0x20, 0x01, 0x3e, 0x01,
    0xee, 0x00, 0x22, 0x18, 0x3f, 0x08, 0x3b, 0x08, 0x59, 0x10, 0x61, 0x01, 0x00, 0x80, 0x00, 0x64,
    0xbf, 0x02, 0x6c, 0x00, 0x6c, 0x00, 0xc6, 0x00, 0x34, 0x01, 0x73, 0x20, 0x45, 0x20, 0x87, 0x10,
    0x56, 0x20, 0xbf, 0x02, 0x00, 0x71, 0x00, 0x64, 0x0b, 0x02, 0x3e, 0x01, 0xe4, 0x00, 0x9e, 0x00,
    0xe4, 0x00, 0x22, 0x18, 0x73, 0x20, 0x9c, 0x10, 0x3a, 0x10, 0x0b, 0x02, 0x00, 0x8f, 0x00, 0x64,
    0x89, 0x01, 0x20, 0x01, 0x0c, 0x01, 0xd0, 0x00, 0x5c, 0x01, 0x4f, 0x18, 0x5e, 0x10, 0x99, 0x08,
    0x26, 0x18, 0x89, 0x01, 0x00, 0x67, 0xca, 0x64, 0x43, 0x01, 0xf8, 0x00, 0x0c, 0x01, 0x48, 0x01,
    0x2a, 0x01, 0x69, 0x20, 0x56, 0x20, 0x3b, 0x08, 0x55, 0x18, 0x43, 0x01, 0x00, 0x79, 0xc9, 0x64,
    0x39, 0x01, 0xc6, 0x00, 0xbc, 0x00, 0x52, 0x01, 0x70, 0x01, 0x5e, 0x10, 0x45, 0x20, 0x56, 0x20,
    0x69, 0x20, 0x39, 0x01, 0x00, 0x41, 0xcc, 0x64, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x2e, 0xdb, 0x7d, 0x61, 0xcb, 0xba, 0x0d, 0x1e, 0x7e, 0x9e, 0x00,
])
