"""Gymnasium environment for Balatro.

Phase 3 scope: in-round decisions only (toggle cards, play, discard).
No shop between blinds — the env auto-advances Small -> Big -> Boss
-> next ante. Shop / money / consumables arrive in later phases.
"""
from __future__ import annotations

from balatro_env.gym_env import BalatroEnv

__all__ = ["BalatroEnv"]
