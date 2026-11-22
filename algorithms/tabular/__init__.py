"""Tabular reinforcement-learning algorithms."""
from .td_control import q_learning, sarsa, epsilon_greedy

__all__ = ["q_learning", "sarsa", "epsilon_greedy"]
