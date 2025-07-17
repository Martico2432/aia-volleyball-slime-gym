from typing import Dict, Any, Sequence, Tuple
import numpy as np
from slime_api.slimestate import VolleyballState
from rlgym.api import StateMutator

from slime_api.common_values import *


class WeightedMutator(StateMutator[VolleyballState]):
    """Controls environment reset and state modifications"""
    def __init__(self, mutators: Sequence[StateMutator], weights: Sequence[float]):
        assert len(mutators) == len(weights)
        self.mutators = mutators
        weights = np.array(weights)
        self.probs = weights / weights.sum()

    @staticmethod
    def from_zipped(*mutator_weights: Tuple[StateMutator, float]):
        mutators, weights = zip(*mutator_weights)
        return WeightedMutator(mutators, weights)

    def apply(self, state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        idx = np.random.choice(len(self.mutators), p=self.probs)
        mutator = self.mutators[idx]
        mutator.apply(state, shared_info)

class IndieDevMutator(WeightedMutator):
    def __init__(self):
        super().__init__([DropMutator(), TossMutator(), HardTossMutator()], [0.2, 0.5, 0.3])

class DropMutator(StateMutator[VolleyballState]):
    def __init__(self):
        pass

    def apply(self, state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        # Random agent and target positions
        is_first_slime = True
        for slime_id in state.slimes:
            slime = state.slimes[slime_id]
            if is_first_slime:
                # Normal slime, X value must be positive, and Y will be 1
                slime.position[0] = np.random.uniform(0.26, 6)
                slime.position[1] = 1
                slime.position[2] = np.random.uniform(-3, 3)
                slime.touches_remaining = 3
            else:
                # just the same, excpet X
                slime.position[0] = np.random.uniform(-6, -0.26)
                slime.position[1] = 1
                slime.position[2] = np.random.uniform(-3, 3)
                slime.touches_remaining = 3

            is_first_slime = not is_first_slime

        state.ball_position[0] = np.random.uniform(-3, 3)
        state.ball_position[1] = np.random.uniform(2, 4)
        state.ball_position[2] = np.random.uniform(-1, 1)

        # X should be -5 to -4, and 4 to 5
        if np.random.uniform(0, 1) < 0.5:
            state.ball_position[0] = np.random.uniform(-5, -4)
        else:
            state.ball_position[0] = np.random.uniform(4, 5)
        state.ball_velocity[1] = np.random.uniform(-1, 3)
        state.ball_velocity[2] = np.random.uniform(-2, 2)

class TossMutator(StateMutator[VolleyballState]):
    def __init__(self):
        pass

    def apply(self, state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        # Follows the game's true toss function
        is_first_slime = True
        for slime_id in state.slimes:
            slime = state.slimes[slime_id]
            if is_first_slime:
                slime.position[0] = 3
                slime.position[1] = 0
                slime.position[2] = 0
                slime.touches_remaining = 3
            else:
                slime.position[0] = -3
                slime.position[1] = 0
                slime.position[2] = 0
                slime.touches_remaining = 3

            is_first_slime = not is_first_slime

        state.ball_position[0] = 0
        state.ball_position[1] = 3
        state.ball_position[2] = 0

        state.ball_velocity[0] = np.random.choice([-1, 1])
        state.ball_velocity[1] = 1
        state.ball_velocity[2] = np.random.uniform(-0.5, 0.5)
        state.ball_velocity = state.ball_velocity * 7 / np.linalg.norm(state.ball_velocity)

class HardTossMutator(StateMutator[VolleyballState]):
    def __init__(self):
        pass

    def apply(self, state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        # Tosses the ball from the other side of the net
        is_first_slime = True
        for slime_id in state.slimes:
            slime = state.slimes[slime_id]
            if is_first_slime:
                slime.position[0] = np.random.uniform(0.26, 6)
                slime.position[1] = 1
                slime.position[2] = np.random.uniform(-3, 3)
                slime.touches_remaining = 3
            else:
                slime.position[0] = np.random.uniform(-6, -0.26)
                slime.position[1] = 1
                slime.position[2] = np.random.uniform(-3, 3)
                slime.touches_remaining = 3

            is_first_slime = not is_first_slime

        state.ball_velocity[0] = np.random.choice([-1, 1])
        state.ball_velocity[1] = np.random.uniform(1, 4)
        state.ball_velocity[2] = np.random.uniform(-0.5, 0.5)

        state.ball_position[0] = np.random.uniform(1, 5) * -state.ball_velocity[0]
        state.ball_position[1] = 2
        state.ball_position[2] = 0

        state.ball_velocity = state.ball_velocity * np.random.uniform(7, 24) / np.linalg.norm(state.ball_velocity)

