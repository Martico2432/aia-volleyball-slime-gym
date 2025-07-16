from typing import Dict, Any
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition

from slime_api.common_values import *

class IndieDevMutator(StateMutator[VolleyballState]):
    """Controls environment reset and state modifications"""

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

            is_first_slime = False

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



