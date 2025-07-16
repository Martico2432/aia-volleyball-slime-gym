from typing import Dict, Any, List
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition

class SlimeActions(ActionParser[int, int, int, VolleyballState, int]):
    """Defines the action space and parsing"""
    def get_action_space(self, agent: int) -> int:
        return 4, 'continuous'  # vec3 and jump
        # return 2
        
    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        pass  # No state to reset
        
    def parse_actions(self, actions: Dict[int, np.ndarray], state: VolleyballState, shared_info: Dict[str, Any]) -> Dict[int, int]:
        # Actions are already in the correct format
        # actions = {agent: [throttle, steer]}
        shared_info['actions'] = actions  # Store actions in shared info if needed
        for agent, action in actions.items():
            slime = state.slimes[agent]
            # if slime pos is < 0, then, we want to invert the first 3 items, so x = -x, and z = -z
            if slime.position[0] < 0:
                action[0] = -action[0]
                action[2] = -action[2]
            
        return actions