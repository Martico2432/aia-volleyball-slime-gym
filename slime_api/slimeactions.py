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
        real_actions = actions
        # Scale first 3 actions by 10, the last one leave it
        real_actions = {agent: [action * 10 if i < 3 else action for i, action in enumerate(actions[agent])] for agent in actions}

        shared_info['actions'] = real_actions  # Store actions in shared info if needed
        for agent, action in actions.items():
            slime = state.slimes[agent]
            # if slime pos is < 0, then, we want to invert the first 3 items, so x = -x, and z = -z
            if slime.position[0] < 0:
                real_actions[agent][0] = -action[0]
                real_actions[agent][2] = -action[2]
            
        return real_actions