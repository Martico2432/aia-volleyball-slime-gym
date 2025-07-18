from typing import Dict, Any, List
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition

class SlimeActions(ActionParser[int, np.ndarray, np.ndarray, VolleyballState, int]):
    """Defines the action space and parsing with fixed side inversion"""
    def get_action_space(self, agent: int) -> int:
        # 3D target + jump flag
        return 4, 'continuous'

    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        # Determine starting side for each agent
        self.started_on_right = {
            agent: (initial_state.slimes[agent].position[0] > 0)
            for agent in agents
        }

    def parse_actions(self, actions: Dict[int, np.ndarray], state: VolleyballState, shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        real_actions: Dict[int, np.ndarray] = {}
        for agent_id, action in actions.items():
            proc = action.astype(np.float32)
            # Scale X, Y, Z targets to [-10, 10]
            proc[:3] *= 10.0

            # Mirror horizontal axes if started on right
            if self.started_on_right.get(agent_id, False):
                proc[0] *= -1.0
                proc[2] *= -1.0

            real_actions[agent_id] = proc

        shared_info['actions'] = real_actions
        return real_actions
