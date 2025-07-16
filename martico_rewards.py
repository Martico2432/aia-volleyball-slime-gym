from typing import List, Dict, Any
import numpy as np
from rlgym.api import RewardFunction
from slime_api.slimestate import Slime, VolleyballState


class PointRward(RewardFunction[int, VolleyballState, float]):
    """
    A RewardFunction that gives a reward of 1 if the agent's team scored a goal, -1 if the opposing team scored a goal,
    """

    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: VolleyballState, is_terminated: Dict[int, bool],
                    is_truncated: Dict[int, bool], shared_info: Dict[str, Any]) -> Dict[int, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent_id: int, state: VolleyballState) -> float:
        if state.point_scored:
            return 1 if state.scoring_slime == agent_id else -1
        else:
            return 0
        

class TouchesReward(RewardFunction[int, VolleyballState, float]):
    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: VolleyballState, is_terminated: Dict[int, bool],
                    is_truncated: Dict[int, bool], shared_info: Dict[str, Any]) -> Dict[int, float]:
        rewards = {}
        for agent in agents:
            rewards[agent] = 0
        for slime_id in state.slimes:
            slime = state.slimes[slime_id]
            rewards[slime_id] = (0.333334 * slime.touches_remaining)

            if slime.touches_remaining <= 0:
                rewards[slime_id] = -1

        return rewards
            

class BallDistanceReward(RewardFunction[int, VolleyballState, float]):

    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[int], state: VolleyballState, is_terminated: Dict[int, bool],
                    is_truncated: Dict[int, bool], shared_info: Dict[str, Any]) -> Dict[int, float]:
        rewards = {}
        for agent in agents:
            rewards[agent] = 0
        for slime_id in state.slimes:
            slime = state.slimes[slime_id]
            slime_pos = slime.position
            ball_pos = state.ball_position
            distance = np.linalg.norm(slime_pos - ball_pos) / 10
            rewards[slime_id] = -distance # just negate distance

        return rewards