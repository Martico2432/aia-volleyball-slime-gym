from typing import Dict, Any, List
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition

class IndieDevTerminalCondition(DoneCondition[int, VolleyballState]):
    """Determines when episodes naturally end (reaching the goal)"""
    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        pass
        
    def is_done(self, agents: List, state: VolleyballState, shared_info: Dict[str, Any]) -> bool:

        results = {}
        for agent in agents:
            results[agent] = False

        if state.point_scored:
            for agent in agents:
                results[agent] = True

        for agent in agents:
            if state.slimes[agent].touches_remaining <= 0:
                for agent in agents:
                    results[agent] = True



        return results