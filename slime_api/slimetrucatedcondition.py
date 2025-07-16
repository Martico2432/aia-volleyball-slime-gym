from typing import Dict, Any, List
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition

class IndieDevTruncatedCondition(DoneCondition[int, VolleyballState]):
    """Determines when episodes are cut short (timeout)"""
    def __init__(self, max_steps: int = 100):
        self.max_steps = max_steps
        
    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        pass
        
    def is_done(self, agents: List, state: VolleyballState, shared_info: Dict[str, Any]) -> bool:
        # Episode is truncated if we exceed max steps
        # Make an empty dictionary to store results
        results = {}
        done = False
        for agent in agents:
            done = state.steps >= self.max_steps

            if done:
                break

        for agent in agents:
            results[agent] = done

        return results  # Return a dictionary indicating if each agent's episode is done