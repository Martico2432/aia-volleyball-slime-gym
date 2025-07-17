from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from dataclasses import dataclass
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition


from slime_api.slimestate import Slime, VolleyballState

from slime_api.sim.main_sim import SlimeVolleyballSim





class IndieDevEngine(TransitionEngine[int, VolleyballState, int]):
    """Handles the core game logic"""
    # def __init__(self, port: int = 5000):
    def __init__(self):
        self._slimes = {}  # These will contain THE slimes from THE SIM
        self._arena = SlimeVolleyballSim(self.create_base_state(), render_mode="human")
        self._state = self._arena.get_state()

    @property
    def agents(self) -> List[int]:
        """
        Returns a list of slimes IDs
        """
        self._slimes = self._arena.get_state().slimes
        return list(self._slimes.keys())

    @property
    def max_num_agents(self) -> int:
        return 2  # This environment only supports 2 agents

    @property
    def state(self) -> VolleyballState:
        self._state = self._arena.get_state()
        return self._state

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @config.setter
    def config(self, value: Dict[str, Any]):
        self._config = value

    def set_state(self, desired_state: VolleyballState, shared_info: Dict[str, Any]) -> VolleyballState:
        self._state = desired_state
        self._arena.set_state(self._state)

        return self._state

    def step(self, actions: Dict[int, int], shared_info: Dict[str, Any]) -> VolleyballState:
        self._state = self._arena.step_game(actions, 6)
        self._state.steps += 6
        #print(f"Engine step with actions: {actions}")
        #print(f"Engine step with state: {self._state}")
        return self._state

    def create_base_state(self) -> VolleyballState:
        # Create a minimal state for the mutator to modify
        return VolleyballState({0: Slime(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 3, False, 0, 0),
                                1: Slime(np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), 3, False, 0, 0)},
                                np.zeros(3, dtype=np.float32),
                                np.zeros(3, dtype=np.float32),
                                False,
                                None)

    def reset(self, initial_state: Optional[VolleyballState] = None) -> None:
        """Reset the engine with an optional initial state"""
        self._state = initial_state if initial_state is not None else self.create_base_state()
        self._arena.set_state(self._state)
        #print(f"Engine reset with state: {self._state}")

    def close(self) -> None:
        pass