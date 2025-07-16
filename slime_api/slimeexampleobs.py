from typing import Dict, Any, List
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import TransitionEngine, StateMutator, ObsBuilder, ActionParser, RewardFunction, DoneCondition

class IndieDevDefaultObs(ObsBuilder[int, np.ndarray, VolleyballState, np.ndarray]):
    """Converts state into agent observations"""
        
    def get_obs_space(self, agent: int) -> np.ndarray:
        self._state = None
        return 'real', -1 # This means it will adapt :D, the -1, not the real, idk what that does :D
        
        
    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        self._state = initial_state
        
    def build_obs(self, agents: List[int], state: VolleyballState, shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        # Build observation for each agent
        if self._state is None:
            self._state = state
        
        observations = {}
        for host_agent in agents:
            obs = np.array([])
            # Add ball position and velocity, relative to us
            invert = state.slimes[host_agent].position[0] < 0
            if invert:
                ball_pos_x = state.ball_position[0] * -1
                ball_pos_z = state.ball_position[2] * -1
            else:
                ball_pos_x = state.ball_position[0]
                ball_pos_z = state.ball_position[2]

            # Do the same for velocity
            if invert:
                ball_vel_x = state.ball_velocity[0] * -1
                ball_vel_z = state.ball_velocity[2] * -1
            else:
                ball_vel_x = state.ball_velocity[0]
                ball_vel_z = state.ball_velocity[2]

            obs = np.append(obs, ball_pos_x)
            obs = np.append(obs, ball_pos_z)
            obs = np.append(obs, ball_vel_x)
            obs = np.append(obs, ball_vel_z)
            obs = np.append(obs, state.ball_position[1])
            obs = np.append(obs, state.ball_velocity[1])


            for agent in agents:
                # Get obs for every agent
                obs = np.append(obs, self._build_obs_for_agent(agent, state, shared_info, host_agent=host_agent))

            observations[host_agent] = obs 
        return observations
    
    def _build_obs_for_agent(self, agent, state: VolleyballState, shared_info, host_agent: int) -> np.ndarray:

        invert = state.slimes[host_agent].position[0] < 0
        obs = np.array([])
        if invert:
            obs = np.append(obs, state.slimes[agent].position[0] * -1)
            obs = np.append(obs, state.slimes[agent].position[2] * -1)
            obs = np.append(obs, state.slimes[agent].velocity[0] * -1)
            obs = np.append(obs, state.slimes[agent].velocity[2] * -1)
        else:
            obs = np.append(obs, state.slimes[agent].position[0])
            obs = np.append(obs, state.slimes[agent].position[2])
            obs = np.append(obs, state.slimes[agent].velocity[0])
            obs = np.append(obs, state.slimes[agent].velocity[2])


        obs = np.append(obs, state.slimes[agent].velocity[1])
        obs = np.append(obs, state.slimes[agent].position[1])

        obs = np.append(obs, (state.slimes[agent].touches_remaining)/3) # We didvide by 3, to  noramlize it
        obs = np.append(obs, state.slimes[agent].can_jump)
        obs = np.append(obs, state.slimes[agent].jump_cooldown)


        return obs