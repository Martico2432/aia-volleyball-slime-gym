from typing import Dict, Any, List
import numpy as np
from slime_api.slimestate import Slime, VolleyballState
from rlgym.api import ObsBuilder

class IndieDevDefaultObs(ObsBuilder[int, np.ndarray, VolleyballState, np.ndarray]):
    """Converts state into agent observations with fixed side inversion"""
    def get_obs_space(self, agent: int) -> np.ndarray:
        # Adaptable observation size
        return 'real', -1

    def reset(self, agents: List[int], initial_state: VolleyballState, shared_info: Dict[str, Any]) -> None:
        # Determine starting side for each agent
        self.started_on_right = {
            agent: (initial_state.slimes[agent].position[0] > 0)
            for agent in agents
        }
        self._state = initial_state

    def build_obs(self, agents: List[int], state: VolleyballState, shared_info: Dict[str, Any]) -> Dict[int, np.ndarray]:
        observations = {}
        for host_agent in agents:
            # Use fixed flag instead of dynamic position
            invert = self.started_on_right[host_agent]
            obs = np.array([], dtype=np.float32)

            # Ball position and velocity
            if invert:
                ball_pos_x = -state.ball_position[0]
                ball_pos_z = -state.ball_position[2]
                ball_vel_x = -state.ball_velocity[0]
                ball_vel_z = -state.ball_velocity[2]
            else:
                ball_pos_x = state.ball_position[0]
                ball_pos_z = state.ball_position[2]
                ball_vel_x = state.ball_velocity[0]
                ball_vel_z = state.ball_velocity[2]

            obs = np.append(obs, [ball_pos_x,
                                  state.ball_position[1],
                                  ball_pos_z,
                                  ball_vel_x,
                                  state.ball_velocity[1],
                                  ball_vel_z])

            # Self observation
            obs = np.append(obs, self._build_obs_for_agent(host_agent, state, invert))

            # Other agent(s)
            for agent in agents:
                if agent == host_agent:
                    continue
                obs = np.append(obs, self._build_obs_for_agent(agent, state, invert))

            observations[host_agent] = obs
        return observations

    def _build_obs_for_agent(self, agent: int, state: VolleyballState, invert: bool) -> np.ndarray:
        data = []
        pos = state.slimes[agent].position
        vel = state.slimes[agent].velocity

        # X, Z position and velocity
        if invert:
            data.extend([-pos[0], -pos[2], -vel[0], -vel[2]])
        else:
            data.extend([pos[0], pos[2], vel[0], vel[2]])

        # Y velocity, Y position, normalized touches, can_jump
        data.append(vel[1])
        data.append(pos[1])
        data.append(state.slimes[agent].touches_remaining / 3)
        data.append(float(state.slimes[agent].can_jump))

        return np.array(data, dtype=np.float32)
