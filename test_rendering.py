
import numpy as np
from slime_api.sim.main_sim import SlimeVolleyballSim
from slime_api.slimemutator import IndieDevMutator
from slime_api.slimestate import VolleyballState, Slime
from time import sleep
from slime_api.common_values import *

GRAVITY = np.float32(GRAVITY[1])

mutator = IndieDevMutator()
arena = None


def predict_position(position, velocity, target_height=1.0):
    pos = np.array(position, dtype=np.float32)
    vel = np.array(velocity, dtype=np.float32)

    a = 0.5 * GRAVITY
    b = vel[1]
    c = pos[1] - target_height

    disc = b * b - 4 * a * c
    if disc < 0:
        return 0, pos

    t = (-b - np.sqrt(disc)) / (2 * a)
    if t <= 0:
        return 0, pos

    disp = vel * t
    disp[1] += 0.5 * GRAVITY * t**2

    return t, (pos + disp).astype(np.float32)


action_left = np.zeros(4, np.float32)
action_right = np.zeros(4, np.float32)
action_right[1] = 1
for i in range(1000):
    if arena is None:
        state = VolleyballState(
            {0: Slime(
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                3, False, 0, 0
            ), 1: Slime(
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                0, False, 0, 0
            )},
            np.zeros(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            False,
            None
        )
        mutator.apply(state, None)
        arena = SlimeVolleyballSim(state, 6, 4, 0, render_mode="human")

    landing_time, landing_position = predict_position(state.ball_position, state.ball_velocity)
    landing_position[1] = 0
    action_left[:-1] = landing_position
    action_right[:-1] = landing_position

    if np.sign(action_left[0]) != np.sign(state.slimes[0].position[0]):
        action_left[0] = -2.5 * np.sign(action_left[0])
        action_left[3] = 0
    else:
        action_left[2] -= 0.3
        action_left[0] += np.sign(action_left[0]) * 0.2
        action_left[3] = 1 if landing_time < 0.1 else 0


    if np.sign(action_right[0]) != np.sign(state.slimes[1].position[0]):
        action_right[0] = -2.5 * np.sign(action_right[0])
        action_right[3] = 0
    else:
        action_right[0] += np.sign(action_right[0]) * 0.2
        action_right[3] = 1 if landing_time < 0.1 else 0

    arena.step_game({0: action_left, 1: action_right})
    arena.render()

    if arena.state.point_scored:
        arena = None