from slime_api.common_values import *
from slime_api.slimestate import VolleyballState, Slime
import numpy as np
from typing import Dict

GRAVITY = np.array(GRAVITY, dtype=np.float32) # Setting it to f32



class SlimeVolleyballSim:
    def __init__(
        self,
        initial_state: VolleyballState,
        max_speed: float = 4.0,
        acceleration: float = 4.0,
        jump_force: float = 3.0
    ):
        # Make sure all of this stuff is f32, if not we crash sometimes
        for slime in initial_state.slimes.values():
            slime.position = slime.position.astype(np.float32)
            slime.velocity = slime.velocity.astype(np.float32)
            slime.touches_remaining = float(slime.touches_remaining)
            slime.jump_cooldown = float(slime.jump_cooldown)
        initial_state.ball_position = initial_state.ball_position.astype(np.float32)
        initial_state.ball_velocity = initial_state.ball_velocity.astype(np.float32)

        self.state = initial_state
        self.max_speed = np.float32(max_speed)
        self.acceleration = np.float32(acceleration)
        self.jump_force = np.float32(jump_force)

    def get_state(self) -> VolleyballState:
        return self.state # Self explanatory

    def set_state(self, goal_state: VolleyballState):
        self.__init__(goal_state, float(self.max_speed), float(self.acceleration), float(self.jump_force))
        # Yeah, we can use init here, and we don't work as hard lol

    def step_game(self, actions: Dict[int, np.ndarray], ticks: int = 1) -> VolleyballState:
        for _ in range(ticks):
            if self.state.point_scored:
                break
            # Reset stuff
            self.state.point_scored = False
            self.state.scoring_slime = None

            # SLIMES
            for sid, slime in self.state.slimes.items():
                act = actions.get(sid, np.zeros(4, dtype=np.float32))
                target = act[:3].astype(np.float32)
                jump_req = bool(act[3])
                is_grounded = slime.position[1] <= np.float32(SLIME_RADIUS)
                # Jump
                if jump_req and is_grounded and slime.jump_cooldown <= 0.0:
                    slime.velocity[1] = self.jump_force
                    slime.jump_cooldown = np.float32(JUMP_COOLDOWN_SECONDS)
                slime.jump_cooldown = max(np.float32(0.0), slime.jump_cooldown - DT)
                # Move toward the otuput, so the target
                dir_vec = (target - slime.position).astype(np.float32)
                dir_vec[1] = 0.0
                dist = np.linalg.norm(dir_vec)
                if dist > np.float32(STOPPING_DISTANCE):
                    desired_vel = (dir_vec / dist) * self.max_speed
                    vel_xz = slime.velocity[[0,2]]
                    dv = desired_vel[[0,2]] - vel_xz
                    max_d = self.acceleration * DT
                    dv_clipped = np.clip(dv, -max_d, max_d)
                    slime.velocity[0] += dv_clipped[0]
                    slime.velocity[2] += dv_clipped[1]
                # Gravity & move
                slime.velocity += GRAVITY * DT
                new_pos = slime.position + slime.velocity * DT
                # We don't want to fade through the floor, and good idea for a gamemode
                if new_pos[1] < SLIME_CENTER_ON_FLOOR:
                    new_pos[1] = SLIME_CENTER_ON_FLOOR
                    slime.velocity[1] = max(np.float32(0.0), slime.velocity[1])
                # Net goes here LOL
                # Slime can't cross
                if new_pos[1] < SLIME_BLOCKER_HEIGHT_FLT:
                    if slime.position[0] < NET_PLANE_X and new_pos[0] + SLIME_RADIUS > NET_PLANE_X - NET_HALF_THICKNESS:
                        new_pos[0] = NET_PLANE_X - NET_HALF_THICKNESS - SLIME_RADIUS
                        slime.velocity[0] = 0.0
                    if slime.position[0] > NET_PLANE_X and new_pos[0] - SLIME_RADIUS < NET_PLANE_X + NET_HALF_THICKNESS:
                        new_pos[0] = NET_PLANE_X + NET_HALF_THICKNESS + SLIME_RADIUS
                        slime.velocity[0] = 0.0
                slime.position = new_pos
                # Ball collision
                rel = self.state.ball_position - slime.position
                db = np.linalg.norm(rel)
                if db <= SLIME_RADIUS + BALL_RADIUS:
                    n = rel / (db + np.float32(1e-6))
                    v = self.state.ball_velocity
                    self.state.ball_velocity = (v - 2 * np.dot(v,n)*n).astype(np.float32)
                    if slime.touches_remaining > 0:
                        slime.touches_remaining -= 1
                    self.state.scoring_slime = sid

            # BALL
            # Net collision for ball
            # if crossing net and below net height
            prev_x = self.state.ball_position[0]
            self.state.ball_velocity += GRAVITY * DT
            self.state.ball_position += self.state.ball_velocity * DT
            # Ball vs net
            bx = self.state.ball_position[0]
            by = self.state.ball_position[1]
            if abs(bx - NET_PLANE_X) < (BALL_RADIUS + NET_HALF_THICKNESS) and by <= NET_HEIGHT_FLT:
                # reflect X velocity
                self.state.ball_velocity[0] *= -BALL_RESTITUTION
                # reposition in walls
                if prev_x < NET_PLANE_X:
                    self.state.ball_position[0] = NET_PLANE_X - NET_HALF_THICKNESS - BALL_RADIUS
                else:
                    self.state.ball_position[0] = NET_PLANE_X + NET_HALF_THICKNESS + BALL_RADIUS

            # Floor collision -> point scored, so we can reset
            if self.state.ball_position[1] <= BALL_RADIUS:
                self.state.ball_position[1] = BALL_RADIUS
                x = self.state.ball_position[0]
                scorer = 1 if x < 0 else 0
                self.state.point_scored = True
                self.state.scoring_slime = scorer

            self.state.steps += 1

        return self.state

    def render(self):
        pass # Yeah, we can render, so if you know, pls do it
