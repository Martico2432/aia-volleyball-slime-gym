from slime_api.common_values import *
from slime_api.slimestate import VolleyballState, Slime
import numpy as np
from typing import Dict, Optional

GRAVITY = np.array(GRAVITY, dtype=np.float32) # Setting it to f32

# Don't import pygame unless we're rendering
pygame = None


class SlimeVolleyballSim:
    def __init__(
        self,
        initial_state: VolleyballState,
        max_speed: float = 6.0,
        acceleration: float = 2.0,
        jump_force: float = 2.0,
        render_mode: Optional[str] = None
    ):
        # Make sure all of this stuff is f32, if not we crash sometimes
        for slime in initial_state.slimes.values():
            slime.position = slime.position.astype(np.float32)
            slime.velocity = slime.velocity.astype(np.float32)
            slime.touches_remaining = float(slime.touches_remaining)
            slime.jump_cooldown = float(slime.jump_cooldown)
            slime.touch_cooldown = float(slime.touch_cooldown)
        initial_state.ball_position = initial_state.ball_position.astype(np.float32)
        initial_state.ball_velocity = initial_state.ball_velocity.astype(np.float32)

        self.state = initial_state

        self.max_speed_points = max_speed
        self.acceleration_points = acceleration
        self.jump_force_points = jump_force
        self.max_speed = MAX_SPEED_RANGE[max_speed]
        self.acceleration = ACCELERATION_RANGE[acceleration]
        self.jump_force = JUMP_FORCE_RANGE[jump_force]

        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self.font = None

        if render_mode is not None:
            global pygame
            import pygame

    def get_state(self) -> VolleyballState:
        return self.state # Self explanatory

    def set_state(self, goal_state: VolleyballState):
        self.__init__(goal_state, float(self.max_speed_points), float(self.acceleration_points), float(self.jump_force_points))
        # Yeah, we can use init here, and we don't work as hard lol

    def step_game(self, actions: Dict[int, np.ndarray], ticks: int = 1) -> VolleyballState:
        def move_towards(current, target, max_delta):
            delta = target - current
            dist = np.linalg.norm(delta)
            if dist <= max_delta or dist == 0:
                return target
            return current + delta / dist * max_delta

        def slime_controller(sid: int, slime: Slime):
            act = actions.get(sid, slime.target)
            target = act[:3].astype(np.float32)
            slime.target = target
            jump_req = bool(act[3])
            is_grounded = slime.position[1] <= np.float32(SLIME_CENTER_ON_FLOOR + 0.1)
            vel = slime.velocity.copy()
            pos = slime.position

            if jump_req and is_grounded and slime.jump_cooldown <= 0.0:
                vel[1] = self.jump_force
                slime.jump_cooldown = np.float32(JUMP_COOLDOWN_SECONDS)

            dir_vec = target - pos
            dir_vec[1] = 0.0
            dist = np.linalg.norm(dir_vec)
            if dist <= np.float32(STOPPING_DISTANCE):
                lerp_factor = self.acceleration * DT
                slime.velocity[0] *= (1 - lerp_factor)
                slime.velocity[2] *= (1 - lerp_factor)
                return

            vel_xz = vel[[0,2]]
            mag_sq = np.dot(vel_xz, vel_xz)
            mag = np.sqrt(mag_sq)
            temp_max_speed = self.max_speed
            if mag > 1e-5:
                threshold = 0.5 * mag_sq / self.acceleration + STOPPING_DISTANCE
                if dist <= threshold:
                    temp_max_speed = np.sqrt(2 * self.acceleration * max(dist - STOPPING_DISTANCE, 0))

            target_vel = dir_vec[[0,2]] / dist * temp_max_speed
            new_vel_xz = move_towards(vel_xz, target_vel, self.acceleration * DT)
            slime.velocity[0] = new_vel_xz[0]
            slime.velocity[2] = new_vel_xz[1]
            slime.velocity[1] = vel[1]

        def sphere_collision_mut(
            pos1: np.ndarray,
            vel1: np.ndarray,
            mass1: float,
            radius1: float,
            pos2: np.ndarray,
            vel2: np.ndarray,
            mass2: float,
            radius2: float,
            restitution: float = 0.6
        ) -> None:
            """
            Resolves collision between two 3D spheres (modifies velocities/positions in-place).
            """
            # Vector from sphere1 to sphere2
            delta_pos = pos2 - pos1
            distance = np.linalg.norm(delta_pos)
            min_distance = radius1 + radius2

            if distance > min_distance:
                return False

            # Normalized collision normal
            collision_normal = delta_pos / distance

            # Relative velocity
            relative_vel = vel2 - vel1

            # Velocity along collision normal
            velocity_along_normal = np.dot(relative_vel, collision_normal)

            # Do not resolve if spheres are moving apart
            if velocity_along_normal > 0:
                return False

            # Impulse calculation
            reduced_mass = 1.0 / (1.0 / mass1 + 1.0 / mass2)
            impulse_magnitude = -(1 + restitution) * velocity_along_normal * reduced_mass
            impulse = impulse_magnitude * collision_normal

            # Apply impulse
            vel1 -= impulse / mass1
            vel2 += impulse / mass2

            # Position correction to prevent overlap
            overlap = min_distance - distance
            if overlap > 0:
                total_mass = mass1 + mass2
                pos1 -= (mass2 / total_mass) * overlap * collision_normal
                pos2 += (mass1 / total_mass) * overlap * collision_normal

            return True

        for _ in range(ticks):
            if self.state.point_scored:
                break
            # Reset stuff
            self.state.point_scored = False
            self.state.scoring_slime = None

            # SLIMES
            for sid, slime in self.state.slimes.items():
                slime.jump_cooldown = max(np.float32(0.0), slime.jump_cooldown - DT)
                slime.touch_cooldown = max(np.float32(0.0), slime.touch_cooldown - DT)

                slime_controller(sid, slime)

                # Gravity & move
                slime.velocity += GRAVITY * DT
                new_pos = slime.position + slime.velocity * DT
                # We don't want to fade through the floor, and good idea for a gamemode
                if new_pos[1] < SLIME_CENTER_ON_FLOOR and slime.velocity[1] < 0:
                    new_pos[1] = SLIME_CENTER_ON_FLOOR
                    slime.velocity[1] = max(np.float32(0.0), slime.velocity[1])

                # Slime can't cross
                if new_pos[1] < SLIME_BLOCKER_HEIGHT_FLT:
                    if slime.position[0] < 0 and new_pos[0] + SLIME_RADIUS > -NET_PLANE_X:
                        new_pos[0] = 0 - NET_HALF_THICKNESS - SLIME_RADIUS
                        slime.velocity[0] = 0.0
                    if slime.position[0] > 0 and new_pos[0] - SLIME_RADIUS < NET_PLANE_X:
                        new_pos[0] = NET_PLANE_X + SLIME_RADIUS
                        slime.velocity[0] = 0.0
                slime.position = new_pos

                # Ball collision
                if sphere_collision_mut(slime.position, slime.velocity, SLIME_MASS, SLIME_RADIUS, self.state.ball_position, self.state.ball_velocity, BALL_MASS, BALL_RADIUS):
                    if slime.touch_cooldown <= 0:
                        slime.touch_cooldown = SLIME_TOUCH_COOLDOWN
                        if slime.touches_remaining > 0:
                            slime.touches_remaining -= 1
                        else:
                            self.state.point_scored = True
                            x = self.state.ball_position[0]
                            scorer = 1 if x < 0 else 0
                            self.state.scoring_slime = scorer

            # BALL
            # Net collision for ball
            # if crossing net and below net height
            prev_x = self.state.ball_position[0]
            self.state.ball_velocity += GRAVITY * DT
            self.state.ball_position += self.state.ball_velocity * DT
            # Ball vs net
            bx = self.state.ball_position[0]
            by = self.state.ball_position[1]
            bz = self.state.ball_position[2]
            # Distance from net
            bndx = abs(bx) - (BALL_RADIUS + NET_HALF_THICKNESS)
            bndy = by - NET_HEIGHT_FLT - BALL_RADIUS

            if bndx <= 0:
                for sid, slime in self.state.slimes.items():
                    slime.touches_remaining = 3

            if bndx < 0 and bndy < 0:
                # We assume whichever wall it's closer to is the one it's bouncing off
                # NB: This does not account for hitting the courner
                if bndx >= bndy:
                    # reflect X velocity
                    self.state.ball_velocity[0] *= -BALL_RESTITUTION
                    # reposition in walls
                    if prev_x < 0:
                        self.state.ball_position[0] = NET_HALF_THICKNESS - BALL_RADIUS
                    else:
                        self.state.ball_position[0] = NET_HALF_THICKNESS + BALL_RADIUS
                else:
                    # reflect X velocity
                    self.state.ball_velocity[1] *= -BALL_RESTITUTION
                    # reposition in walls
                    self.state.ball_position[1] = NET_HEIGHT_FLT + BALL_RADIUS
            elif abs(bx) > STAGE_RADIUS[0] - BALL_RADIUS:
                # Back wall collision
                self.state.ball_velocity[0] *= -BALL_RESTITUTION
                self.state.ball_position[0] = np.sign(bx) * (STAGE_RADIUS[0] - BALL_RADIUS)
            if abs(bz) > STAGE_RADIUS[1] - BALL_RADIUS:
                # Side wall collision
                self.state.ball_velocity[2] *= -BALL_RESTITUTION
                self.state.ball_position[2] = np.sign(bz) * (STAGE_RADIUS[1] - BALL_RADIUS)

            # Floor collision -> point scored, so we can reset
            if self.state.ball_position[1] <= 0:
                self.state.ball_position[1] = 0
                x = self.state.ball_position[0]
                scorer = 1 if x < 0 else 0
                self.state.point_scored = True
                self.state.scoring_slime = scorer

            self.state.steps += 1

        return self.state

    def render(self):
        if self.render_mode is None:
            return

        padding = 50

        # Compute aspect ratios for the views based on stage dimensions
        side_aspect = (STAGE_RADIUS[0] * 2) / (NET_HEIGHT_FLT * 8 * 0.8)  # width / height (scaled similarly to to_screen)
        top_aspect = (STAGE_RADIUS[0] * 2) / (STAGE_RADIUS[1] * 2)      # width / height

        # Target width for views (max width to fit window comfortably)
        target_width = 700

        # Compute heights preserving aspect ratios
        side_height = int(target_width / side_aspect)
        top_height = int(target_width / top_aspect)

        # Total window height (two views stacked + padding in between and top/bottom)
        window_width = target_width + padding * 2
        window_height = side_height + top_height + padding * 3  # extra padding for spacing

        if self.window is None and self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((window_width, window_height))
            pygame.display.set_caption("Slime Volleyball - Side and Top Views")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)

        canvas = pygame.Surface((window_width, window_height)) if self.render_mode == "rgb_array" else self.window
        canvas.fill((20, 20, 35))  # Dark blue background

        # Draw view labels centered horizontally
        label_side = self.font.render("Side View (X-Y Plane)", True, (255, 255, 255))
        label_top = self.font.render("Top View (X-Z Plane)", True, (255, 255, 255))
        label_side_pos = ((window_width - label_side.get_width()) // 2, padding // 2)
        label_top_pos = ((window_width - label_top.get_width()) // 2, side_height + padding * 2)

        canvas.blit(label_side, label_side_pos)
        canvas.blit(label_top, label_top_pos)

        # Draw the two views stacked vertically with padding
        self._draw_side_view(canvas, pygame.Rect(padding, padding, target_width, side_height))
        self._draw_top_view(canvas, pygame.Rect(padding, side_height + padding * 2, target_width, top_height))

        if self.render_mode == "human":
            pygame.event.pump()
            self.window.blit(canvas, (0, 0))
            pygame.display.update()
            self.clock.tick(60)
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def _draw_side_view(self, surface, rect):
        # Draw court outline
        pygame.draw.rect(surface, (30, 30, 50), rect)
        pygame.draw.rect(surface, (200, 200, 200), rect, 2)

        # Convert game coords to screen coords
        def to_screen(x, y):
            screen_x = rect.left + (x + STAGE_RADIUS[0]) / (STAGE_RADIUS[0] * 2) * rect.width
            screen_y = rect.bottom - y / (NET_HEIGHT_FLT*3) * rect.height * 0.8
            return (int(screen_x), int(screen_y))

        # Draw floor
        floor_y = to_screen(-STAGE_RADIUS[0], 0)[1]
        pygame.draw.line(
            surface, (80, 180, 80),
            (rect.left, floor_y),
            (rect.right, floor_y),
            3
        )

        # Draw net
        net_top = to_screen(0, NET_HEIGHT_FLT)
        net_bottom = to_screen(0, 0)
        pygame.draw.line(surface, (220, 20, 60), net_bottom, net_top, 4)

        # Draw center line
        pygame.draw.line(
            surface, (100, 100, 150),
            to_screen(0, 0),
            to_screen(0, NET_HEIGHT_FLT*2.5),
            1
        )

        # Draw slimes
        for sid, slime in self.state.slimes.items():
            pos = to_screen(slime.position[0], slime.position[1])
            radius = int(SLIME_RADIUS / (STAGE_RADIUS[0] * 2) * rect.width * 1.5)
            color = (0, 150, 255) if sid == 0 else (50, 255, 100)
            pygame.draw.circle(surface, color, pos, radius)
            pygame.draw.circle(surface, (255, 255, 255), pos, radius, 2)

            # Draw jump status
            if slime.jump_cooldown > 0:
                jump_pos = (pos[0], pos[1] - radius - 10)
                pygame.draw.circle(surface, (255, 215, 0), jump_pos, 5)

            # Draw movement target
            target_radius = radius / 5
            target_pos = to_screen(slime.target[0], slime.target[1])
            pygame.draw.circle(surface, color, target_pos, target_radius)

        # Draw ball
        ball_pos = to_screen(
            self.state.ball_position[0],
            self.state.ball_position[1]
        )
        ball_radius = int(BALL_RADIUS / (STAGE_RADIUS[0] * 2) * rect.width * 2)
        pygame.draw.circle(surface, (255, 255, 255), ball_pos, ball_radius)
        pygame.draw.circle(surface, (200, 30, 30), ball_pos, ball_radius, 2)

    def _draw_top_view(self, surface, rect):
        # Draw court outline
        pygame.draw.rect(surface, (30, 30, 50), rect)
        pygame.draw.rect(surface, (200, 200, 200), rect, 2)

        # Convert game coords to screen coords
        def to_screen(x, z):
            screen_x = rect.left + (x + STAGE_RADIUS[0]) / (STAGE_RADIUS[0] * 2) * rect.width
            screen_z = rect.bottom - (z + STAGE_RADIUS[1]) / (STAGE_RADIUS[1] * 2) * rect.height
            return (int(screen_x), int(screen_z))

        # Draw net
        net_left = to_screen(-NET_HALF_THICKNESS, -STAGE_RADIUS[1])
        net_right = to_screen(NET_HALF_THICKNESS, STAGE_RADIUS[1])
        pygame.draw.rect(
            surface, (220, 20, 60),
            (net_left[0], rect.top, net_right[0]-net_left[0], rect.height)
        )

        # Draw center line
        center_top = to_screen(0, -STAGE_RADIUS[1])
        center_bottom = to_screen(0, STAGE_RADIUS[1])
        pygame.draw.line(
            surface, (100, 100, 150), center_top, center_bottom, 1
        )

        # Draw slimes
        for sid, slime in self.state.slimes.items():
            pos = to_screen(slime.position[0], slime.position[2])
            radius = int(SLIME_RADIUS / (STAGE_RADIUS[0] * 2) * rect.width * 1.2)
            color = (0, 150, 255) if sid == 0 else (50, 255, 100)
            pygame.draw.circle(surface, color, pos, radius)
            pygame.draw.circle(surface, (255, 255, 255), pos, radius, 2)

            # Draw movement direction
            if np.linalg.norm(slime.velocity) > 0.1:
                end_pos = (
                    pos[0] + int(slime.velocity[0] * 10),
                    pos[1] - int(slime.velocity[2] * 10)
                )
                pygame.draw.line(surface, (255, 255, 0), pos, end_pos, 2)

            # Draw movement target
            target_radius = SLIME_RADIUS / 3
            target_pos = to_screen(slime.target[0], slime.target[2])
            pygame.draw.circle(surface, color, target_pos, target_radius)

        # Draw ball
        ball_pos = to_screen(
            self.state.ball_position[0],
            self.state.ball_position[2]
        )
        ball_radius = int(BALL_RADIUS / (STAGE_RADIUS[0] * 2) * rect.width * 1.5)
        pygame.draw.circle(surface, (255, 255, 255), ball_pos, ball_radius)
        pygame.draw.circle(surface, (200, 30, 30), ball_pos, ball_radius, 2)

        # Draw velocity vector
        if np.linalg.norm(self.state.ball_velocity) > 0.1:
            end_pos = (
                ball_pos[0] + int(self.state.ball_velocity[0] * 8),
                ball_pos[1] - int(self.state.ball_velocity[2] * 8)
            )
            pygame.draw.line(surface, (255, 100, 100), ball_pos, end_pos, 3)

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
