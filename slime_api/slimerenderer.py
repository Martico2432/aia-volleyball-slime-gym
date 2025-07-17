from typing import Dict, Any
import numpy as np
from rlgym.api import Renderer
from slime_api.slimestate import VolleyballState
import pygame
from slime_api.common_values import *



class SlimeRenderer(Renderer[VolleyballState]):
    """A simple renderer that shows the game."""
    def __init__(self, render_mode: str):
        # Render stuff, we can use from shared_info actions, so shared_info.get("actions") and it returns a 2 item dict, with the slime id, and also the 4 actions
        self.padding = 50
        self.render_mode = render_mode

        # Compute aspect ratios for the views based on stage dimensions
        self.side_aspect = (STAGE_RADIUS[0] * 2) / (NET_HEIGHT_FLT * 8 * 0.8)  # width / height (scaled similarly to to_screen)
        self.top_aspect = (STAGE_RADIUS[0] * 2) / (STAGE_RADIUS[1] * 2)      # width / height

        # Target width for views (max width to fit window comfortably)
        self.target_width = 700

        # Compute heights preserving aspect ratios
        self.side_height = int(self.target_width / self.side_aspect)
        self.top_height = int(self.target_width / self.top_aspect)

        # Total window height (two views stacked + padding in between and top/bottom)
        self.window_width = self.target_width + self.padding * 2
        self.window_height = self.side_height + self.top_height + self.padding * 3  # extra padding for spacing

        if self.render_mode == "human":
            pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            pygame.display.set_caption("Slime Volleyball - Side and Top Views")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.SysFont("Arial", 24)

    def render(self, state: VolleyballState, shared_info: Dict[str, Any]) -> Any:
        self.state = state
        

        canvas = pygame.Surface((self.window_width, self.window_height)) if self.render_mode == "rgb_array" else self.window
        canvas.fill((20, 20, 35))  # Dark blue background

        # Draw view labels centered horizontally
        label_side = self.font.render("Side View (X-Y Plane)", True, (255, 255, 255))
        label_top = self.font.render("Top View (X-Z Plane)", True, (255, 255, 255))
        label_side_pos = ((self.window_width - label_side.get_width()) // 2, self.padding // 2)
        label_top_pos = ((self.window_width - label_top.get_width()) // 2, self.side_height + self.padding * 2)

        canvas.blit(label_side, label_side_pos)
        canvas.blit(label_top, label_top_pos)

        # Draw the two views stacked vertically with padding
        self._draw_side_view(canvas, pygame.Rect(self.padding, self.padding, self.target_width, self.side_height))
        self._draw_top_view(canvas, pygame.Rect(self.padding, self.side_height + self.padding * 2, self.target_width, self.top_height))

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
        """Called when the environment is closed."""
        pass