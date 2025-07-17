from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class Slime:
    position: np.ndarray  # Position in 3D space (x, y, z)
    velocity: np.ndarray
    target: np.ndarray
    touches_remaining: float
    can_jump: bool
    jump_cooldown: float
    touch_cooldown: float

@dataclass
class VolleyballState:
    slimes: Dict[int, Slime]
    ball_position: np.ndarray
    ball_velocity: np.ndarray
    point_scored: bool
    scoring_slime: Optional[int]

    steps: int = 0

    def __repr__(self):
        return f"VolleyballState(slimes={self.slimes}, ball_position={self.ball_position}, ball_velocity={self.ball_velocity})"

