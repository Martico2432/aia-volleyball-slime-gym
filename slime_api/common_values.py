import numpy as np

# Info: collision unstuck in wall is just normal direction
# Info: but on slime it's the same


SLIME_RADIUS = 0.75
SLIME_MASS = 1
SLIME_DRAG = 0
SLIME_CoM = np.array([0, 0.05, 0],  dtype=np.float32)
SLIME_CENTER_ON_FLOOR = -0.045

JUMP_COOLDOWN_SECONDS = 0.1 # In seconds
JUMP_FORCE_BASE = 6
JUMP_FORCE_MAX = 12
JUMP_FORCE_RANGE = np.linspace(6, 12, 11,  dtype=np.float32)

MAX_SPEED_BASE = 4
MAX_SPEED_MAX = 12
MAX_SPEED_RANGE = np.linspace(4, 12, 11, dtype=np.float32)

ACCELERATION_BASE = 50
ACCELERATION_MAX = 90
ACCELERATION_RANGE = np.linspace(50, 90, 11, dtype=np.float32)


JUMP_HEIGHTS = {
    0: 0.9546,
    1: 1.1710,
    2: 1.4082,
    3: 1.6666,
    4: 1.9470,
    5: 2.2482,
    6: 2.5702,
    7: 2.9130,
    8: 3.2782,
    9: 3.6642,
    10: 4.0710
}

STOPPING_DISTANCE = 0.01

BALL_RADIUS = 0.3
BALL_MASS = 0.25
BALL_DRAG = 0
BALL_CoM = np.array([0, 0, 0], dtype=np.float32)

STAGE_RADIUS = np.array([6.8, 3.5], dtype=np.float32) # It's a rectangle, width and width, so 6.8 is length and 3.5 is width
NET_RADIUS_FOR_SLIME = 0.25 # (TBV)
NET_RADIUS_FOR_BALL = np.array([0.1, 5], dtype=np.float32)
NET_HEIGHT = 0.95

SLIME_BLOCKER_HEIGHT = 30
# Net parameters
NET_PLANE_X = np.float32(0.0)
NET_HALF_THICKNESS = np.float32(NET_RADIUS_FOR_BALL[0])
NET_HEIGHT_FLT = np.float32(NET_HEIGHT)
SLIME_BLOCKER_HEIGHT_FLT = np.float32(SLIME_BLOCKER_HEIGHT)

GRAVITY = np.array([0, -17, 0], dtype=np.float32)

BALL_RESTITUTION = 0.6
BALL_FRICTION = 0

CEILING_HEIGHT = 14.49


DT = np.array([1.0 / 60.0], dtype=np.float32) # 1.0 / 60.0  # 60 fps