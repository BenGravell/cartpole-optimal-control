# Dimensions
DIM_STATE = 4  # number of states
DIM_ACTION = 1  # number of actions

STATE_FIELDS = ["position", "velocity", "angle", "angular_velocity"]
ACTION_FIELDS = ["force"]
ALL_FIELDS = STATE_FIELDS + ACTION_FIELDS

# Animation
WORLD_HALF_WIDTH_METERS = 2.0  # should be larger than position_min and position_max extents TODO add unit test
WORLD_WIDTH_METERS = 2 * WORLD_HALF_WIDTH_METERS
WORLD_WIDTH_PX = 800
WORLD_HEIGHT_PX = 600

WORLD_CENTER_X_PX = WORLD_WIDTH_PX // 2
WORLD_CENTER_Y_PX = WORLD_HEIGHT_PX // 2

WORLD_SCALE = WORLD_WIDTH_PX / WORLD_WIDTH_METERS

POLE_WIDTH_PX = 32
POLE_RADIUS_PX = POLE_WIDTH_PX // 2
CART_WIDTH_PX = 100
CART_HEIGHT_PX = 60
AXLE_RADIUS_PX = int(0.5 * POLE_RADIUS_PX)

# Color palette
# https://coolors.co/palette/154274-1368C5-4EA7BA-BEE9E7-F68B3F-FFFFFF
# Generated using http://colormind.io/ and fixing the 2nd-to-last color as 1368C5,
# the primaryColor in the streamlit theme in .streamlit/config.toml
DARK_BLUE = [21, 66, 116]  # 154274
MEDIUM_BLUE = [19, 104, 197]  # 1368C5
LIGHT_BLUE = [78, 167, 186]  # 4EA7BA
PALE_BLUE = [190, 233, 231]  # BEE9E7
ORANGE = [246, 139, 63]  # F68B3F
WHITE = [255, 255, 255]  # FFFFFF

# Element colors
BACKGROUND_COLOR = WHITE
BORDER_COLOR = DARK_BLUE
CART_COLOR = MEDIUM_BLUE
POLE_COLOR = LIGHT_BLUE
AXLE_COLOR = PALE_BLUE
FORCE_COLOR = ORANGE
CONSTRAINT_COLOR = DARK_BLUE
