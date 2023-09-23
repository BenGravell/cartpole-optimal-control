import io

import numpy as np

import pygame
from pygame import gfxdraw
from PIL import Image

import streamlit as st

import constants
import app_options as ao


# Initialize pygame
pygame.init()


def get_pygame_mono_font():
    """Load a monospace font."""
    return pygame.font.Font("fonts/SpaceMono/SpaceMono-Regular.ttf", 16)


def pillgon(length, width, num_points_per_arc=10):
    """Coordinates for a pill-shaped polygon that combines arc endcaps with straight edges."""
    angles_top_arc = np.linspace(0, np.pi, num_points_per_arc)
    angles_bot_arc = np.linspace(np.pi, 2 * np.pi, num_points_per_arc)

    r = width / 2

    x_top_arc = r * np.cos(angles_top_arc)
    y_top_arc = r * np.sin(angles_top_arc) + length
    coords_top_arc = [(x, y) for x, y in zip(x_top_arc, y_top_arc)]

    x_bot_arc = r * np.cos(angles_bot_arc)
    y_bot_arc = r * np.sin(angles_bot_arc)
    coords_bot_arc = [(x, y) for x, y in zip(x_bot_arc, y_bot_arc)]

    return coords_top_arc + coords_bot_arc


def arrowgon(a=0.1, b=0.7, c=0.1, d=0.3, scale=1.0):
    """Coordinates for an arrow-shaped polygon that points to the left."""
    sa = scale * a
    sb = scale * b
    sc = scale * c
    sd = scale * d
    coords = [
        (0.0, 0.0),
        (sd, sa + sc),
        (sd, sa),
        (sb + sd, sa),
        (sb + sd, -sa),
        (sd, -sa),
        (sd, -(sa + sc)),
    ]
    return coords


def draw_cartpole(surface, state, action, model_parameter_options: ao.ModelParameterOptions, ghost=False):
    if state is None:
        return None

    POLE_LENGTH_PX = constants.WORLD_SCALE * model_parameter_options.length_pole

    # Add opacity as needed
    cart_color = constants.CART_COLOR
    pole_color = constants.POLE_COLOR
    axle_color = constants.AXLE_COLOR
    force_color = constants.FORCE_COLOR
    if ghost:
        opacity = 127  # Transparent
        cart_color = cart_color + [opacity]
        pole_color = pole_color + [opacity]
        axle_color = axle_color + [opacity]
        force_color = force_color + [opacity]

    # Draw cart
    left = -constants.CART_WIDTH_PX // 2
    right = constants.CART_WIDTH_PX // 2
    top = constants.CART_HEIGHT_PX // 2
    bottom = -constants.CART_HEIGHT_PX // 2

    cart_x = int(state[0] * constants.WORLD_SCALE + constants.WORLD_CENTER_X_PX)  # MIDDLE OF CART
    cart_y = constants.WORLD_CENTER_Y_PX  # MIDDLE OF CART
    cart_coords = [(left, bottom), (left, top), (right, top), (right, bottom)]
    cart_coords = [(c[0] + cart_x, c[1] + cart_y) for c in cart_coords]
    gfxdraw.aapolygon(surface, cart_coords, cart_color)

    if not ghost:
        gfxdraw.filled_polygon(surface, cart_coords, cart_color)

    # Draw pole
    pole_coords_base = pillgon(POLE_LENGTH_PX, constants.POLE_WIDTH_PX)
    pole_coords = []
    for coord in pole_coords_base:
        coord = pygame.math.Vector2(coord).rotate_rad(-state[2])
        coord = (coord[0] + cart_x, coord[1] + cart_y)
        pole_coords.append(coord)
    gfxdraw.aapolygon(surface, pole_coords, pole_color)
    if not ghost:
        gfxdraw.filled_polygon(surface, pole_coords, pole_color)

    # Draw axle
    axle_x = cart_x
    axle_y = cart_y
    axle_radius = constants.AXLE_RADIUS_PX
    gfxdraw.aacircle(
        surface,
        axle_x,
        axle_y,
        axle_radius,
        axle_color,
    )
    if not ghost:
        gfxdraw.filled_circle(
            surface,
            axle_x,
            axle_y,
            axle_radius,
            axle_color,
        )

    # Draw force
    if action is not None and not np.isnan(action):
        force = action
        force_coords_base = arrowgon(scale=-0.1 * (np.sign(force)) * (np.abs(force) ** 0.5) * constants.WORLD_SCALE)
        force_coords = []
        for coord in force_coords_base:
            coord = pygame.math.Vector2(coord)
            force_xshift = -np.sign(force) * ((constants.CART_WIDTH_PX / 2) + 10)
            coord = (coord[0] + cart_x + force_xshift, coord[1] + cart_y)
            force_coords.append(coord)
        gfxdraw.aapolygon(surface, force_coords, force_color)
        if not ghost:
            gfxdraw.filled_polygon(surface, force_coords, force_color)
    return


def pretty_str_float_field(field, value, unit):
    value_round = round(value, 2)
    if abs(value_round) < 0.01:
        value_round = 0.0
    return f"{field:>16s}: {value_round:6.2f} {unit}"


def draw_scene(surface, time, state, action, app_options: ao.AppOptions):
    cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
    (force,) = action

    if app_options.animation_options.show_border:
        # Draw the border
        # Top border
        pygame.gfxdraw.line(surface, 0, 0, constants.WORLD_WIDTH_PX - 1, 0, constants.BORDER_COLOR)
        # Bottom border
        pygame.gfxdraw.line(
            surface,
            0,
            constants.WORLD_HEIGHT_PX - 1,
            constants.WORLD_WIDTH_PX - 1,
            constants.WORLD_HEIGHT_PX - 1,
            constants.BORDER_COLOR,
        )
        # Left border
        pygame.gfxdraw.line(surface, 0, 0, 0, constants.WORLD_HEIGHT_PX - 1, constants.BORDER_COLOR)
        # Right border
        pygame.gfxdraw.line(
            surface,
            constants.WORLD_WIDTH_PX - 1,
            0,
            constants.WORLD_WIDTH_PX - 1,
            constants.WORLD_HEIGHT_PX - 1,
            constants.BORDER_COLOR,
        )

    pos_min_screen_coords = int(
        app_options.constraint_options.position.min * constants.WORLD_SCALE
        + constants.WORLD_CENTER_X_PX
        - constants.CART_WIDTH_PX / 2
    )
    pos_max_screen_coords = int(
        app_options.constraint_options.position.max * constants.WORLD_SCALE
        + constants.WORLD_CENTER_X_PX
        + constants.CART_WIDTH_PX / 2
    )

    cart_y = constants.WORLD_CENTER_Y_PX  # MIDDLE OF CART
    cart_top = cart_y - int(constants.CART_HEIGHT_PX / 2)
    cart_bot = cart_y + int(constants.CART_HEIGHT_PX / 2)

    if app_options.animation_options.show_constraint_box:
        # Draw constraint box
        gfxdraw.hline(
            surface, pos_min_screen_coords, pos_max_screen_coords, cart_top, constants.CONSTRAINT_COLOR + [127]
        )
        gfxdraw.hline(
            surface, pos_min_screen_coords, pos_max_screen_coords, cart_bot, constants.CONSTRAINT_COLOR + [127]
        )

        gfxdraw.vline(surface, pos_min_screen_coords, cart_top, cart_bot, constants.CONSTRAINT_COLOR + [127])
        gfxdraw.vline(surface, pos_max_screen_coords, cart_top, cart_bot, constants.CONSTRAINT_COLOR + [127])

    if app_options.animation_options.show_terminal_state:
        # Draw the ghosted terminal state
        draw_cartpole(
            surface,
            app_options.terminal_state.numpy,
            action=None,
            model_parameter_options=app_options.model_parameter_options,
            ghost=True,
        )

    # Draw the actual cartpole
    action_for_draw = action if app_options.animation_options.show_force else None
    draw_cartpole(
        surface,
        state,
        action_for_draw,
        model_parameter_options=app_options.model_parameter_options,
        ghost=False,
    )

    # Render the text overlay
    if app_options.animation_options.show_text_overlay:
        fields = [
            "Time",
            "Position",
            "Velocity",
            "Angle",
            "Angular Velocity",
            "Force",
        ]
        values = [
            time,
            cart_position,
            cart_velocity,
            pole_angle,
            pole_angular_velocity,
            force,
        ]
        units = [
            "s",
            "m",
            "m/s",
            "rad",
            "rad/s",
            "N",
        ]
        text_overlay_strs = [
            pretty_str_float_field(field, value, unit) for field, value, unit in zip(fields, values, units)
        ]
        font = get_pygame_mono_font()
        for i, s in enumerate(text_overlay_strs):
            label = font.render(s, 1, (0, 0, 0))
            surface.blit(label, (10, 10 + i * 20))


def create_frame(time, state, action, app_options: ao.AppOptions):
    surface = pygame.Surface((constants.WORLD_WIDTH_PX, constants.WORLD_HEIGHT_PX))
    surface.fill(constants.BACKGROUND_COLOR)
    draw_scene(surface, time, state, action, app_options)

    size = surface.get_size()
    data = pygame.image.tobytes(surface, "RGBA")
    return Image.frombytes("RGBA", size, data)


# NOTE: Although we are using app_options in the function signature,
# which would ordinarily possibly result in cache misses when unused options are changed,
# most of the time it will not matter since we use ani_state_action_time_series as well,
# which changes almost all the time when any of the non-animation app_options are changed.
@st.cache_data(max_entries=10)
def animate(ani_state_action_time_series, app_options: ao.AppOptions):
    with st.spinner("Creating frames..."):
        frames = [
            create_frame(time, state, action, app_options) for time, state, action in ani_state_action_time_series
        ]
        durations = [app_options.animation_options.duration_single_ms] * len(frames)
    if app_options.animation_options.duration_end_hold_sec > 0:
        # Add a repeat of the last frame
        frames += [frames[-1]]
        durations += [1000 * app_options.animation_options.duration_end_hold_sec]
    with st.spinner("Saving animation..."):
        # Create an in-memory byte stream
        byte_stream = io.BytesIO()
        frames[0].save(byte_stream, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=durations)
        # Go to the start of the byte stream
        byte_stream.seek(0)
    return byte_stream
