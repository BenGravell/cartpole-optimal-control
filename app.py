from dataclasses import dataclass
from typing import Any
import io

import numpy as np
import pandas as pd
import casadi

import pygame
from pygame import gfxdraw
from PIL import Image

import streamlit as st
import plotly.express as px


@dataclass
class MinMax:
    """Class to hold min and max constraint values."""

    min: Any
    max: Any


st.set_page_config(layout="wide")


# Based on http://underactuated.mit.edu/acrobot.html
def cartpole_dynamics(state, action, params):
    gravity, mass_cart, mass_pole, length_pole = (
        params["gravity"],
        params["mass_cart"],
        params["mass_pole"],
        params["length_pole"],
    )

    # Extract each element of the state with a human-radable alias
    _, x_dot, theta, theta_dot = state[0], state[1], state[2], state[3]
    # Extract each element of the action with a human-radable alias
    f_x = action

    # Compute intermediate quantities
    sin_theta = casadi.sin(theta)
    cos_theta = casadi.cos(theta)

    denom_1 = mass_cart + mass_pole * sin_theta * sin_theta

    denom_x_dot_dot = 1.0 / denom_1
    denom_theta_dot_dot = 1.0 / (length_pole * denom_1)

    numer_x_dot_dot = f_x + mass_pole * sin_theta * (length_pole * theta_dot * theta_dot + gravity * cos_theta)
    numer_theta_dot_dot = (
        -f_x * cos_theta
        - mass_pole * length_pole * theta_dot * theta_dot * cos_theta * sin_theta
        - (mass_cart + mass_pole) * gravity * sin_theta
    )

    # Compute derivatives of state with respect to time
    x_dot_dot = numer_x_dot_dot / denom_x_dot_dot
    theta_dot_dot = numer_theta_dot_dot / denom_theta_dot_dot

    return casadi.vertcat(x_dot, x_dot_dot, theta_dot, theta_dot_dot)


st.title("Cartpole Optimal Control")

with st.expander("Description & Explanation", expanded=False):
    st.header("Summary")
    st.write(
        "This app demonstrates optimal control of a cartpole system. The task is to apply a sequence of inputs to drive"
        " the system from the initial state to the target terminal state while minimizing an objective function."
    )

    st.header("Optimal Control")
    st.write(
        "The optimal control problem is formulated using [CasADi](https://web.casadi.org/) and the direct"
        " multiple-shooting technique to transcribe it to a nonlinear program (NLP), which is solved using the"
        " [IPOPT](https://github.com/coin-or/Ipopt) solver. See https://web.casadi.org/blog/ocp/ for more details."
    )

    st.header("Dynamics")
    st.subheader("Equations of Motion")
    st.write(
        "The equations of motion for the cartpole system are derived using Lagrangian mechanics. The details are found"
        " at http://underactuated.mit.edu/acrobot.html."
    )
    st.subheader("Numerical Integration of the Dynamics Ordinary Differential Equation")
    st.write(
        "The ordinary differential equation (ODE) governing the dynamics of the system are integrated numerically using"
        " the Runge-Kutta 4th-order method (RK4)."
    )

    st.header("Animation")
    st.write(
        "The cartpole is animated as a GIF using PyGame as a lightweight rendering engine. The animation rendering is"
        " an improved version of that found in the [Gymnasium cartpole"
        " environment](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py)."
    )

# Dimensions
n = 4  # number of states
m = 1  # number of actions

position_threshold = 2.0

with st.expander("Options", expanded=False):
    with st.form("options_form"):
        st.header("Options", anchor=False)

        option_cols_row1 = st.columns(3)
        option_cols_row2 = st.columns(3)

        with option_cols_row1[0]:
            st.subheader("Model Parameters", anchor=False)
            gs = st.slider("Gravitational Acceleration (G's)", min_value=0.0, max_value=4.0, value=1.0, step=0.1)
            gravity = 9.81 * gs
            mass_cart = st.slider("Mass of cart (kg)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
            mass_pole = st.slider("Mass of pole (kg)", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
            length_pole = st.slider("Length of pole (m)", min_value=0.4, max_value=1.2, value=1.0, step=0.1)
            model_params = {
                "gravity": gravity,
                "mass_cart": mass_cart,
                "mass_pole": mass_pole,
                "length_pole": length_pole,
            }

        with option_cols_row1[1]:
            st.subheader("Simulation Options", anchor=False)
            T = st.slider("Simulation Duration (seconds)", min_value=1, max_value=20, value=5, step=1)
            sim_fps = st.select_slider("Simulation Frame Rate (frames per second)", options=[5, 10, 25, 50], value=10)
            N = int(sim_fps * T)  # Number of control intervals

        with option_cols_row1[2]:
            st.subheader("Animation Options", anchor=False)
            duration_end_hold_sec = st.slider(
                "Animation Duration Pause at End (sec)", min_value=0, max_value=5, value=2
            )
            ani_fps = st.select_slider("Animation Frame Rate (frames per second)", options=[5, 10, 25, 50], value=25)
            show_animation = st.toggle("Show Animation", value=True)
            show_force = st.toggle("Show Force Arrow", value=True)
            show_target_state = st.toggle("Show Target State Outline", value=True)
            show_guidelines = st.toggle("Show Cart Rail Guideline", value=True)
            show_text_overlay = st.toggle("Show Text Overlay", value=True)
            show_border = st.toggle("Show Border", value=True)

            animation_options = {
                "show_force": show_force,
                "show_target_state": show_target_state,
                "show_guidelines": show_guidelines,
                "show_text_overlay": show_text_overlay,
                "show_border": show_border,
                "duration_end_hold_sec": duration_end_hold_sec,
            }

        with option_cols_row2[0]:
            st.subheader("Initial States", anchor=False)
            position_0 = st.slider(
                "Position (m)", min_value=-position_threshold, max_value=position_threshold, value=0.0, step=0.1
            )
            veloicty_0 = st.slider("Velocity (m/s)", min_value=-4.0, max_value=4.0, value=0.0, step=0.1)
            angle_0_deg = st.slider("Angle (deg)", min_value=-180, max_value=180, value=0, step=10)
            angle_0 = angle_0_deg * (2 * np.pi / 360)
            angular_velocity_0_rps = st.slider(
                "Angular Velocity (rev/s)", min_value=-2.0, max_value=2.0, value=0.0, step=0.1
            )
            angular_velocity_0 = angular_velocity_0_rps * (2 * np.pi)

            # Initial and terminal states
            x0 = np.array([position_0, veloicty_0, angle_0, angular_velocity_0])  # Slight angle to the pole

            # TODO handle off by 2*pi by encoding end points using sin, cos representation
            xT = np.array([0, 0, np.pi, 0])  # Straight up, centered

        with option_cols_row2[1]:
            st.subheader("Constraint Options", anchor=False)
            position_min, position_max = st.slider(
                "Position Constraint (m)", min_value=-2.0, max_value=2.0, value=(-2.0, 2.0), step=0.1
            )
            velocity_min, velocity_max = st.slider(
                "Velocity Constraint (m/s)", min_value=-10.0, max_value=10.0, value=(-10.0, 10.0), step=0.5
            )
            angle_min_deg, angle_max_deg = st.slider(
                "Angle Constraint (deg)", min_value=-360, max_value=360, value=(-360, 360), step=10
            )
            angular_velocity_min, angular_velocity_max = st.slider(
                "Angular Velocity Constraint (rev/s)", min_value=-2.0, max_value=2.0, value=(-2.0, 2.0), step=0.1
            )

            angle_min, angle_max = angle_min_deg * (2 * np.pi / 360), angle_max_deg * (2 * np.pi / 360)
            angular_velocity_min, angular_velocity_max = angular_velocity_min * (2 * np.pi), angular_velocity_max * (
                2 * np.pi
            )

            force_min, force_max = st.slider(
                "Force Constraint (N)", min_value=-50, max_value=50, value=(-20, 20), step=5
            )

            constraint_options = {
                "position": MinMax(position_min, position_max),
                "velocity": MinMax(velocity_min, velocity_max),
                "angle": MinMax(angle_min, angle_max),
                "angular_velocity": MinMax(angular_velocity_min, angular_velocity_max),
                "force": MinMax(force_min, force_max),
            }

        with option_cols_row2[2]:
            st.subheader("Objective Options", anchor=False)
            position_penalty = st.slider("Position Penalty", min_value=0, max_value=10, value=1, step=1)
            velocity_penalty = st.slider("Velocity Penalty", min_value=0, max_value=10, value=1, step=1)
            angle_penalty = st.slider("Angle Penalty", min_value=0, max_value=10, value=1, step=1)
            angular_velocity_penalty = st.slider("Angular Velocity Penalty", min_value=0, max_value=10, value=4, step=1)
            force_penalty = st.slider("Force Penalty", min_value=0, max_value=10, value=2, step=1)
            penalty_function = st.selectbox("Penalty Function", options=["square", "smooth_abs"])
            penalty_options = {
                "position": position_penalty,
                "velocity": velocity_penalty,
                "angle": angle_penalty,
                "angular_velocity": angular_velocity_penalty,
                "force": force_penalty,
                "penalty_function": penalty_function,
            }

        st.form_submit_button()


def casadi_square(x):
    return casadi.dot(x, x)


def casadi_smoothabs(x, eps=0.01):
    return casadi.sqrt(casadi_square(x) + eps)


penalty_func_map = {"square": casadi_square, "smooth_abs": casadi_smoothabs}


@st.cache_data(max_entries=10, show_spinner=False)
def solve_optimal_control_problem(x0, xT, N, T, model_params, penalty_options, constraint_options):
    # An optimal control problem (OCP),
    # solved with direct multiple-shooting.
    # For more information see: https://web.casadi.org/blog/ocp/

    opti = casadi.Opti()  # Optimization problem

    # ---- decision variables ---------
    X = opti.variable(n, N + 1)  # state trajectory
    U = opti.variable(m, N)  # control trajectory (throttle)
    # T = opti.variable()      # final time

    state_fields = ["position", "velocity", "angle", "angular_velocity"]
    action_fields = ["force"]

    state_field_vars = {field: X[i, :] for i, field, in enumerate(state_fields)}
    action_field_vars = {field: U[i, :] for i, field in enumerate(action_fields)}
    # all_field_vars = {**state_field_vars, **action_field_vars}

    # ---- objective          ---------
    objective = 0
    penalty_func = penalty_func_map[penalty_options["penalty_function"]]
    for i, field in enumerate(state_fields):
        delta_series = state_field_vars[field] - xT[i]
        objective += penalty_options[field] * penalty_func(delta_series)
    for field in action_fields:
        series = action_field_vars[field]
        objective += penalty_options[field] * penalty_func(series)
    opti.minimize(objective)

    # ---- dynamic constraints --------
    def f(x, u):
        return cartpole_dynamics(x, u, model_params)  # dx/dt = f(x,u)

    dt = T / N  # length of a control interval
    for k in range(N):  # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:, k], U[:, k])
        k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
        k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
        k4 = f(X[:, k] + dt * k3, U[:, k])
        x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # # ---- path constraints -----------
    for i, field in enumerate(state_fields):
        opti.subject_to(opti.bounded(constraint_options[field].min, X[i, :], constraint_options[field].max))
    for i, field in enumerate(action_fields):
        opti.subject_to(opti.bounded(constraint_options[field].min, U[i, :], constraint_options[field].max))

    # ---- boundary conditions --------
    opti.subject_to(X[:, 0] == x0)  # initial state
    opti.subject_to(X[:, -1] == xT)  # terminal state
    # opti.subject_to(casadi.sin(X[2, -1]) == 0)
    # opti.subject_to(casadi.cos(X[2, -1]) == -0.9)

    # ---- initial values for solver ---
    opti.set_initial(X[0, :], np.linspace(x0[0], xT[0], N + 1))
    opti.set_initial(X[1, :], np.linspace(x0[1], xT[1], N + 1))
    opti.set_initial(X[2, :], np.linspace(x0[2], xT[2], N + 1))
    opti.set_initial(X[3, :], np.linspace(x0[3], xT[3], N + 1))
    opti.set_initial(U[0, :], 0)

    # ---- solve NLP              ------
    # solver_options = {
    #     'ipopt': {
    #         'print_level': 0
    #     },
    #     'print_time': 0,
    #     'verbose': False,
    #     'error_on_fail': True
    # }
    solver_options = {"ipopt": {"max_iter": 1000}}
    opti.solver("ipopt", solver_options)  # set numerical backend
    sol = opti.solve()  # actual solve

    state_out = {field: sol.value(state_field_vars[field]) for field in state_fields}
    ocp_df = pd.DataFrame.from_dict(state_out, orient="columns")
    ocp_df["time"] = np.arange(N + 1) * dt

    action_out = {field: sol.value(action_field_vars[field]) for field in action_fields}
    ocp_df["force"] = action_out["force"].tolist() + [0]
    return ocp_df


with st.spinner("Solving optimal control problem..."):
    ocp_df = solve_optimal_control_problem(x0, xT, N, T, model_params, penalty_options, constraint_options)
# st.write(ocp_df)

state_fields = ["position", "velocity", "angle", "angular_velocity"]
action_fields = ["force"]
all_fields = state_fields + action_fields

with st.expander("Results", expanded=True):
    st.header("Results", anchor=False)

    plot_cols = st.columns(2)

    with plot_cols[0]:
        st.subheader("Time-series Plot", anchor=False)
        fig = px.line(ocp_df, x="time", y=all_fields)
        st.plotly_chart(fig, use_container_width=True)

    with plot_cols[1]:
        st.subheader("Phase-space Plot", anchor=False)
        subcols = st.columns(2)
        with subcols[0]:
            x_field = st.selectbox("x-axis field", options=all_fields, index=0)
        with subcols[1]:
            y_field = st.selectbox("y-axis field", options=all_fields, index=1)
        fig = px.line(ocp_df, x=x_field, y=y_field, hover_data=["time"])
        st.plotly_chart(fig, use_container_width=True)

    # Constants
    WIDTH, HEIGHT = 800, 600
    BACKGROUND_COLOR = (255, 255, 255)

    # Color palette
    # https://coolors.co/palette/8ecae6-219ebc-023047-ffb703-fb8500
    LIGHT_BLUE = [142, 202, 230]
    MEDIUM_BLUE = [33, 158, 188]
    DARK_BLUE = [2, 48, 71]
    GOLD = [255, 183, 3]
    ORANGE = [251, 133, 0]

    BORDER_COLOR = DARK_BLUE
    CART_COLOR = DARK_BLUE
    POLE_COLOR = MEDIUM_BLUE
    AXLE_COLOR = LIGHT_BLUE
    FORCE_COLOR = ORANGE

    pygame.init()

    # Load a monospace font
    font = pygame.font.Font("fonts/SpaceMono/SpaceMono-Regular.ttf", 16)

    def pillgon(length, width, num_points_per_arc=10):
        # Coordinates for a pill-shaped polygon that combines arc endcaps with straight edges
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
        # Coordinates for an arrow-shaped polygon that points to the left
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

    def draw_cartpole(surface, state, action, ghost=False):
        screen_width = WIDTH
        screen_height = HEIGHT

        world_width = position_threshold * 2
        scale = screen_width / world_width
        pole_width = 30.0
        pole_length = scale * model_params["length_pole"]
        cart_width = 100.0
        cart_height = 60.0

        if state is None:
            return None

        cart_color = CART_COLOR
        pole_color = POLE_COLOR
        axle_color = AXLE_COLOR
        force_color = FORCE_COLOR
        if ghost:
            opacity = 127  # transparent
            cart_color = cart_color + [opacity]
            pole_color = pole_color + [opacity]
            axle_color = axle_color + [opacity]
            force_color = force_color + [opacity]

        # Draw cart
        left, right, top, bottom = -cart_width / 2, cart_width / 2, cart_height / 2, -cart_height / 2
        cart_x = int(state[0] * scale + screen_width / 2)  # MIDDLE OF CART
        cart_y = int(screen_height / 2)  # TOP OF CART
        cart_coords = [(left, bottom), (left, top), (right, top), (right, bottom)]
        cart_coords = [(c[0] + cart_x, c[1] + cart_y) for c in cart_coords]
        gfxdraw.aapolygon(surface, cart_coords, cart_color)

        if not ghost:
            gfxdraw.filled_polygon(surface, cart_coords, cart_color)

        # Draw pole
        pole_coords_base = pillgon(pole_length, pole_width)
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
        axle_radius = int(0.5 * 0.5 * pole_width)
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
            force_coords_base = arrowgon(scale=-0.1 * (np.sign(force)) * (np.abs(force) ** 0.5) * scale)
            force_coords = []
            for coord in force_coords_base:
                coord = pygame.math.Vector2(coord)
                force_xshift = -np.sign(force) * ((cart_width / 2) + 10)
                coord = (coord[0] + cart_x + force_xshift, coord[1] + cart_y)
                force_coords.append(coord)
            gfxdraw.aapolygon(surface, force_coords, force_color)
            if not ghost:
                gfxdraw.filled_polygon(surface, force_coords, force_color)
        return

    def draw_scene(surface, time, state, action, target_state, animation_options):
        cart_position, cart_velocity, pole_angle, pole_angular_velocity = state
        (force,) = action

        screen_width = WIDTH
        screen_height = HEIGHT

        cart_y = screen_height // 2  # TOP OF CART

        if animation_options["show_border"]:
            # Draw the border
            # Top border
            pygame.gfxdraw.line(surface, 0, 0, WIDTH - 1, 0, BORDER_COLOR)
            # Bottom border
            pygame.gfxdraw.line(surface, 0, HEIGHT - 1, WIDTH - 1, HEIGHT - 1, BORDER_COLOR)
            # Left border
            pygame.gfxdraw.line(surface, 0, 0, 0, HEIGHT - 1, BORDER_COLOR)
            # Right border
            pygame.gfxdraw.line(surface, WIDTH - 1, 0, WIDTH - 1, HEIGHT - 1, BORDER_COLOR)

        if animation_options["show_guidelines"]:
            # Draw guidelines
            gfxdraw.hline(surface, 0, screen_width, cart_y, (0, 0, 0, 127))

        if animation_options["show_target_state"]:
            # Draw the ghosted target state
            draw_cartpole(surface, target_state, action=None, ghost=True)

        # Draw the actual cartpole
        action_for_draw = action if animation_options["show_force"] else None
        draw_cartpole(surface, state, action_for_draw, ghost=False)

        # Render the text overlay
        if animation_options["show_text_overlay"]:
            text_overlay_strs = []
            text_overlay_strs.append(f"            Time: {time:6.2f} s")
            text_overlay_strs.append(f"        Position: {cart_position:6.2f} m")
            text_overlay_strs.append(f"        Velocity: {cart_velocity:6.2f} m/s")
            text_overlay_strs.append(f"           Angle: {pole_angle:6.2f} rad")
            text_overlay_strs.append(f"Angular Velocity: {pole_angular_velocity:6.2f} rad/s")
            text_overlay_strs.append(f"           Force: {force:6.2f} N")

            for i, s in enumerate(text_overlay_strs):
                label = font.render(s, 1, (0, 0, 0))
                surface.blit(label, (10, 10 + i * 20))

    def create_frame(time, state, action, target_state, animation_options):
        surface = pygame.Surface((WIDTH, HEIGHT))
        surface.fill(BACKGROUND_COLOR)
        draw_scene(surface, time, state, action, target_state, animation_options)

        size = surface.get_size()
        data = pygame.image.tobytes(surface, "RGBA")
        return Image.frombytes("RGBA", size, data)

    @st.cache_data(max_entries=10)
    def animate(ani_state_action_time_series, target_state, fps, animation_options):
        with st.spinner("Creating frames..."):
            frames = [
                create_frame(time, state, action, target_state, animation_options)
                for time, state, action in ani_state_action_time_series
            ]
        if animation_options["duration_end_hold_sec"] > 0:
            num_frames_end_hold = int(fps * animation_options["duration_end_hold_sec"])
            frames += [frames[-1]] * num_frames_end_hold
        # target fps = 50 Hz
        # https://wunkolo.github.io/post/2020/02/buttery-smooth-10fps/
        duration = 1000 // fps  # this is the duration of each frame in milliseconds
        with st.spinner("Saving animation..."):
            # Create an in-memory byte stream
            byte_stream = io.BytesIO()
            frames[0].save(
                byte_stream, format="GIF", save_all=True, append_images=frames[1:], loop=0, duration=duration
            )
            # Go to the start of the byte stream
            byte_stream.seek(0)
        return byte_stream

    plot_cols = st.columns(2)
    with plot_cols[0]:
        st.subheader("Animation", anchor=False)
        if show_animation:
            # Use linear interpolation to resample the signals at the fps for animation
            N_ani = int(ani_fps * T)
            ani_df = pd.DataFrame({"time": np.arange(N_ani + 1) / ani_fps})

            for field in all_fields:
                ani_df[field] = np.interp(ani_df.time, ocp_df.time, ocp_df[field])

            ani_state_action_time_series = [
                (t, s, a) for t, s, a in zip(ani_df.time, ani_df[state_fields].values, ani_df[action_fields].values)
            ]
            target_state = xT

            animation = animate(ani_state_action_time_series, target_state, ani_fps, animation_options)
            st.image(animation, use_column_width=True)
        else:
            st.info('Enable "Show Animation" in the options to see an animation here.', icon="i")
