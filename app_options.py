from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

import streamlit as st


@dataclass
class MinMax:
    """Class to hold min and max values of any type."""

    min: Any
    max: Any


@dataclass
class ModelParameterOptions:
    gravity_acceleration: float
    mass_cart: float
    mass_pole: float
    length_pole: float


@dataclass
class SimulationOptions:
    duration: int
    fps: int
    num_intervals: int = field(init=False)
    dt: float = field(init=False)

    def __post_init__(self):
        self.num_intervals = self.fps * self.duration
        self.dt = self.duration / self.num_intervals


@dataclass
class AnimationOptions:
    show_force: bool
    show_terminal_state: bool
    show_constraint_box: bool
    show_text_overlay: bool
    show_border: bool
    fps: int
    playback_rate_scale: float
    duration_end_hold_sec: float
    duration_single_ms: int = field(init=False)

    def __post_init__(self):
        # Duration of a single frame in milliseconds
        self.duration_single_ms = int((1000 / self.fps) / self.playback_rate_scale)


@dataclass
class DynamicsState:
    position: float | None
    velocity: float | None
    angle: float | None
    angular_velocity: float | None
    numpy: np.ndarray = field(init=False)

    def __post_init__(self):
        self.numpy = np.array([self.position, self.velocity, self.angle, self.angular_velocity])


@dataclass
class ConstraintOptions:
    position: MinMax
    velocity: MinMax
    angle: MinMax
    angular_velocity: MinMax
    force: MinMax


@dataclass
class PenaltyOptions:
    position: float
    velocity: float
    angle: float
    angular_velocity: float
    force: float
    function: Callable


@dataclass
class SolverOptions:
    max_iter: int


@dataclass
class AppOptions:
    model_parameter_options: ModelParameterOptions
    simulation_options: SimulationOptions
    animation_options: AnimationOptions
    initial_state: DynamicsState
    terminal_state: DynamicsState
    constraint_options: ConstraintOptions
    penalty_options: PenaltyOptions
    solver_options: SolverOptions


def get_model_parameter_options_from_ui():
    gravity_acceleration_gs = st.slider(
        "Gravitational Acceleration (G's)", min_value=0.0, max_value=4.0, value=1.0, step=0.1
    )
    gravity_acceleration = 9.81 * gravity_acceleration_gs
    mass_cart = st.slider("Mass of cart (kg)", min_value=0.0, max_value=2.0, value=1.0, step=0.1)
    mass_pole = st.slider("Mass of pole (kg)", min_value=0.0, max_value=2.0, value=0.8, step=0.1)
    length_pole = st.slider("Length of pole (m)", min_value=0.4, max_value=1.2, value=1.0, step=0.1)
    return ModelParameterOptions(
        gravity_acceleration,
        mass_cart,
        mass_pole,
        length_pole,
    )


def get_simulation_options_from_ui():
    duration = st.slider("Simulation Duration (seconds)", min_value=1, max_value=20, value=6, step=1)
    fps = st.select_slider(
        "Simulation Frame Rate (frames per second)",
        options=[5, 10, 25, 50],
        value=10,
        help=(
            "This controls the rate at which the dynamics evolve, i.e how frequently states and actions are"
            " updated. Larger values will result in a larger optimal control problem that takes longer to"
            " solve, but will be more precise."
        ),
    )
    return SimulationOptions(duration, fps)


def get_animation_options_from_ui():
    toggle_col, slider_col = st.columns([2, 2])

    with slider_col:
        duration_end_hold_sec = st.slider(
            "Animation Duration Pause at End (sec)",
            min_value=0,
            max_value=5,
            value=1,
        )
        # 50 fps is the max practical frame rate for GIF
        # https://wunkolo.github.io/post/2020/02/buttery-smooth-10fps/
        fps = st.select_slider(
            "Animation Frame Rate (frames per second)",
            options=[5, 10, 25, 50],
            value=25,
            help=(
                "This controls the rate at which animation frames are displayed. Simulation result data is linearly"
                " interpolated from simulation time to animation time. Larver values will result in more frames"
                " that take longer to render and save, but will play back more smoothly. There is a hard limit at"
                " 50 FPS due to technical limitations of GIFs."
            ),
        )
        playback_rate_scale = st.select_slider(
            "Playback Rate Scale",
            options=[0.1, 0.25, 0.5, 1.0],
            value=1.0,
            help=(
                "Use this to play back the animation more slowly than real-time to give yourself more time to see"
                " what is happening."
            ),
        )

    with toggle_col:
        show_force = st.toggle("Show Force Arrow", value=True)
        show_terminal_state = st.toggle("Show Terminal State Outline", value=True)
        show_constraint_box = st.toggle("Show Cart Constraint Outline", value=True)
        show_text_overlay = st.toggle("Show Text Overlay", value=True)
        show_border = st.toggle("Show Border", value=False)

    return AnimationOptions(
        show_force,
        show_terminal_state,
        show_constraint_box,
        show_text_overlay,
        show_border,
        fps,
        playback_rate_scale,
        duration_end_hold_sec,
    )


def get_state_options_from_ui(default_angle=0, max_angle=180, suffix=""):
    field_title = "Position"
    unit = "m"
    cols = st.columns([2, 2])
    with cols[0]:
        constrain_position = st.toggle(f"Enforce {field_title}", value=True, key=f"{field_title}_{suffix}_toggle")
    with cols[1]:
        position_from_slider = st.slider(
            f"{field_title} ({unit})",
            min_value=-1.5,
            max_value=1.5,
            value=0.0,
            step=0.1,
            key=f"position_{suffix}_slider",
        )
    if constrain_position:
        position = position_from_slider
    else:
        position = None

    field_title = "Velocity"
    unit = "m/s"
    cols = st.columns([2, 2])
    with cols[0]:
        constrain_velocity = st.toggle(f"Enforce {field_title}", value=True, key=f"{field_title}_{suffix}_toggle")
    with cols[1]:
        velocity_from_slider = st.slider(
            f"{field_title} ({unit})",
            min_value=-4.0,
            max_value=4.0,
            value=0.0,
            step=0.1,
            key=f"velocity_{suffix}_slider",
        )
    if constrain_velocity:
        velocity = velocity_from_slider
    else:
        velocity = None

    field_title = "Angle"
    unit = "deg"
    cols = st.columns([2, 2])
    with cols[0]:
        constrain_angle = st.toggle(f"Enforce {field_title}", value=True, key=f"{field_title}_{suffix}_toggle")
    with cols[1]:
        angle_deg_from_slider = st.slider(
            f"{field_title} ({unit})",
            min_value=-max_angle,
            max_value=max_angle,
            value=default_angle,
            step=10,
            key=f"angle_deg_{suffix}_slider",
        )
    if constrain_angle:
        angle_deg = angle_deg_from_slider
        angle = angle_deg * (2 * np.pi / 360)
    else:
        angle = None

    field_title = "Angular Velocity"
    unit = "rev/s"
    cols = st.columns([2, 2])
    with cols[0]:
        constrain_angular_velocity = st.toggle(
            f"Enforce {field_title}", value=True, key=f"{field_title}_{suffix}_toggle"
        )
    with cols[1]:
        angular_velocity_rps_from_slider = st.slider(
            f"{field_title} ({unit})",
            min_value=-2.0,
            max_value=2.0,
            value=0.0,
            step=0.1,
            key=f"angular_velocity_rps_{suffix}_slider",
        )
    if constrain_angular_velocity:
        angular_velocity_rps = angular_velocity_rps_from_slider
        angular_velocity = angular_velocity_rps * (2 * np.pi)
    else:
        angular_velocity = None

    return DynamicsState(position, velocity, angle, angular_velocity)


def get_initial_state_options_from_ui():
    return get_state_options_from_ui(default_angle=0, max_angle=180, suffix="initial")


def get_terminal_state_options_from_ui():
    return get_state_options_from_ui(default_angle=180, max_angle=360, suffix="terminal")


def get_constraint_options_from_ui():
    position_min, position_max = st.slider(
        "Position Constraint (m)", min_value=-1.5, max_value=1.5, value=(-1.0, 1.0), step=0.1
    )
    velocity_min, velocity_max = st.slider(
        "Velocity Constraint (m/s)", min_value=-20.0, max_value=20.0, value=(-15.0, 15.0), step=0.5
    )
    angle_min_deg, angle_max_deg = st.slider(
        "Angle Constraint (deg)", min_value=-360, max_value=360, value=(-270, 270), step=10
    )
    angular_velocity_min, angular_velocity_max = st.slider(
        "Angular Velocity Constraint (rad/s)", min_value=-40, max_value=40, value=(-20, 20), step=1
    )

    angle_min, angle_max = angle_min_deg * (2 * np.pi / 360), angle_max_deg * (2 * np.pi / 360)

    force_min, force_max = st.slider("Force Constraint (N)", min_value=-50, max_value=50, value=(-30, 30), step=5)

    return ConstraintOptions(
        position=MinMax(position_min, position_max),
        velocity=MinMax(velocity_min, velocity_max),
        angle=MinMax(angle_min, angle_max),
        angular_velocity=MinMax(angular_velocity_min, angular_velocity_max),
        force=MinMax(force_min, force_max),
    )


def get_objective_options_from_ui():
    position_penalty = st.slider("Position Penalty", min_value=0, max_value=10, value=1, step=1)
    velocity_penalty = st.slider("Velocity Penalty", min_value=0, max_value=10, value=2, step=1)
    angle_penalty = st.slider("Angle Penalty", min_value=0, max_value=10, value=1, step=1)
    angular_velocity_penalty = st.slider("Angular Velocity Penalty", min_value=0, max_value=10, value=8, step=1)
    force_penalty = st.slider("Force Penalty", min_value=0, max_value=10, value=4, step=1)
    penalty_function = st.selectbox(
        "Penalty Function",
        options=["square", "smooth_abs"],
        help=(
            "For each element in the sequence, the penalty function specified here is applied and the result is"
            " added to the objective. The `square` function computes the square of an element. The `smooth_abs`"
            " function computes `sqrt(square(x) + eps)` where `eps` is a small number. Generally, the"
            " `smooth_abs` function is much more expensive to use, since it leads to a more complicated and"
            " less well-conditioned optimization problem."
        ),
    )

    return PenaltyOptions(
        position_penalty,
        velocity_penalty,
        angle_penalty,
        angular_velocity_penalty,
        force_penalty,
        penalty_function,
    )


def get_solver_options_from_ui():
    max_iter = st.select_slider(
        "Maximum Iterations for IPOPT Solver",
        options=[10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000],
        value=1000,
    )
    return SolverOptions(max_iter)


@dataclass
class TabInfo:
    name: str
    func: Callable
    tab: Any = None  # streamlit container, technically a DeltaGenerator
    options_object: Any = None  # technically a union type of all the <Xyz>Options classes


def get_app_options_from_ui():
    # Note: The order in which these are listed determines the order they show up in the UI from left to right.
    tab_infos = [
        TabInfo("Model Parameters", get_model_parameter_options_from_ui),
        TabInfo("Simulation", get_simulation_options_from_ui),
        TabInfo("Animation", get_animation_options_from_ui),
        TabInfo("Initial State", get_initial_state_options_from_ui),
        TabInfo("Terminal State", get_terminal_state_options_from_ui),
        TabInfo("Constraints", get_constraint_options_from_ui),
        TabInfo("Objective", get_objective_options_from_ui),
        TabInfo("Solver", get_solver_options_from_ui),
    ]
    # Create the actual tabs
    tab_names = [tab_info.name for tab_info in tab_infos]
    tabs = st.tabs(tab_names)

    # Get the options from inside each tab
    for tab_info, tab in zip(tab_infos, tabs):
        tab_info.tab = tab
        with tab:
            tab_info.options_object = tab_info.func()

    return AppOptions(*[tab_info.options_object for tab_info in tab_infos])
