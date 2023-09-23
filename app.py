import numpy as np
import pandas as pd

import streamlit as st
import plotly.express as px

import app_options as ao
import constants
import optimal_control as oc
import animation as ani


st.set_page_config(layout="wide")
st.title("🛒💈🎛️ Cartpole Optimal Control")


def execute_ui_section_description():
    st.header("Summary")
    st.write(
        "This app demonstrates optimal control of a cartpole system. The task is to apply a sequence of inputs to drive"
        " the system from the initial state to the terminal state while minimizing an objective function."
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


def ani_df_from_ocp_df(ocp_df: pd.DataFrame, app_options: ao.AppOptions):
    """Use linear interpolation to resample the signals at the fps for animation."""
    num_intervals_ani = app_options.animation_options.fps * app_options.simulation_options.duration
    ani_df = pd.DataFrame({"time": np.arange(num_intervals_ani + 1) / app_options.animation_options.fps})

    for field in constants.ALL_FIELDS:
        ani_df[field] = np.interp(ani_df.time, ocp_df.time, ocp_df[field])

    return ani_df


def state_action_time_series_from_df(df: pd.DataFrame) -> list[tuple[float, np.ndarray, np.ndarray]]:
    """Convert a DataFrame to a list of time-state-action tuples."""
    return [
        (t, s, a) for t, s, a in zip(df.time, df[constants.STATE_FIELDS].values, df[constants.ACTION_FIELDS].values)
    ]


def execute_ui_section_animation(ocp_df: pd.DataFrame, app_options: ao.AppOptions):
    if app_options.animation_options.show_animation:
        ani_df = ani_df_from_ocp_df(ocp_df, app_options)
        ani_state_action_time_series = [
            (t, s, a)
            for t, s, a in zip(
                ani_df.time, ani_df[constants.STATE_FIELDS].values, ani_df[constants.ACTION_FIELDS].values
            )
        ]
        animation_byte_stream = ani.animate(ani_state_action_time_series, app_options)

        # Center the animation horizontally on the page, leaving small buffer columns on either side
        cols = st.columns([2, 4, 2])
        with cols[1]:
            st.image(animation_byte_stream, use_column_width=True)
    else:
        st.info('Enable "Show Animation" in the options to see an animation here.', icon="🛒")


def execute_ui_section_results(ocp_df, app_options):
    animation_container = st.container()
    plot_container = st.container()

    with plot_container:
        st.subheader("Time-series Plot", anchor=False)
        fig = px.line(ocp_df, x="time", y=constants.ALL_FIELDS)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Phase-space Plot", anchor=False)
        subcols = st.columns(2)
        with subcols[0]:
            x_field = st.selectbox(
                "x-axis field", options=constants.ALL_FIELDS, index=constants.ALL_FIELDS.index("angle")
            )
        with subcols[1]:
            y_field = st.selectbox(
                "y-axis field", options=constants.ALL_FIELDS, index=constants.ALL_FIELDS.index("angular_velocity")
            )
        fig = px.line(ocp_df, x=x_field, y=y_field, hover_data=["time"])
        st.plotly_chart(fig, use_container_width=True)

    with animation_container:
        st.subheader("Animation", anchor=False)
        execute_ui_section_animation(ocp_df, app_options)


def main():
    with st.expander("Description & Explanation", expanded=False):
        execute_ui_section_description()

    with st.expander("Options", expanded=False):
        with st.form("options_form"):
            # Create a container for the form submit button first so it appears at the top,
            # which ensures it does not move around as different tabs in get_app_options_from_ui() are opened
            button_container = st.container()
            # Get the options
            app_options = ao.get_app_options_from_ui()
            # Show the form submit button
            with button_container:
                st.form_submit_button("Update Options", type="primary")

    with st.spinner("Solving optimal control problem..."):
        ocp_df, exception = oc.solve_optimal_control_problem(
            app_options.model_parameter_options,
            app_options.simulation_options,
            app_options.initial_state,
            app_options.terminal_state,
            app_options.constraint_options,
            app_options.penalty_options,
            app_options.solver_options,
        )

    if exception:
        st.error("Exception encountered while solving the optimal control problem.")
        st.exception(exception)
        st.info(
            "Try changing the options to make the optimal control problem solvable. Common sources of infeasibility"
            " include overly restrictive constraints, overly challenging initial states, and too low solver maximum"
            " iterations."
        )
    else:
        with st.expander("Results", expanded=True):
            execute_ui_section_results(ocp_df, app_options)


main()
