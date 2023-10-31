import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

import welcome
import app_options as ao
import constants
import optimal_control as oc
import animation as ani


st.set_page_config(page_title="Cartpole Optimal Control", page_icon="🛒", layout="centered")


def execute_ui_section_description():
    st.header("Optimal Control", anchor=False)
    st.write(
        "The optimal control problem is formulated using [CasADi](https://web.casadi.org/) and the direct"
        " multiple-shooting technique to transcribe it to a nonlinear program (NLP), which is solved using the"
        " [IPOPT](https://github.com/coin-or/Ipopt) solver. See https://web.casadi.org/blog/ocp/ for more details."
    )

    st.header("Dynamics", anchor=False)
    st.subheader("Equations of Motion", anchor=False)
    st.write(
        "The equations of motion for the cartpole system are derived using Lagrangian mechanics. The details are found"
        " at http://underactuated.mit.edu/acrobot.html."
    )
    st.subheader("Numerical Integration of the Dynamics Ordinary Differential Equation", anchor=False)
    st.write(
        "The ordinary differential equation (ODE) governing the dynamics of the system are integrated numerically using"
        " the Runge-Kutta 4th-order method (RK4)."
    )

    st.header("Animation", anchor=False)
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
    show_animation = st.toggle("Show Animation", value=True)
    if show_animation:
        ani_df = ani_df_from_ocp_df(ocp_df, app_options)
        ani_state_action_time_series = [
            (t, s, a)
            for t, s, a in zip(
                ani_df.time, ani_df[constants.STATE_FIELDS].values, ani_df[constants.ACTION_FIELDS].values
            )
        ]
        animation_byte_stream = ani.animate(ani_state_action_time_series, app_options)

        # Center the animation horizontally on the page, leaving small buffer columns on either side
        st.image(animation_byte_stream)
    else:
        st.info('Enable "Show Animation" to see an animation here.', icon="🛒")


def execute_ui_section_results(ocp_df, ocp_exception, ocp_captured_output, app_options):
    tab_names = ["Animation", "Time-series Plot", "Phase-space Plot", "Captured Output from Solver"]
    tabs = st.tabs(tab_names)

    with tabs[tab_names.index("Captured Output from Solver")]:
        if ocp_captured_output is not None:
            st.text(ocp_captured_output)
        else:
            st.info("No captured output to show.")

    for i in range(len(tab_names) - 1):
        with tabs[i]:
            if ocp_exception:
                st.error("Exception encountered while solving the optimal control problem.")
                st.exception(ocp_exception)
                st.info(
                    "Try changing the options to make the optimal control problem solvable. Common sources of"
                    " infeasibility include overly restrictive constraints, overly challenging initial and terminal"
                    " states, and too low solver maximum iterations."
                )

    for i in range(len(tab_names) - 1):
        with tabs[i]:
            if ocp_df is None:
                st.info("No optimal control solution available yet.")

    if ocp_exception or ocp_df is None:
        return

    with tabs[tab_names.index("Time-series Plot")]:
        fig = px.line(ocp_df, x="time", y=constants.ALL_FIELDS)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[tab_names.index("Phase-space Plot")]:
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

    with tabs[tab_names.index("Animation")]:
        execute_ui_section_animation(ocp_df, app_options)


def solve_optimal_control_problem_in_ui(app_options):
    with st.spinner("Solving optimal control problem..."):
        df, exception, captured_output = oc.solve_optimal_control_problem(
            app_options.model_parameter_options,
            app_options.simulation_options,
            app_options.initial_state,
            app_options.terminal_state,
            app_options.constraint_options,
            app_options.penalty_options,
            app_options.solver_options,
        )
    st.session_state.ocp_df = df
    st.session_state.ocp_exception = exception
    st.session_state.ocp_captured_output = captured_output


def main():
    tab_names = [
        "Welcome",
        "Options",
        "Results",
        "Help",
    ]

    tabs = st.tabs(tab_names)

    with tabs[tab_names.index("Welcome")]:
        welcome.run()

    with tabs[tab_names.index("Help")]:
        execute_ui_section_description()

    with tabs[tab_names.index("Options")]:
        with st.form("options_form"):
            # Create a container for the form submit button first so it appears at the top,
            # which ensures it does not move around as different tabs in get_app_options_from_ui() are opened
            button_container = st.container()
            # Get the options
            app_options = ao.get_app_options_from_ui()
            # Show the form submit button
            with button_container:
                st.form_submit_button("Update Options")

    with tabs[tab_names.index("Results")]:
        if st.session_state.get("ocp_exception") is None:
            st.session_state.ocp_exception = None

        if st.session_state.get("ocp_df") is None:
            st.session_state.ocp_df = None

        if st.session_state.get("ocp_captured_output") is None:
            st.session_state.ocp_captured_output = None

        solve = st.button("Solve Optimal Control Problem")
        if solve:
            solve_optimal_control_problem_in_ui(app_options)

        execute_ui_section_results(
            st.session_state.ocp_df, st.session_state.ocp_exception, st.session_state.ocp_captured_output, app_options
        )


if __name__ == "__main__":
    main()
