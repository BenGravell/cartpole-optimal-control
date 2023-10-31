import numpy as np
import pandas as pd
import casadi

import streamlit as st

import constants
import app_options as ao


def casadi_square(x):
    return casadi.dot(x, x)


def casadi_smooth_abs(x, eps: float = 0.1):
    return casadi.sqrt(casadi_square(x) + eps)


def cartpole_dynamics(state: np.ndarray, action: np.ndarray, model_parameter_options: ao.ModelParameterOptions):
    """Dynamics function for a cartpole.

    Based on http://underactuated.mit.edu/acrobot.html
    """

    # Convenience aliases
    gravity = model_parameter_options.gravity_acceleration
    mass_cart = model_parameter_options.mass_cart
    mass_pole = model_parameter_options.mass_pole
    length_pole = model_parameter_options.length_pole

    state[constants.STATE_FIELDS.index("position")]
    x_dot = state[constants.STATE_FIELDS.index("velocity")]
    theta = state[constants.STATE_FIELDS.index("angle")]
    theta_dot = state[constants.STATE_FIELDS.index("angular_velocity")]

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


@st.cache_data(max_entries=10, show_spinner=False)
def solve_optimal_control_problem(
    model_parameter_options: ao.ModelParameterOptions,
    simulation_options: ao.SimulationOptions,
    initial_state: ao.DynamicsState,
    terminal_state: ao.DynamicsState,
    constraint_options: ao.ConstraintOptions,
    penalty_options: ao.PenaltyOptions,
    solver_options: ao.SolverOptions,
):
    """Solve an optimal control problem (OCP) with direct multiple-shooting.

    For more information see: https://web.casadi.org/blog/ocp/
    """

    opti = casadi.Opti()  # Optimization problem

    # Convenience aliases
    N = simulation_options.num_intervals
    dt = simulation_options.dt
    x0 = initial_state.numpy
    xT = terminal_state.numpy

    # Decision variables
    X = opti.variable(constants.DIM_STATE, N + 1)  # state trajectory
    U = opti.variable(constants.DIM_ACTION, N)  # control trajectory

    # Convenience aliases
    state_field_vars = {field: X[i, :] for i, field, in enumerate(constants.STATE_FIELDS)}
    action_field_vars = {field: U[i, :] for i, field in enumerate(constants.ACTION_FIELDS)}

    # Objective
    objective = 0
    penalty_func_map = {"square": casadi_square, "smooth_abs": casadi_smooth_abs}
    penalty_func = penalty_func_map[penalty_options.function]
    for i, field in enumerate(constants.STATE_FIELDS):
        if xT[i] is not None:
            delta_series = state_field_vars[field] - xT[i]
            objective += getattr(penalty_options, field) * penalty_func(delta_series)
    for field in constants.ACTION_FIELDS:
        series = action_field_vars[field]
        objective += getattr(penalty_options, field) * penalty_func(series)
    opti.minimize(objective)

    # Dynamic constraints
    def f(x, u):
        return cartpole_dynamics(x, u, model_parameter_options)  # dx/dt = f(x,u)

    for k in range(N):  # loop over control intervals
        # Runge-Kutta 4 integration
        k1 = f(X[:, k], U[:, k])
        k2 = f(X[:, k] + dt / 2 * k1, U[:, k])
        k3 = f(X[:, k] + dt / 2 * k2, U[:, k])
        k4 = f(X[:, k] + dt * k3, U[:, k])
        x_next = X[:, k] + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        opti.subject_to(X[:, k + 1] == x_next)  # close the gaps

    # Path constraints
    for i, field in enumerate(constants.STATE_FIELDS):
        constraint_values = getattr(constraint_options, field, None)
        if constraint_values is not None:
            min_value = constraint_values.min
            max_value = constraint_values.max
            if min_value is not None and max_value is not None:
                opti.subject_to(opti.bounded(min_value, X[i, :], max_value))
    for i, field in enumerate(constants.ACTION_FIELDS):
        constraint_values = getattr(constraint_options, field, None)
        if constraint_values is not None:
            min_value = constraint_values.min
            max_value = constraint_values.max
            if min_value is not None and max_value is not None:
                opti.subject_to(opti.bounded(min_value, U[i, :], max_value))

    # Boundary conditions
    for i, field in enumerate(constants.STATE_FIELDS):
        # Initial state
        # st.write(f"initial state {field} = {x0[i]}")
        if x0[i] is not None:
            opti.subject_to(X[i, 0] == x0[i])
        # Terminal state
        # DEBUG
        # st.write(f"terminal state {field} = {xT[i]}")
        if xT[i] is not None:
            opti.subject_to(X[i, -1] == xT[i])

    # Initial values for solver
    # TODO expose solver initialization methods via UI:
    # 1. zeros
    # 2. linear ramp from initial state to terminal state
    # 3. random between constraint bounds w/ zeros fallback
    for i, field in enumerate(constants.STATE_FIELDS):
        constraint_values = getattr(constraint_options, field, None)
        if constraint_values is not None:
            min_value = constraint_values.min
            max_value = constraint_values.max
            if min_value is not None and max_value is not None:
                # TODO
                # # Zeros initialization
                # opti.set_initial(X[i, :], 0.0)
                # # Linear ramp from initial to terminal value initialization
                if x0[i] is not None and xT[i] is not None:
                    opti.set_initial(X[i, :], np.linspace(x0[i], xT[i], N + 1))
                # # Random initialization
                # opti.set_initial(X[i, :], np.random.uniform(min_value, max_value, N + 1))

    for i, field in enumerate(constants.ACTION_FIELDS):
        constraint_values = getattr(constraint_options, field, None)
        if constraint_values is not None:
            min_value = constraint_values.min
            max_value = constraint_values.max
            if min_value is not None and max_value is not None:
                # # TODO
                # Zeros initialization
                opti.set_initial(U[i, :], 0.0)
                # # Random initialization
                # opti.set_initial(U[i, :], np.random.uniform(min_value, max_value, N))

    # Solve NLP
    opti_solver_options = {"ipopt": {"max_iter": solver_options.max_iter}}
    opti.solver("ipopt", opti_solver_options)  # set numerical backend

    try:
        sol = opti.solve()  # actual solve
    except RuntimeError as exception:
        return None, exception

    state_out = {field: sol.value(state_field_vars[field]) for field in constants.STATE_FIELDS}
    ocp_df = pd.DataFrame.from_dict(state_out, orient="columns")
    ocp_df["time"] = np.round(
        np.arange(N + 1) * dt, 9
    )  # sub-nanosecond time resolution not needed, mitigate float rounding issues

    action_out = {field: sol.value(action_field_vars[field]) for field in constants.ACTION_FIELDS}
    ocp_df["force"] = action_out["force"].tolist() + [0]

    return ocp_df, None
