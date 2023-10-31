import streamlit as st


def run():
    st.title("🛒 Cartpole Optimal Control", anchor=False)
    st.caption("Control a cart with a swinging pole, optimally!")
    cols = st.columns([2, 1])
    with cols[0]:
        st.subheader("What is Cartpole Optimal Control?", divider="blue", anchor=False)
        st.markdown(
            "This app demonstrates optimal control of a cartpole system. The task is to apply a sequence of inputs to"
            " drive the system from the initial state to the terminal state while minimizing an objective function."
        )
    with cols[1]:
        st.image("assets/welcome.jpg")
