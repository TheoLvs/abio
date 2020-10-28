import streamlit as st

from abio.cellular_automata import CellularAutomata1D
from abio.cellular_automata import GameOfLife

st.sidebar.write("# Cellular Automata demo")

# Add a selectbox to the sidebar:
dimension = st.sidebar.selectbox(
    'Cellular Automata Type',
    ('1D', '2D')
)

if dimension == "1D":

    ca = CellularAutomata1D()
    rule = st.sidebar.number_input("Rule (-1 for random)", min_value=-1, max_value=256, value=-1, step=1)
    size = st.sidebar.number_input("Size", min_value=50, max_value=1000, value=100, step=50)
    p_init = st.sidebar.number_input("% Init", min_value=0.01, max_value=1.00, value=0.02, step=0.01)

    run = st.sidebar.button("Run")

    rule = None if rule == -1 else rule

    if run:
        fig = ca.run_random(rule = rule,size = size,p_init = p_init,n_steps = size,return_fig = True)
        st.pyplot(fig)

elif dimension == "2D":

    ca = GameOfLife()
    size = st.sidebar.number_input("Size", min_value=50, max_value=1000, value=100, step=50)
    n_steps = st.sidebar.number_input("Steps", min_value=100, max_value=1000, value=500, step=50)
    p_init = st.sidebar.number_input("% Init", min_value=0.05, max_value=1.00, value=0.3, step=0.05)
    persistence = not st.sidebar.checkbox("Persistence")
    fps = st.sidebar.number_input("FPS", min_value=5, max_value=30, value=10, step=5)
    run = st.sidebar.button("Run")

    if run:

        states = ca.run(n_steps,p_init = p_init,init_size = size)
        states.transform(only_alive = persistence,resize = (400,400),method = "nearest")
        filepath = states.save_video("streamlit_capture.mp4",fps = fps)

        st.video(filepath)