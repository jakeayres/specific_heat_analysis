import streamlit as st


def init_session_state():
    if 'barechip_calibration' not in st.session_state:
        st.session_state['barechip_calibration'] = None

def main():


    init_session_state()

    st.title('Specific Heat Analysis')
    st.write('A set of utilities for running and analysing long relaxation specific heat measurements.')
    st.write('... Add some sparse theory/method notes here ...')

    st.write('Hello')

    st.text_input('Test')


if __name__ == "__main__":
    main()