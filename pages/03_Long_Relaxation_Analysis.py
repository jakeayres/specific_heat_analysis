import streamlit as st


def main():

    st.title('Long Relaxation Analysis')
    st.write('Analysis tool for obtaining the temperature dependent specific heat.')
    st.code('Inputs: Voltage vs time during relaxation sweeps.', language=None, wrap_lines=True)
    st.code('Outputs: Heat capacity vs temperature', language=None, wrap_lines=True)


if __name__ == "__main__":
    main()