import streamlit as st


def main():

    st.title('Long Relaxation Analysis')
    st.write('Analysis tool for obtaining the temperature dependent specific heat.')
    st.write('Inputs: Voltage vs time during relaxation sweeps.')
    st.write('Outputs: Heat capacity vs temperature')


if __name__ == "__main__":
    main()