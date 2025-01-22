import streamlit as st


def main():

    st.title('Barechip Calibration')
    st.write('A utility for producing the barechip calibration files.')
    st.write('Inputs: Resistance vs current of a barechip cernox at fixed temperatures.')
    st.write('Outputs: Resistance vs temperature in the absence of self heating.')


if __name__ == "__main__":
    main()