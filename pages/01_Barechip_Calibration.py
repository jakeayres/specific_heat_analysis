import streamlit as st


def main():

    st.title('Barechip Calibration')
    st.write('A utility for producing the barechip calibration files from current dependent resistance data.')
    st.code('Inputs: Resistance vs current of a barechip cernox at fixed temperatures.', language=None, wrap_lines=True)
    st.code('Outputs: Resistance vs temperature in the absence of self heating.', language=None, wrap_lines=True)


if __name__ == "__main__":
    main()