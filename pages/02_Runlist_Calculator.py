import streamlit as st


def main():

    st.title('Runlist Calculator')
    st.write('A tool for producing a list of measurement parameters for the long relaxation experiment.')
    st.write('Outputs: Low and high currents for a desired temperature rise, aquisition time for each relaxation, expected preamplifier gain, number of repeats etc')


if __name__ == "__main__":
    main()