import streamlit as st


def init_session_state():
    if 'barechip_calibration' not in st.session_state:
        st.session_state['barechip_calibration'] = None

def main():


    init_session_state()

    st.title('Specific Heat Analysis')
    st.write('This is a set of utilities for analysing long relaxation specific heat data.')
    st.write('Step through the pages in the menu on the left.')

    st.header('Workflow')

    st.subheader('Create barechip calibration')
    st.write('1. Measure resistance vs current at fixed temperatures. Will likely need multiple parameters to cover full temperature range.')
    st.write('2. Use the "Barechip Calibration" page to produce a calibration file (R vs T)')

    st.subheader('Analyze relaxation sweeps')
    st.write('3. Measure relaxation sweeps (voltage vs time) covering relevant temperature ranges.')
    st.write('4. Use the "Long Relaxation Analysis" to produce C vs T for each relaxation sweep')
    st.write('5. Use the "Heat Capacity Averaging" to average multiple sweeps into one smooth C vs T')


if __name__ == "__main__":
    main()