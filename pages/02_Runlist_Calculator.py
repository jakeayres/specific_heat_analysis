import streamlit as st


def main():

    st.title('Runlist Calculator')

    st.error('Under construction')

    st.write('A tool for producing a list of measurement parameters for the long relaxation experiment.')
    st.code('Outputs: Low and high currents for a desired temperature rise, aquisition time for each relaxation, expected preamplifier gain, number of repeats etc', language=None, wrap_lines=True)


    st.write('Get thermal conductivity by appling incrementally large heating powers above a base temperature. Take a derivative to get collapse.')
    st.write(r'$\Delta T = \frac{P}{\kappa_B}$')

    st.write('Get time directly from a relaxation')
    st.write(r'$T(t) = T_B + \Delta T (1 - e^{-t/\tau_B})$')
    st.write(r'$T(t) = T_B + \Delta Te^{-t/\tau_B}$')


if __name__ == "__main__":
    main()