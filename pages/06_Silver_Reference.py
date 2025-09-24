import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def read_silver_reference():
    PATH = 'example_data/reference_silver/silver.data'
    df = pd.read_csv(PATH, sep='\t', names=['temperature', 'specific_heat', 'H', 'G', 'I', 'J', 'L'], skiprows=1, comment='#')
    return df


def get_files():
    try:
        st.divider()
        st.title('Upload heat capacity data')
        filepaths = st.file_uploader("Select datafiles", accept_multiple_files=True)
        if len(filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in filepaths]
            st.success(f'{len(filepaths)} Uploaded')
        else:
            st.warning(f'No data yet loaded')
            return 0
    except Exception as e:
        st.error(e)
        raise e
    return filepaths, dfs


def main():

    st.title('Silver Reference Calibration')

    silver_data = read_silver_reference()

    st.write('# Reference Data')

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(silver_data['temperature'], silver_data['specific_heat'], 'k-')
    ax[0].set_xlabel('Temperature (K)')
    ax[0].set_ylabel('Specific Heat (J/K/mol)')
    ax[0].set_xlim([0, 25])
    ax[0].set_ylim([0, 4])

    ax[1].plot(silver_data['temperature']**2, silver_data['specific_heat']/silver_data['temperature'], 'k-')
    ax[1].set_xlabel('Temperature^2 (K^2)')
    ax[1].set_ylabel('C/T (J/K$^2$/mol)')
    ax[1].set_xlim([0, 25*25])
    ax[1].set_ylim([0, 4/25])

    fig.tight_layout()
    st.pyplot(fig)


    # Get data files from user
    try:
        filenames, dfs = get_files()
        mass = st.number_input('Mass of silver reference (ug)', value=1000.0, step=0.01)
        mr_silver = 107.8682
        moles = mass*1e-6/mr_silver
        st.write(f'Moles of silver: {moles:.6f} mol')
    except Exception as e:
        st.error(e)
        return -2
    

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(silver_data['temperature'], silver_data['specific_heat'], 'k-')
    ax[0].set_xlabel('Temperature (K)')
    ax[0].set_ylabel('Specific Heat (J/K/mol)')
    ax[0].set_xlim([0, 25])
    ax[0].set_ylim([0, 4])

    for df in dfs:
        ax[0].plot(df['temperature'], df['heat_capacity']/moles, '.')

    st.pyplot(fig)






if __name__ == "__main__":
    main()