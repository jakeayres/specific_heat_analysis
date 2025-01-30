import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatter 
import numpy as np
from scipy.optimize import curve_fit

from src.chebychev_fitting import fit_chebyshev_polynomials, resistance_to_temperature, temperature_to_resistance


def fit_resistances(x, y):
    f = lambda x, a, b, c: a + b*x*x + c*x*x
    p0 = [300, -1]
    popt, _ = curve_fit(f, x, y)
    return f, popt


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def main():

    st.title('Barechip Calibration')
    st.write('A utility for producing the barechip calibration files from current dependent resistance data.')
    st.code('Inputs: Resistance vs current of a barechip cernox at fixed temperatures. Expects columnar data with "setpoint", "temperature", "current" and "resistance" columns. ', language=None, wrap_lines=True)
    st.code('Outputs: Resistance vs temperature in the absence of self heating. A barechip calibration file with "temperature" and "resistance" columns', language=None, wrap_lines=True)


    #
    # Read in raw data into a list of dataframes
    #
    try:
        st.divider()
        st.header('Upload raw data')
        filepaths = st.file_uploader("Select a datafile", accept_multiple_files=True)
        if len(filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in filepaths]
            st.success(f'{len(filepaths)} Uploaded')
        else:
            st.warning(f'No data yet loaded')
            return 0
    except Exception as e:
        st.error(e)
        return 0

    #
    # Build a list of data groups at each temperature (setpoint, not measured temperature)
    # as well as a list of colors for ease of visualization
    #
    try:
        st.divider()  
        st.header('Raw data')
        groups = [df.groupby('setpoint') for df in dfs]
        cmap = plt.get_cmap("tab10")
        colors = [[cmap(i)]*int(len(groups[i])) for i in range(len(groups))]
        for group in groups:
            st.write(len(group), 'temperatures in file between', group.size().index.min(), 'and', group.size().index.max())
        groups = [group for group in groups for group in group]
        colors = [color for color in colors for color in color]
    except Exception as e:
        st.error(e)
        return 0


    #
    #   Plot out the raw resistance vs current at each temperature and 
    #   make fits to find zero current resistances
    #
    try:
        resistances = []
        temperatures = []

        rows = (len(groups)+2)//3
        fig, axes = plt.subplots(rows, 3, figsize=(10, 2*rows))
        #fig.suptitle(r'Resistance ($\Omega$) vs current ($\mathrm{\mu A}$)'+'\n', fontweight='bold', fontsize=15)
        axes = axes.flatten()

        for i, (temperature, data) in enumerate(groups):
            ax = axes[i]
            ax.plot(data['current']*1e6, data['resistance'], '.', color=colors[i])
            ax.text(0.05, 0.05, f'{temperature:.3}K', transform=ax.transAxes, fontweight='bold')
            ax.grid(True, which="both", linestyle="--", linewidth=0.5)

            f, popt = fit_resistances(data['current']*1e6, data['resistance'])
            x = np.linspace(0, data['current'].max()*1e6, 50)
            ax.plot(x, f(x, *popt), 'k--', linewidth=0.5)
            temperatures.append(temperature)
            resistances.append(popt[0])

        calibration = pd.DataFrame(data={
            'temperature': temperatures,
            'resistance': resistances,
        })


        fig.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(e)
        return 0


    #
    #   Plot out the calibration and do chebychev fitting
    #
    try:
        st.divider()  
        st.header('Calibration')
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].scatter(calibration['temperature'], calibration['resistance'], color=colors, marker='.')
        axes[1].scatter(calibration['temperature'], calibration['resistance'], color=colors, marker='.')

        cheb_fit_R_to_T, cheb_fit_T_to_R = fit_chebyshev_polynomials(calibration, degree=5)
        x = np.linspace(calibration['temperature'].min(), calibration['temperature'].max(), 50)
        axes[0].plot(x, temperature_to_resistance(x, cheb_fit_T_to_R), 'k--', linewidth=0.5)
        axes[1].plot(x, temperature_to_resistance(x, cheb_fit_T_to_R), 'k--', linewidth=0.5)

        axes[0].grid(True, which="both", linestyle="--", linewidth=0.5)

        axes[1].set_xscale("log")
        axes[1].set_yscale("log")
        axes[1].xaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
        axes[1].yaxis.set_major_formatter(LogFormatter(labelOnlyBase=False))
        axes[1].grid(True, which="both", linestyle="--", linewidth=0.5)

        fig.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.error(e)
        return 0

    #
    #   Play with or save the calibration
    #
    try:
        columns = st.columns(2, gap='large')

        with columns[0]:
            st.subheader('T to R')
            user_temperature = st.number_input('Temperature')
            if st.button('Convert T2R'):
                st.write(temperature_to_resistance(user_temperature, cheb_fit_T_to_R))

        with columns[1]:
            st.subheader('R to T')
            user_resistance = st.number_input('Resistance')
            if st.button('Convert R2T'):
                st.write(resistance_to_temperature(user_resistance, cheb_fit_R_to_T))

        st.subheader('Save calibration')
        user_filename = st.text_input('Filename', value='calibration.csv')
        st.download_button(
           "Press to Download",
           convert_df(calibration),
           f"{user_filename}",
           "text/csv",
           key='download-csv'
        )


    except Exception as e:
        st.error(e)
        return 0



if __name__ == "__main__":
    main()