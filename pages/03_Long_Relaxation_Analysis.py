import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from src.chebychev_fitting import fit_chebyshev_polynomials, resistance_to_temperature, temperature_to_resistance


def get_files():
    try:
        st.divider()
        st.header('Upload raw data')
        filepaths = st.file_uploader("Select a datafile", accept_multiple_files=True)[::-1]
        if len(filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in filepaths]
            st.success(f'{len(filepaths)} Uploaded')
            fig, ax = plt.subplots(1, 4, figsize=(10, 3))
            ax[0].set_ylabel('Voltage (V)')
            ax[0].set_xlabel('Time (s)')
            ax[1].set_xlabel('Time (s)')
            ax[2].set_xlabel('Time (s)')
            ax[3].set_xlabel('Time (s)')
            ax[0].plot(dfs[0]['time'], dfs[0]['voltage'], color='tab:blue')
            ax[1].plot(dfs[1]['time'], dfs[1]['voltage'], color='tab:orange')
            ax[2].plot(dfs[2]['time'], dfs[2]['voltage'], color='tab:green')
            ax[3].plot(dfs[3]['time'], dfs[3]['voltage'], color='tab:red')
            ax[0].set_title('Warming +')
            ax[1].set_title('Cooling +')
            ax[2].set_title('Warming -')
            ax[3].set_title('Cooling -')
            fig.tight_layout()
            st.pyplot(fig)

        else:
            st.warning(f'No data yet loaded')
            return 0
    except Exception as e:
        st.error(e)
        raise e
    return dfs


def get_calibration():
    try:
        st.divider()
        st.header('Upload barechip calibration')
        filepath = st.file_uploader("Select a barechip calibration")
        if filepath is not None:
            calibration = pd.read_csv(filepath)
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.plot(calibration['temperature'], calibration['resistance'], '.')
            st.pyplot(fig)
    except Exception as e:
        st.error(e)
        raise e
    return calibration


def compute_resistances(
    positive_warming, 
    positive_cooling, 
    negative_warming, 
    negative_cooling
    ):
    try:
        time = positive_warming['time']
        warming_voltage = (positive_warming['voltage']-negative_warming['voltage'])/2
        cooling_voltage = (positive_cooling['voltage']-negative_cooling['voltage'])/2
        warming_resistance = (warming_voltage/positive_warming['gain'])/(negative_warming['current'])
        cooling_resistance = (cooling_voltage/positive_cooling['gain'])/(negative_cooling['current'])
    except Exception as e:
        st.error(e)
        raise e
    return pd.DataFrame(data={
        'time': time,
        'warming_voltage': warming_voltage,
        'cooling_voltage': cooling_voltage,
        'warming_resistance': warming_resistance,
        'cooling_resistance': cooling_resistance,
        'warming_current': positive_warming['current'],
        'cooling_current': positive_cooling['current']
        })


def compute_temperature(data, calibration):
    try:
        cheb_fit_R_to_T, cheb_fit_T_to_R = fit_chebyshev_polynomials(calibration, degree=5)
        data['warming_temperature'] = resistance_to_temperature(data['warming_resistance'], cheb_fit_R_to_T)
        data['cooling_temperature'] = resistance_to_temperature(data['cooling_resistance'], cheb_fit_R_to_T)
    except Exception as e:
        st.error(e)
        raise e
    return data


def perform_filtering(data, window):
    try:
        time_step = data['time'].diff().mean()
        data['warming_temperature'] = scipy.signal.savgol_filter(
            data['warming_temperature'], window, polyorder=2, deriv=0, delta=time_step, mode='nearest')
        data['cooling_temperature'] = scipy.signal.savgol_filter(
            data['cooling_temperature'], window, polyorder=2, deriv=0, delta=time_step, mode='nearest')
    except Exception as e:
        st.error(e)
        raise e
    return data


def compute_temperature_derivative(data, window):
    try:
        time_step = data['time'].diff().mean()
        data['warming_derivative'] = scipy.signal.savgol_filter(
            data['warming_temperature'], window, polyorder=2, deriv=1, delta=time_step, mode='nearest')
        data['cooling_derivative'] = scipy.signal.savgol_filter(
            data['cooling_temperature'], window, polyorder=2, deriv=1, delta=time_step, mode='nearest')
    except Exception as e:
        st.error(e)
        raise e
    return data


def compute_heat_capacity(data, temperature_window=0.1, ignore_ends=2):
    try:
        min_temperature = data[["warming_temperature", "cooling_temperature"]].min().min()
        max_temperature = data[["warming_temperature", "cooling_temperature"]].max().max()
        temperatures = np.arange(min_temperature, max_temperature, temperature_window)
        temperatures = temperatures[ignore_ends:-ignore_ends]

        warming_derivatives = []
        cooling_derivatives = []
        warming_powers = []
        cooling_powers = []
        heat_capacities = []

        for T in temperatures:
            warming_data = data.where(np.abs(data['warming_temperature']-T) < temperature_window).dropna()
            coeffs = np.polyfit(warming_data["warming_temperature"], warming_data["warming_derivative"], deg=2)
            warming_derivative = np.poly1d(coeffs)(T)
            warming_derivatives.append(warming_derivative)

            coeffs = np.polyfit(warming_data["warming_temperature"], warming_data["warming_resistance"], deg=2)
            warming_resistance = np.poly1d(coeffs)(T)
            warming_power = np.power(warming_data['warming_current'].mean(), 2) * warming_resistance
            warming_powers.append(warming_power)

            cooling_data = data.where(np.abs(data['cooling_temperature']-T) < temperature_window).dropna()
            coeffs = np.polyfit(cooling_data["cooling_temperature"], cooling_data["cooling_derivative"], deg=2)
            cooling_derivative = np.poly1d(coeffs)(T)
            cooling_derivatives.append(cooling_derivative)

            coeffs = np.polyfit(cooling_data["cooling_temperature"], cooling_data["cooling_resistance"], deg=2)
            cooling_resistance = np.poly1d(coeffs)(T)
            cooling_power = np.power(cooling_data['cooling_current'].mean(), 2) * cooling_resistance
            cooling_powers.append(cooling_power)

            heat_capacity = (warming_power - cooling_power) / (warming_derivative - cooling_derivative)
            heat_capacities.append(heat_capacity)

    except Exception as e:
        st.error(e)
        raise e
    return pd.DataFrame(data={
        'temperature': [T for T in temperatures],
        'warming_derivatives': warming_derivatives,
        'cooling_derivatives': cooling_derivatives,
        'heat_capacity': heat_capacities,
        })


def main():

    try:
        st.title('Long Relaxation Analysis')
        st.write('Analysis tool for obtaining the temperature dependent specific heat.')
        st.code('Inputs: Voltage vs time during relaxation sweeps.', language=None, wrap_lines=True)
        st.code('Outputs: Heat capacity vs temperature', language=None, wrap_lines=True)

        dfs = get_files()

        calibration = get_calibration()

        C0_SR_PR = dfs[0] 
        C0_SR_PF = dfs[1]
        C0_SR_NR = dfs[2]
        C0_SR_NF = dfs[3]


        st.divider()
        st.title('Analyse Sweep')

        cols = st.columns(2)

        st.subheader('Filtering')
        skip_rows = st.number_input('Skip starting rows', value=0, min_value=0, max_value=1000, step=1)
        window_size = st.number_input('Savgol filter width', value=100, step=1)


        # Make figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Average positive/negative current
        averaged = compute_resistances(C0_SR_PR, C0_SR_PF, C0_SR_NR, C0_SR_NF)
        averaged = averaged.iloc[skip_rows:]

        # Compute temperature from resistance
        averaged = compute_temperature(averaged, calibration)

        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Resistance (Ohms)')
        axes[0].plot(averaged['time'], averaged['warming_resistance'])
        axes[0].plot(averaged['time'], averaged['cooling_resistance'])

        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].plot(averaged['time'], averaged['warming_temperature'])
        axes[1].plot(averaged['time'], averaged['cooling_temperature'])

        # Perform filtering
        averaged = perform_filtering(averaged, window=window_size)
        axes[1].plot(averaged['time'], averaged['warming_temperature'], 'k-', linewidth=0.75)
        axes[1].plot(averaged['time'], averaged['cooling_temperature'], 'k-', linewidth=0.75)

        st.pyplot(fig)

        # Compute derivatives
        averaged = compute_temperature_derivative(averaged, window=window_size)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(averaged['warming_temperature'], averaged['warming_derivative'], '-', linewidth=0.75)
        axes[0].plot(averaged['cooling_temperature'], averaged['cooling_derivative'], '-', linewidth=0.75)


        st.subheader('Temperature interpolation')
        temperature_window = st.number_input('Temperature interpolation window', value=0.1, step=0.0001)
        ignore_ends = st.number_input('Ignore end temperatures', value=2, step=1)

        # Compute heat capacity
        cp = compute_heat_capacity(averaged, temperature_window=temperature_window)
        axes[0].plot(cp['temperature'], cp['warming_derivatives'], '.')
        axes[0].plot(cp['temperature'], cp['cooling_derivatives'], '.')
        axes[1].plot(cp['temperature'], cp['heat_capacity'], '.')
        axes[0].set_xlabel('Temperature (K)')
        axes[0].set_ylabel('dT/dt (K/s)')
        axes[1].set_xlabel('Temperature (K)')
        axes[1].set_xlabel('Temperature (K)')
        axes[1].set_ylabel('Heat Capacity (J/K)')
        st.pyplot(fig)


    except Exception as e:
        st.error(e)
        return 0


if __name__ == "__main__":
    main()