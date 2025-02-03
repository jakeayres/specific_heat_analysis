import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

from src.chebychev_fitting import fit_chebyshev_polynomials, resistance_to_temperature, temperature_to_resistance


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def order_dataframes(dfs):
    """
    Orders dataframes [warming +, cooling +, warming -, cooling -]
    """
    means = [df['voltage'].mean() for df in dfs]
    sorted_indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
    custom_order = [sorted_indices[0], sorted_indices[1], sorted_indices[3], sorted_indices[2]]
    return [dfs[i] for i in custom_order]



def get_files():
    try:
        st.divider()
        st.header('Upload raw data')
        filepaths = st.file_uploader("Select datafiles", accept_multiple_files=True)
        if len(filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in filepaths]
            dfs = order_dataframes(dfs)
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
        else:
            st.warning('Upload calibration to perform analysis')
            return None
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
        warming_resistance = np.abs((warming_voltage/positive_warming['gain'])/(negative_warming['current']))
        cooling_resistance = np.abs((cooling_voltage/positive_cooling['gain'])/(negative_cooling['current']))
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


def perform_filtering(data, window_warming, window_cooling):
    try:
        time_step = data['time'].diff().mean()
        data['warming_temperature'] = scipy.signal.savgol_filter(
            data['warming_temperature'], window_warming, polyorder=2, deriv=0, delta=time_step, mode='nearest')
        data['cooling_temperature'] = scipy.signal.savgol_filter(
            data['cooling_temperature'], window_cooling, polyorder=2, deriv=0, delta=time_step, mode='nearest')
    except Exception as e:
        st.error(e)
        raise e
    return data


def compute_temperature_derivative(data, window_warming, window_cooling):
    try:
        time_step = data['time'].diff().mean()
        data['warming_derivative'] = scipy.signal.savgol_filter(
            data['warming_temperature'], window_warming, polyorder=2, deriv=1, delta=time_step, mode='nearest')
        data['cooling_derivative'] = scipy.signal.savgol_filter(
            data['cooling_temperature'], window_cooling, polyorder=2, deriv=1, delta=time_step, mode='nearest')
    except Exception as e:
        st.error(e)
        raise e
    return data


def compute_heat_capacity(data, temperature_window=0.1, ignore_ends=2):
    try:
        min_temperature = data[["warming_temperature", "cooling_temperature"]].min().max()
        max_temperature = data[["warming_temperature", "cooling_temperature"]].max().min()
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
        window_warming = st.number_input('Savgol filter width warming', value=100, step=1)
        window_cooling = st.number_input('Savgol filter width cooling', value=100, step=1)

        # Make figure
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Average positive/negative current
        averaged = compute_resistances(C0_SR_PR, C0_SR_PF, C0_SR_NR, C0_SR_NF)
        averaged = averaged.iloc[skip_rows:]



        # Compute temperature from resistance
        averaged = compute_temperature(averaged, calibration)

        st.write('From warming temperature: Start T ~', np.min(averaged['warming_temperature']), 'End T ~', np.max(averaged['warming_temperature']))
        st.write('Change in T ~', ((np.max(averaged['warming_temperature']) - np.min(averaged['warming_temperature']))/np.min(averaged['warming_temperature']))*100, "%")
        if  ((np.max(averaged['warming_temperature']) - np.min(averaged['warming_temperature']))/np.min(averaged['warming_temperature']))*100 < 30:
            st.write('OK: Change in T is within 30% of base temperature, acceptable T range')
        else:
            st.write('NO: Change in T is NOT within 30% of base temperature, CHANGE CURRENT')

        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Resistance (Ohms)')
        axes[0].plot(averaged['time'], averaged['warming_resistance'], label='Warming', c='red')
        axes[0].plot(averaged['time'], averaged['cooling_resistance'], label='Cooling', c='blue')
        axes[0].legend()

        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Temperature (K)')
        axes[1].plot(averaged['time'], averaged['warming_temperature'], label='Warming', c='red')
        axes[1].plot(averaged['time'], averaged['cooling_temperature'], label='Cooling', c='blue')
        axes[1].legend()

        # Perform filtering
        averaged = perform_filtering(averaged, window_warming=window_warming, window_cooling=window_cooling)
        axes[1].plot(averaged['time'], averaged['warming_temperature'], 'k-', linewidth=0.75, c='black')
        axes[1].plot(averaged['time'], averaged['cooling_temperature'], 'k-', linewidth=0.75, c='black')

        st.pyplot(fig)

        # Compute derivatives
        averaged = compute_temperature_derivative(averaged, window_warming=window_warming, window_cooling=window_cooling)

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(averaged['warming_temperature'], averaged['warming_derivative'], '-', linewidth=0.75, c='red')
        axes[0].plot(averaged['cooling_temperature'], averaged['cooling_derivative'], '-', linewidth=0.75, c='blue')



        st.subheader('Temperature interpolation')
        temperature_window = st.number_input('Temperature interpolation window', value=0.02, step=0.0001)
        ignore_ends = st.number_input('Ignore end temperatures', value=2, step=1)

        # Compute heat capacity
        cp_data = compute_heat_capacity(averaged, temperature_window=temperature_window)
        axes[0].plot(cp_data['temperature'], cp_data['warming_derivatives'], '.')
        axes[0].plot(cp_data['temperature'], cp_data['cooling_derivatives'], '.')
        axes[1].plot(cp_data['temperature'], cp_data['heat_capacity'], '.')
        axes[0].set_xlabel('Temperature (K)')
        axes[0].set_ylabel('dT/dt (K/s)')
        axes[1].set_xlabel('Temperature (K)')
        axes[1].set_xlabel('Temperature (K)')
        axes[1].set_ylabel('Heat Capacity (J/K)')
        st.pyplot(fig)


        try:
            st.divider()
            st.title('Save Data')
            cols = st.columns(2)
            with cols[0]:
                st.subheader('Save to new file')
                cp_data['run'] = 1
                st.download_button(
                   "Save Data",
                   convert_df(cp_data),
                   mime="text/csv",
                   key='download-csv',
                   type="primary"
                )
            with cols[1]:
                st.subheader('Append to existing file')
                existing_filename = st.file_uploader("Select a datafile")
                if st.button('Append data', type='primary'):
                    existing_df = pd.read_csv(existing_filename)
                    new_run_value = existing_df['run'].max() + 1 if 'run' in existing_df else 1
                    cp_data['run'] = new_run_value   
                    combined_df = pd.concat([existing_df, cp_data], ignore_index=True)
                    combined_df.to_csv(existing_filename, index=False)
                    st.success(f"Data has been successfully appended to {existing_filename.name}")
                else:
                    st.warning('Select file to append data to')


        except Exception as e:
            st.error(e)
            return 0


    except Exception as e:
        st.error(e)
        return 0


if __name__ == "__main__":
    main()