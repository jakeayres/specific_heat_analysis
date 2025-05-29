import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from collections import defaultdict


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
        st.title('Upload raw data')
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


def get_calibration():
    try:
        st.divider()
        st.title('Upload barechip calibration')
        filepath = st.file_uploader("Select a barechip calibration")

        if filepath is not None:
            st.info('Using selected calibration')
            calibration = pd.read_csv(filepath)
            st.session_state['barechip_calibration'] = calibration
            fig, ax = plt.subplots(1, 1, figsize=(10, 3))
            ax.plot(calibration['temperature'], calibration['resistance'], '.')
            st.pyplot(fig)
        elif st.session_state['barechip_calibration'] is not None:
            st.info('Getting last used calibration')
            calibration = st.session_state['barechip_calibration']
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


def get_temperatures_from_filenames(filenames):
    try:
        temperatures = []
        for n, filename in enumerate(filenames):
            stem = filename.name.split('_')[0]
            temperatures.append(stem)
        temperatures = list(set(temperatures))
        st.info(f'{len(temperatures)} Temperatures found: {", ".join(temperatures)}')
        return temperatures
    except Exception as e:
        st.error(e)
        raise e


def group_files_by_run_number(filenames, dfs):
    try:
        groups = defaultdict(list)
        for n, filename in enumerate(filenames):
            repeat = filename.name.split('n_')[1].split('_cal')[0]
            repeat = int(repeat)
            groups[repeat].append(dfs[n])

        complete = [group for k, group in groups.items() if len(group) == 4]
        incomplete = [group for k, group in groups.items() if len(group) != 4]
        if len(incomplete) > 0:
            st.warning(f'{len(incomplete)} Incomplete Sets of sweeps found')
        return groups
    except Exception as e:
        st.error(e)
        raise e


def group_files_by_run_and_temperature(temperatures, filenames, dfs):

    try:
        temperature_groups = {}

        for temperature in temperatures:

            st.subheader(temperature)

            temperature_filenames = []
            temperature_dataframes = []
            for filename, df in zip(filenames, dfs):
                if temperature in filename.name:
                    temperature_filenames.append(filename)
                    temperature_dataframes.append(df)

            grouped_files = group_files_by_run_number(temperature_filenames, temperature_dataframes)

            ordered_groups = {}
            for i, group_dfs in grouped_files.items():
                ordered_dfs = order_dataframes(group_dfs)
                ordered_groups[i] = ordered_dfs

                fig, ax = plt.subplots(1, 4, figsize=(10, 2.5))
                ax[0].set_ylabel('Voltage (V)')
                ax[0].set_xlabel('Time (s)')
                ax[1].set_xlabel('Time (s)')
                ax[2].set_xlabel('Time (s)')
                ax[3].set_xlabel('Time (s)')
                ax[0].plot(ordered_dfs[0]['time'], ordered_dfs[0]['voltage'], color='tab:blue')
                ax[1].plot(ordered_dfs[1]['time'], ordered_dfs[1]['voltage'], color='tab:orange')
                ax[2].plot(ordered_dfs[2]['time'], ordered_dfs[2]['voltage'], color='tab:green')
                ax[3].plot(ordered_dfs[3]['time'], ordered_dfs[3]['voltage'], color='tab:red')
                ax[0].set_title(f'Run {i}: Warming +')
                ax[1].set_title(f'Run {i}: Cooling +')
                ax[2].set_title(f'Run {i}: Warming -')
                ax[3].set_title(f'Run {i}: Cooling -')
                fig.tight_layout()
                st.pyplot(fig)

            temperature_groups[temperature] = ordered_groups    

        return temperature_groups

    except Exception as e:
        st.error(e)
        raise e



def process_and_filter_sweeps(temperature_groups, calibration):

    try:

        st.divider()
        st.title('Process and Filter Sweeps')


        processed_sweeps = {k: None for k, _ in temperature_groups.items()}

        for temperature, temperature_group in temperature_groups.items():

            averaged_list = []

            for i, group in temperature_group.items():
                cols = st.columns(2, gap='medium')
                with cols[0]:
                    st.subheader(f'{temperature}: Run {i}')

                    # Average positive/negative current
                    averaged = compute_resistances(*group)

                    # Attempt to automatically find rows to skip
                    warming_end_resistance = averaged.iloc[-20:]['warming_resistance'].mean()
                    first_index = (averaged['cooling_resistance'] < warming_end_resistance).idxmax() + 10

                    skip_rows = st.number_input('Skip starting rows', value=first_index, min_value=0, max_value=1000, step=1, key=f'{temperature}_{i}_skip_rows_input')
                    window_warming = st.number_input('Savgol filter width warming', value=100, step=1, key=f'{temperature}_{i}_warming_window_input')
                    window_cooling = st.number_input('Savgol filter width cooling', value=100, step=1, key=f'{temperature}_{i}_cooling_windo_input')

                with cols[1]:

                    # Average positive/negative current
                    averaged = compute_resistances(*group)
                    averaged = averaged.iloc[skip_rows:]

                    # Compute temperature from resistance
                    averaged = compute_temperature(averaged, calibration)
                    averaged_list.append(averaged)

                    # Make figure
                    fig, axes = plt.subplots(2, 1, figsize=(6, 6))

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

                    fig.tight_layout()
                    st.pyplot(fig)

            processed_sweeps[temperature] = averaged_list

        return processed_sweeps

    except Exception as e:
        st.error(e)
        return 0



def perform_chebychev_fit(calibration):
    try:
        cheb_fit_R_to_T, cheb_fit_T_to_R = fit_chebyshev_polynomials(calibration, degree=5)
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

    except Exception as e:
        st.error(e)
        return 0

    return cheb_fit_R_to_T, cheb_fit_T_to_R


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

    # PRINT FRONT MATTER
    try:
        st.title('Long Relaxation Analysis')
        st.write('Analysis tool for obtaining the temperature dependent specific heat.')
        st.info('Inputs: Voltage vs time during relaxation sweeps.')
        st.info('Outputs: Heat capacity vs temperature')
    except Exception as e:
        st.error(e)


    # Get barechip calibration file from user and fit chebychev
    try:
        calibration = get_calibration()
        cheb_fit_R_to_T, cheb_fit_T_to_R = perform_chebychev_fit(calibration)
    except Exception as e:
        return -1


    # Get data files from user
    try:
        filenames, dfs = get_files()
    except Exception as e:
        st.error(e)
        return -2


    # Check for multiple of 4 files
    try:
        if len(dfs)%4 == 0:
            pass
        else:
            st.error('Expecting multiple of 4 data files')
            return 0
    except Exception as e:
        st.error(e)
        return -3


    # Group the files by run number and temperature. Order them warming+, cooling+, warming-, cooling-
    try:
        st.divider()
        temperatures = get_temperatures_from_filenames(filenames)
        temperature_groups = group_files_by_run_and_temperature(temperatures, filenames, dfs)

    except Exception as e:
        st.error(e)
        return -4


    try:
        processed_sweeps = process_and_filter_sweeps(temperature_groups, calibration)
    except Exception as e:
        st.error(e)
        return -5



    try:

        st.title('Temperature Interpolation')

        final_fig, final_axes = plt.subplots(figsize=(10, 4))
        final_dataframes = []

        for temperature, dataframes in processed_sweeps.items():
        
            for i, averaged in enumerate(dataframes):

                cols = st.columns(2, gap='medium')

                with cols[0]:
                    st.subheader(f'{temperature}: Run {i}')
                    temperature_window = st.number_input('Temperature interpolation window', value=0.02, step=0.0001, key=f'{temperature}_{i}_temperature_window_input')
                    ignore_ends = st.number_input('Ignore end temperatures', value=2, step=1, key=f'{temperature}_{i}_temperature_ignore_input')


                with cols[1]:
                    # Compute derivatives
                    averaged = compute_temperature_derivative(averaged, window_warming=st.session_state[f'{temperature}_{i}_warming_window_input'], window_cooling=st.session_state[f'{temperature}_{i}_warming_window_input'])

                    fig, axes = plt.subplots(figsize=(5, 4))
                    axes.plot(averaged['warming_temperature'], averaged['warming_derivative'], '-', linewidth=0.75, c='red')
                    axes.plot(averaged['cooling_temperature'], averaged['cooling_derivative'], '-', linewidth=0.75, c='blue')

                    # Compute heat capacity
                    cp_data = compute_heat_capacity(averaged, temperature_window=temperature_window, ignore_ends=ignore_ends)
                    final_dataframes.append(cp_data)
                    axes.plot(cp_data['temperature'], cp_data['warming_derivatives'], '.')
                    axes.plot(cp_data['temperature'], cp_data['cooling_derivatives'], '.')
                    axes.set_xlabel('Temperature (K)')
                    axes.set_ylabel('dT/dt (K/s)')

                    final_axes.plot(cp_data['temperature'], cp_data['heat_capacity'], '.')
                    final_axes.set_xlabel('Temperature (K)')
                    final_axes.set_xlabel('Temperature (K)')
                    final_axes.set_ylabel('Heat Capacity (J/K)')

                    fig.tight_layout()
                    st.pyplot(fig)

        st.pyplot(final_fig)

    except Exception as e:
        st.error(e)
        return 0


    try:
        st.divider()
        st.title('Save Data')
        
        if len(final_dataframes) > 1:
            data = pd.concat(final_dataframes)
        else:
            data = final_dataframes

        st.download_button(
            "Save All Data",
            convert_df(data),
            mime="text/csv",
            key='download-csv',
            type="primary"
        )

    except Exception as e:
        st.error(e)
        return 0



if __name__ == "__main__":
    main()