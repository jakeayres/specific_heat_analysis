import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx


def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def bin_dataframe(df, bin_width):
    bin_edges = np.arange(df['temperature'].min(), df['temperature'].max() + bin_width, bin_width)  # Bins from min(x) to max(x)
    # Bin x values
    df['bin_temperature'] = pd.cut(df['temperature'], bins=bin_edges, include_lowest=True)

    # Aggregate y-values in each bin (e.g., mean y-value per x-bin)
    binned_df = df.groupby('bin_temperature').mean().reset_index()
    return binned_df


def get_files():
    try:
        st.divider()
        st.header('Upload raw data')
        filepaths = st.file_uploader("Select datafiles", accept_multiple_files=True)
        if len(filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in filepaths]
            st.success(f'{len(filepaths)} Uploaded')
            return dfs
        else:
            st.warning(f'No data yet loaded')
            return 0
    except Exception as e:
        st.error(e)
        return None

def main():

    try:
        st.title('Heat Capacity Averaging')
        st.write('Average the computed specific heat for each sweep.')
        st.code('Inputs: Heat capacity vs temperature (multiple files)', language=None, wrap_lines=True)
        st.code('Outputs: Heat capacity vs temperature (single averaged file)', language=None, wrap_lines=True)

        try:
            dfs = get_files()
        except Exception as e:
            st.error('Could not get files')
            st.error(e)


        try:

            bin_width = st.number_input('Bin width', value=0.1, step=1e-6)

            fig, ax = plt.subplots()

            for df in dfs:
                ax.plot(df['temperature'], df['heat_capacity'], '.', alpha=0.25, markersize=10, markerfacecolor='none')

            total_df = pd.concat(dfs)

            total_df = bin_dataframe(total_df, bin_width)

            ax.plot(total_df['temperature'], total_df['heat_capacity'], 'k.', markersize=3.0)

            st.pyplot(fig)
        except Exception as e:
            st.error('Failed to plot')
            st.error(e)

    except Exception as e:
        st.error(e)
        raise(e)


if __name__ == "__main__":
    main()



# st.title('Average specific heat')
# st.subheader('Average c(T) curves (for one calorimeter, one T range)')
# filepaths_avg = st.file_uploader('Upload files', accept_multiple_files=True)
# if len(filepaths_avg) > 0:
#     dfs = [pd.read_csv(filepath) for filepath in filepaths_avg]
#     dfs = order_dataframes(dfs)
#     st.success(f'{len(filepaths_avg)} files uploaded and averaged!')
# else:
#   st.write("Error with file input.")

# st.divider()

# all_T_mins = np.zeros(len(dfs))
# all_T_maxs = np.zeros(len(dfs))

# '''Find min and max of common T range'''
# for i in range(0, len(dfs)):

#   all_T_mins[i] = np.min(dfs[i]['temperature'])
#   all_T_maxs[i] = np.max(dfs[i]['temperature'])

# T_min = np.max(all_T_mins)
# T_max = np.min(all_T_maxs)

# T_matched = []
# C_matched = []
# problem_indice = 100 # If no problem, use 100
# '''Match endpoints (and therefore lengths of arrays)'''
# for i in range(0, len(dfs)):

#   df_start_T, df_start_idx = find_nearest(dfs[i]['temperature'], T_min)
#   df_end_T, df_end_idx = find_nearest(dfs[i]['temperature'], T_max)

#   '''Added this in case of some weird non-matching array problem (i.e. len i0 and i2 = 99 but len i1 = 100)'''
#   if i == problem_indice:
#       T_matched.append(np.array(dfs[i]['temperature'][df_start_idx:df_end_idx-1]))
#       C_matched.append(np.array(dfs[i]['heat_capacity'][df_start_idx:df_end_idx-1]))

#   else:
#       T_matched.append(np.array(dfs[i]['temperature'][df_start_idx:df_end_idx]))
#       C_matched.append(np.array(dfs[i]['heat_capacity'][df_start_idx:df_end_idx]))

# C_avg = np.zeros(len(C_matched[0]))
# for i in range(0, len(dfs)):
#   C_avg += np.array(C_matched[i])

# C_avg = C_avg/len(dfs)
# specific_heat = pd.DataFrame({'temperature': T_matched[0], 'heat_capacity': C_avg})

# fig, axes = plt.subplots(1, 1, figsize=(10, 4))
# axes.scatter(specific_heat['temperature'], specific_heat['heat_capacity'], linewidth=0.75, c='blue',s=5, label='Average')
# axes.scatter(specific_heat['temperature'], C_matched[0], linewidth=0.75, c='red',s=5, alpha=0.25, label='Repeat 0')
# axes.set_xlabel("T (K)")
# axes.set_ylabel("C (J/K)")
# plt.legend()
# st.pyplot(fig)

# st.divider()
# st.title('Save Data')
# cols = st.columns(2)
# with cols[0]:
#     st.subheader('Save to new file')
#     st.download_button(
#        "Save Data",
#        convert_df(specific_heat),
#        mime="text/csv",
#        key='download-csv',
#        type="primary"
#     )