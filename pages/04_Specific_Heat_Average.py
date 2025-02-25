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


