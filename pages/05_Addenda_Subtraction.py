import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy

def get_addenda_files():
    try:
        st.divider()
        st.header('Addenda Data')
        addenda_filepaths = st.file_uploader("Select Addenda Datafiles", accept_multiple_files=True)
        if len(addenda_filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in addenda_filepaths]
            st.success(f'{len(addenda_filepaths)} Uploaded')
            return dfs
        else:
            st.warning(f'No data yet loaded')
            return None
    except Exception as e:
        st.error(e)
        return None


def get_sample_files():
    try:
        st.divider()
        st.header('Sample Data')
        sample_filepaths = st.file_uploader("Select Sampl Datafiles", accept_multiple_files=True)
        if len(sample_filepaths) > 0:
            dfs = [pd.read_csv(filepath) for filepath in sample_filepaths]
            st.success(f'{len(sample_filepaths)} Uploaded')
            return dfs
        else:
            st.warning(f'No data yet loaded')
            return None
    except Exception as e:
        st.error(e)
        return None


def main():

    try:
        st.title('Addenda Subtraction')
        st.write('Take addenda and sample C(T) data and perform the background subtraction.')
        st.info('Input 1: Addenda heat capacity vs temperature (multiple files allowed)')
        st.info('Input 2: Addenda + Sample heat capacity vs temperature (multiple files allowed)')
        st.info('Outputs: Sample heat capacity vs temperature (single averaged file)')


        cols = st.columns(2)

        with cols[0]:

            try:
                addenda_dfs = get_addenda_files()
            except Exception as e:
                st.error('Could not get files')
                st.error(e)


        with cols[1]:

            try:
                sample_dfs = get_sample_files()
            except Exception as e:
                st.error('Could not get files')
                st.error(e)


        fig, ax = plt.subplots()
        
        try:
            if addenda_dfs is not None:
                for df in addenda_dfs:
                    ax.plot(df['temperature'], df['heat_capacity'], '.', alpha=0.25, markersize=10, markerfacecolor='none')
        except Exception as e:
            st.error('Failed to plot addenda data')
            st.error(e)


        try:
            if sample_dfs is not None:
                for df in sample_dfs:
                    ax.plot(df['temperature'], df['heat_capacity'], '.', alpha=0.25, markersize=10, markerfacecolor='none')
        except Exception as e:
            st.error('Failed to plot sample data')
            st.error(e)



        st.pyplot(fig)



    except Exception as e:
        st.error(e)
        raise(e)


if __name__ == "__main__":
    main()
