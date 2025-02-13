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


def order_dataframes(dfs):
    """
    Orders dataframes [temperature, c, warming + , cooling -, run]
    """
    means = [df['temperature'].mean() for df in dfs]
    sorted_indices = sorted(range(len(means)), key=lambda i: means[i], reverse=True)
  
    return [dfs[i] for i in sorted_indices]
    st.write(sorted_indices)

st.title('Average specific heat')
st.subheader('Average c(T) curves (for one calorimeter, one T range)')
filepaths_avg = st.file_uploader('Upload files to be appended', accept_multiple_files=True)
if len(filepaths_avg) > 0:
    dfs = [pd.read_csv(filepath) for filepath in filepaths_avg]
    dfs = order_dataframes(dfs)

    st.success(f'{len(filepaths_avg)} Files uploaded!')
else:
	st.warning("Please input file! Pleaseeeeeee!")


df_combined = np.concat(dfs, axis=0)
specific_heat = pd.DataFrame({'temperature': df_combined[:,0], 'heat_capacity': df_combined[:,1]})


fig, axes = plt.subplots(1, 1, figsize=(10, 4))
axes.scatter(specific_heat['temperature'], specific_heat['heat_capacity'], linewidth=0.75, c='blue',s=5, label='Average')
axes.set_xlabel("T (K)")
axes.set_ylabel("C (J/K)")
plt.legend()
st.pyplot(fig)