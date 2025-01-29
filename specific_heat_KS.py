#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 09:13:12 2025

@author: ks17567
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

# =============================================================================
# Bare chip calibration
# =============================================================================

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx

plt.close('all')

barechip_cali = pd.read_csv(
    'SpecificHeat/barechip_calibration.dat', delimiter='\t')

plt.figure()
plt.scatter(barechip_cali['temperature'],
            barechip_cali['resistance'], s=4, c='black')
plt.xlabel("Temperature ($K$)")
plt.ylabel("Resistance (Ohms)")

'''Get T from R'''
z = np.polynomial.polynomial.Polynomial.fit(
    barechip_cali['resistance'], barechip_cali['temperature'], 15)
fit_data = z(barechip_cali['resistance'])
R_range = np.linspace(barechip_cali['resistance'][0], barechip_cali['resistance'][len(
    barechip_cali['resistance'])-1], 10000)
plt.plot(z(R_range), R_range, c='green', label='RT Barechip calibration')
plt.legend()

# =============================================================================
# Calorimeter_0 Voltages
# =============================================================================
start_point = 25
C0_SR_NF = pd.read_csv(
    'SpecificHeat/calorimeter_0_single_relaxation_KS_negative_falling.dat')[start_point:]
C0_SR_PF = pd.read_csv(
    'SpecificHeat/calorimeter_0_single_relaxation_KS_positive_falling.dat')[start_point:]
C0_SR_NR = pd.read_csv(
    'SpecificHeat/calorimeter_0_single_relaxation_KS_negative_rising.dat')[start_point:]
C0_SR_PR = pd.read_csv(
    'SpecificHeat/calorimeter_0_single_relaxation_KS_positive_rising.dat')[start_point:]

time = C0_SR_NF['time']
# plt.figure()

C0_SR_rising_v = (C0_SR_PR['voltage']-C0_SR_NR['voltage'])/2
C0_SR_falling_v = (C0_SR_PF['voltage']-C0_SR_NF['voltage'])/2

'''divide this by a negative current for a +ve R'''
'''V/I = R, P = I^2R'''
C0_SR_rising_R = (C0_SR_rising_v/100)/(C0_SR_NR['current'])  # 100 is gain!
C0_SR_falling_R = (C0_SR_falling_v/100)/(C0_SR_NF['current'])
# plt.plot(time, C0_SR_NR['resistance'], label='Uncorrected Resistance', c='black', alpha=0.5)
# plt.plot(C0_SR_PR['time'], C0_SR_rising_R, label='Corrected Resistance', c='red')
# plt.plot(time, C0_SR_PF['resistance'], label='Uncorrected Resistance', c='black', alpha=0.5)
# plt.plot(C0_SR_PR['time'], C0_SR_falling_R, label='Corrected Resistance', c='blue')
# plt.legend()
# plt.show()
'''Convert from R to T'''
C0_SR_rising_T = z(C0_SR_rising_R)
C0_SR_falling_T = z(C0_SR_falling_R)
'''Savgol filter via window size'''
savgol_window1 = 1000
savgol_window2 = 1600
C0_SR_rising_T_filter = scipy.signal.savgol_filter(
    C0_SR_rising_T, savgol_window1, 2, deriv=0)
C0_SR_falling_T_filter = scipy.signal.savgol_filter(
    C0_SR_falling_T, savgol_window2, 2, deriv=0)

plt.figure()
plt.plot(time,
         C0_SR_falling_T, c='blue')
plt.plot(time,
         C0_SR_falling_T_filter, c='black')
plt.plot(time,
         C0_SR_rising_T, c='red')
plt.plot(time,
         C0_SR_rising_T_filter, c='black')
plt.xlabel("Time (s)")
plt.ylabel("T (K)")
plt.show()

del_R = time[101] - time[100]

C0_SR_rising_T_filter_d1 = scipy.signal.savgol_filter(
    C0_SR_rising_T_filter, savgol_window1, 2, deriv=1, delta=del_R)
C0_SR_falling_T_filter_d1 = scipy.signal.savgol_filter(
    C0_SR_falling_T_filter, savgol_window2, 2, deriv=1, delta=del_R)
plt.figure()
plt.plot(C0_SR_rising_T_filter, C0_SR_rising_T_filter_d1,
         c='red', label='dT/dt Rising')
plt.plot(C0_SR_falling_T_filter, C0_SR_falling_T_filter_d1,
         c='blue', label='dT/dt Falling')
plt.xlabel("Temperature (K)")
plt.ylabel("dT/dt")
plt.legend()
plt.show()


def CalculatePower(I, R_array):

    return np.array(I**2 * R_array)


Power_Rising_wrt_time = CalculatePower(
    C0_SR_PR['current'], C0_SR_rising_T_filter)
Power_Falling_wrt_time = CalculatePower(
    C0_SR_PF['current'], C0_SR_falling_T_filter)


def SpecificHeat(t,
                 T_rising, T_falling,
                 P_rising, P_falling,
                 dTdt_rising, dTdt_falling):

    num = P_rising - P_falling
    denom = dTdt_rising - dTdt_falling

    return num/denom


'''Establish min, max, and interpolate for T'''
if np.min(C0_SR_rising_T_filter) < np.min(C0_SR_falling_T_filter):
    T_lower = np.min(C0_SR_falling_T_filter)
else:
    T_lower = np.min(C0_SR_rising_T_filter)

if np.max(C0_SR_rising_T_filter) < np.max(C0_SR_falling_T_filter):
    T_upper = np.max(C0_SR_rising_T_filter)
else:
    T_upper = np.max(C0_SR_falling_T_filter)

delta_T = 0.001
T_array = np.arange(T_lower, T_upper, step=delta_T)

'''Interpolate power over a common T'''
Power_Rising_wrt_time_Tinterp = np.interp(
    T_array, C0_SR_rising_T_filter, Power_Rising_wrt_time)

Power_Falling_wrt_time_Tinterp = np.interp(
    T_array, C0_SR_falling_T_filter[::-1], Power_Falling_wrt_time[::-1]) # Can't interpolate decreasing function - temp fix.

plt.figure()
plt.plot(T_array, Power_Rising_wrt_time_Tinterp)
plt.plot(T_array, Power_Falling_wrt_time_Tinterp)

'''Interpolate derivatives over a common T'''
C0_SR_rising_T_filter_d1_Tinterp = np.interp(
    T_array, C0_SR_rising_T_filter, C0_SR_rising_T_filter_d1)
C0_SR_falling_T_filter_d1_Tinterp = np.interp(
    T_array, C0_SR_falling_T_filter[::-1], C0_SR_falling_T_filter_d1[::-1])

plt.figure()
plt.plot(T_array, C0_SR_rising_T_filter_d1_Tinterp)
plt.plot(T_array, C0_SR_falling_T_filter_d1_Tinterp)

def specificheat(T_array, P_up, P_down, dTdt_up, dTdt_down):

    return (P_up - P_down)/(dTdt_up - dTdt_down)

c_T = specificheat(T_array,
                    Power_Rising_wrt_time_Tinterp, Power_Falling_wrt_time_Tinterp,
                    C0_SR_rising_T_filter_d1_Tinterp, C0_SR_falling_T_filter_d1_Tinterp)

cp_add = pd.read_csv(
    'SpecificHeat/cp_addenda.dat', delimiter='\t')

plt.figure()
plt.plot(T_array, 1*c_T)
# plt.plot(cp_add['T(K)'], cp_add['C'])
plt.show()



