import numpy as np
from numpy.polynomial.chebyshev import Chebyshev


def fit_chebyshev_polynomials(df, degree=7):
    """
    Fits Chebyshev polynomials to log-log data for resistance vs. temperature.
    
    Parameters:
        df (pandas.DataFrame): DataFrame containing 'resistance' and 'temperature' columns.
        degree (int): Degree of the Chebyshev polynomial.
    
    Returns:
        tuple: (Chebyshev object for R->T, Chebyshev object for T->R)
    """
    log_R = np.log10(df['resistance'])
    log_T = np.log10(df['temperature'])
    
    cheb_fit_R_to_T = Chebyshev.fit(log_R, log_T, degree)
    cheb_fit_T_to_R = Chebyshev.fit(log_T, log_R, degree)
    
    return cheb_fit_R_to_T, cheb_fit_T_to_R


# Fit the polynomials (modify the degree if needed)
#cheb_fit_R_to_T, cheb_fit_T_to_R = fit_chebyshev_polynomials(df, degree=3)


def resistance_to_temperature(resistance, cheb_fit_R_to_T):
    """
    Convert resistance to temperature using the Chebyshev fit.
    Parameters:
        resistance (float): Resistance in ohms.
        cheb_fit_R_to_T (Chebyshev): Chebyshev fit object for R->T.
    Returns:
        float: Temperature in Kelvin.
    """
    log_R = np.log10(resistance)
    log_T = cheb_fit_R_to_T(log_R)
    return 10**log_T



def temperature_to_resistance(temperature, cheb_fit_T_to_R):
    """
    Convert temperature to resistance using the Chebyshev fit.
    Parameters:
        temperature (float): Temperature in Kelvin.
        cheb_fit_T_to_R (Chebyshev): Chebyshev fit object for T->R.
    Returns:
        float: Resistance in ohms.
    """
    log_T = np.log10(temperature)
    log_R = cheb_fit_T_to_R(log_T)
    return 10**log_R
