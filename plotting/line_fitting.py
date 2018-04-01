"""
line_fitting
============

Python code to fit emission lines and determine S/N

Provide description for code here.
"""

from os.path import exists

import numpy as np

from astropy.table import Table
from scipy.optimize import curve_fit

from ..extract import gauss1d

def main(wave, spec1d, OH_arr, ax):

    '''
    Main() function for line_fitting

    Parameters
    ----------
    wave : np.array
      Array of wavelengths to fit

    spec1d : dict
      Dictionary containing wavelengths and fluxes
    
    OH_arr : np.array
      Array of 0 and 1 for where pixel is affected by OH skyline emission

    ax : matplotlib.axes._subplots.AxesSubplot
      Matplotlib axes from plt.subplots()

    Returns
    -------
    ax with Gaussian fits overlaid

    Notes
    -----
    Created by Chun Ly, 1 April 2018
     - Use OH skyline mask to determine median
     - Set boundaries for curve_fit fitting
    '''

    lamb0 = spec1d['wave']
    flux0 = spec1d['flux']

    n_lines = len(wave)

    unmask = np.where(OH_arr == 0)[0]
    med0 = np.median(flux0[unmask])
    #ax.axhline(y=med0,color='magenta')

    for ii in range(n_lines):
        idx = [xx for xx in range(len(lamb0)) if
               np.abs(lamb0[xx]-wave[ii]) < 1.0]
        max0 = max(flux0[idx])
        p0 = [med0, max0, wave[ii], 0.4]
        param_bounds = ((0.995*med0, 0.9*max0, wave[ii]-0.2, 0),
                        (1.005*med0, 1.1*max0, wave[ii]+0.2, np.inf))

        popt, pcov = curve_fit(gauss1d, lamb0, flux0, p0=p0, bounds=param_bounds)
        center0 = popt[2]
        sigma0  = popt[3]
        sig_idx = np.where(np.abs(lamb0-center0)/sigma0 <= 2.5)[0]
        int_flux = np.sum(flux0[sig_idx] - popt[0])
        print ii, wave[ii], center0, sigma0, int_flux
        ax.plot(lamb0, gauss1d(lamb0, *popt), color='g')

    return ax
#enddef

