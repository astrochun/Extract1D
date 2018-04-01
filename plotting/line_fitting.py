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

def main(wave, spec1d, ax):

    '''
    Main() function for line_fitting

    Parameters
    ----------
    wave : np.array
      Array of wavelengths to fit

    spec1d : dict
      Dictionary containing wavelengths and fluxes
    
    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 1 April 2018
    '''

    lamb0 = spec1d['wave']
    flux0 = spec1d['flux']

    n_lines = len(wave)

    for ii in range(n_lines):
        idx = [xx for xx in range(len(lamb0)) if
               np.abs(lamb0[xx]-wave[ii]) < 1.0]
        max0 = max(flux0[idx])
        p0 = [0.0, max0, wave[ii], 0.4]
        popt, pcov = curve_fit(gauss1d, lamb0, flux0, p0=p0)
        center0 = popt[2]
        sigma0  = popt[3]
        print popt
        sig_idx = np.where(np.abs(lamb0-center0)/sigma0 <= 2.5)[0]
        int_flux = np.sum(flux0[sig_idx] - popt[0])
        print ii, wave[ii], center0, sigma0, int_flux
        ax.plot(lamb0, gauss1d(lamb0, *popt))

    return ax
#enddef

