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

bbox_props = dict(boxstyle="square,pad=0.15", fc="white", alpha=0.9, ec="none")

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
     - Compute one rms for all lines
     - Mask around emission lines
     - Plot rms
    Modified by Chun Ly, 9 April 2018
     - Annotate plot with line fitting properties
    Modified by Chun Ly, 20 August 2018
     - Handle negative peak in curve_fit
     - Fix annotated text for no measurements
    Modified by Chun Ly, 23 August 2018
     - Move annotation to upper right
    '''

    lamb0 = spec1d['wave']
    flux0 = spec1d['flux']

    n_lines = len(wave)

    for ii in range(n_lines):
        idx = [xx for xx in range(len(lamb0)) if
               np.abs(lamb0[xx]-wave[ii]) < 1.0]
        print ii, len(idx)
        if len(idx) > 0: OH_arr[idx] = 1

    unmask = np.where(OH_arr == 0)[0]
    med0 = np.median(flux0[unmask])
    #ax.axhline(y=med0,color='magenta')

    rms0 = np.std(flux0[unmask])
    print rms0

    fit_annot = ['', 'flux = ', r'$\sigma(\AA)$ = ', 'S/N = ']

    for ii in range(n_lines):
        idx = [xx for xx in range(len(lamb0)) if
               np.abs(lamb0[xx]-wave[ii]) < 1.0]
        max0 = max(flux0[idx])
        p0 = [med0, max0, wave[ii], 0.4]
        if max0 > 0:
            param_bounds = ((0.995*med0, 0.9*max0, wave[ii]-0.2, 0),
                            (1.005*med0, 1.1*max0, wave[ii]+0.2, np.inf))

            popt, pcov = curve_fit(gauss1d, lamb0, flux0, p0=p0, bounds=param_bounds)
            center0 = popt[2]
            sigma0  = popt[3]
            sig_idx = np.where(np.abs(lamb0-center0)/sigma0 <= 2.5)[0]
            int_flux = np.sum(flux0[sig_idx] - popt[0])
        else:
            print('Negative value for peak')

            center0  = -1
            sigma0   = -1
            int_flux = 0
            sig_idx  = []

        print ii, wave[ii], center0, sigma0, int_flux, len(sig_idx), rms0, \
            rms0*np.sqrt(len(sig_idx))
        ax.plot(lamb0, gauss1d(lamb0, *popt), color='g')
        if ii == 0: Ha_flux = int_flux

        fit_annot[0] += '%.1f, ' % center0
        fit_annot[1] += '%.3f, ' % (int_flux / Ha_flux)
        fit_annot[2] += '%.1f, ' % (sigma0 * 10)
        if len(sig_idx) > 0:
            fit_annot[3] += '%.1f, ' % (int_flux / (rms0*np.sqrt(len(sig_idx))))
        else:
            fit_annot[3] += '%.1f, ' % (int_flux)

    #endfor
    fit_annot0 = '\n'.join([a[:-2] for a in fit_annot])
    ax.annotate(fit_annot0, (0.95,0.90), ha='right', va='top', color='orange',
                bbox=bbox_props, xycoords='axes fraction', zorder=6,
                fontsize=10)

    ax.axhspan(med0-rms0,med0+rms0, facecolor='red', alpha=0.2)
    return ax
#enddef

