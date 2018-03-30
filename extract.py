"""
extract
=======

Main module for extraction of 1-D spectra from 2-D data
"""

import sys, os

from os.path import exists

from astropy.io import fits

import numpy as np

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
from glob import glob

def gauss1d(x, a0, a, x0, sigma):
    '''
    1-D gaussian function for curve_fit

    Parameters
    ----------
    x : list or np.array
      Pixel value along spectral direction

    a0 : float
      continuum level

    a : float
      Amplitude of Gaussian

    x0 : float
      Center of Gaussian

    sigma : float
      Gaussian sigma width

    Notes
    -----
    Created by Chun Ly, 30 March 2018
    '''

    return a0 + a * np.exp(-(x - x0)**2 / (2 * sigma**2))
#enddef

def main(path0='', filename='', Instr='', coords=[], direction=''):

    '''
    Main function to perform extraction for input data

    Parameters
    ----------
    path0 : str
      Full path to directory

    filename : str
      Filename for 2-D images. Specify full path if path0 is not given

    Instr : str
      Instrument name. Options are 'MMIRS', 'GNIRS', 'MOSFIRE'
      By setting this, glob searches for default values

    coords : list
      List of lists containing x,y coordinates in the image for each extraction.
      Provide x or y only to indicate a continuum source.
      Providing both x and y is intended for emission lines

    direction : str
      Direction of extraction along spectra.  Either 'x' or 'y'
      If Instr is specified, direction is not needed

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 30 March 2018
     - Bug fix: coordinates -> coords
     - Add direction keyword input
     - Change coords handling style for continuum and non-continuum spectra
     - Get FITS data, coords length check handling, compute median for
       continuum case
     - Call gauss1d() to get aperture location for continuum case
    '''

    if path0 == '' and filename == '' and Instr == '' and len(coords)==0:
        print("### Call: main(path0='/path/to/data/', filename='spectra.fits',"+\
              "Instr='', coords=[[x1,y1],[x2,y2],[y3]])")
        print("### Must specify either path0 and Instr, path0 and filename, or"+\
              "filename, AND include list of coordinates")
        print("### Exiting!!!")
        return
    #endif

    if path0 != '':
        if path0[-1] != '/': path0 = path0 + '/'

    if Instr == '':
        if path0 == '': filename0 = filename
        if path0 != '' and filename != '':
            filename0 = path0 + filename
    else:
        if Instr == 'MMIRS':
            direction = 'x'
            srch0 = glob(path0+'sum_obj-sky_slits_corr.fits')
            if len(srch0) == 0:
                print("## File not found!!!")
            else:
                filename0 = srch0[0]
        #endif


    print("## Filename : "+filename0)
    spec2d, spec2d_hdr = fits.getdata(filename0, header=True)

    n_aper = len(coords)

    if n_aper == 0:
        print("## No aperture provided")
        print("Exiting")
        return

    for nn in range(n_aper):
        if len(coords[nn]) == 1: sp_type = 'cont'
        if len(coords[nn]) == 2: sp_type = 'line'

        if direction == 'x': axis=1 # median over columns
        if direction == 'y': axis=0 # meidan over rows
        if sp_type == 'cont':
            med0 = np.nanmedian(spec2d, axis=axis)
            bad0 = np.where(np.isnan(med0))[0]
            if len(bad0) > 0: med0[bad0] = 0.0
            t_coord = coords[nn][0]
            p0 = [0.0, med0[t_coord], t_coord, 2.0]
            x0 = np.arange(len(med0))
            popt, pcov = curve_fit(gauss1d, x0, med0, p0=p0)
            center0 = popt[2]
            sigma0  = popt[3]

    #endfor

#enddef

