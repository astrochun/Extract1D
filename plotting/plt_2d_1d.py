"""
plt_2d_1d
=========

Code to plot 2-D data on top and 1-D on bottom
"""

import sys, os

from os.path import exists

from astropy.io import fits

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob

wave0 = [6564.614, 6585.27, 6549.86, 6718.29, 6732.68]
name0 = [r'H$\alpha$', '[NII]', '[NII]', '[SII]', '[SII]']

def main(path0='', Instr='', zspec=[]):

    '''
    Main function to plot both the 2-D and 1-D spectra on the same PDF

    Parameters
    ----------
    path0 : str
      Full path to directory

    filename : str
      Filename for 2-D images. Specify full path if path0 is not given

    Instr : str
      Instrument name. Options are 'MMIRS', 'GNIRS', 'MOSFIRE'
      By setting this, glob searches for default values

    Returns
    -------
    Write PDF files to path0 with filename of 'extract_1d_[AP].pdf'
     Where AP is integer from 01 and one

    Notes
    -----
    Created by Chun Ly, 30 March 2018

    Modified by Chun Ly, 31 March 2018
     - Write PDF files
     - Plotting aesthetics, subplots_adjust for same x size
     - Switch to using GridSpec for subplots settings for 2D and 1D panels
    '''

    if path0 == '' and Instr == '':
        log.warn("### Call: main(path0='/path/to/data/', Instr='')")
        log.warn("### Exiting!!!")
        return
    #endif

    if path0 != '':
        if path0[-1] != '/': path0 = path0 + '/'

    fits_file_1d = path0 + 'extract_1d.fits'
    fits_file_2d = path0 + 'extract_2d.fits'

    print('### Reading : '+fits_file_1d)
    data_1d, hdr_1d = fits.getdata(fits_file_1d, header=True)

    print('### Reading : '+fits_file_2d)
    data_2d, hdr_2d = fits.getdata(fits_file_2d, header=True)

    n_apers = hdr_1d['NAXIS2']

    l0_min, l0_max = 645.0, 680.0 # Minimum and maximum rest-frame wavelength (nm)
    crval1, cdelt1 = hdr_2d['CRVAL1'], hdr_2d['CDELT1']
    lam0_arr = crval1 + cdelt1 * np.arange(hdr_2d['NAXIS1'])

    for nn in [1]: #range(n_apers):
        gs1 = GridSpec(3, 1)

        ax1 = plt.subplot(gs1[0])
        ax2 = plt.subplot(gs1[1:3])

        l_min = l0_min * (1+zspec[nn])
        l_max = l0_max * (1+zspec[nn])
        x_idx = np.where((lam0_arr >= l_min) & (lam0_arr <= l_max))[0]
        l_min, l_max = lam0_arr[x_idx[0]], lam0_arr[x_idx[-1]]

        tmpdata = data_2d[nn,10:20,x_idx].transpose()
        ax1.imshow(tmpdata, extent=[l_min,l_max,0,tmpdata.shape[0]],
                   cmap='gray')
        ax1.set_ylabel(r'$y$ [pix]')
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        tx, ty = lam0_arr[x_idx], data_1d[nn,x_idx]
        ax2.plot(tx, ty, 'k')
        ax2.set_xlim(l_min,l_max)
        ax2.set_ylim(1.2*min(ty), 1.3*max(ty))
        ax2.set_xlabel('Wavelengths (nm)')
        ax2.set_ylabel('Relative Flux')
        ax2.set_yticklabels([])

        for wave,name in zip(wave0,name0):
            x_val = (1+zspec[nn])*wave/10.
            ax1.axvline(x=x_val, color='red', linestyle='--')
            ax2.axvline(x=x_val, color='red', linestyle='--')
            ax2.annotate(name, (x_val,1.05*max(ty)), xycoords='data',
                               va='bottom', ha='center', rotation=90,
                               color='blue')
        #endfor

        plt.subplots_adjust(left=0.025, bottom=0.025, top=0.975, right=0.975,
                            wspace=0.03, hspace=0.03)

        out_pdf = '%sextract_1d_%02i.pdf' % (path0, nn+1)
        print('### Writing : '+out_pdf)
        fig = plt.gcf()
        fig.savefig(out_pdf, bbox_inches='tight')
    #endfor

#enddef

