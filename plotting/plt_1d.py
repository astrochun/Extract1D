"""
plt_1d
======

Code to plot 1-D spectra
"""

from os.path import exists, dirname

from astropy.io import fits
import astropy.io.ascii as asc

import numpy as np

import matplotlib.pyplot as plt
import glob

co_filename = __file__
co_dir = dirname(co_filename)
rousselot_file = co_dir + '/' + 'rousselot2000.dat'
rousselot_data = asc.read(rousselot_file, format='commented_header')

from plt_2d_1d import wave0, name0
import line_fitting

bbox_props = dict(boxstyle="square,pad=0.15", fc="white",
                  alpha=1.0, ec="none")

def main(path0='', Instr='', zspec=[], Rspec=3000):

    '''
    Main function to plot both the 1-D spectra on the same PDF

    Parameters
    ----------
    path0 : str
      Full path to directory

    filename : str
      Filename for 2-D images. Specify full path if path0 is not given

    Instr : str
      Instrument name. Options are 'MMIRS', 'GNIRS', 'MOSFIRE'
      By setting this, glob searches for default values

    Rspec : float
      Spectral resolution for OH skyline shading

    Returns
    -------
    Write PDF files to path0 with filename of 'extract_1d_[AP].pdf'
     Where AP is integer from 01 and one

    Notes
    -----
    Created by Chun Ly, 31 March 2018
     - Started as a copy of plt_2d_1d.py
    Modified by Chun Ly, 1 April 2018
     - Import plotting.line_fitting and call line_fitting.main to fit lines
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

    for nn in [1]:
        fig, ax = plt.subplots()

        l_min = l0_min * (1+zspec[nn])
        l_max = l0_max * (1+zspec[nn])
        x_idx = np.where((lam0_arr >= l_min) & (lam0_arr <= l_max))[0]
        l_min, l_max = lam0_arr[x_idx[0]], lam0_arr[x_idx[-1]]

        tx, ty = lam0_arr[x_idx], data_1d[nn,x_idx]
        ax.plot(tx, ty, 'k')
        ax.axhline(y=0, linestyle='dotted', color='black')
        ax.set_xlim(l_min,l_max)
        ax.set_ylim(1.2*min(ty), 1.15*max(ty))
        ax.set_xlabel('Wavelengths (nm)')
        ax.set_ylabel('Relative Flux')
        ax.set_yticklabels([])

        for wave,name in zip(wave0,name0):
            x_val = (1+zspec[nn])*wave/10.
            ax.axvline(x=x_val, color='blue', linestyle='--')
            ax.annotate(name, (x_val,1.05*max(ty)), xycoords='data',
                               va='bottom', ha='center', rotation=90,
                               color='blue', bbox=bbox_props)
        #endfor

        wave = np.array(wave0)*(1+zspec[nn])/10.0
        spec1d = {'wave': tx, 'flux': ty}
        ax = line_fitting.main(wave, spec1d, ax)

        #Shade OH skyline
        OH_in = np.where((rousselot_data['lambda']/10.0 >= l_min) &
                         (rousselot_data['lambda']/10.0 <= l_max))[0]
        max_OH = max(rousselot_data['flux'][OH_in])

        max_in = np.where(rousselot_data['flux'][OH_in] >= 0.25*max_OH)[0]
        OH_idx = OH_in[max_in]
        for ii in range(len(OH_idx)):
            t_wave = rousselot_data['lambda'][OH_idx[ii]]/10.0
            FWHM = t_wave/Rspec
            ax.axvspan(t_wave-FWHM/2.0,t_wave+FWHM/2.0, alpha=0.20,
                       facecolor='black', edgecolor='none')
        #endfor

        plt.subplots_adjust(left=0.025, bottom=0.025, top=0.975, right=0.975,
                            wspace=0.03, hspace=0.03)

        fig.set_size_inches(8,6)

        out_pdf = '%sextract_1d_%02i.spec.pdf' % (path0, nn+1)
        print('### Writing : '+out_pdf)
        fig.savefig(out_pdf, bbox_inches='tight')
    #endfor

#enddef

