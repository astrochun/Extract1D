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

from ..extract import mlog

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
     - Define OH masking array; Pass to line_fitting.main()
     - Mask all OH skylines, not just those that are strongest
    Modified by Chun Ly, 3 April 2018
     - Look over all apertures
     - Handle case with zspec = -1 (no redshift available)
    Modified by Chun Ly, 9 April 2018
     - Handle NaN for plot limits for full spectral plots
     - Plot aesthetics: Limit y-axis lower end
     - Plot aesthetics: 3-panel plot for full spectral plots
    Modified by Chun Ly, 17 August 2018
     - Define l_scale for wavelength transformation
     - Fix ax.axvspan bug for zspec=-1
     - Fix spectral range (proper units)
     - Correct wave (proper units)
    Modified by Chun Ly,  5 September 2018
     - Write line fitting table
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

    # Mask all OH lines
    OH_arr = np.zeros(len(lam0_arr))
    for ii in range(len(rousselot_data)):
        if Instr == 'MMIRS': l_scale = 10.0
        if Instr == 'GNIRS': l_scale = 1.0
        t_wave = rousselot_data['lambda'][ii]/l_scale

        FWHM = 1.2*t_wave/Rspec
        o_idx = np.where((lam0_arr >= t_wave-FWHM/2.0) &
                         (lam0_arr <= t_wave+FWHM/2.0))[0]
        if len(o_idx) > 0: OH_arr[o_idx] = 1

    for nn in range(n_apers):
        # Mod on 03/04/2018
        if zspec[nn] == -1:
            l_min, l_max = min(lam0_arr), max(lam0_arr)
        else:
            l_min = l0_min * (10./l_scale) * (1+zspec[nn])
            l_max = l0_max * (10./l_scale) * (1+zspec[nn])

        x_idx = np.where((lam0_arr >= l_min) & (lam0_arr <= l_max))[0]
        l_min, l_max = lam0_arr[x_idx[0]], lam0_arr[x_idx[-1]]

        tx, ty = lam0_arr[x_idx], data_1d[nn,x_idx]
        nan_idx = np.where((np.isnan(ty) == True) | (np.isinf(ty) == True))[0]
        if len(nan_idx) > 0: ty[nan_idx] = 0.0

        ymin, ymax = -0.1*max(ty), 1.15*max(ty)

        if zspec[nn] != -1:
            fig, ax = plt.subplots()

            ax.plot(tx, ty, 'k')
            ax.axhline(y=0, linestyle='dotted', color='black')
            ax.set_xlim(l_min,l_max)
            ax.set_ylim([ymin, ymax])
            ax.set_xlabel('Wavelengths (nm)')
            ax.set_ylabel('Relative Flux')
            ax.set_yticklabels([])
            hspace = 0.03
        else:
            fig, ax_arr = plt.subplots(nrows=3)

            dx0 = (l_max-l_min) / 3.0
            for aa in range(3):
                ax_arr[aa].plot(tx, ty, 'k')
                ax_arr[aa].axhline(y=0, linestyle='dotted', color='black')
                ax_arr[aa].set_ylim([ymin, ymax])
                ax_arr[aa].set_xlim([l_min+dx0*aa,l_min+dx0*(aa+1)])
                ax_arr[aa].set_yticklabels([])

            ax_arr[1].set_ylabel('Relative Flux')
            ax_arr[2].set_xlabel('Wavelengths (nm)')
            hspace = 0.125

        # Mod on 03/04/2018
        if zspec[nn] != -1:
            for wave,name in zip(wave0,name0):
                x_val = (1+zspec[nn])*wave/l_scale
                ax.axvline(x=x_val, color='blue', linestyle='--')
                ax.annotate(name, (x_val,1.05*max(ty)), xycoords='data',
                            va='bottom', ha='center', rotation=90,
                            color='blue', bbox=bbox_props)
            #endfor
        #endif

        #Shade OH skyline
        OH_in = np.where((rousselot_data['lambda']/l_scale >= l_min) &
                         (rousselot_data['lambda']/l_scale <= l_max))[0]
        max_OH = max(rousselot_data['flux'][OH_in])

        max_in = np.where(rousselot_data['flux'][OH_in] >= 0.25*max_OH)[0]
        OH_idx = OH_in[max_in]
        for ii in range(len(OH_idx)):
            t_wave = rousselot_data['lambda'][OH_idx[ii]]/l_scale
            FWHM = t_wave/Rspec

            if zspec[nn] != -1:
                ax.axvspan(t_wave-FWHM/2.0,t_wave+FWHM/2.0, alpha=0.20,
                           facecolor='black', edgecolor='none')
        #endfor

        # Mod on 03/04/2018
        if zspec[nn] != -1:
            wave = np.array(wave0)*(1+zspec[nn])/l_scale
            spec1d = {'wave': tx, 'flux': ty}
            ax, tab0 = line_fitting.main(wave, spec1d, OH_arr[x_idx], ax)

            tab_outfile = '%sextract_fit_%02i.tbl' % (path0, nn+1)
            print('### Writing : '+tab_outfile)
            tab0.write(tab_outfile, format='ascii.fixed_width_two_line')
        #endif

        plt.subplots_adjust(left=0.025, bottom=0.025, top=0.975, right=0.975,
                            wspace=0.03, hspace=hspace)

        fig.set_size_inches(8,6)

        out_pdf = '%sextract_1d_%02i.spec.pdf' % (path0, nn+1)
        print('### Writing : '+out_pdf)
        fig.savefig(out_pdf, bbox_inches='tight')

    #endfor

#enddef

