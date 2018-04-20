"""
plt_2d_1d
=========

Code to plot 2-D data on top and 1-D on bottom
"""

from os.path import exists, dirname

from astropy.io import fits
import astropy.io.ascii as asc

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import glob

from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()

co_filename = __file__
co_dir = dirname(co_filename)
rousselot_file = co_dir + '/' + 'rousselot2000.dat'
rousselot_data = asc.read(rousselot_file, format='commented_header')

wave0 = [6564.614, 6585.27, 6549.86, 6718.29, 6732.68]
name0 = [r'H$\alpha$', '[NII]', '[NII]', '[SII]', '[SII]']

def main(path0='', Instr='', zspec=[], Rspec=3000):

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

    Rspec : float
      Spectral resolution for OH skyline shading

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
     - Adjust grayscale normalization to zscale for imshow()
     - Add Rspec keyword input; Shade regions affected by OH skylines

    Modified by Chun Ly, 19 April 2018
     - Read in negative images; Display negative images in plots
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

    # + on 19/04/2018
    fits_file_2d_N1 = path0 + 'extract_2d_neg1.fits'
    fits_file_2d_N2 = path0 + 'extract_2d_neg2.fits'

    print('### Reading : '+fits_file_1d)
    data_1d, hdr_1d = fits.getdata(fits_file_1d, header=True)

    print('### Reading : '+fits_file_2d)
    data_2d, hdr_2d = fits.getdata(fits_file_2d, header=True)

    # + on 19/04/2018
    print('### Reading : '+fits_file_2d_N1)
    data_2d_N1, hdr_2d_N1 = fits.getdata(fits_file_2d_N1, header=True)

    # + on 19/04/2018
    print('### Reading : '+fits_file_2d_N2)
    data_2d_N2, hdr_2d_N2 = fits.getdata(fits_file_2d_N2, header=True)

    n_apers = hdr_1d['NAXIS2']

    l0_min, l0_max = 645.0, 680.0 # Minimum and maximum rest-frame wavelength (nm)
    crval1, cdelt1 = hdr_2d['CRVAL1'], hdr_2d['CDELT1']
    lam0_arr = crval1 + cdelt1 * np.arange(hdr_2d['NAXIS1'])

    for nn in [0]: #range(n_apers):
        gs1 = GridSpec(6, 1) # Mod on 19/04/2018

        ax1 = plt.subplot(gs1[1])
        ax2 = plt.subplot(gs1[3:])

        # + on 19/04/2018
        axN1 = plt.subplot(gs1[0])
        axN2 = plt.subplot(gs1[2])

        l_min = l0_min * (1+zspec[nn])
        l_max = l0_max * (1+zspec[nn])
        x_idx = np.where((lam0_arr >= l_min) & (lam0_arr <= l_max))[0]
        l_min, l_max = lam0_arr[x_idx[0]], lam0_arr[x_idx[-1]]

        tmpdata = data_2d[nn,10:20,x_idx].transpose()
        z1, z2 = zscale.get_limits(tmpdata)
        norm = ImageNormalize(vmin=z1, vmax=z2)
        ax1.imshow(tmpdata, extent=[l_min,l_max,0,tmpdata.shape[0]],
                   cmap='gray', norm=norm)
        ax1.set_ylabel(r'$y$ [pix]')
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        # Plot above negative image | + on 19/04/2018
        tmpdata = data_2d_N1[nn,10:20,x_idx].transpose()
        z1, z2 = zscale.get_limits(tmpdata)
        norm = ImageNormalize(vmin=z1, vmax=z2)
        axN1.imshow(tmpdata, extent=[l_min,l_max,0,tmpdata.shape[0]],
                    cmap='gray', norm=norm)
        axN1.set_ylabel('')
        axN1.set_yticklabels([])
        axN1.set_xticklabels([])

        # Plot below negative image | + on 19/04/2018
        tmpdata = data_2d_N2[nn,10:20,x_idx].transpose()
        z1, z2 = zscale.get_limits(tmpdata)
        norm = ImageNormalize(vmin=z1, vmax=z2)
        axN2.imshow(tmpdata, extent=[l_min,l_max,0,tmpdata.shape[0]],
                    cmap='gray', norm=norm)
        axN2.set_ylabel('')
        axN2.set_yticklabels([])
        axN2.set_xticklabels([])

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

        #Shade OH skyline | + on 31/03/2018
        OH_in = np.where((rousselot_data['lambda']/10.0 >= l_min) &
                         (rousselot_data['lambda']/10.0 <= l_max))[0]
        max_OH = max(rousselot_data['flux'][OH_in])

        max_in = np.where(rousselot_data['flux'][OH_in] >= 0.25*max_OH)[0]
        OH_idx = OH_in[max_in]
        for ii in range(len(OH_idx)):
            t_wave = rousselot_data['lambda'][OH_idx[ii]]/10.0
            FWHM = t_wave/Rspec
            ax2.axvspan(t_wave-FWHM/2.0,t_wave+FWHM/2.0, alpha=0.20,
                        facecolor='black', edgecolor='none')
        #endfor

        plt.subplots_adjust(left=0.025, bottom=0.025, top=0.975, right=0.975,
                            wspace=0.03, hspace=0.03)

        out_pdf = '%sextract_1d_%02i.pdf' % (path0, nn+1)
        print('### Writing : '+out_pdf)
        fig = plt.gcf()
        fig.savefig(out_pdf, bbox_inches='tight')
    #endfor

#enddef

