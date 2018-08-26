"""
extract
=======

Main module for extraction of 1-D spectra from 2-D data
"""

import sys, os

from os.path import exists, dirname

from astropy.io import fits

import numpy as np

import itertools
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
from glob import glob

from astropy.visualization import ZScaleInterval
zscale = ZScaleInterval()
from astropy.visualization.mpl_normalize import ImageNormalize

from astropy import log

import logging
formatter = logging.Formatter('%(asctime)s - %(module)12s.%(funcName)20s - %(levelname)s: %(message)s')
sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(formatter)

class mlog:
    '''
    Main class to log information to stdout and ASCII file

    To execute:
    mylog = mlog(path0)._get_logger()

    Parameters
    ----------
    path0 : str
      Full path for where raw files are

    Returns
    -------

    Notes
    -----
    Created by Chun Ly, 30 March 2018
    '''

    def __init__(self,path0):
        self.LOG_FILENAME = path0 + 'extract.log'
        self._log = self._get_logger()

    def _get_logger(self):
        loglevel = logging.INFO
        log = logging.getLogger(self.LOG_FILENAME)
        if not getattr(log, 'handler_set', None):
            log.setLevel(logging.INFO)
            sh = logging.StreamHandler()
            sh.setFormatter(formatter)
            log.addHandler(sh)

            fh = logging.FileHandler(self.LOG_FILENAME)
            fh.setLevel(logging.INFO)
            fh.setFormatter(formatter)
            log.addHandler(fh)

            log.setLevel(loglevel)
            log.handler_set = True
        return log
#enddef

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

def find_negative_images(x0, t_spec0, center0, peak, mylogger=None):
    '''
    Identify location of negative images

    Parameters
    ----------
    x0 : list or np.array
      List or array of wavelengths or pixels

    t_spec0 : np.array
      Array of 1-D spectral cut of target/object

    center0 : float
      Location of center of positive spectra

    peak : float
      Amplitude of peak in positive spectra

    mylogger : None type or class object
      Class for stdout. If None given, then astropy.log used instead

    Returns
    -------
    neg_offset : float
      Location of negative image relative to positive image in pixel units

    Notes
    -----
    Created by Chun Ly, 10 April 2018

    Modified by Chun Ly, 19 April 2018
     - Add mylogger keyword input: Implement stdout and ASCII logging
     - Return neg_offset
    '''

    if type(mylogger) == type(None):
        mylog = log
    else:
        mylog = mylogger

    dist0 = x0 - center0

    cen_pix = np.where(np.abs(dist0) == np.min(np.abs(dist0)))[0][0]

    # Average in distance from peak to improve S/N. Flip for positive signal
    # This assumes that the dithering is consistently spaced
    cen_avg = -1*(t_spec0[0:cen_pix][::-1] + t_spec0[cen_pix:cen_pix*2])/2.0

    new_x = np.arange(len(cen_avg))
    np.savetxt('/Users/cly/Downloads/find_negative_images.txt',
               np.c_[new_x, cen_avg])

    neg_pos_guess = np.argmax(cen_avg)
    p0 = [0.0, 0.5*peak, neg_pos_guess, 0.2]
    popt, pcov = curve_fit(gauss1d, new_x, cen_avg, p0=p0)
    mylog.info(str(popt))

    neg_offset = popt[2]
    return neg_offset
#enddef

def db_index(center0, coords, sigma0, distort_shift, spec2d_shape, direction=''):
    '''
    Computes indexing array using a distortion solution

    center0 : float
      Central value to use

    coords : list
      (x,y) coordinate. This is to determine the necessary distortion shift
      relative to the emission line. Generally from ds9, counts from 1
      for first pixel

    sigma0 : float
      Gaussian sigma from curve_fit [selection is done out to 3-sigma]

    distort_shift: numpy.array
      Numpy array containing the distortion as a function of wavelength
      Generated from calling np.poly1d array:
        pd = np.poly1d(db_best_fit)
        distort_shift = pd(np.arange(n_pix)
    '''

    n_pix = len(distort_shift)
    ds_inter = interp1d(1+np.arange(n_pix), distort_shift)
    if direction == 'y':
        ds_offset = ds_inter(coords[1])
        x0 = 1+np.arange(spec2d_shape[1])
    else:
        ds_offset = ds_inter(coords[0])
        x0 = 1+np.arange(spec2d_shape[0])

    ds_trace = center0 + (distort_shift - ds_offset)

    tmp_idx = np.zeros(spec2d_shape)
    for nn in range(n_pix):
        t_idx = np.where(np.abs(x0-ds_trace[nn])/sigma0 <= 3.0)[0]
        if len(t_idx) > 0:
            tmp_idx[nn,t_idx] = 1

    distort_idx = np.where(tmp_idx == 1)

    return distort_idx, tmp_idx, ds_trace
#enddef

def main(path0='', filename='', Instr='', coords=[], direction='', dbfile=''):

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

    dbfile : str
      Filename to numpy file containing solution. File is relative to path0.
      The solution should be a polynomial fit and provided in the numpy array
      as 'best_fit'. For continuum source, we use 'fit_arr'

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
     - Extract 1-D spectra for continuum case
     - Define wavelength solution from FITS header
     - Bug fix: axis flipped for extraction via np.sum call
     - Extract 1-D spectra for emission-line case
     - Write 2-D FITS file of 1-D extracted spectra to file
     - Write 2-D FITS datacube containing 2-D spectra for each target
     - Implement stdout and ASCII logging with mlog()
    Modified by Chun Ly, 10 April 2018
     - Call find_negative_images()
    Modified by Chun Ly, 19 April 2018
     - Typo fixed for NAXIS for y extraction
     - Get location of negative images using median of full 2-D image
     - Get 2-D spectra of negative images; Write to FITS files
    Modified by Chun Ly, 31 July 2018
     - Add dbfile keyword; read in dbfile
    Modified by Chun Ly, 13 August 2018
     - Change MMIRS final stack file to use
     - Define np.poly1d solution from db
    Modified by Chun Ly, 15 August 2018
     - Code documentation
     - Add Instr == 'GNIRS' case
     - Index x0 from 1 instead of 0
     - Call db_index()
     - Proper indexing for spec1d with distortion dbfile
     - Use nearest solution for continuum extraction case
    Modified by Chun Ly, 16 August 2018
     - Fix mylog typo
     - Fix to get proper poly1d fit for cont case
     - Call db_index() for cont case
     - Use np.multiply to np.sum with proper axis
     - Use np.transpose for spec2d (direction='y')
    Modified by Chun Ly, 17 August 2018
     - Change dbfile to filename (relative to path0)
     - Change med0 (use center) for cont case; Call db_index correctly
    Modified by Chun Ly, 22 August 2018
     - Force peak coordinates as integer
    Modified by Chun Ly, 24 August 2018
     - Handle 2-D edge extraction (broadcast input array error) [cont'd]
    Modified by Chun Ly, 25 August 2018
     - Plot results of distortion solution
     - Plot 2-D spectra with distortion solutions overlaid
     - Plot aesthetics: Specify line color, modify line styles, add legend
    '''

    if path0 == '' and filename == '' and Instr == '' and len(coords)==0:
        log.warn("### Call: main(path0='/path/to/data/', filename='spec.fits',"+\
                 "Instr='', coords=[[x1,y1],[x2,y2],[y3]])")
        log.warn("### Must specify either path0 and Instr, path0 and filename,"+\
                 " or filename, AND include list of coordinates")
        log.warn("### Exiting!!!")
        return
    #endif

    if path0 != '':
        if path0[-1] != '/': path0 = path0 + '/'

    mylog = mlog(path0)._get_logger()

    mylog.info('Begin main ! ')
    if Instr == '':
        if path0 == '': filename0 = filename
        if path0 != '' and filename != '':
            filename0 = path0 + filename
    else:
        if Instr == 'MMIRS':
            direction = 'x'
            srch0 = glob(path0+'sum_all_obj-sky_slits_corr.fits')
            if len(srch0) == 0:
                mylog.warn('File not found!!!')
            else:
                filename0 = srch0[0]
        if Instr == 'GNIRS':
            direction = 'y'
            srch0 = glob(path0+'obj_comb.fits')
            if len(srch0) == 0:
                mylog.warn('File not found!!!')
            else:
                filename0 = srch0[0]


    mylog.info("Input 2-D FITS file : ")
    mylog.info(filename0)
    spec2d, spec2d_hdr = fits.getdata(filename0, header=True)

    n_aper = len(coords)

    if n_aper == 0:
        mylog.warn("No aperture provided")
        mylog.warn("Exiting!!!")
        return

    if direction == 'x':
        lam0_min  = spec2d_hdr['CRVAL1']
        lam0_delt = spec2d_hdr['CDELT1']
        n_pix     = spec2d_hdr['NAXIS1']
    if direction == 'y':
        lam0_min  = spec2d_hdr['CRVAL2']
        lam0_delt = spec2d_hdr['CDELT2']
        n_pix     = spec2d_hdr['NAXIS2']
    lam0_arr = lam0_min + lam0_delt*np.arange(n_pix)

    spec1d_arr  = np.zeros((n_aper, len(lam0_arr)))
    spec2d_arr  = np.zeros((n_aper, 30, len(lam0_arr)))
    spec2d_neg1 = np.zeros((n_aper, 30, len(lam0_arr)))
    spec2d_neg2 = np.zeros((n_aper, 30, len(lam0_arr)))

    # Moved up on 19/04/2018
    if direction == 'x': axis=1 # median over row
    if direction == 'y': axis=0 # median over columns

    # Read in distortion database file | + on 31/07/2018
    if dbfile != '':
        mylog.info('Reading : '+path0+dbfile)
        distort_db = np.load(path0+dbfile)
        db_xcen     = distort_db['xcen_arr'][0]
        db_best_fit = distort_db['best_fit']
        pd = np.poly1d(db_best_fit)
        distort_shift = pd(np.arange(n_pix))

    # Find negative images (need one source with continuum | + on 19/04/2018
    t_spec2 = np.nanmedian(spec2d, axis=axis)
    bad0 = np.where(np.isnan(t_spec2))[0]
    if len(bad0) > 0: t_spec2[bad0] = 0.0
    t_x0    = np.arange(len(t_spec2))
    center0 = np.argmax(t_spec2)
    neg_off = find_negative_images(t_x0, t_spec2, center0, np.max(t_spec2))

    out_pdf = path0+'extract_QA.pdf'
    fig, ax = plt.subplots()

    z1, z2 = zscale.get_limits(spec2d)
    norm = ImageNormalize(vmin=z2, vmax=z1)
    ax.imshow(spec2d, cmap='Greys', origin='lower', norm=norm)

    colors = itertools.cycle(["r", "b", "g", "m"])

    for nn in range(n_aper):
        if len(coords[nn]) == 1: sp_type = 'cont'
        if len(coords[nn]) == 2: sp_type = 'line'

        if sp_type == 'cont':
            if direction == 'y':
                med0 = spec2d[n_pix/2] #np.nanmedian(spec2d, axis=axis)
            if direction == 'x':
                med0 = spec2d[:,n_pix/2]

            bad0 = np.where(np.isnan(med0))[0]
            if len(bad0) > 0: med0[bad0] = 0.0
            t_coord = np.int(coords[nn][0])
            p0 = [0.0, med0[t_coord], t_coord, 2.0]
            x0 = 1+np.arange(len(med0))
            popt, pcov = curve_fit(gauss1d, x0, med0, p0=p0)
            center0 = popt[2]
            sigma0  = popt[3]

            if dbfile == '':
                idx0 = np.where(np.abs(x0 - center0)/sigma0 <= 3.0)[0]
            else:
                c_diff = np.abs(center0 - db_xcen)
                idx_near = np.where(c_diff == np.min(c_diff))[0][0]
                mylog.info('Nearest solution found : '+\
                              str(db_xcen[idx_near]))
                cont_best_fit = distort_db['fit_arr'][0][idx_near]
                cont_pd = np.poly1d(cont_best_fit)
                cont_distort_shift = cont_pd(np.arange(n_pix))

                idx0, tmp_idx, \
                    ds_trace = db_index(center0, [coords[nn],n_pix/2], sigma0,
                                        cont_distort_shift, spec2d.shape,
                                        direction=direction)

        if sp_type == 'line':
            if direction == 'x':
                t_spec0 = spec2d[:,np.int(coords[nn][0]-1)]
                t_coord = np.int(coords[nn][1])
                t_peak  = t_spec0[t_coord]

            if direction == 'y':
                t_spec0 = spec2d[np.int(coords[nn][1]-1),:]
                t_coord = np.int(coords[nn][0])
                t_peak  = t_spec0[t_coord]
            p0 = [0.0, t_peak, t_coord, 2.0]
            x0 = 1+np.arange(len(t_spec0))

            bad0 = np.where(np.isnan(t_spec0))[0]
            if len(bad0) > 0: t_spec0[bad0] = 0.0
            popt, pcov = curve_fit(gauss1d, x0, t_spec0, p0=p0)
            center0 = popt[2]
            sigma0  = popt[3]

            if dbfile == '':
                idx0 = np.where(np.abs(x0 - center0)/sigma0 <= 3.0)[0]
            else:
                idx0, tmp_idx, \
                    ds_trace = db_index(center0, coords[nn], sigma0,
                                        distort_shift, spec2d.shape,
                                        direction=direction)

        if dbfile != '':
            y_temp = np.arange(len(ds_trace))
            ctype = next(colors)
            ax.plot(ds_trace, y_temp, color=ctype,
                    linestyle='dashed', alpha=0.5, linewidth=1.0,
                    label='Aper #'+str(nn+1))
            ax.plot(ds_trace+neg_off, y_temp, color=ctype,
                    linestyle='dotted', alpha=0.5, linewidth=1.0)
            ax.plot(ds_trace-neg_off, y_temp, color=ctype,
                    linestyle='dotted', alpha=0.5, linewidth=1.0)

        idx1 = np.where(np.abs(x0 - center0) <= 15.0)[0]

        # + on 19/04/2018
        idxN1 = np.where(np.abs(x0 - (center0+neg_off)) <= 15.0)[0]
        idxN2 = np.where(np.abs(x0 - (center0-neg_off)) <= 15.0)[0]
        if axis==1: #dispersion along x
            if dbfile == '':
                spec1d = np.sum(spec2d[idx0,:], axis=0)
            else:
                spec1d = np.sum(np.multiply(spec2d, tmp_idx), axis=0)
            t_spec2d = spec2d[idx1,:]

            # + on 19/04/2018
            t_spec2d_N1 = spec2d[idxN1,:]
            t_spec2d_N2 = spec2d[idxN2,:]
        if axis==0: #dispersion along y
            if dbfile == '':
                spec1d = np.sum(spec2d[:,idx0], axis=1)
            else:
                spec1d = np.sum(np.multiply(spec2d, tmp_idx), axis=1)
            t_spec2d = np.transpose(spec2d[:,idx1])

            # + on 19/04/2018
            t_spec2d_N1 = np.transpose(spec2d[:,idxN1])
            t_spec2d_N2 = np.transpose(spec2d[:,idxN2])

        spec1d_arr[nn,:] = spec1d

        n_width = t_spec2d.shape[0]
        spec2d_arr[nn, 0:n_width,:] = t_spec2d

        n_width_N1 = t_spec2d_N1.shape[0]
        spec2d_neg1[nn,0:n_width_N1,:] = t_spec2d_N1

        n_width_N2 = t_spec2d_N2.shape[0]
        spec2d_neg2[nn,0:n_width_N2,:] = t_spec2d_N2
    #endfor

    ax.legend(loc='upper center', fontsize=6)
    mylog.info('Writing : '+out_pdf)
    fig.savefig(out_pdf, bbox_inches='tight')

    spec2d_hdr['CRVAL1'] = lam0_min
    spec2d_hdr['CDELT1'] = lam0_delt
    spec2d_hdr['CD1_1']  = lam0_delt

    if direction == 'y':
        del spec2d_hdr['CRVAL2']
        del spec2d_hdr['CDELT2']
        del spec2d_hdr['CD2_2']

    out_fits_file = dirname(filename0)+'/extract_1d.fits'
    mylog.info('Writing : '+out_fits_file)
    fits.writeto(out_fits_file, spec1d_arr, spec2d_hdr, overwrite=True)

    out_2d_fits_file = dirname(filename0)+'/extract_2d.fits'
    mylog.info('Writing : '+out_2d_fits_file)
    fits.writeto(out_2d_fits_file, spec2d_arr, spec2d_hdr, overwrite=True)

    # Write above negative image | + on 19/04/2018
    out_2d_fits_file = dirname(filename0)+'/extract_2d_neg1.fits'
    mylog.info('Writing : '+out_2d_fits_file)
    fits.writeto(out_2d_fits_file, spec2d_neg1, spec2d_hdr, overwrite=True)

    # Write below negative image | + on 19/04/2018
    out_2d_fits_file = dirname(filename0)+'/extract_2d_neg2.fits'
    mylog.info('Writing : '+out_2d_fits_file)
    fits.writeto(out_2d_fits_file, spec2d_neg2, spec2d_hdr, overwrite=True)

    mylog.info('End main ! ')

#enddef
