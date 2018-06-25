#!/usr/bin/python2.7
from imports import *

global c, h, bands, path2sigRV
c, h = 299792458., 6.62607004e-34
bands = ['U','B','V','R','I','Y','J','H','K']
path2sigRV = '/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV'


def get_reduced_spectrum(Teff, logg, Z, vsini, band_str, R,
                         centralwl, SNRtarget):
    '''
    Download a PHOENIX stellar model spectrum and reduce the spectrum over 
    a particular spectral bin via convolution with the instrumental resolution 
    and a stellar rotation kernel.

    Parameters
    ----------
    `Teff': scalar
        The stellar effective temperature in Kelvin
    `logg': scalar
        The stellar logg in cgs units
    `Z': scalar
        The stellar metallicity [Fe/H] in solar units
    `vsini': scalar
        The projected stellar rotation velocity in km/s
    `band_str': str
        The letter designating the spectral band under consideration. Must be 
        in ['U','B','V','R','I','Y','J','H','K']
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)
    `centralwl': scalar
        The reference wavelength where the instrument resolution and SNR are 
        specified in microns
    `SNRtarget': scalar
        Target SNR per resolution element to be obtained during an exposure

    Returns
    -------
    `wl': numpy.array
        Spectral array of wavelengths in microns
    `spec': numpy.array
        Stellar spectrum array in Nphotons/s/cm^2/cm

    '''
    wl = _get_wavelengthgrid()
    _, spectrum = _get_full_spectrum(float(Teff), float(logg), float(Z))    
    wl_conv, spec_conv = _convolve_band_spectrum(wl, spectrum, band_str, R)
    if vsini > 0:
        spec_conv = _rotational_convolution(wl_conv, spec_conv, vsini)
    wl_resamp, spec_resamp = _resample_spectrum(wl_conv, spec_conv, R,
                                                centralwl)
    spec_scaled = _cgs2Nphot(wl, spectrum, wl_resamp, spec_resamp, centralwl,
                             SNRtarget)
    return wl_resamp, spec_scaled


def _get_wavelengthgrid():
    '''
    Read-in the wavelength grid for the PHOENIX model spectra and return it.
    
    Returns
    -------
    `wl': numpy.array
        Full spectral array of wavelengths in microns (i.e. full spectral 
        coverage)

    '''
    fname = "%s/HiResFITS/WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"%path2sigRV
    #fname = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/" + \
    #        "WAVE_PHOENIX-ACES-AGSS-COND-2011.fits"
    wl_fits = fits.open(fname)[0]
    return np.ascontiguousarray(wl_fits.data) * 1e-4  # in microns


def _get_full_spectrum(Teff, logg, Z):
    '''
    Read-in a model stellar spectrum from the PHOENIX library and return the 
    header and  the full spectral data (i.e. all available wavelengths).

    Parameters
    ----------
    `Teff': scalar
        The stellar effective temperature in Kelvin
    `logg': scalar
        The stellar logg in cgs units
    `Z': scalar
        The stellar metallicity [Fe/H] in solar units

    Returns
    -------
    `header': astropy.io.fits.header.Header
        The fits header of the PHOENIX stellar model
    `data': numpy.array
        Stellar spectrum array in erg/s/cm^2/cm

    '''
    Teffs = np.append(np.arange(23e2,7e3,1e2), np.arange(7e3,121e2,2e2)) 
    assert Teff in Teffs
    loggs = np.arange(0,6.1,.5)
    assert logg in loggs
    Zs = np.append(np.arange(-4,-2,1), np.arange(-2,1.5,.5))
    assert Z in Zs
    if Z == 0:
	Zstr = '-0.0'
    elif Z > 0:
	Zstr = '+%.1f'%Z
    else:
	Zstr = '%.1f'%Z

    # Download the spectrum
    prefix = "%s/HiResFITS/PHOENIX-ACES-AGSS-COND-2011/"%path2sigRV
    #prefix = "ftp://phoenix.astro.physik.uni-goettingen.de/HiResFITS/" + \
    #         "PHOENIX-ACES-AGSS-COND-2011/"
    fname = "Z%s/lte%.5d-%.2f%s"%(Zstr, Teff, logg, Zstr) + \
            ".PHOENIX-ACES-AGSS-COND-2011-HiRes.fits"
    spec_fits = fits.open(prefix+fname)[0]

    return spec_fits.header, spec_fits.data


def _convolve_band_spectrum(wl_microns, spectrum, band_str, R):
    '''
    Convolve the spectrum in a given band with a Gaussian profile whose FWHM 
    is specified by the spectral resolution of the instrument.

    Parameters
    ----------
    `wl_microns': array-like
        Spectral array of wavelengths in microns
    `spectrum': array-like
        Stellar spectrum array in erg/s/cm^2/cm
    `band_str': str
        The letter designating the spectral band under consideration. Must be 
        in ['U','B','V','R','I','Y','J','H','K']
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)

    Returns
    -------
    `wl': numpy.array 
        Spectral array of wavelengths in microns over the spectral band
    `spec': numpy.array 
        Stellar spectrum array in erg/s/cm^2/cm over the spectral band

    '''
    # Create equidistant wl array
    wl_microns2 = np.linspace(wl_microns.min(), wl_microns.max(),
                              wl_microns.size)
    fint = interp1d(wl_microns, spectrum)
    spectrum2 = fint(wl_microns2)
    
    # Isolate wavelength range
    if band_str not in bands:
        raise ValueError('Unknown passband: %s'%band_str)
    wlmin, wlmax, wl_central_microns = _get_band_range(band_str)
    in_band = (wl_microns2 >= wlmin) & (wl_microns2 <= wlmax)
    wl_band, spectrum_band = wl_microns2[in_band], spectrum2[in_band]

    # Convolve to instrument resolution
    FWHM_microns = wl_central_microns / float(R)
    sigma_microns = FWHM_microns / (2*np.sqrt(2*np.log(2)))
    print 'Convolving the %s band stellar spectrum to the '%band_str + \
        'instrumental resolution (R = %i)'%R
    try:
        spectrum_conv = broadGaussFast(wl_band, spectrum_band, sigma_microns)
    except NameError: # no convolution is bad
        spectrum_conv = spectrum_band
        
    return wl_band, spectrum_conv


def _rotational_convolution(wl_band, spec_band, vsini, epsilon=0.6):
    '''
    Convolve the spectrum with rotational kernel based on the star's projected 
    rotation velocity and assuming a constant linear limb-darkening across the 
    stellar disk.

    Parameters
    ----------
    `wl_band': array-like
        Spectral array of wavelengths in microns
    `spec_band': array-like
        Stellar spectrum array in erg/s/cm^2/cm
    `vsini': scalar
        The projected stellar rotation velocity in km/s
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)

    Returns
    -------
    `spec': numpy.array 
        Stellar spectrum array in erg/s/cm^2/cm over the spectral band

    '''
    try:
        spec_conv = rotBroad(wl_band, spec_band, epsilon, vsini)
    except NameError:   # no convolution is bad
        spec_conv = spec_band

    return spec_conv


def _resample_spectrum(wl, spec, R, centralwl):
    '''
    Resample the input wavelength array and spectrum to the width of the 
    resolution element.
    
    Parameters
    ----------
    `wl': array-like
        Spectral array of wavelengths in microns
    `spec': array-like
        Stellar spectrum array in erg/s/cm^2/cm
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)
    `centralwl': scalar
        The reference wavelength where the instrument resolution and SNR are 
        specified in microns

    Returns
    -------
    `wl': numpy.array 
        Resampled spectral array of wavelengths in microns
    `spec': numpy.array 
        Resampled stellar spectrum array in erg/s/cm^2/cm

    '''
    dl = centralwl / R
    wl_resamp = np.arange(wl.min(), wl.max(), dl)
    fint = interp1d(wl, spec)
    return wl_resamp, fint(wl_resamp)


def _get_band_range(band_str):
    '''
    Define the wavelength range for a given band and its central wavelength.

    Parameters
    ----------
    `band_str': str
        The letter designating the spectral band under consideration. Must be 
        in ['U','B','V','R','I','Y','J','H','K']
        
    Returns
    -------
    `wlmin': float
        The smallest wavelength in the spectral band in microns
    `wlmax': float
        The largest wavelength in the spectral band in microns
    `wlcentral': float
        The smallest wavelength in the spectral band in microns

    '''
    # bin widths are chosen to have the same number of wavelength elements
    # per band for flux normalization purposes
    # http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse&gname=Generic&gname2=Johnson
    # http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse&gname=CFHT&gname2=Wircam
    if band_str == 'U':
        wlwidth, wlcentral = .06570, .35311
    elif band_str == 'B':
        wlwidth, wlcentral = .09727, .44304
    elif band_str == 'V':
        wlwidth, wlcentral = .08897, .55372
    elif band_str == 'R':
        wlwidth, wlcentral = .20700, .69396
    elif band_str == 'I':
        wlwidth, wlcentral = .23160, .87807
    elif band_str == 'Y':
        wlwidth, wlcentral = .10842, 1.02588
    elif band_str == 'J':
        wlwidth, wlcentral = .15479, 1.25446
    elif band_str == 'H':
        wlwidth, wlcentral = .28857, 1.63099
    elif band_str == 'K':
        wlwidth, wlcentral = .32086, 2.14975
    else:
        raise ValueError('Unknown bandpass: %s'%band_str)

    wlmin, wlmax = wlcentral-wlwidth/2, wlcentral+wlwidth/2 
    return wlmin, wlmax, wlcentral
    

def _cgs2Nphot(wl_full_microns, spec_full_cgs, wl_band_microns, spec_band_cgs,
	       centralwl, SNRtarget):
    '''
    Convert the input spectrum from cgs units (erg/s/cm^2/cm) to a the number 
    of photons with a fixed SNR at the center of the J band (1.25 microns).

    Parameters
    ----------
    `wl_full_microns': array-like
        Spectral array of wavelengths (in microns) across the full PHOENIX 
        spectral coverage
    `spec_full_cgs': array-like
        Stellar spectrum array (in erg/s/cm^2/cm) across the full PHOENIX 
        spectral coverage
    `wl_band_microns': array-like
        Spectral array of wavelengths (in microns) across a single spectral band
    `spec_band_cgs': array-like
        Stellar spectrum array (in erg/s/cm^2/cm) across a single spectral band
    `centralwl': scalar
        The reference wavelength where the instrument resolution and SNR are 
        specified in microns
    `SNRtarget': scalar
        Target SNR per resolution element to be obtained during an exposure

    Results
    -------
    `spec_Nphot_scaled': numpy.array
        Stellar spectrum array in Nphotons/s/cm^2/cm in a single spectral band

    '''
    assert wl_full_microns.size == spec_full_cgs.size
    assert wl_band_microns.size == spec_band_cgs.size
    wl_full_cm, wl_band_cm = wl_full_microns * 1e-4, wl_band_microns * 1e-4
    energy_full_erg, energy_band_erg = h*c / (wl_full_cm*1e-2) * 1e7, \
                                       h*c / (wl_band_cm*1e-2) * 1e7
    spec_full_Nphot, spec_band_Nphot = spec_full_cgs / energy_full_erg, \
                                       spec_band_cgs / energy_band_erg
    centralindex = np.where(abs(wl_full_microns-centralwl) == \
                            np.min(abs(wl_full_microns-centralwl)))[0][0]
    # SNR = sqrt(Nphot)
    norm = SNRtarget**2 / \
           np.max(spec_full_Nphot[centralindex-3:centralindex+3])
    spec_Nphot_scaled = norm * spec_band_Nphot
    
    return spec_Nphot_scaled
                    

def _rescale_sigmaRV(sigmaRV, mag, band_str, texp_min, aperture, throughput, R,
                     SNRtarget):
    '''
    Rescale sigmaRV from SNR=100 per resolution element to whatever SNR is 
    achieved in the input band over a given integration time.

    Parameters
    ----------
    `sigmaRV': scalar
        The photon-noise-limited RV precision in m/s
    `mag': scalar
        The stellar magnitude in spectral band given in `band_str'
    `band_str': str
        The letter designating the spectral band under consideration. Must be 
        in ['U','B','V','R','I','Y','J','H','K']
    `texp_min': scalar
        The integration time in minutes
    `aperture': float
        The telescope's aperture diameter in meters
    `throughput': scalar
        The quantum efficiency of the detector (0<throughput<=1)
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)
    `SNRtarget': scalar
        Target S/N per resolution element to be obtained during an exposure
    
    Returns
    -------
    `SNR': float
        The rescaled SNR of the spectrum

    '''
    snr = get_snr(mag, band_str, texp_min, aperture, throughput, R)
    return sigmaRV * SNRtarget / snr
    

def get_snr(mag, band_str, texp_min, aperture, throughput, R):
    '''
    Compute the SNR of the spectrum in a certain band (e.g. 'J').

    Parameters
    ----------
    `mag': scalar
        The stellar magnitude in spectral band given in `band_str'
    `band_str': str
        The letter designating the spectral band under consideration. Must be 
        in ['U','B','V','R','I','Y','J','H','K']
    `texp_min': scalar
        The integration time in minutes
    `aperture': float
        The telescope's aperture diameter in meters
    `throughput': scalar
        The quantum efficiency of the detector (0<throughput<=1)
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)

    Returns
    -------
    `SNR': float
        The SNR of the spectrum in a certain band

    '''
    # Define constants
    area_cm2 = np.pi * (aperture*1e2/2)**2
    res_kms = c/R
    texp_s = texp_min * 60.
    c_As = c * 1e10
    c_kms = c_As * 1e-13
    h_ergss = h * 1e7

    # Get flux density zeropoint (for m=0) in ergs s^-1 A^-1 cm^-2,
    # wavelength in angstroms and bandwidth in microns
    # http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse&gname=Generic&gname2=Johnson
    # http://svo2.cab.inta-csic.es/theory/fps3/index.php?mode=browse&gname=CFHT&gname2=Wircam
    if band_str == 'U':
        Fl = 3.678e-9
        l  = 3531.1
    elif band_str == 'B':
        Fl = 6.293e-9
        l  = 4430.4
    elif band_str == 'V':
        Fl = 3.575e-9
        l  = 5537.2
    elif band_str == 'R':
        Fl = 1.882e-9
        l  = 6939.6
    elif band_str == 'I':
        Fl = 9.329e-10
        l  = 8780.7
    elif band_str == 'Y':
        Fl = 5.949e-10
        l  = 10258.8
    elif band_str == 'J':
        Fl = 2.985e-10
        l  = 12544.6
    elif band_str == 'H':
        Fl = 1.199e-10
        l  = 16309.9
    elif band_str == 'K':
        Fl = 4.442e-11
        l  = 21497.5
    else:
        raise ValueError('Unknown passband: %s'%band_str)

    fl = Fl * 10**(-.4 * mag)
    Ephot = h_ergss * c_As / l
    Nphot_A = fl * texp_s * area_cm2 * throughput / Ephot
    
    # Get # photons per resolution element (dl/l)
    Nphot_res = Nphot_A * (l / R)

    # Add photon noise and dark current to noise budget
    ##darkcurrent, footprint = 1e-2, 12     # electrons/s, pixels
    ##SNR = Nphot_res / np.sqrt(Nphot_res + darkcurrent*footprint*texp_s)
    SNR = np.sqrt(Nphot_res)

    return SNR


def exposure_time_calculator_per_band(mags, band_strs, aperture, throughput, R,
                                      SNRtarget, texpmin=10, texpmax=60):
    '''
    Compute the exposure time required to reach a target SNR per resolution 
    element in a particular band. Either V for visible spectrographs or J for 
    nIR spectrographs.
    
    Parameters
    ----------
    `mags': list of scalars
        A list of the stellar magnitudes for the star in each spectral band
    `band_strs': list of strs
        A list of the spectral bands that span the wavelength coverage of the 
        spectrograph used to measure the radial velocities of the TESS star. 
        All band_strs entries must be in 
        ['U','B','V','R','I','Y','J','H','K']
        and at least one of ['V','J'] as a reference band
    `aperture': float
        The telescope's aperture diameter in meters
    `throughput': scalar
        The quantum efficiency of the detector (0<throughput<=1)
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)
    `texpmin': scalar
        The minimum exposure time in minutes. Required to mitigate the effects 
        of stellar pulsations and granulation
    `texpmax': scalar
        The maximum exposure time in minutes. Used to moderate the limit the 
        observational time that can be dedicated to a single star
    
    Returns
    -------
    `texp': float
        The minimum exposure time in minutes required to achieve a desired SNR 
        in a particular reference band (either 'V' or 'J')

    '''
    if 'V' in band_strs:
        reference_band = 'V'
    elif 'J' in band_strs:
        reference_band = 'J'
    else:
        raise ValueError("No reference band. Neither 'V' nor 'J' are " + \
                         'included in band_strs.')
    mags, band_strs = np.ascontiguousarray(mags), np.ascontiguousarray(band_strs)
    reference_mag = float(mags[band_strs == reference_band])
    
    texps = np.arange(texpmin, texpmax+.1, .1)  # minutes
    SNRs = np.zeros(texps.size)
    for i in range(texps.size):
        SNRs[i] = get_snr(reference_mag, reference_band, texps[i],
                          aperture, throughput, R)

    if SNRs.min() > SNRtarget:
        return float(texpmin)
    elif SNRs.max() < SNRtarget:
        return float(texpmax)
    else:
        fint = interp1d(SNRs, texps)
        return float(fint(SNRtarget))


def _remove_tellurics_from_W(band_str, wl_band, W, transmission_threshold,
                             wl_telluric, trans_telluric):
    '''
    Remove wavelengths that are sampled at wavelengths affected by tellurics 
    at the level more than a specified threshold.

    Parameters
    ----------
    `wl_band': array-like
        Spectral array of wavelengths in microns
    `W': array-like
        Spectral weighting function from Eq. X in Bouchy et al 2001 in 
        Nphot/s/cm^2/cm
    `transmission_threshold': scalar
        Maximum fractional absorption from tellurics. Only keep where 
        transmission is greater than 1 - `transmission_threshold'

    Returns
    -------
    `Wout': numpy.array()
        Spectral weighting function only where the atmospheric transmission is 
        favourable

    '''
    assert wl_band.size == W.size
    print 'Masking %s band telluric regions where '%band_str + \
        'transmission > %.2f percent'%(1. - transmission_threshold)
    
    # remove rayleigh continuum via boxcar smoothing if passband is in
    # that regime
    if np.any(wl_band <=.8):
        boxsteps = np.arange(wl_telluric.min(), wl_telluric.max(), 1e-2)
        trans_telluric_continuum = np.zeros(boxsteps.size-1)
    	for i in range(boxsteps.size-1):
	    trans_telluric_continuum[i] = \
                    trans_telluric[(wl_telluric >= boxsteps[i]) \
                                   & (wl_telluric <= boxsteps[i+1])].max()
	fint = interp1d(boxsteps[1:]-np.diff(boxsteps)[0]*.5,
                        trans_telluric_continuum,
                        bounds_error=False, fill_value=.878)
	trans_telluric = trans_telluric - fint(wl_telluric) + 1.

    # resample the wavelength grid
    fint = interp1d(wl_telluric, trans_telluric, bounds_error=False,
                    fill_value=0)
    transmission = fint(wl_band)

    # only keep where transmission > 1-threshold
    notellurics = np.where(transmission > (1.-transmission_threshold))[0]
    return W[notellurics]


def compute_W(wl_band, spec_band):
    '''
    Compute the weighting function W from the input spectrum using Eq. 8 from 
    Bouchy+2001.

    Parameters
    ----------
    `wl_band': numpy.array 
        Spectral array of wavelengths in microns over the spectral band
    `spec_band': numpy.array
        Stellar spectrum array in Nphotons/s/cm^2/cm over the spectral band
    
    Returns
    -------
    `W': numpy.array
        The weighting function as a function of wavelength in Nphotons/s/cm^2/cm

    '''
    # over-sample the spectrum
    wl2 = np.linspace(wl_band.min(), wl_band.max(), wl_band.size*10)
    fint = interp1d(wl_band, spec_band)
    A = fint(wl2)

    # compute W
    # note: np.gradient is consistent with scipy.misc.derivative method
    dwl = np.diff(wl2)[0]
    W = wl2**2 * np.gradient(A, dwl)**2 / A
    
    # resample W to the native resolution
    fint = interp1d(wl2, W)
    return fint(wl_band)


def compute_sigmaRV(wl_band, spec_band, mag, band_str, texp, aperture,
                    throughput, R, transmission_threshold, wl_telluric,
                    trans_telluric, SNRtarget):
    '''
    Compute the photon-noise limit of the RV precision from the information 
    content in the spectrum, over a particular band, and the characteristics 
    of the observing setup.

    Parameters
    ----------
    `wl_band': numpy.array 
        Spectral array of wavelengths in microns over the spectral band
    `spec_band': numpy.array
        Stellar spectrum array in Nphotons/s/cm^2/cm over the spectral band
    `mag': scalar
        The stellar magnitude in spectral band given in `band_str'
    `band_str': str
        The letter designating the spectral band under consideration. Must be 
        in ['U','B','V','R','I','Y','J','H','K']
    `texp': scalar
        The integration time in minutes
    `aperture': float
        The telescope's aperture diameter in meters
    `throughput': scalar
        The quantum efficiency of the detector (0<throughput<=1)
    `R': scalar
        The spectral resolution of the spectrograph (lambda / d_lambda)
    `transmission_threshold': scalar
        Maximum fractional absorption from tellurics. Only keep where 
        transmission is greater than 1 - `transmission_threshold'
    `SNRtarget': scalar
        Target SNR per resolution element to be obtained during an exposure

    Returns
    -------
    `sigmaRV': float
        The RV precision scaled from the reference SNR of 100 to the SNR given 
        the magnitude, exposure time, and telescope aperture
 
    '''
    W = compute_W(wl_band, spec_band)
    
    # remove tellurics
    W_clean = _remove_tellurics_from_W(band_str, wl_band, W,
                                       transmission_threshold, wl_telluric,
                                       trans_telluric)
    g = np.arange(W_clean.size) #np.arange(4, W_clean.size-4, dtype=int)
    sigmaRV = c / np.sqrt(np.sum(W_clean[g]))

    sigmaRV_scaled = _rescale_sigmaRV(sigmaRV, mag, band_str, texp,
                                      aperture, throughput, R, SNRtarget)
    return sigmaRV_scaled



def compute_sigmaRV_grid(wl_band, spec_band, band_str, R, transmission_threshold, 
			 wl_telluric, trans_telluric):
    W = compute_W(wl_band, spec_band)
    W_clean = _remove_tellurics_from_W(band_str, wl_band, W, transmission_threshold,
                                       wl_telluric, trans_telluric)
    sigmaRV = c / np.sqrt(np.sum(W_clean))
    return sigmaRV
