#!/usr/bin/python2.7
from imports import *
from compute_sigmaRV import *
from sigmaRV_activity import *
from sigmaRV_planets import *
from compute_nRV_GP import *
from create_pdf import *
from Teff2color import *
import rvs_custom as rvs

global G, NGPmax
G, NGPmax = 6.67e-11, 1e2

def nRV_calculator(Kdetsig,
                   input_planet_fname='/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputFiles/user_planet.in',
                   input_star_fname='/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputFiles/user_star.in',
                   input_spectrograph_fname='/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputFiles/user_spectrograph.in',
                   input_sigRV_fname='/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputFiles/user_sigRV.in',
                   output_fname='/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/Results/RVFollowupCalculator.txt',
                   duration=100, NGPtrials=0, verbose_results=True):
    '''
    Compute the number of RV measurements required to detect an input 
    transiting planet around an input star with an input spectrograph at a 
    given significance.

    Parameters
    ----------
    `Kdetsig': scalar
        The desired RV semi-amplitude detection significance measured as 
        the semi-amplitude over its measurement uncertainty 
        (i.e. Kdetsig = K / sigmaK)
    `duration': scalar
        Duration (in days) of the assumed uniform time-series for calculations 
        of nRV with a GP. 
    `NGPtrials': scalar
        Number of times to compute Nrv with a GP as these results can vary
        during repeated trials. Returned results are the median values (e.g. 
        median(Nrv_GP) (*recommended)
    `runGP': boolean
        If True, compute nRV with a GP. Significantly faster if False. 

    '''    
    # get inputs
    texp, sigRV_phot, sigRV_act, sigRV_planet, sigRV_eff = \
                                        _read_sigRV_input(input_sigRV_fname)
    wlmin, wlmax, R, aperture, throughput, RVnoisefloor, maxtelluric,\
        toverhead = _read_spectrograph_input(input_spectrograph_fname)
    P, rp, mp = _read_planet_input(input_planet_fname)
    mag, Ms, Rs, Teff, Z, vsini, Prot = _read_star_input(input_star_fname)
    clean_input_files(input_sigRV_fname, input_spectrograph_fname, 
    		      input_planet_fname, input_star_fname)

    # checks
    if NGPtrials > NGPmax:
    	raise ValueError('Cannot exceed the maximum number of GP trials (i.e. 100).')
    runGP = True if NGPtrials > 0 else False
    if runGP and (duration < P):
        raise ValueError('Time-series duration must be longer than the' + \
                         "planet's orbital period.")
    
    # compute sigRV_eff from other sources if not specified
    if sigRV_eff < 0:
        
        # compute sigRV_phot once if needed
        if sigRV_phot <= 0:
            # get central band
            Vcen, Jcen = False, False
	    if (wlmin <= .555 <= wlmax):
	        centralwl_nm = 555
		Vcen = True
	    elif (wlmin <= 1.250 <= wlmax):
		centralwl_nm = 1250
		Jcen = True
	    else:
		raise ValueError('Spectral coverage does not include the V or J-band.')

    	    # get spectral bands corresponding to the wavelength range
            band_strs = _get_spectral_bands(wlmin, wlmax)

            # get mags for each spectral bin based on reference magnitude and Teff
            logg = float(unp.nominal_values(_compute_logg(Ms, Rs)))
	    magsfull = V2all(mag, Teff, logg, Z) if Vcen else J2all(mag, Teff, logg, Z)
	    all_band_strs = np.array(['U','B','V','R','I','Y','J','H','K'])
	    mags = magsfull[np.in1d(all_band_strs, band_strs)]
	    g = np.isnan(mags)
	    if g.sum() > 0:
	        gfull = np.isnan(magsfull)
		mags[g] = float(.5 * (magsfull[np.where(gfull)[0]-1] + magsfull[np.where(gfull)[0]+1]))

	    do_checks(throughput, maxtelluric)

            transmission_fname = 'tapas_000001.ipac'
            wlTAPAS, transTAPAS = np.loadtxt('/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputData/%s'%transmission_fname,
                                             skiprows=23).T
            wlTAPAS *= 1e-3  # microns
            SNRtarget, sigRV_phot = _compute_sigRV_phot(band_strs, mags, Teff, logg,
                                                        Z, vsini, texp, R, aperture,
                                                        throughput, RVnoisefloor,
                                                        centralwl_nm, maxtelluric,
                                                        wlTAPAS, transTAPAS)
            
            # get RV noise sources
            Bmag, Vmag = _get_magnitudes(band_strs, mags, Ms)
            B_V = Bmag - Vmag
            if sigRV_act < 0:
            	sigRV_act = get_sigmaRV_activity(Teff, Ms, Prot, B_V)
            if sigRV_planet < 0:
            	sigRV_planet = get_sigmaRV_planets(P,rp,Teff,Ms,sigRV_phot)

	else:
	    if sigRV_act < 0:
	        logg = float(unp.nominal_values(_compute_logg(Ms, Rs)))
	        B_V = get_B_V(Teff, logg, Z)
	    	sigRV_act = get_sigmaRV_activity(Teff, Ms, Prot, B_V)
	    if sigRV_planet < 0:
		sigRV_planet = get_sigmaRV_planets(P,rp,Teff,Ms,sigRV_phot)

        # compute sigRV_eff
        sigRV_eff = np.sqrt(RVnoisefloor**2 + \
                            sigRV_phot**2 + \
                            sigRV_act**2 + \
                            sigRV_planet**2)

    # get target K measurement uncertainty
    mp = float(_get_planet_mass(rp)) if mp <= 0 else mp
    K, sigK_target = _get_sigK(Kdetsig, P, Ms, mp)

    # compute number of RVs required for a white and red noise model
    print 'Computing nRVs...'
    nRV = 2. * (sigRV_eff / sigK_target)**2

    if runGP:
	#if sigRV_act < 0:
	#    logg = float(unp.nominal_values(_compute_logg(Ms, Rs)))
	#    B_V = get_B_V(Teff, logg, Z)
	#    sigRV_act = get_sigmaRV_activity(Teff, Ms, Prot, B_V)
	#if sigRV_planet < 0:
	#    sigRV_planet = get_sigmaRV_planets(P,rp,Teff,Ms,sigRV_phot)
    	NGPtrials = int(NGPtrials)
        nRVGPs = np.zeros(NGPtrials)
        for i in range(NGPtrials):
            aGP = sigRV_act if sigRV_act > 0 else sigRV_eff
	    if aGP < 0:
	    	raise ValueError('aGP cannot be negative.')
            lambdaGP = Prot * (3 + np.random.randn() * .1)
	    if lambdaGP <= 0:
	    	raise ValueError('lGP cannot be <= 0.')
            GammaGP = 2 + np.random.randn() * .1
            GPtheta = aGP, lambdaGP, GammaGP, Prot, sigRV_planet
            keptheta = P, K
	    print GPtheta
            nRVGPs[i] = compute_nRV_GP(GPtheta, keptheta, sigRV_phot,
                                       sigK_target, duration=duration)
        nRVGP = np.median(nRVGPs)
    else:
        nRVGP = 0.
        
    # compute total observing time in hours
    tobs = nRV * (texp + toverhead) / 60.
    tobsGP = nRVGP * (texp + toverhead) / 60.

    # is SNRtarget set?
    try:
        _ = SNRtarget
    except NameError:
        SNRtarget = np.nan
        
    # write results to file
    NGPtrials = int(NGPtrials) if runGP else 0
    try:
      	_ = mags
    except NameError:
        mags, band_strs, centralwl_nm = np.zeros(1), [''], 0
    output = [P, rp, mp, K,
              mags, Ms, Rs, Teff, Z, vsini, Prot,
              band_strs, R, aperture, throughput, RVnoisefloor,
              centralwl_nm*1e-3, maxtelluric, toverhead, texp,
              SNRtarget, sigRV_phot, sigRV_act, sigRV_planet, sigRV_eff,
              sigK_target, nRV, nRVGP, NGPtrials, tobs, tobsGP]
    ##_write_results2file(output_fname, output)
    ##create_pdf(output_fname, output)
    if verbose_results:
        #_print_results(output, output_fname)
	print output_fname
	_save_RVFC(output, output_fname)
    return output


def do_checks(throughput, maxtelluric):
    if (throughput <= 0) or (throughput >= 1):
        raise ValueError('Throughput must be between 0-1.')
    if (maxtelluric <= 0) or (throughput >= 1):
        raise ValueError('Maxtelluric must be between 0-1.')


def _read_planet_input(input_planet_fname):
    '''
    Read-in planetary data from the input file.
    '''
    f = open(input_planet_fname, 'r')
    g = f.readlines()
    f.close()
    return float(g[3]), float(g[5]), float(g[7])


def _read_star_input(input_star_fname):
    '''
    Read-in stellar data from the input file.
    '''
    f = open(input_star_fname, 'r')
    g = f.readlines()
    f.close()
    return float(g[3]), float(g[5]), float(g[7]), float(g[9]), \
        float(g[11]), float(g[13]), float(g[15])


def _read_spectrograph_input(input_spectrograph_fname):
    '''
    Read-in spectrograph data from the input file.
    '''
    f = open(input_spectrograph_fname, 'r')
    g = f.readlines()
    f.close()
    return float(g[3])*1e-3, float(g[5])*1e-3, float(g[7]), \
        float(g[9]), float(g[11]), float(g[13]), float(g[15]), float(g[17])


def _read_sigRV_input(input_sigRV_fname):
    '''
    Read-in RV noise source data from the input file.
    '''
    f = open(input_sigRV_fname, 'r')
    g = f.readlines()
    f.close()
    return float(g[3]), float(g[5]), float(g[7]), float(g[9]), float(g[11])


def _get_spectral_bands(wlmin, wlmax):
    band_strs = np.array(['U','B','V','R','I','Y','J','H','K'])
    wlcens = np.array([.3531, .4430, .5537, .694, .8781, 1.0259, 1.2545,
                       1.631, 2.1498])
    wlwidth = np.array([.0657, .0973, .089, .207, .2316, .1084, .1548,
                        .2886, .3209])
    # define boundaries
    tolerance = .0
    lower_bnds, upper_bnds = (wlcens - wlwidth) * (1-tolerance), \
                            (wlcens + wlwidth) * (1+tolerance)
    # get bands
    bnds = np.append(np.where(abs(lower_bnds-wlmin) == \
                              np.min(abs(lower_bnds-wlmin))),
                     np.where(abs(upper_bnds-wlmax) == \
                              np.min(abs(upper_bnds-wlmax))))
    # expand if necessary
    if (lower_bnds[bnds[0]] > wlmin) and (bnds[0] != 0):
        bnds[0] -= 1
    if (upper_bnds[bnds[1]] < wlmax) and (bnds[1] != band_strs.size-1):
        bnds[1] += 1
    inds = np.arange(bnds.min(), bnds.max()+1)
    
    if inds.size == 0:
        raise ValueError('No spectral bands found.')

    return band_strs[inds]
    

def _compute_logg(Ms, Rs):
    '''
    Compute stellar logg in cgs units.
    '''
    Ms, Rs = rvs.Msun2kg(Ms), rvs.Rsun2m(Rs)
    return unp.log10(G * Ms / Rs**2 * 1e2)


def _compute_sigRV_phot(band_strs, mags, Teff, logg, Z, vsini, texp, R,
                        aperture, throughput, RVnoisefloor, centralwl_nm,
                        transmission_threshold, wl_telluric, trans_telluric):
    '''
    Calculate the photon-noise limited RV precision over the spectrograph's 
    full spectral domain.
    '''
    # get round values for PHOENIX stellar models
    Teffs = np.append(np.arange(23e2,7e3,1e2), np.arange(7e3,121e2,2e2))
    Teff_round = Teffs[abs(Teffs-Teff) == np.min(abs(Teffs-Teff))][0]
    loggs = np.arange(0, 6.1, .5)
    logg_round = loggs[abs(loggs-logg) == np.min(abs(loggs-logg))][0]
    Zs = np.append(np.arange(-4,-1,dtype=float), np.arange(-1.5,1.5,.5))
    Z_round = Zs[abs(Zs-Z) == np.min(abs(Zs-Z))][0]
    
    # compute sigmaRV in each band for a fixed texp
    sigmaRVs, SNRtargets = np.zeros(mags.size), np.zeros(mags.size)
    for i in range(sigmaRVs.size):
        t0 = time.time()
        SNRtargets[i] = get_snr(mags[i], band_strs[i], texp, aperture, throughput, R)
        wl, spec = get_reduced_spectrum(Teff_round, logg_round, Z_round, vsini,
                                        band_strs[i], R, centralwl_nm*1e-3,
                                        SNRtargets[i])
        sigmaRVs[i] = compute_sigmaRV(wl, spec, mags[i], band_strs[i], texp,
                                      aperture, throughput, R,
                                      transmission_threshold, wl_telluric,
                                      trans_telluric, SNRtargets[i])
        print 'Took %.1f seconds\n'%(time.time()-t0)

    # compute SNRtarget
    SNRtarget = SNRtargets.mean()

    # compute sigmaRV over all bands
    sigRV_phot = 1 / np.sqrt(np.sum(1. / sigmaRVs**2))
    sigRV_phot = sigRV_phot if sigRV_phot > RVnoisefloor \
                 else float(RVnoisefloor)
    return SNRtarget, sigRV_phot


def _get_planet_mass(rps, Fs=336.5):
    '''
    '''
    rps, Fs = np.ascontiguousarray(rps), np.ascontiguousarray(Fs)
    assert rps.size == Fs.size

    # isolate different regimes
    Fs = Fs*1367*1e7*1e-4   # erg/s/cm^2
    rocky = rps < 1.5
    small = (1.5 <= rps) & (rps < 4)
    neptune = (4 <= rps) & (rps < 13.668)
    giant = rps >= 13.668

    # compute mean mass in each regime
    mps = np.zeros(rps.size)
    mps[rocky]   = .44*rps[rocky]**3 + .614*rps[rocky]**4
    mps[small]   = 2.69 * rps[small]**(.93)
    mps[neptune] = (rps[neptune]*Fs[neptune]**.03 / 1.78)**(1/.53)
    mps[giant]   = np.random.uniform(150,2e3)

    return mps


def _get_sigK(Kdetsig, P, Ms, mp):
    '''
    Compute the desired semi-amplitude detection measurement uncertainty.
    '''
    K = rvs.RV_K(P, Ms, mp)
    return K, K / float(Kdetsig)


def _get_magnitudes(band_strs, mags, Ms):
    # Use isochrone colours to compute mags in each band of interest
    # solar metallicity at a fixed age of 10^9 yrs
    MU, MB, MV, MR, MI, MY, MJ, MH, MK = _get_absolute_stellar_magnitudes(Ms)

    # only consider V or J as reference bands. could use more...
    assert ('V' in band_strs) or ('J' in band_strs)
    if 'V' in band_strs:
        ref_band, ref_mag, ref_absmag = 'V', mags[band_strs == 'V'], MV
    else:
        ref_band, ref_mag, ref_absmag = 'J', mags[band_strs == 'J'], MJ

    Bmag = MB - ref_absmag + ref_mag
    Vmag = MV - ref_absmag + ref_mag

    return Bmag, Vmag


def _get_absolute_stellar_magnitudes(Ms):
    '''
    Get the absolute magnitudes of a star with a given stellar mass at a 
    given age using the isochrones from 2005A&A...436..895G

    Parameters
    ----------
    `Ms': scalar
        The stellar mass in MSun
    
    Returns
    -------
    `mags': numpy.array
        The absolute magnitudes of the star 

    '''
    # Appoximate MS lifetime to see at what age to obtain colours
    logtMS_yrs = np.log10(1e10 * (1./Ms)**(2.5))
    logage = round(logtMS_yrs*.77 / 5e-2) * 5e-2   # get ~3/4 through MS

    # First set of isochrones (ubvri)
    logages,Mss,Mus,Mbs,Mvs,Mrs,Mis,Mjs,Mhs,Mks = \
                                np.loadtxt('/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputData/isoc_z019_ubvrijhk.dat',
                                usecols=(0,1,7,8,9,10,11,12,13,14)).T
    g = abs(logages-logage) == np.min(abs(logages-logage))
    Mss,Mus,Mbs,Mvs,Mrs,Mis,Mjs,Mhs,Mks = Mss[g],Mus[g],Mbs[g],Mvs[g],Mrs[g], \
                                          Mis[g],Mjs[g],Mhs[g],Mks[g]
    g = abs(Mss-Ms) == np.min(abs(Mss-Ms))
    if g.sum() > 1:
	g = np.where(g)[0][0]
    Mu,Mb,Mv,Mr,Mi,Mj,Mh,Mk = Mus[g],Mbs[g],Mvs[g],Mrs[g],Mis[g],Mjs[g], \
                              Mhs[g],Mks[g]
    # Second set of isochrones (ZYJHK)
    logages2,Mss2,MZs,MYs,MJs,MHs,MKs = \
                                np.loadtxt('/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputData/isoc_z019_ZYJHK.dat',
                                usecols=(0,1,7,8,9,10,11)).T
    g = abs(logages2-logage) == np.min(abs(logages2-logage))
    Mss2,MZs,MYs,MJs,MHs,MKs = Mss2[g],MZs[g],MYs[g],MJs[g],MHs[g],MKs[g]
    g = abs(Mss2-Ms) == np.min(abs(Mss2-Ms))
    if g.sum() > 1:
	g = np.where(g)[0][0]
    MZ,MY,MJ,MH,MK = MZs[g],MYs[g],MJs[g],MHs[g],MKs[g]    

    return Mu, Mb, Mv, Mr, Mi, MY, MJ, MH, MK 
    

def _scale_sigmaRV_phot(sigRV_phot, texp_old, texp_new):
    return sigRV_phot * np.sqrt(float(texp_old) / texp_new)


def _write_results2file(output_fname, magiclistofstuff2write):
    '''
    Write the resulting parameters to a .dat file.
    '''
    # create header with indices
    maglabels = ''
    for i in range(magiclistofstuff2write[3].size):
        maglabels += '%s magnitude\n'%magiclistofstuff2write[9][i]
    hdr = 'Orbital period (days)\nPlanetary radius (Earth radii)\nPlanetary mass(Earth masses)\nRV semi-amplitude(m/s)\n%sStellar mass (Solar masses)\nStellar Radius (Solar radii)\nEffective temperature (K)\n[Fe/H] (Solar units)\nProjected rotation velocity (km/s)\nRotation period (days)\nSpectral resolution\nAperture (meters)\nThroughput\nRV noise floor (m/s)\nReference wavelength (microns)\nMaximum fractional telluric absorption\nMinimum exposure time (minutes)\nMaximum exposure time (minutes)\nOverhead (minutes)\nPhoton noise limited RV (m/s)\nRV activity rms (m/s)\nAdditional planet RV rms (m/s)\nEffective RV rms (m/s)\nTarget K measurement uncertainty (m/s)\nExposure time (minutes)\nNumber of RV measurements\nNumber of RV measurements with GP\nTotal observing time (hours)\nTotal observing time with GP (hours)'%maglabels
    hdr, hdrv2 = hdr.split('\n'), ''
    for i in range(len(hdr)):
        hdrv2 += '# %i: %s\n'%(i, hdr[i])

    g = hdrv2
    for i in range(len(magiclistofstuff2write)):
        if i == 3:
            for j in range(magiclistofstuff2write[i].size):
                g += '%.4e\n'%magiclistofstuff2write[i][j]
        elif i == 9:
            pass
        else:
            g += '%.4e\n'%magiclistofstuff2write[i]

    # write .dat file
    try:
        os.mkdir('Results')
    except OSError:
        pass
    f = open(output_fname, 'w')
    f.write(g)
    f.close()


def _save_RVFC(output, output_fname):
    P, rp, mp, K, mags, Ms, Rs, Teff, Z, vsini, Prot, band_strs, R, aperture, throughput, RVnoisefloor, centralwl_microns, maxtelluric, toverhead, texp, SNRtarget, sigRV_phot, sigRV_act, sigRV_planet, sigRV_eff, sigK_target, nRV, nRVGP, NGPtrials, tobs, tobsGP = output
    g = '%.7f,%.4f,%.4f,%s,%s,%.4f,%.4f,%i,%.3f,%.3f,%.3f,%i,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%i,%.1f,%.1f,%.3f,%.3f'%(P,rp,mp,'-'.join(mags.astype(str)),''.join(band_strs),Ms,Rs,Teff,Z,vsini,Prot,R,aperture,throughput,RVnoisefloor,centralwl_microns,maxtelluric,toverhead,texp,sigRV_phot,sigRV_act,sigRV_planet,sigRV_eff,sigK_target,NGPtrials,nRV,nRVGP,tobs,tobsGP)
    f = open(output_fname, 'w')
    f.write(g)
    f.close()


def _print_results(output, output_fname=''):
    # get data
    P, rp, mp, K, mags, Ms, Rs, Teff, Z, vsini, Prot, band_strs, R, aperture, throughput, RVnoisefloor, centralwl_microns, maxtelluric, toverhead, texp, SNRtarget, sigRV_phot, sigRV_act, sigRV_planet, sigRV_eff, sigK_target, nRV, nRVGP, NGPtrials, tobs, tobsGP = output
    
    # get string to print
    g = '\n' + '#'*50
    g += '\n#\tPlanet parameters:\n'
    g += '# P  = %.3f days\n# rp = %.2f REarth\n# mp = %.2f MEarth\n# K  = %.2f m/s'%(P,rp,mp,K)
    g += '\n\n#\tStellar parameters:\n'
    g += '# mags  = %s (%s)\n# Ms    = %.2f MSun\n# Rs    = %.2f RSun\n# Teff  = %i K\n# vsini = %.1f km/s'%(', '.join(mags.astype(str)),''.join(band_strs),Ms,Rs,Teff,vsini)
    g += '\n\n#\tSpectrograph parameters:\n'
    g += '# R           = %i\n# Aperture    = %.1f m\n# Throughput  = %.2f\n# Noise floor = %.2f m/s'%(R,aperture,throughput,RVnoisefloor)
    g += '\n\n#\tRV noise parameters:\n'
    g += '# texp           = %.1f min\n# toverhead      = %.1f min\n# SNRtarget      = %.1f \n# sigRV_photon   = %.2f m/s\n# sigRV_activity = %.2f m/s\n# sigRV_planets  = %.2f m/s\n# sigRV_eff      = %.2f m/s'%(texp,toverhead,SNRtarget,sigRV_phot,sigRV_act,sigRV_planet,sigRV_eff)
    g += '\n' + '#'*50
    g += '\n\n#\tResults:  (NGPtrials = %i)\n'%NGPtrials
    g += '# Desired sigK = %.2f m/s  (%.1f sigma K detection)\n# Nrv          = %.1f\n# tobs         = %.2f hours\n# tobs         = %.2f nights\n# Nrv_GP       = %.1f\n# tobs_GP      = %.2f hours\n# tobs_GP      = %.2f nights\n'%(sigK_target,K/sigK_target,nRV,tobs,tobs/7.,nRVGP,tobsGP,tobsGP/7.)
    
    print g
    
    # save text file if desired
    if output_fname != '':
        h = open(output_fname, 'w')
        h.write(g)
        h.close()


def _save_results(output):
    # get data
    P, rp, mp, K, \
    mags, Ms, Rs, Teff, Z, vsini, Prot, \
    band_strs, R, aperture, throughput, RVnoisefloor, \
    centralwl_microns, maxtelluric, toverhead, \
    texp, sigRV_phot, sigRV_act, sigRV_planet, sigRV_eff, \
    sigK_target, nRV, nRVGP, NGPtrials, tobs, tobsGP = output

    # save mags to csv file first
    tocsv = np.zeros(9, dtype='str')
    all_band_strs = np.array(['U','B','V','R','I','Y','J','H','K'])
    tocsv[np.in1d(all_band_strs, band_strs)] = mags
    tocsv = ','.join(tocsv.astype(str))

    # add remaining parameters
    tocsv = tocsv + ',' + ','.join(np.array(output[:4]).astype(str))
    tocsv = tocsv + ',' + ','.join(np.array(output[5:11]).astype(str))
    tocsv = tocsv + ',' + ','.join(np.array(output[12:]).astype(str))

    # save to csv file for uploading into the RVFC for repeated calculations
    fname = ('%.6f_%.6f'%(time.time(), np.random.rand())).replace('.','d') + '.csv'
    f = open(fname, 'w')
    f.write(tocsv)
    f.close()


def clean_input_files(input_sigRV_fname, input_spectrograph_fname, 
		      input_planet_fname, input_star_fname):
    os.system('rm %s'%input_sigRV_fname)
    os.system('rm %s'%input_spectrograph_fname)
    os.system('rm %s'%input_planet_fname)
    os.system('rm %s'%input_star_fname)
