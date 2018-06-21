from RVFollowupCalculator import *

global SNRtarget
SNRtarget = 1e2

def compute_stellar_SNRs(band_str, R, Teff):
    '''Compute a grid of sigmaRVs for a single band and at a fixed spectral resolution.'''
    # get round values for PHOENIX stellar models
    #Teffs = np.append(np.arange(23e2,7e3,1e2), np.arange(7e3,121e2,2e2))
    loggs = np.arange(0, 6.1, .5)
    Zs = np.append(np.arange(-4,-1,dtype=float), np.arange(-1.5,1.5,.5))
    vsinis = np.array([.05,.1,.5,1.,5.,10.,50.])

    # get central wavelength
    band_strs = ['U','B','V','R','I','Y','J','H','K']
    assert band_str in band_strs
    centralwl_microns = .555 if band_str in band_strs[:5] else 1.25

    # get telluric spectrum
    transmission_fname = 'tapas_000001.ipac'
    wlTAPAS, transTAPAS = np.loadtxt('/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputData/%s'%transmission_fname,
                                     skiprows=23).T
    wlTAPAS *= 1e-3  # microns

    # compute sigmaRV for various stellar types
    sigmaRVs = np.zeros((loggs.size, Zs.size, vsinis.size))
    for j in range(loggs.size):
	for k in range(Zs.size):
	    for l in range(vsinis.size):
	    	wl, spec = get_reduced_spectrum(Teff, loggs[j], Zs[k], vsinis[l],
						band_str, R, centralwl_microns, SNRtarget)
		sigmaRVs[j,k,l] = compute_sigmaRV_grid(wl, spec, band_str, 
						       R, .02, wlTAPAS, transTAPAS)

    # save grid
    hdu = fits.PrimaryHDU(sigmaRVs)
    hdu.writeto('sigmaRVgrids/sigmaRVgrid_band%s_R%i_Teff%i'%(band_str,R,Teff), clobber=True)



if __name__ == '__main__':
    band_str = sys.argv[1]
    R = int(sys.argv[2])
    Teff = int(sys.argv[3])
    # use mag, texp, aperture, throughput to set the S/R and scale the results from a fixed S/R=100
    compute_stellar_SNRs(band_str, R, Teff)
