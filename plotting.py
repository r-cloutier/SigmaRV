from imports import *

global band_strs, Nband, NR, NTeff, Nlogg, NZ, Nvsini
band_strs = ['U','B','V','R','I','Y','J','H','K']
Nband, NR, NTeff, Nlogg, NZ, Nvsini = 9, 8, 68, 9, 9, 7


def get_oneband(band_str):
    # define grids
    band_strs = ['B','V','R','Y','J','H','K','U','I']
    Rs = np.arange(20e3,161e3,2e4)
    Teffs = np.append(np.arange(28e2,7e3,1e2), np.arange(7e3,121e2,2e2))[47:49]
    loggs = np.arange(2, 6.1, .5)
    Zs = np.append(np.arange(-4,-1,dtype=float), np.arange(-1.5,1.5,.5))
    vsinis = np.array([.05,.1,.5,1.,5.,10.,50.])

    assert band_str in band_strs
    sigRV_grid = np.zeros((NR, NTeff, Nlogg, NZ, Nvsini))
    for i in range(NR):
        for j in range(NTeff):
            print float(j)/NTeff
            for k in range(Nlogg):
                try:
                    f = fits.open('sigmaRVgrids/sigmaRVgrid_band%s_R%i_Teff%i_logg%.1f'%(band_str,Rs[i],Teffs[j],loggs[k]))[0].data
                    assert f.shape == (NR, Nvsini)
                    sigRV_grid[i,j,k,:,:] = f
                except IOError:
                    sigRV_grid[i,j,k,:,:] = np.nan

    return sigRV_grid

