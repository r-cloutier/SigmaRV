from imports import *
from matplotlib.colors import LogNorm
from scipy.ndimage.filters import gaussian_filter


global band_strs, Nband, NR, NTeff, Nlogg, NZ, Nvsini
band_strs = ['U','B','V','R','I','Y','J','H','K']
Nband, NR, NTeff, Nlogg, NZ, Nvsini = 9, 8, 68, 9, 9, 7

cbticks = np.append(np.append(np.arange(.5,1,.1),np.arange(1,10,1)),
                    np.arange(10,101,10))
cbticklabels = np.repeat('', cbticks.size).astype('|S3')
cbticklabels[np.array([5,7,14,16,23])] = ['1','3','10','30','100']



def get_missing_files():
    # define grids
    Rs = np.arange(20e3, 161e3, 2e4)
    Teffs = np.append(np.arange(28e2,7e3,1e2), np.arange(7e3,121e2,2e2))
    loggs = np.arange(2, 6.1, .5)

    missing, missing_arr = '', np.zeros(0).astype('|S60')
    for b in band_strs:
        for r in Rs:
            for t in Teffs:
                for l in loggs:
                    fname = 'sigmaRVgrids/sigmaRVgrid_band%s_R%i_Teff%i_logg%.1f'%(b, r, t, l)
                    if os.path.exists(fname):
                        pass
                    else:
                        missing += '%s\n'%fname
                        missing_arr = np.append(missing_arr, fname)

    # write missing files
    h = open('missing_sigmaRV_files', 'w')
    h.write(missing)
    h.close()

    return missing_arr
        


def get_oneband(band_str):
    # define grids
    Rs = np.arange(20e3, 161e3, 2e4)
    Teffs = np.append(np.arange(28e2,7e3,1e2), np.arange(7e3,121e2,2e2))
    loggs = np.arange(2, 6.1, .5)
    Zs = np.append(np.arange(-4,-1,dtype=float), np.arange(-1.5,1.5,.5))
    vsinis = np.array([.05,.1,.5,1.,5.,10.,50.])

    assert band_str in band_strs
    sigRV_grid = np.zeros((NR, NTeff, Nlogg, NZ, Nvsini))
    for i in range(NR):
        for j in range(NTeff):
            for k in range(Nlogg):
                try:
                    f = fits.open('sigmaRVgrids/sigmaRVgrid_band%s_R%i_Teff%i_logg%.1f'%(band_str,Rs[i],Teffs[j],loggs[k]))[0].data
                    assert f.shape == (NZ, Nvsini)
                    sigRV_grid[i,j,k,:,:] = f
                except IOError:
                    sigRV_grid[i,j,k,:,:] = np.nan

    # save
    hdu = fits.PrimaryHDU(sigRV_grid)
    hdu.writeto('SigmaRV_Grids/SigmaRV_%sgrid'%band_str, overwrite=True)



def plot_grid(pltt=True, label=False, vmin=.5, vmax=5e2):
    # define grids
    Rs = np.arange(20e3, 161e3, 2e4)
    Teffs = np.append(np.arange(28e2,7e3,1e2), np.arange(7e3,121e2,2e2))
    loggs = np.arange(2, 6.1, .5)
    Zs = np.append(np.arange(-4,-1,dtype=float), np.arange(-1.5,1.5,.5))
    vsinis = np.array([.05,.1,.5,1.,5.,10.,50.])
    
    fig = plt.figure(figsize=(10,5.5))
    for i in range(Nband):
    #for i in [0,4,8]:

        # get band data
        try:
            fname = 'SigmaRV_Grids/SigmaRV_%sgrid'%band_strs[i]
            sigRVgrid = fits.open(fname)[0].data
            assert sigRVgrid.shape == (NR,NTeff,Nlogg,NZ,Nvsini)
            
            # plot Teff v R
            axr = fig.add_subplot(4,Nband,i+1)
            axr.text(.5, 1.07, '%s'%band_strs[i], weight='normal',
                     horizontalalignment='center', transform=axr.transAxes)
            cax = axr.pcolormesh(Teffs, Rs,
                                 reduce_dimension(sigRVgrid,(2,3,4)),
                                 cmap=plt.get_cmap('rainbow'),
                                 norm=LogNorm(vmin=vmin, vmax=vmax))
            axr.set_yticks(np.arange(5e4,16e4,5e4))
            if i == 0:
                axr.set_yticklabels(['50,000','100,000','150,000'], fontsize=9)
                axr.set_ylabel('Spectral\nResolution', fontsize=10)
            else:
                axr.set_yticklabels('')
            axr.set_ylim((2e4,16e4)), axr.set_xlim((28e2,12e3))
            axr.set_xticklabels('')
                
            if i == 0:
                cbar_axes = fig.add_axes([.25,.08,.5,.03])
                cbar = fig.colorbar(cax, cax=cbar_axes, ticks=cbticks,
                                    orientation='horizontal')
                cbar.set_label('$\sigma_{RV}$ [m s$^{-1}$]', labelpad=-1)
                cbar.ax.set_xticklabels(cbticklabels)
                
            # plot Teff v logg
            axl = fig.add_subplot(4,Nband,i+10)
            axl.pcolormesh(Teffs, loggs,
                           reduce_dimension(sigRVgrid,(0,3,4)).T,
                           cmap=plt.get_cmap('rainbow'),
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            axl.set_yticks(range(2,7,2))
            if i == 0:
                axl.set_yticklabels(range(2,7,2), fontsize=9)
                axl.set_ylabel('log g', fontsize=10)
            else:
                axl.set_yticklabels('')
            axl.set_ylim((2,6)), axl.set_xlim((28e2,12e3))
            axl.set_xticklabels('')

            # plot Teff v Z
            axz = fig.add_subplot(4,Nband,i+19)
            axz.pcolormesh(Teffs, Zs,
                           reduce_dimension(sigRVgrid,(0,2,4)).T,
                           cmap=plt.get_cmap('rainbow'),
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            axz.set_yticks(range(-4,2,2))
            if i == 0:
                axz.set_yticklabels(range(-4,2,2), fontsize=9)
                axz.set_ylabel('Z', fontsize=10)
            else:
                axz.set_yticklabels('')
            axz.set_ylim((-4,1)), axz.set_xlim((28e2,12e3))
            axz.set_xticklabels('')
            
            # plot Teff v vsini
            axv = fig.add_subplot(4,Nband,i+28)
            axv.pcolormesh(Teffs, vsinis,
                           reduce_dimension(sigRVgrid,(0,2,3)).T,
                           cmap=plt.get_cmap('rainbow'),
                           norm=LogNorm(vmin=vmin, vmax=vmax))
            axv.set_yscale('log'), axv.set_yticks(np.logspace(-1,1,3))
            if i == 0:
                axv.set_yticklabels(['0.1','1','10'], fontsize=9)
                axv.set_ylabel('v$_s$sini$_s$\n[km s$^{-1}$]', fontsize=10)
            elif i == 4:
                axv.set_yticklabels('')
                axv.text(-.75, -.4, 'T$_{eff}$ x 10$^{-3}$\t[K]', fontsize=12,
                         transform=axv.transAxes)
            else:
                axv.set_yticklabels('')
            axv.set_ylim((.05,50)), axv.set_xlim((28e2,12e3))
            axv.set_xticks(np.arange(3e3,13e3,3e3))
            axv.set_xticklabels(range(3,13,3), fontsize=9)
            #axv.set_xlabel('T$_{eff}$\n[x10$^3$ K]', fontsize=10)
            
        except IOError:
            pass
    
    fig.subplots_adjust(left=.09, bottom=.2, top=.95, right=.99, hspace=.07, wspace=.07)
    if label:
        plt.savefig('plots/sigRV_grid.png')
    if pltt:
        plt.show()
    plt.close('all')



def reduce_dimension(arr, axes, sigma=0):
    axes = np.sort(list(axes))[::-1]
    for i in range(len(axes)):        
        arr = np.nanmedian(arr, axis=axes[i])
    if sigma == 0:
        return arr
    else:
        return gaussian_filter(arr, sigma)
