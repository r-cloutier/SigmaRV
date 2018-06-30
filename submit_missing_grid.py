import os
import numpy as np

def get_params_from_fname(fname):
    band = fname.split('band')[-1].split('_')[0]
    R = int(fname.split('R')[-1].split('_')[0])
    Teff = int(fname.split('Teff')[-1].split('_')[0])
    logg = float(fname.split('logg')[-1])
    return band, R, Teff, logg

d = np.genfromtxt('missing_sigmaRV_files', dtype='|S60')
start, end = 0, 1000

for i in range(start,end):

    f = open('jobscript_template', 'r')
    g = f.read()
    f.close()

    band_str, R, Teff, logg = get_params_from_fname(d[i])
    g = g.replace('<<band_str>>', band_str)
    g = g.replace('<<R>>', '%i'%R)
    g = g.replace('<<Teff>>', '%i'%Teff)
    g = g.replace('<<logg>>', '%.1f'%logg)

    h = open('jobscript', 'w')
    h.write(g)
    h.close()

    #os.system('cat jobscript')
    os.system('qsub jobscript')
    os.system('rm jobscript')
