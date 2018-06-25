import os
import numpy as np

band_strs = ['B','V','R','Y','J','H','K','U','I']
Rs = np.arange(20e3,161e3,2e4)
Teffs = np.append(np.arange(28e2,7e3,1e2), np.arange(7e3,121e2,2e2))[0:1]
loggs = np.arange(2, 6.1, .5)

for i in range(len(band_strs)):
    for j in range(Rs.size):
        for k in range(Teffs.size):
	    for l in range(loggs.size):

            	f = open('jobscript_template', 'r')
            	g = f.read()
            	f.close()

            	g = g.replace('<<band_str>>', band_strs[i])
            	g = g.replace('<<R>>', '%i'%Rs[j])
            	g = g.replace('<<Teff>>', '%i'%Teffs[k])
		g = g.replace('<<logg>>', '%.1f'%loggs[l])

            	h = open('jobscript', 'w')
            	h.write(g)
            	h.close()

            	#os.system('cat jobscript')
            	os.system('qsub jobscript')
            	os.system('rm jobscript')
