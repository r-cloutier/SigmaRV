#!/usr/bin/python2.7
from imports import *


def create_pdf(output_fname, magiclistofstuff2write):

    # make sure the output names are not common with any scripts
    assert output_fname not in [s[:-3] for s in glob.glob('*py')]

    # create plots if necessary
    figures = False
    sigRV_acts, sigRV_planets, sigRV_effs,_,_,nRVs, nRVGPs, tobss, tobsGPs = \
                                                    magiclistofstuff2write[-9:]
    if sigRV_acts.size > 1:
        plot_hist(np.array([sigRV_acts, sigRV_planets, sigRV_effs]).T,
                  ['RV activity', 'RV planets', 'Effective RV'], 
                  'RV noise sources [m/s]', '%s_sigRVs.png'%output_fname)
        plot_hist(np.array([nRVs, nRVGPs]).T, ['nRV (white)', 'nRV (w/ GP)'],
                  'Number of RVs', '%s_nRVs.png'%output_fname)
        plot_hist(np.array([tobss, tobsGPs]).T,
                  ['Observing time (white)','Observing time (w/ GP)'],
                  'Total observing time [hours]', '%s_tobs.png'%output_fname)
        figures = True

    # add parameters to tex file
    g = write_pdf_str(output_fname, magiclistofstuff2write, figures=figures)
        
    # write latex file
    tex_fname = '%s.tex'%output_fname
    h = open(tex_fname, 'w')
    h.write(g)
    h.close()

    # create the pdf and clean up
    os.system('pdflatex %s'%tex_fname)
    os.system('mv %s.pdf Results/'%output_fname)
    os.system('rm %s*'%output_fname)


    
def write_pdf_str(output_fname, magiclistofstuff2write, figures=False):
    P, rp, mp, K, mags, Ms, Rs, Teff, Z, vsini, Prot, band_strs, R, aperture, throughput, RVnoisefloor, centralwl_microns, SNRtarget, transmission_threshold, texpmin, texpmax, toverhead, sigRV_phot, sigRV_acts, sigRV_planets, sigRV_effs, sigK_target, texp, nRVs, nRVGPs, tobss, tobsGPs = magiclistofstuff2write
    Kdetsig = K / sigK_target
    
    # open/modify latex template
    f = open('outputpdf_template.tex', 'r')
    g = f.read()
    f.close()

    # add parameters
    title = output_fname.replace('_', '\_')
    g = g.replace('<<title>>', title)
    magstr = ', '.join(['%s = %.2f'%(band_strs[i], mags[i])
                        for i in range(len(mags))])
    g = g.replace('<<mags>>', magstr)
    fastlist = ['P', 'rp', 'mp', 'K', 'Ms', 'Rs', 'Teff', 'Z', 'vsini', 'Prot',
                'R', 'aperture', 'throughput', 'RVnoisefloor',
                'centralwl_microns', 'SNRtarget', 'transmission_threshold',
                'texpmin', 'texpmax', 'toverhead', 'sigRV_phot', 'Kdetsig',
                'sigK_target', 'texp']
    for i in range(len(fastlist)):
        if fastlist[i] in ['Teff','R']:
            fmt = '%i'
        elif fastlist[i] in ['Ms','Rs','sigK_target']:
            fmt = '%.3f'
        elif fastlist[i] in ['Kdetsig']:
            fmt = '%.1f'
        else:
            fmt = '%.2f'
        g = g.replace('<<%s>>'%fastlist[i], fmt%eval(fastlist[i]))

    # add sigRVs depending on size
    if sigRV_acts.size > 1:
        g = g.replace('<<sigRV_act>>',
            'Median RV rms from activity [m/s] & %.2f'%(np.median(sigRV_acts)))
        g = g.replace('<<sigRV_planets>>',
'Median RV rms from additional planets [m/s] & %.2f'%(np.median(sigRV_planets)))
        g = g.replace('<<sigRV_eff>>',
        'Median effective RV precision [m/s] & %.2f'%(np.median(sigRV_effs)))
    else:
        g = g.replace('<<sigRV_act>>',
                      'RV rms from activity [m/s] & %.2f'%float(sigRV_acts))
        g = g.replace('<<sigRV_planets>>',
            'RV rms from additional planets [m/s] & %.2f'%float(sigRV_planets))
        g = g.replace('<<sigRV_eff>>',
        'Effective RV precision [m/s] & %.2f'%float(sigRV_effs))        

    # add median results
    g = g.replace('<<nRV>>', '%.1f'%(np.median(nRVs)))
    g = g.replace('<<nRVGP>>', '%.1f'%(np.median(nRVGPs)))
    g = g.replace('<<tobs>>', '%.2f'%(np.median(tobss)))
    g = g.replace('<<tobsGP>>', '%.2f'%(np.median(tobsGPs)))
    g = g.replace('<<tobs_nights>>', '%.2f'%(np.median(tobss)/7.))
    g = g.replace('<<tobsGP_nights>>', '%.2f'%(np.median(tobsGPs)/7.))

    # add figures if available
    if figures:
        figstr = '\\begin{figure}[h]\n\centering\n'
        figstr += '\includegraphics[width=\hsize]{%s_sigRVs.png}\n'%output_fname
        figstr += '\caption{sigRVs}\n'
        figstr += '\end{figure}\n\n\\begin{figure}\n\centering\n'
        figstr += '\includegraphics[width=\hsize]{%s_nRVs.png}\n'%output_fname
        figstr += '\caption{nRVs}\n'
        figstr += '\end{figure}\n\n\\begin{figure}\n\centering\n'
        figstr += '\includegraphics[width=\hsize]{%s_tobs.png}\n'%output_fname
        figstr += '\caption{tobs}\n'
        figstr += '\end{figure}\n'

    else:
        figstr = ''   

    g = g.replace('<<figures>>', figstr)

    return g


def plot_hist(arr, labels, xlabel, fname):
    NGPtrials, Narr = arr.shape
    assert len(labels) == Narr
    
    fig = plt.figure(figsize=(6,4))
    ax = fig.add_subplot(111)
    cols = ['b', 'g', 'r', 'k', 'c']
    for i in range(Narr):
        ax.hist(arr[:,i], bins=10, histtype='step',
                color=cols[i], lw=3, label=labels[i])

    ax.set_xlabel(xlabel), ax.set_ylabel('Number of trials')
    ax.legend()
    plt.savefig(fname)
    plt.close('all')
