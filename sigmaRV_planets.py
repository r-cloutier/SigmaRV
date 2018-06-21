#!/usr/bin/python2.7
import numpy as np
import rvs_custom as rvs
import massradius as mr


def draw_FGK_planets(P, rp, nplanets, Ms):
    '''
    Draw planets from around FGK stars from the occurrence rates derived in 
    Fressin et al 2013. Ensure that each pair in the multi-planet system is 
    Lagrange stable given circular orbits.
    '''
    assert nplanets > 1
    Pgrid = np.array([.8,2.,3.4,5.9,10.,17.,29.,50.,85.,145.,245.,418.])
    rpgrid = np.array([.8,1.25,2.,4.,6.,22.])
    occurrence = np.array([[.015,.067,.17,.18,.27,.23,.35,.71,1.25,.94,1.05],
                           [.004,.006,.11,.091,.29,.32,.49,.66,.43,.53,.24],
                           [.035,.18,.73,1.93,3.67,5.29,6.45,5.25,4.31,3.09,1.29],
                           [.17,.74,1.49,2.90,4.3,4.49,5.29,3.66,6.54,4.65,3.01],
                           [.18,.61,1.72,2.7,2.7,2.93,4.08,3.46,4.55,5.25,4.3]]).T
    occurrence *= 1e-2

    # sample planets until we get the correct multiplicity
    Ps, mp = np.array([P]), mr.radF2mass(rp)
    while Ps.size < nplanets:
        
        Ps, rps = np.array([P]), np.array([rp])
        mps = mr.radF2mass(rps, Fs=np.repeat(336.5, rps.size))
        for i in range(Pgrid.size-1):
            for j in range(rpgrid.size-1):

                if np.random.rand() <= occurrence[i,j]:

                    # draw a system
                    Ps  = np.append(Ps, np.random.uniform(Pgrid[i], Pgrid[i+1]))
                    rps = np.append(rps, np.random.uniform(rpgrid[j], rpgrid[j+1]))
                    mps = mr.radF2mass(rps, Fs=np.repeat(336.5, rps.size))

        sort = np.argsort(Ps)
        Ps, rps, mps = Ps[sort], rps[sort], mps[sort]

        # get stable pairs
        if Ps.size >= nplanets:
            lagrangestable = rvs.is_Lagrangestable(Ps, Ms, mps,
                                                   np.zeros(Ps.size))
            if lagrangestable[-1] == 1:
                lagrangestable = np.append(lagrangestable.astype(bool), True)
            else:
                lagrangestable = np.append(lagrangestable.astype(bool), False)
            Ps, rps, mps = Ps[lagrangestable], rps[lagrangestable], \
                           mps[lagrangestable]

    # exclude TESS planet
    Ps, rps, mps = Ps[1:], rps[1:], mps[1:]

    # get correct number of planets
    if Ps.size > nplanets-1:
        inds = np.arange(Ps.size)
        np.random.shuffle(inds)
        Ps, rps, mps = Ps[inds][:nplanets-1], rps[inds][:nplanets-1], \
                       mps[inds][:nplanets-1]
    
    # compute semi-amplitudes
    Ks = rvs.RV_K(Ps, Ms, mps)
            
    return Ps, rps, mps, Ks


def draw_M_planets(P, rp, nplanets, Ms):
    '''
    Draw planets from around M stars from the occurrence rates derived in 
    Dressing & Charbonneau 2015. Ensure that each pair in the multi-planet 
    system is Lagrange stable given circular orbits.
    '''
    assert nplanets > 1
    rpgrid = np.linspace(.5, 4, 8)
    Pgrid = np.array((.5, 1.7, 5.5, 18.2, 60.3, 200))
    occurrence = np.array(((1.38, 8.42, 20.59, 0, 0),
                           (1.95, 9.94, 0, 26.85, 28.85),
                           (0.41, 4.15, 0, 24.59, 19.98),
                           (0, 2.72, 18.73, 27.58, 18.08),
                           (0, 1.59, 8.29, 14.51, 8.61),
                           (0, 0.65, 3.25, 3.37, 1.97),
                           (0, 0.38, 1.05, 0.56, 0))).T
    occurrence *= 1e-2

    # sample planets until we get the correct multiplicity
    Ps, mp = np.array([P]), mr.radF2mass(rp)
    while Ps.size < nplanets:

        Ps, rps = np.array([P]), np.array([rp])
        mps = mr.radF2mass(rps, Fs=np.repeat(336.5, rps.size))
        for i in range(Pgrid.size-1):
            for j in range(rpgrid.size-1):

                if np.random.rand() <= occurrence[i,j]:

                    # draw a system
                    Ps  = np.append(Ps, np.random.uniform(Pgrid[i], Pgrid[i+1]))
                    rps = np.append(rps, np.random.uniform(rpgrid[j],
                                                           rpgrid[j+1]))
                    mps = mr.radF2mass(rps, Fs=np.repeat(336.5, rps.size))
                    
        sort = np.argsort(Ps)
        Ps, rps, mps = Ps[sort], rps[sort], mps[sort]

        # get stable pairs
        if Ps.size >= nplanets:
            lagrangestable = rvs.is_Lagrangestable(Ps, Ms, mps,
                                                   np.zeros(Ps.size))
            if lagrangestable[-1] == 1:
                lagrangestable = np.append(lagrangestable.astype(bool), True)
            else:
                lagrangestable = np.append(lagrangestable.astype(bool), False)
            Ps, rps, mps = Ps[lagrangestable], rps[lagrangestable], \
                           mps[lagrangestable]

    # exclude TESS planet
    Ps, rps, mps = Ps[1:], rps[1:], mps[1:]

    # get correct number of planets
    if Ps.size > nplanets-1:
        inds = np.arange(Ps.size)
        np.random.shuffle(inds)
        Ps, rps, mps = Ps[inds][:nplanets-1], rps[inds][:nplanets-1], \
                       mps[inds][:nplanets-1]
    
    # compute semi-amplitudes
    Ks = rvs.RV_K(Ps, Ms, mps)
            
    return Ps, rps, mps, Ks


def _compute_sigmaRV_planets(Ks):
    rms = np.sin(np.linspace(0,2*np.pi,100)).std()
    return float(np.sqrt(np.sum((Ks*rms)**2)))


# rvs.is_Lagrangestable(Ps, Ms, mps, eccs)
def get_sigmaRV_planets(P, rp, Teff, Ms, sigmaRV_phot):
    '''
    Compute the RV rms due to additional planets with P > TESS planet P. 
    Additional planets are drawn from the known planet occurrences rates. 
    Sampled planets with K > sigmaRV_phot, are assumed to be detected in RV, 
    and modelled such that they do not contribute to the effective RV rms.
    '''
    # FGK star
    if Teff > 3800:
        mult = int(np.round(1.1 + np.random.randn() * 0.1))
	if mult > 0:
            _,_,_,Ks = draw_FGK_planets(P, rp, mult+1, Ms)
	else:
	   Ks = np.zeros(0)
    # M dwarf
    else:
        mult = int(np.round(2.5 + np.random.randn() * 0.2 ))
	if mult > 0:
	    _,_,_,Ks = draw_M_planets(P, rp, mult+1, Ms)
	else:
	    Ks = np.zeros(0)

    # only include undetected planets with K < sigmaRV_phot
    Ks = Ks[Ks < sigmaRV_phot]

    return _compute_sigmaRV_planets(Ks)
