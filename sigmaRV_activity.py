#!/usr/bin/python2.7
import numpy as np
import rvs_custom as rvs
from scipy.interpolate import interp1d
from astropy.io import ascii
from Teff2color import *

global path2sigRV
path2sigRV = '/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV'

def F82sigmaRV(F8, Teff):
    '''
    Convert the F8 flicker to an expected sigmaRV using Eqs. 3 & 4 in Cegla et 
    al. 2014.

    Parameters
    ----------
    `F8': array-like
        The 8-hour flicker or photometric variability on a timescale of 8 hours
        measured in parts-per-thousand.
    `Teff': array-like
        The stellar effective temperature in kelvin

    Returns
    -------
    `sigmaRVs': numpy.array
        The expected RV noise from activity as probed by the 8-hour photometric 
        flicker in m/s

    '''
    F8, Teff = np.ascontiguousarray(F8), np.ascontiguousarray(Teff)
    assert F8.size == Teff.size
    sigmaRVs = np.zeros(F8.size)

    g1 = Teff >= 6e3
    sigmaRVs[g1] = 84.23 * F8[g1] - 3.35

    g2 = Teff < 6e3
    sigmaRVs[g2] = 18.04 * F8[g2] - .98

    return sigmaRVs


def _get_F8(Teff):
    '''
    Get F8 values from Bastien+2013 after isolating dwarfs based on their 
    Xcrossing value describing the complexity of the light curve.
    '''
    Teffs, F8, R, Z = np.loadtxt('%s/InputData/nature12419-s2.txt'%path2sigRV,
                                usecols=(2,3,4,5)).T
    g = np.log10(Z) <= 1.85
    Teffs, F8 = Teffs[g], F8[g]

    # draw a value of F8
    if Teff < Teffs.min():
	g = np.argsort(Teffs)[:10]
    elif Teff > Teffs.max():
	g = np.argsort(Teffs)[-10::]
    else:
    	g = (Teffs <= Teff*1.1) & (Teffs >= Teff*.9)
    return np.random.choice(F8[g]) + np.random.rand() * 2e-2


def _get_logRhk_OLD(Prot, Ms):
    '''
    Draw a value for logRhk given the star's rotation period. 
    FGK stars from the ditrubiton from Lovis+2011 and M stars from the 
    relationship from Astudillo-Defru+2017.
    '''
    # 'M dwarf'
    if Ms <= .8:
        logRhk = -1.509*np.log10(Prot) - 2.55 if Prot > 10 else -4.045 
        logRhk += np.random.randn()*.2
        
    # FGK star
    else:
        lRhk1, Prot1 = np.loadtxt('%s/InputData/Lovisetal2011_withoutBcycle.dat'%path2sigRV,
                                  usecols=(4,9)).T
        lRhk2, Prot2 = np.loadtxt('%s/InputData/Lovisetal2011_withBcycle.dat'%path2sigRV,
                                  usecols=(4,9)).T
        logRhks, Prots = np.append(lRhk1, lRhk2), np.append(Prot1, Prot2)
        # boundary conditions
        if (Prot > Prots.max()):
            logRhk = -5.5
        elif (Prot < Prots.min()):
            logRhk = -4.2
        else:
            g = (Prots <= Prot*1.1) & (Prots >= Prots*.9)
            logRhk = np.random.choice(logRhks[g])
        logRhk += np.random.randn()

    return logRhk


def _get_logRhk_Mdwarf(Prot, Ms):
    '''
    Draw a value for logRhk given the star's rotation period. 
    FGK stars from the ditrubiton from Lovis+2011 and M stars from the 
    relationship from Astudillo-Defru+2017.
    '''
    assert Ms <= .8
    logRhk = -1.509*np.log10(Prot) - 2.55 if Prot > 10 else -4.045 
    logRhk += np.random.randn()*.2
    return logRhk


def logRhk2sigmaRV(logRhk, Teff, mean_only=False):
    '''
    Convert the logRhk chromospheric index to an expected sigmaRV using Eqs. 2, 
    3, and 4 from Santos et al. 2000.

    Parameters
    ----------
    `logRhk': array-like
        The chromospheric activity index logRhk
    `Teff': array-like
        The stellar effective temperature in kelvin
    `mean_only': boolean
        If True, the residual rms in the reported relation is not used to 
        resample the mean sigmaRV

    Returns
    -------
    `sigmaRVs': numpy.array
        The expected RV noise from activity as probed by logRhk in m/s
    
    '''
    R5, Teff = np.ascontiguousarray(1e5*10**logRhk), np.ascontiguousarray(Teff)
    assert R5.size == Teff.size
    sigmaRVs = np.zeros(R5.size)

    g1 = (Teff < 6650) & (Teff >= 6e3)
    sigmaRVs[g1] = 9.2 * R5[g1]**(.75)
    
    g2 = (Teff >= 52e2) & (Teff < 6e3)
    sigmaRVs[g2] = 7.9 * R5[g2]**(.55)

    g3 = (Teff >= 38e2) & (Teff < 52e2)
    sigmaRVs[g3] = 7.8 * R5[g3]**(.12)

    g4 = Teff < 38e2
    sigmaRVs[g4] = 2. * R5[g4]**(.1)

    if not mean_only:
        sigmaRVs[g1] += np.random.randn(sigmaRVs[g1].size) * 10**(.17)
        sigmaRVs[g2] += np.random.randn(sigmaRVs[g2].size) * 10**(.18)
        sigmaRVs[g3] += np.random.randn(sigmaRVs[g3].size) * 10**(.19)
        sigmaRVs[g4] += np.random.randn(sigmaRVs[g4].size) * 10**(.2)

    return sigmaRVs


def Prot2logRhk_M(Prot, Ms, mean_only=False):
    '''
    Convert the stellar rotation period to the logRhk chromospheric index for 
    stars with masses less than 0.8 solar masses from Astudillo-Defru et al. 
    2017.

    Parameters
    ----------
    `Prot': array-like
        The stellar rotation period in days
    `Ms': array-like
        The stellar mass in solar masses
    `mean_only': boolean
        If True, the residual rms in the reported relation is not used to 
        resample the mean logRhk

    Returns
    -------
    `logRhk': numpy.array
        The chromospheric activity index logRhk
    
    '''
    Prot, Ms = np.ascontiguousarray(Prot), np.ascontiguousarray(Ms)
    assert Prot.size == Ms.size
    if np.any(Ms > .8):
        raise ValueError("Relationship is only valid for stars with masses " + \
                         "less than or equal to 0.8 solar masses.")
    logRhks = np.zeros(Ms.size)

    sigm, sigb, siglogRhk = .007, .02, .093 if not mean_only else 0, 0, 0
    m = -1.509 + np.random.randn(Ms.size) * sigm
    b = -2.550 + np.random.randn(Ms.size) * sigb

    g1 = Prot >= 10    
    logRhks[g1] = m[g1] * np.log10(Prot[g1]) + b[g1]

    g2 = Prot < 10
    logRhks[g2] = -4.045 + np.random.randn(Ms[g2].size) * siglogRhk

    return logRhks

# logRhk for low-mass stars: http://adsabs.harvard.edu/abs/2017A%26A...600A..13A
# lgoRhk for FGK stars: http://adsabs.harvard.edu/abs/2011arXiv1107.5325L 


def get_prot_kepler(Teff, seed=None):
    '''
    Draw a stellar rotation period based on the measured distribution from
    McQuillan+2014 (2014ApJS..211...24M).

    Parameters
    ----------
    `Teff': scalar
        The effective temperature of the star whose rotation period is being
        sampling
    `protseed': scalar
        Seed for the random number generator used to draw the stellar rotation
        period which is not know a-priori for the TESS stars from Sullivan

    Returns
    -------
    `Prot': float
        The star's sampled rotation period in days

    '''
    Teffs, Prots = np.loadtxt('%s/InputData/asu.tsv'%path2sigRV, skiprows=37).T
    # Isolate range of effective temperatures
    dT = 1e2
    if Teff > Teffs.max():
        g = Teffs >= Teffs.max()-dT
    elif Teff < Teffs.min():
        g = Teffs <= Teffs.min()+dT
    else:
        g = (Teffs >= Teff-dT) & (Teffs <= Teff+dT)
    # Set seed
    if seed != None:
        np.random.seed(int(seed))
    return np.random.choice(Prots[g]) + np.random.randn() * .1


def draw_prot_empirical(Ms):
    '''
    Draw a rotation period from either the empirical distribution from 
    Pizzolato+2003 (FGK) or Newton+2016 (M dwarfs).
    '''
    if Ms > .6:
        B_Vs, Prots, Mss = np.loadtxt('%s/InputData/PizzolatoData.tsv'%path2sigRV,
                                      skiprows=35).T
    else:
        Prots, Mss, As, _ = read_newtondata()
        # remove so have only 25% fast rotators
        slow = Prots > 10
        Nfast = int(np.round(Prots[slow].size * .25 / (1-.25)))
        fast = Prots < 10
        inds = np.arange(Prots[fast].size)
        np.random.shuffle(inds)
        Prots = np.append(Prots[slow], Prots[fast][inds][:Nfast])
        Mss = np.append(Mss[slow], Mss[fast][inds][:Nfast])
                          
    # sample Prot
    g = (Mss >= Ms*.9) & (Mss <= Ms*1.1)
    Prot = np.random.choice(Prots[g]) if Mss[g].size > 0 else \
           np.random.uniform(.5,20)
    return Prot + np.random.randn() * Prot * .2 


def get_prot_gyrochronology(Teff, B_V):
    '''
    Draw the stellar rotation period using gyrochronology based on the star's 
    B-V colour and its drawn age for a field star.

    Parameters
    ----------
    `B_V': scalar
        The B-V colour of the star

    Returns
    -------
    `Prot': float
        The star's sampled rotation period in days

    '''
    # gyrochronology: Barnes 2003, Barnes & Kim 2010
    # Meibom et al 2006 (good up to 2.5 Gyrs)
    f = .77*(B_V-.47)**(.55) if B_V >= .47 else 0.
    t_Myr = draw_age()
    print t_Myr
    return np.sqrt(t_Myr) * f


def draw_age():
    '''
    Get age from the stellar age distribution of exoplanet hosting Solar-type 
    stars from Silva-Aguirre et al 2015.
    '''
    ages, eages = np.loadtxt('%s/InputData/ExoplanetHostStarAges.tsv'%path2sigRV,
                             skiprows=37).T
    ind = np.random.choice(np.arange(ages.size))
    return (ages[ind] + np.random.randn() * eages[ind]) * 1e3  # in Myr


def read_newtondata():
    '''
    Read the M dwarf rotation periods, masses, and photmetric variability 
    amplitudes into numpy arrays.
    '''
    data = ascii.read('%s/InputData/newton16.txt'%path2sigRV)
    nstars = data['P'].size
    Prots = np.zeros(0)
    Mss = np.zeros(0)
    As = np.zeros(0)
    grades = np.zeros(0)
    for i in range(nstars):
        grade = data['Type'][i]
        if grade == 'A' or grade == 'B':
            Prots = np.append(Prots, data['P'][i])
            ms = data['Mass'][i] if type(data['Mass'][i]) == np.float64 else 0.
            Mss = np.append(Mss, ms)
            As = np.append(As, data['a'][i])
            if grade.strip() == 'A':
                grades = np.append(grades, 1)
            else:
                grades = np.append(grades, 2)
    # Remove bad masses (not measured)
    good = np.where(Mss > 0)[0]
    return Prots[good], Mss[good], As[good], grades[good]


def BV2logtau(B_V):
    '''
    Convert B-V colour to the log convective turnover time in days using 
    Noyes et al 1984 Eq 4.
    '''
    x = 1. - B_V
    if x > 0:
        logtau = 1.362 - .166*x + .025*x**2 - 5.323*x**3
    else:
        logtau = 1.362 - .14*x
    return logtau


def Prot2logRhk(Prot, B_V):
    '''
    Convert the stellar rotation period and B-V colour into an 
    expected logRhk using Eq 3 from Noyes et al 1984.
    '''
    logtau = BV2logtau(B_V)
    logRhks = np.linspace(-4,-6,1000)
    logR5 = np.log10(1e5 * 10**logRhks)
    Prots = 10**(.324 - .4*logR5 - .283*logR5**2 - 1.325*logR5**3 + logtau)
    # interpolate to input Prot
    fint = interp1d(Prots, logRhks, bounds_error=False,
                    fill_value=(logRhks.min(), logRhks.max()))
    return float(fint(Prot)) + np.random.randn() * .2


def get_B_V(Teff, logg, FeH):
    Teffs,_,_,_,B_Vs,_,_,_,_,_,_,_ = get_data()
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], B_Vs[g])
    B_V = fint(Teff)
    return B_V


def get_sigmaRV_activity(Teff, Ms, Prot, B_V):
    '''
    Estimate the RV rms due to stellar activity (measured in the optical).
    '''   
    # slowly rotating FGK star
    if Teff >= 4e3 and Prot > 10:
        F8 = _get_F8(Teff)
        sigmaRV_act = F82sigmaRV(F8, Teff)        

    # rapidly rotating FGK star
    elif Teff >= 4e3:
        logRhk = Prot2logRhk(Prot, B_V)
        sigmaRV_act = logRhk2sigmaRV(logRhk, Teff)

    # M dwarf
    else:
        logRhk = _get_logRhk_Mdwarf(Prot, Ms)
        sigmaRV_act = logRhk2sigmaRV(logRhk, Teff)

    return float(abs(sigmaRV_act))
