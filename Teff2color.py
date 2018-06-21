#!/usr/bin/python2.7
from imports import *

def get_data():
    '''get theoretical data from Lejeune+1998'''
    d = np.loadtxt('/mnt/scratch-lustre/cloutier/SigmaRV/SigmaRV/InputData/LCB98_corrected.tsv',
    		   delimiter=';', skiprows=51).T
    g = d[0] <= 1e4
    Teff,logg,FeH,U_B,B_V,V_R,V_I,V_K,R_I,J_H,H_K,J_K = d[:,g]
    return Teff, logg, FeH, U_B, B_V, V_R, V_I, V_K, R_I, J_H, H_K, J_K

def isolate_logg_FeH(logg, FeH):
    _,loggs,FeHs,_,_,_,_,_,_,_,_,_ = get_data()
    return (abs(loggs-logg) == np.min(abs(loggs-logg))) & \
        (abs(FeHs-FeH) == np.min(abs(FeHs-FeH)))


# MAGNITUDE CONVERSIONS
Teffs,loggs,FeHs,U_Bs,B_Vs,V_Rs,V_Is,V_Ks,R_Is,J_Hs,H_Ks,J_Ks = get_data()

def _V2U(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], U_Bs[g])
    assert 2e3 <= Teff <= 1e4
    U_B = fint(Teff)
    Bmag = _V2B(Vmag, Teff, logg, FeH)
    Umag = U_B + Bmag
    return Umag

def _V2B(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], B_Vs[g])
    assert 2e3 <= Teff <= 1e4
    B_V = fint(Teff)
    Bmag = B_V + Vmag
    #plt.scatter(Teffs[g], B_Vs[g]), plt.plot([Teff], B_V, 'ro'), plt.show()
    return Bmag

def _V2R(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Rs[g])
    assert 2e3 <= Teff <= 1e4
    V_R = fint(Teff)
    Rmag = Vmag - V_R
    return Rmag

def _V2I(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Is[g])
    assert 2e3 <= Teff <= 1e4
    V_I = fint(Teff)
    Imag = Vmag - V_I
    return Imag

def _V2J(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], J_Ks[g])
    assert 2e3 <= Teff <= 1e4
    J_K = fint(Teff)
    Kmag = _V2K(Vmag, Teff, logg, FeH)
    Jmag = J_K + Kmag
    return Jmag

def _V2H(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], J_Hs[g])
    assert 2e3 <= Teff <= 1e4
    J_H = fint(Teff)
    Jmag = _V2J(Vmag, Teff, logg, FeH)
    Hmag = Jmag - J_H
    return Hmag

def _V2K(Vmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Ks[g])
    assert 2e3 <= Teff <= 1e4
    V_K = fint(Teff)
    Kmag = Vmag - V_K
    return Kmag

def _J2U(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Ks[g])
    assert 2e3 <= Teff <= 1e4
    V_K = fint(Teff)
    Kmag = _J2K(Jmag, Teff, logg, FeH)
    Vmag = V_K + Kmag
    Umag = _V2U(Vmag, Teff, logg, FeH)
    return Umag

def _J2B(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Ks[g])
    assert 2e3 <= Teff <= 1e4
    V_K = fint(Teff)
    Kmag = _J2K(Jmag, Teff, logg, FeH)
    Vmag = V_K + Kmag
    Bmag = _V2B(Vmag, Teff, logg, FeH)
    return Bmag

def _J2V(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Ks[g])
    assert 2e3 <= Teff <= 1e4
    V_K = fint(Teff)
    Kmag = _J2K(Jmag, Teff, logg, FeH)
    Vmag = V_K + Kmag
    return Vmag

def _J2R(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Ks[g])
    assert 2e3 <= Teff <= 1e4
    V_K = fint(Teff)
    Kmag = _J2K(Jmag, Teff, logg, FeH)
    Vmag = V_K + Kmag
    Rmag = _V2R(Vmag, Teff, logg, FeH)
    return Rmag

def _J2I(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], V_Ks[g])
    assert 2e3 <= Teff <= 1e4
    V_K = fint(Teff)
    Kmag = _J2K(Jmag, Teff, logg, FeH)
    Vmag = V_K + Kmag
    Imag = _V2I(Vmag, Teff, logg, FeH)
    return Imag

def _J2H(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], J_Hs[g])
    assert 2e3 <= Teff <= 1e4
    J_H = fint(Teff)
    Hmag = Jmag - J_H
    return Hmag

def _J2K(Jmag, Teff, logg, FeH):
    g = isolate_logg_FeH(logg, FeH)
    fint = interp1d(Teffs[g], J_Ks[g])
    assert 2e3 <= Teff <= 1e4
    J_K = fint(Teff)
    Kmag = Jmag - J_K
    return Kmag


def V2all(Vmag, Teff, logg, FeH):
    Vmag = float(Vmag)
    Umag = _V2U(Vmag, Teff, logg, FeH)
    Bmag = _V2B(Vmag, Teff, logg, FeH)
    Rmag = _V2R(Vmag, Teff, logg, FeH)
    Imag = _V2I(Vmag, Teff, logg, FeH)
    Ymag = np.nan
    Jmag = _V2J(Vmag, Teff, logg, FeH)
    Hmag = _V2H(Vmag, Teff, logg, FeH)
    Kmag = _V2K(Vmag, Teff, logg, FeH)
    return np.array([Umag, Bmag, Vmag, Rmag, Imag, Ymag, Jmag, Hmag, Kmag])

def J2all(Jmag, Teff, logg, FeH):
    Jmag = float(Jmag)
    Umag = _J2U(Jmag, Teff, logg, FeH)
    Bmag = _J2B(Jmag, Teff, logg, FeH)
    Vmag = _J2V(Jmag, Teff, logg, FeH)
    Rmag = _J2R(Jmag, Teff, logg, FeH)
    Imag = _J2I(Jmag, Teff, logg, FeH)
    Ymag = np.nan
    Hmag = _J2H(Jmag, Teff, logg, FeH)
    Kmag = _J2K(Jmag, Teff, logg, FeH)
    return np.array([Umag, Bmag, Vmag, Rmag, Imag, Ymag, Jmag, Hmag, Kmag])
