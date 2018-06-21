#!/usr/bin/python2.7
from imports import *
from PyAstronomy.pyasl import foldAt
from compute_sigK_QPGP import compute_sigmaK_GP


def compute_nRV_GP(GPtheta, keptheta, sigRV_phot, sigK_target, duration=100):
    assert len(GPtheta) == 5
    assert len(keptheta) == 2
    a, l, G, Pgp, s = GPtheta
    P, K = keptheta

    # search for target sigma by iterating over Nrv coarsely
    Nrvs = np.arange(10, 1001, 90)
    sigKs = np.zeros(Nrvs.size)
    for i in range(Nrvs.size):
        # get rv activity model
        gp = george.GP(a*(george.kernels.ExpSquaredKernel(l) + \
                          george.kernels.ExpSine2Kernel(G,Pgp)))
        t = _uniform_window_function(duration, Nrvs[i])
        #t = _uniform_gaps_window_function(duration, Nrvs[i], 1, 10)
        erv = np.repeat(sigRV_phot, t.size)
        gp.compute(t, np.sqrt(erv**2 + s**2))
        rv_act = gp.sample(t)

        # get planet model
        rv_kep = -K*np.sin(2*np.pi*foldAt(t, P))

        # get total rv signal with noise
        rv = rv_act + rv_kep + np.random.randn(t.size) * sigRV_phot

        # compute sigK
        theta = P, 0, K, a, l, G, Pgp, s
        sigKs[i] = compute_sigmaK_GP(theta, t, rv, erv)

    # fit powerlaw in log space to estimate Nrv for the target sigK
    g = np.isfinite(sigKs)
    p = np.poly1d(np.polyfit(np.log(sigKs[g]), np.log(Nrvs[g]), 1))
    Nrv = np.exp(p(np.log(sigK_target)))
    return Nrv


def _uniform_window_function(duration, Nrv):
    t = np.random.rand(Nrv) * duration
    return np.sort(t)


def _uniform_gaps_window_function(duration, Nrv, Ngaps, gap_duration):
    assert int(Ngaps) > 0
    assert duration > Ngaps*gap_duration
    t = np.random.rand(Nrv*50*Ngaps) * duration
    for i in range(Ngaps):
        tin, tout = float(duration)/(Ngaps+1)*(i+1) + \
                    np.linspace(-gap_duration, gap_duration,2)/2.
        t = np.delete(t, np.where((t >= tin) & (t <= tout))[0])
    # remove excess measurements
    inds = np.arange(t.size)
    np.random.shuffle(inds)
    return np.sort(t[inds][:int(Nrv)])
