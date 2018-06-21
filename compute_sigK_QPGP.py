#!/usr/bin/python2.7
import numpy as np
import sympy
from PyAstronomy.pyasl import foldAt

sympy.init_printing()


def compute_sigmaK_GP(theta, t_arr, rv_arr, erv_arr):
    '''P, T0, Krv, a, l, G, Pgp, s_jitter = theta'''
    B = _compute_Fisher_information_GP(theta, t_arr, rv_arr, erv_arr)
    C = np.linalg.inv(B)
    sigK = np.sqrt(np.diag(C)[0])
    return sigK


def _covariance_matrix(theta, t, sig=np.zeros(0)):  # this is working
    a, l, G, P, s = theta
    t = t.reshape(t.size, 1)
    K = np.zeros ((t.size, t.size))
    K += -.5/l**2 * (np.tile(t[:,0], t.shape) - \
                     np.tile(t[:,0], t.shape).T)**2
    K += -G**2 * np.sin(np.pi/P * abs(np.tile(t[:,0], t.shape) - \
                                      np.tile(t[:,0], t.shape).T))**2
    K = a*a*np.exp(K)
    s2 = s**2 + sig**2 if sig.size == t.size else s**2
    return K + s2 * np.eye(t.size)


def _compute_Fisher_information_GP(theta, t_arr, rv_arr, erv_arr):
    '''Compute the Fisher information matrix term-by-term for a circular keplerian 
    RV model and a QP GP red noise model.
    theta = list of parameter values (P, T0, Krv, a, l, G, Pgp, s)
    '''
    # get orbital phase array
    assert len(theta) == 8
    P, T0 = theta[:2]
    sort = np.argsort(t_arr)
    t_arr, rv_arr, erv_arr = t_arr[sort], rv_arr[sort], erv_arr[sort]
    rv_arr -= np.median(rv_arr)  # roughly center on zero
    phi_arr = 2*np.pi * foldAt(t_arr, P, T0)
    thetavals = theta[2:]
    
    # define variables
    Krv, a, l, G, P, s = sympy.symbols('Krv a lambda Gamma P s')
    symbol_vals = Krv, a, l, G, P, s
    assert len(symbol_vals) == len(thetavals)
    
    # define time-series symbols
    dt, phi, rv, erv, deltafunc = sympy.symbols('dt phi RV sigma delta')
    y = rv - (-Krv * sympy.sin(phi))  # residual vector

    # compute QP covariance function
    k = a*a*sympy.exp(-.5*dt**2 / l**2 - G*G*sympy.sin(np.pi*abs(dt)/P)**2) + \
        deltafunc * (erv**2 + s**2)
    symbol_arrs = dt, phi, rv, erv, y, k, deltafunc

    # get arrays
    Kinv = np.linalg.inv(_covariance_matrix(theta[3:], t_arr, erv_arr))
    thetaarrs = t_arr, phi_arr, rv_arr, erv_arr, Kinv

    # compute the element-wise Fisher matrix 
    Nparams = len(thetavals)
    B = np.zeros((Nparams, Nparams))
    for i in range(Nparams):
        for j in range(Nparams):
            if i > j:
                B[i,j] = B[j,i]
            else:
                B[i,j] = _compute_Fisher_entry(symbol_vals[i], symbol_vals[j],
                                               symbol_vals, symbol_arrs,
                                               thetavals, thetaarrs)
    return B


def _compute_Fisher_entry(symbol_i, symbol_j, symbol_values, symbol_arrays,
                          thetavals, thetaarrs):
    '''
    Compute the Fisher information entry for one pair of model 
    parameters. I.e. the partial of the lnlikelihood wrt to each symbol.
    '''
    # compute partial expressions
    Krv_sym, a_sym, l_sym, G_sym, P_sym, s_sym = symbol_values
    dt_sym, phi_sym, rv_sym, erv_sym, y_sym, K_sym, deltafunc_sym = symbol_arrays
    dy_didj = sympy.lambdify([Krv_sym, phi_sym, rv_sym],
                             sympy.diff(y_sym, symbol_i, symbol_j), 'numpy')
    dy_di = sympy.lambdify([Krv_sym, phi_sym, rv_sym],
                           sympy.diff(y_sym, symbol_i), 'numpy')
    dy_dj = sympy.lambdify([Krv_sym, phi_sym, rv_sym],
                           sympy.diff(y_sym, symbol_j), 'numpy')
    dK_didj = sympy.lambdify([a_sym, l_sym, G_sym, P_sym, s_sym,
                              deltafunc_sym, dt_sym, erv_sym],
                             sympy.diff(K_sym, symbol_i, symbol_j), 'numpy')
    dK_di = sympy.lambdify([a_sym, l_sym, G_sym, P_sym, s_sym,
                            deltafunc_sym, dt_sym, erv_sym],
                           sympy.diff(K_sym, symbol_i), 'numpy')
    dK_dj = sympy.lambdify([a_sym, l_sym, G_sym, P_sym, s_sym,
                            deltafunc_sym, dt_sym, erv_sym],
                           sympy.diff(K_sym, symbol_j), 'numpy')

    # evaluate partials at input values
    K_val, a_val, l_val, G_val, P_val, s_val = thetavals
    t_arr, phi_arr, rv_arr, erv_arr, Kinv =  thetaarrs
    N = t_arr.size
    deltat_mat = np.tile(t_arr, (N,1)) - np.tile(t_arr, (N,1)).T
    deltafunc_mat = np.eye(N)
    erv_mat = np.eye(N)*erv_arr
    dy_didj = _intovector(dy_didj(K_val, phi_arr, rv_arr), N)
    dy_di = _intovector(dy_di(K_val, phi_arr, rv_arr), N)
    dy_dj = _intovector(dy_dj(K_val, phi_arr, rv_arr), N)
    dK_didj = _intomatrix(dK_didj(a_val, l_val, G_val, P_val, s_val,
                                  deltafunc_mat, deltat_mat, erv_mat), N)
    dK_di = _intomatrix(dK_di(a_val, l_val, G_val, P_val, s_val,
                              deltafunc_mat, deltat_mat, erv_mat), N)
    dK_dj = _intomatrix(dK_dj(a_val, l_val, G_val, P_val, s_val,
                              deltafunc_mat, deltat_mat, erv_mat), N)

    # get Fisher terms to sum
    y_arr = _intovector(rv_arr - (-K_val*np.sin(phi_arr)), N)
    terms = np.zeros(11)
    terms[0] = np.dot(dy_didj.T, np.dot(Kinv, y_arr))
    terms[1] = -np.dot(dy_di.T, np.dot(Kinv, np.dot(dK_dj, np.dot(Kinv, y_arr))))
    terms[2] = np.dot(dy_di.T, np.dot(Kinv, dy_dj))
    terms[3] = -np.dot(dy_dj.T,np.dot(Kinv, np.dot(dK_di, np.dot(Kinv, y_arr))))
    terms[4] = np.dot(y_arr.T, np.dot(Kinv,
                                     np.dot(dK_dj,
                                            np.dot(Kinv,
                                                   np.dot(dK_di,
                                                          np.dot(Kinv, y_arr))))))
    terms[5] = -np.dot(y_arr.T, np.dot(Kinv, np.dot(dK_didj, np.dot(Kinv, y_arr))))
    terms[6] = np.dot(y_arr.T, np.dot(Kinv,
                                     np.dot(dK_di,
                                            np.dot(Kinv,
                                                   np.dot(dK_dj,
                                                          np.dot(Kinv, y_arr))))))
    terms[7] = -np.dot(y_arr.T, np.dot(Kinv, np.dot(dK_di, np.dot(Kinv, dy_dj))))
    terms[8] = np.dot(dy_dj.T, np.dot(Kinv, dy_di))
    terms[9] = -np.dot(y_arr.T, np.dot(Kinv, np.dot(dK_dj, np.dot(Kinv, dy_di))))
    terms[10] = np.dot(y_arr.T, np.dot(Kinv, dy_didj))

    return .5 * np.sum(terms)



def _intovector(x, size):
    '''
    Input from partial derivatives may be a 1D vector or a scalar. Change 
    dimensions to do future dot product calculations.
    '''
    if type(x) == int:  # x == 0
        return np.zeros((size, 1))
    else:
        return x.reshape(size, 1)


def _intomatrix(x, size):
    '''
    Input from partial derivatives may be a 2D matrix or a scalar. Change
    dimensions to do future dot product calculations.
    '''
    if type(x) == int:  # x == 0
        return np.zeros((size, size))
    else:
        return x
