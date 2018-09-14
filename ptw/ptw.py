#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import lstsq


def interpol(t, sig):
    """
    Interpolation of 1D signal.
    """

    siglen = sig.size

    # Calculate the range of t points within limits
    siglim = (t > 0) & (t < siglen-1)
    ti = np.floor(t[siglim]).astype('int')

    tr = t[siglim] - ti
    # Gradient of the signal at ti
    grad = sig[ti + 1] - sig[ti]
    # Interpolated signal
    siginterp = sig[ti] + tr[:, None]*grad

    # The three output vectors have the same size
    return siginterp, siglim, grad


def basis(v, order, length):
    """
    Polynomial basis construction.
    """

    B = np.array([(v**p) / (length**(p-1)) for p in range(order+1)]).T

    return B


def pconvert(v, order, length, a):
    """
    Conversion formula for the positional correspondence.
    """

    B = basis(v, order, length)
    conv = B.dot(a)

    return conv


def timeWarp(sigstill, sigmov, order=2, maxiter=100, tol=1e-6, **kwds):
    """
    Polynomial warping of the coordinate axis for signal alignment.
    """

    siglen = max(sigstill.size, sigmov.size)
    t = np.linspace(0, siglen, siglen)

    # Construct basis set
    B = basis(t, order, siglen)

    # Initialize the a coefficients
    ainit = np.zeros((order+1,))
    ainit[1] += 1.
    a = kwds.pop('guess_coeffs', ainit)

    rms_last = 0.
    for it in range(maxiter):

        w = B.dot(a)
        xinterp, siglim, grad = interpol(w, sigstill)

        # Compute RMS residuals and check for convergence
        sigdiff = sigmov[siglim] - xinterp
        rms = norm(sigdiff) / siglen
        drms = np.abs((rms - rms_last) / (rms + 1e-10))
        #print(drms)
        if drms < tol:
            break

        rms_last = rms

        # Improve coeffcients with linear regression
        G = np.tile(grad, (1, B.shape[1]))
        Q = G * B[siglim, :]
        da = lstsq(Q, sigdiff, rcond=None)[0].ravel()
        a += da

    for i in range(2, order+1):
        a[i] /= siglen**i

    return w, siglim, a, siglen


def align(sigstill, sigmov, order=2, ret='result', **kwds):
    """ Align two one-dimensional signals.
    """

    sigstill = np.atleast_2d(sigstill).T
    sigmov = np.atleast_2d(sigmov).T
    w, siglim, acoeffs, siglen = timeWarp(sigstill, sigmov, order=order, **kwds)
    xsigstill, _, _ = interpol(w, sigstill)

    if ret == 'result':
        return acoeffs, xsigstill, sigmov[siglim]

    elif ret == 'func':
        pfunc = partial(pconvert, order=order, length=siglen, a=acoeffs)
        return pfunc
