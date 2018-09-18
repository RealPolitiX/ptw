#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import lstsq


def interpol(t, sig):
    """
    Interpolation of 1D signal on the new time axis.

    :Parameters:
        t : 1D array
            The updated time axis.
        sig : 1D array
            Signal to be interpolated

    :Returns:
        siginterp : 1D array
            Interpolated signal.
        siglim : 1D bool array
            Signal limits.
        grad : 1D array
            Signal gradient.
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

    :Parameters:
        v : 1D array
            Vector basis.
        order : int
            Highest polynomial order in the transform.
        length : int
            Length of the signal

    :Return:
        B : 2D array
            Polynomial basis matrix.
    """

    B = np.array([(v**p) / (length**(p-1)) for p in range(order+1)]).T

    return B

def pconvert(v, order, length, a):
    """
    Conversion formula for the positional correspondence.

    :Parameters:
        v, order, length -- see ptw.ptw.basis()
        a : 1D array
            Coefficients for the polynomial transform.

    :Return:
        conv : 1D array
            Coordinates after the polynomial transform.
    """

    B = basis(v, order, length)
    conv = B.dot(a)

    return conv

def timeWarp(sigfix, sigmov, order=2, maxiter=100, tol=1e-6, **kwds):
    """
    Polynomial warping of the coordinate axis for signal alignment.
    """

    siglen = max(sigfix.size, sigmov.size)
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
        xinterp, siglim, grad = interpol(w, sigfix)

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

def reshape1D(trace):
    """ Reshape 1D trace to 2D with unitary dimension at position 1.
    """

    trsh = list(trace.shape)
    if 1 in trsh:
        if trsh[0] == 1:
            trace = trace.T
    elif 1 not in trsh:
        trace = np.atleast_2d(trace).T

    return trace

def paralign(sigfix, sigmov, order=2, ret='result', **kwds):
    """ Align two one-dimensional signals.

    :Parameters:
        sigfix, sigmov : 1D array, 1D array
            Fixed and moving signal.
        order : int | 2
            Highest polynomial order in the transform.
        ret : str | 'result'
            Return options.
        **kwds : keyword arguments
    """

    sigfix, sigmov = list(map(reshape1D, [sigfix, sigmov]))
    w, siglim, acoeffs, siglen = timeWarp(sigfix, sigmov, order=order, **kwds)
    xsigfix, _, _ = interpol(w, sigfix)

    if ret == 'result':
        return acoeffs, xsigfix, sigmov[siglim]

    elif ret == 'func':
        pfunc = partial(pconvert, order=order, length=siglen, a=acoeffs)
        return pfunc
