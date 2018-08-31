#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from numpy.linalg import lstsq


def interpol(t, sig):
    """
    Interpolation of 1D signal
    """

    siglen = sig.size

    # Calculate the range of t points within limits
    siglim = (t > 0) & (t < siglen-1)
    ti = np.floor(t[siglim]).astype('int')

    tr = t[siglim] - ti
    # Gradient of signal at ti
    grad = sig[ti + 1] - sig[ti]
    # Interpolated signal
    siginterp = sig[ti] + tr[:, None]*grad

    # The three output vectors have the same size
    return siginterp, siglim, grad

def timewarp(siga, sigb, order=2, maxiter=100, tol=1e-6, **kwds):
    """
    Polynomial warping of the coordinate axis
    """

    siglen = max(siga.size, sigb.size)
    t = np.linspace(0, siglen, siglen)

    # Construct basis set
    B = np.zeros((siglen, order+1))
    B[:, 0] = 1.
    B[:, 1] = t
    B[:, 2] = (t/siglen)**2
    #B[:, 3] = (t/siglen)**3

    # Initialize the a coefficients
    a = kwds.pop('guess_coeffs', np.array([0., 1., 0.]))

    rms_last = 0.
    for it in range(maxiter):

        w = B.dot(a)
        xinterp, siglim, grad = interpol(w, siga)

        # Compute RMS residuals and check for convergence
        sigdiff = sigb[siglim] - xinterp
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

    a[2] /= siglen**2
    #a[3] /= siglen**3

    return w, siglim, a
