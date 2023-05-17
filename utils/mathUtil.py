# General utilities

import scipy
import numpy as np


def diagonal_stack_sparse_matrices(A, B):
    """
    Return a diagonal stack of sparse matrices

    Parameters:
        A: scipy.sparse.csr
            first sparse matrix
    
        B: scipy.sparse.csr
            second sparse matrix
    
    Return:
        scipy.sparse.csr
        .. highligh:: python
        .. code-block:: python

            | A 0 |
            | 0 B |
    """

    output = scipy.sparse.vstack(
        (
            scipy.sparse.hstack((A, scipy.sparse.csr_matrix((A.shape[0], B.shape[1]), dtype=A.dtype))).tocsr(),
            scipy.sparse.hstack((scipy.sparse.csr_matrix((B.shape[0], A.shape[1]), dtype=A.dtype), B)).tocsr(),
        )
    )

    return output

# Functions for geodesic smoothing
# --------------------------------

def fwhm2sigma(fwhm):
    """
    Return the sigma of a Gaussian profile for a given full width at half maximum (fwhm).

    Parameters:
        fwhm: float
            full width at half maximum

    Returns:
        sigma: float
            sigma for Gaussian profile given fwhm
    """
    return fwhm / np.sqrt(8 * np.log(2))


def max_smoothing_distance(sigma, epsilon, dim):
    """
    Return the distance of the smoothing kernel that will miss an epsilon proportion of
    the smoothed signal energy. 

    Parameters:
        sigma: float
            sigma parameter of the Gaussian smoothing function

        epsilon: float
            proportion of signal (in mm) to smooth over

        dim: int
            dimension of kernel

    Returns:
        distance: float
            distance in mm of the smoothing kernel
    """
    return sigma * (-scipy.stats.norm.ppf((1 - (1 - epsilon) ** (1/dim)) / 2))
