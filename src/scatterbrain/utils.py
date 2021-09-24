"""basic hack utilities"""
try:
    import cupy as cp
    from cupy import sparse
except ImportError:
    import numpy as cp
    from scipy import sparse


def _spline_basis_vector(x, degree, i, knots):
    """Recursive function to create a single spline basis vector for an input x,
    for the ith knot.
    See https://en.wikipedia.org/wiki/B-spline for a definition of B-spline
    basis vectors
    Parameters
    ----------
    x : cp.ndarray
        Input x
    degree : int
        Degree of spline to calculate basis for
    i : int
        The index of the knot to calculate the basis for
    knots : cp.ndarray
        Array of all knots
    Returns
    -------
    B : cp.ndarray
        A vector of same length as x containing the spline basis for the ith knot
    """
    if degree == 0:
        B = cp.zeros(len(x))
        B[(x >= knots[i]) & (x <= knots[i + 1])] = 1
    else:
        da = knots[degree + i] - knots[i]
        db = knots[i + degree + 1] - knots[i + 1]
        if (knots[degree + i] - knots[i]) != 0:
            alpha1 = (x - knots[i]) / da
        else:
            alpha1 = cp.zeros(len(x))
        if (knots[i + degree + 1] - knots[i + 1]) != 0:
            alpha2 = (knots[i + degree + 1] - x) / db
        else:
            alpha2 = cp.zeros(len(x))
        B = (_spline_basis_vector(x, (degree - 1), i, knots)) * (alpha1) + (
            _spline_basis_vector(x, (degree - 1), (i + 1), knots)
        ) * (alpha2)
    return B


def get_star_mask(f):
    """False where stars are. Keep in mind this might be a bad
    set of hard coded parameters for some TESS images!"""
    # This removes pixels where there is a steep flux gradient
    star_mask = (cp.hypot(*cp.gradient(f)) < 30) & (f < 9e4)
    # This broadens that mask by one pixel on all sides
    star_mask = (
        ~(cp.asarray(cp.gradient(star_mask.astype(float))) != 0).any(axis=0) & star_mask
    )
    return star_mask
