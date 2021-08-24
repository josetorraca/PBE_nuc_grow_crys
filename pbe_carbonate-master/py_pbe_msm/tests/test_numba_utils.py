import numpy as np
from py_pbe_msm import numba_utils

def test_factorial():
    assert(numba_utils.factorial(5) == 120)
    assert(numba_utils.factorial(1) == 1)
    assert(numba_utils.factorial(0) == 1)

def test_interpolation():
    x = np.array([1, 2.5, 4, 5.5, 7])
    y = np.array([np.log(i) for i in x])
    u = 6
    estim = numba_utils.newton_interpolation(x, y, u)
    exact = np.log(u)
    assert(np.isclose(estim, exact, rtol=1e-2))

def test_extrapolation():
    x = np.array([1, 2.5, 4, 5.5, 7])
    y = np.array([np.log(i) for i in x])
    u = 8.0
    estim = numba_utils.newton_interpolation(x, y, u)
    exact = np.log(u)
    assert(np.isclose(estim, exact, rtol=1e-1))

def test_linear_interp_array_interp():
    x = np.array([1, 2.5, 4, 5.5, 7])
    y = np.log(x)
    xB = np.linspace(2.0, 6.0, 3)
    exact = np.log(xB)
    estim = numba_utils.linear_interp_arr(x, y, xB)
    assert np.sum((estim - exact)**2) < 1e-2

def test_linear_interp_array_extrap_with_newton():
    x = np.array([1, 2.5, 4, 5.5, 7])
    y = np.log(x)
    xB = np.linspace(0.5, 8.0, 8)
    exact = np.log(xB)
    estim = numba_utils.linear_inter_extrap_arr(x, y, xB)
    assert np.sum((estim - exact)**2) < 1e-1
    xB = np.linspace(0.5, 5.0, 8)
    exact = np.log(xB)
    estim = numba_utils.linear_inter_extrap_arr(x, y, xB)
    assert np.sum((estim - exact)**2) < 1e-1
    xB = np.linspace(3.0, 9.0, 8)
    exact = np.log(xB)
    estim = numba_utils.linear_inter_extrap_arr(x, y, xB)
    assert np.sum((estim - exact)**2) < 0.5
