import numpy as np
import numba


## ---------------------------------
## UTILS:
## ---------------------------------
@numba.njit
def delta_kron(i,j):
    """Just a simple auxiliary function for computing kronecker delta"""
    if i == j:
        return 1
    else:
        return 0

@numba.njit
def factorial(n: int):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f

@numba.njit
def product(a):
  p = 1
  for i in a: p *= i
  return p

## ---------------------------------
## INTERPOLATION
## ---------------------------------
# @numba.njit
# def calc_alpha_interp(t, tn, h):
#     alpha = (t-tn)/h
#     return

@numba.njit
def newton_interpolation(x, y, u):
  '''
  Parameters
  ----------
  x : errado list of floats
  y : errado list of floats
  u : float

  Returns
  -------
  float
      an estimate at the point u
    Ref: https://trinket.io/python/b13a541c25
  '''
  g = y.copy()
  s = g[0]
  for i in range(len(y)-1):
    g = np.array([(g[j+1]-g[j])/(x[j+i+1]-x[j]) for j in range(len(g)-1)])
    s += g[0] * product([u-x[j] for j in range(i+1)])
  return s

@numba.njit
def linear_interp(x1, x2, y1, y2, x):
    return y1 + (x - x1)*(y2 - y1)/(x2 - x1)

@numba.njit
def linear_interp_arr(xA, yA, xB):
    iclB = np.searchsorted(xB, xA[0])
    icuB = np.searchsorted(xB, xA[-1])
    # iclB = iclB - 1
    # icuB = icuB - 1
    yB = np.zeros_like(xB)
    # ial = 0
    # for i in range(iclB, icuB):
    #     for j in range(ial, xA.shape[0] - 1):
    #         if xB[i] < xA[j+1]:
    #             yB[i] = linear_interp(xA[j], xA[j+1], yA[j], yA[j+1], xB[i])
    #             # ial = j + 1
    #             ial = j
    #             break
    linear_interp_arr_aux(xA, yA, xB, yB)
    return yB

@numba.njit
def linear_interp_arr_aux(xA, yA, xB, yB):
    if xB[0] < xA[0] or xB[-1] > xA[-1]:
        raise ValueError('No extrapolation allowed.')
    if yB.shape != xB.shape:
        raise ValueError('yB shape has to mach xB shape.')
    ial = 0
    for i in range(0, xB.shape[0]):
        for j in range(ial, xA.shape[0] - 1):
            if xB[i] < xA[j+1]:
                yB[i] = linear_interp(xA[j], xA[j+1], yA[j], yA[j+1], xB[i])
                # ial = j + 1
                ial = j
                break
    return yB

@numba.njit
def linear_inter_extrap_arr(xA, yA, xB, n_extrap = 5):
    yB = np.zeros_like(xB)
    linear_inter_extrap_arr_aux(xA, yA, xB, n_extrap, yB)
    return yB

@numba.njit
def linear_inter_extrap_arr_aux(xA, yA, xB, n_extrap, yB):
    if yB.shape != xB.shape:
        raise ValueError('yB shape has to mach xB shape.')
    iclB = np.searchsorted(xB, xA[0])
    icuB = np.searchsorted(xB, xA[-1])
    linear_interp_arr_aux(xA, yA, xB[iclB:icuB], yB[iclB:icuB])
    if iclB > 0:
        i_ex = n_extrap if n_extrap < xA.shape[0] else xA.shape[0]
        for i in range(iclB):
            yB[i] = newton_interpolation(xA[0:i_ex], yA[0:i_ex], xB[i])
    if icuB < xB.shape[0]:
        i_ex = n_extrap if n_extrap < xA.shape[0] else xA.shape[0]
        for i in range(icuB, xB.shape[0]):
            yB[i] = newton_interpolation(xA[-i_ex:], yA[-i_ex:], xB[i])
    return yB
