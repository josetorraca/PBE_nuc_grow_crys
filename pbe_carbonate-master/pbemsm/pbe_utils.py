import numpy as np
#import numba

from scipy import integrate

def integrate_rk3(t0, ti, y, fun):
    h = ti - t0
    k1 = h * fun(t0, y)
    k2 = h * fun(t0 + h/2, y + k1/2)
    k3 = h * fun(t0 + h, y - k1 + 2*k2)
    ynew = y + 1/6*(k1 + 4.*k2 + k3)
    return ynew

def integrate_rkgill(t0, ti, y, fun):
    sq2 = np.sqrt(2)
    a = (sq2 - 1)/2
    b = (2 - sq2)/2
    c = -sq2/2
    d = 1.0 + sq2/2
    h = ti - t0
    # fun_ = lambda t,y: fun(t, y, *f_args) if f_args is not None else fun

    k1 = h*fun(t0, y)
    k2 = h*fun(t0 + h/2, y + 1/2*k1)
    k3 = h * fun(t0 + h/2, y + a*k1 + b*k2)
    k4 = h * fun(t0 + h, y + c*k2 + d*k3)
    ynew = y + 1/6.0*(k1 + k4) + 1/3.0*(b*k2 + d*k3)
    return ynew

def integrate_euler(t0, ti, y, fun):
    return y + (ti - t0) * fun(t0, y)


