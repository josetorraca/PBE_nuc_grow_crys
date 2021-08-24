import numpy as np
import numba
from scipy import integrate
from . import numba_utils

try:
    from scikits.odes.odeint import odeint
    SCIKITSODES_INSTALLED = True
except ImportError:
    SCIKITSODES_INSTALLED = False


USE_NJIT = True
CACHED_JIT = False

def if_njit(*args, **kwargs):
    def decorator(func):
        if not USE_NJIT:
            # Return the function unchanged, not decorated.
            return func
        return numba.njit(func, *args,
            cache=CACHED_JIT,
            **kwargs)
    return decorator

def conditional_decorator(dec, condition):
    def decorator(func):
        if not condition:
            return func
        return dec(func)
    return decorator


######
######
 # Integrators
######
######

def integrate_rk3(t0, ti, y, fun):
    h = ti - t0
    k1 = h * fun(t0, y)
    k2 = h * fun(t0 + h/2, y + k1/2)
    k3 = h * fun(t0 + h, y - k1 + 2*k2)
    ynew = y + 1/6*(k1 + 4.*k2 + k3)
    return ynew

@if_njit()
def integrate_rkgill(t0, ti, y, fun, *args):
    sq2 = np.sqrt(2)
    a = (sq2 - 1)/2
    b = (2 - sq2)/2
    c = -sq2/2
    d = 1.0 + sq2/2
    h = ti - t0
    # fun_ = lambda t,y: fun(t, y, *f_args) if f_args is not None else fun

    k1 = h*fun(t0, y, *args)
    k2 = h*fun(t0 + h/2, y + 1/2*k1, *args)
    k3 = h * fun(t0 + h/2, y + a*k1 + b*k2, *args)
    k4 = h * fun(t0 + h, y + c*k2 + d*k3, *args)
    ynew = y + 1/6.0*(k1 + k4) + 1/3.0*(b*k2 + d*k3)
    return ynew

def create_integrate_rkgill(fun):

    @if_njit()
    def integrate_rkgill_numba(t0, ti, y, *args):
        sq2 = np.sqrt(2)
        a = (sq2 - 1)/2
        b = (2 - sq2)/2
        c = -sq2/2
        d = 1.0 + sq2/2
        h = ti - t0

        k1 = h*fun(t0, y, *args)
        k2 = h*fun(t0 + h/2, y + 1/2*k1, *args)
        k3 = h * fun(t0 + h/2, y + a*k1 + b*k2, *args)
        k4 = h * fun(t0 + h, y + c*k2 + d*k3, *args)
        ynew = y + 1/6.0*(k1 + k4) + 1/3.0*(b*k2 + d*k3)
        return ynew
    return integrate_rkgill_numba

@if_njit()
def integrate_rkgill_numba_mdl(t0, ti, y, mdl):
    sq2 = np.sqrt(2)
    a = (sq2 - 1)/2
    b = (2 - sq2)/2
    c = -sq2/2
    d = 1.0 + sq2/2
    h = ti - t0

    k1 = h * mdl.calc_mdl_rhs(t0, y)
    k2 = h * mdl.calc_mdl_rhs(t0 + h/2, y + 1/2*k1)
    k3 = h * mdl.calc_mdl_rhs(t0 + h/2, y + a*k1 + b*k2)
    k4 = h * mdl.calc_mdl_rhs(t0 + h, y + c*k2 + d*k3)
    ynew = y + 1/6.0*(k1 + k4) + 1/3.0*(b*k2 + d*k3)
    return ynew

def integrate_rkgill_mdl(t0, ti, y, mdl):
    sq2 = np.sqrt(2)
    a = (sq2 - 1)/2
    b = (2 - sq2)/2
    c = -sq2/2
    d = 1.0 + sq2/2
    h = ti - t0

    k1 = h * mdl.calc_mdl_rhs(t0, y)
    k2 = h * mdl.calc_mdl_rhs(t0 + h/2, y + 1/2*k1)
    k3 = h * mdl.calc_mdl_rhs(t0 + h/2, y + a*k1 + b*k2)
    k4 = h * mdl.calc_mdl_rhs(t0 + h, y + c*k2 + d*k3)
    ynew = y + 1/6.0*(k1 + k4) + 1/3.0*(b*k2 + d*k3)
    return ynew

def integrate_euler(t0, ti, y, fun):
    return y + (ti - t0) * fun(t0, y)


## DASSLC INTEGRATION


## SUNDIALS WITH SCYKITS.ODES
# if SCIKITSODES_INSTALLED:
def step_sundials(t0, tf, y, mdl):
    output = odeint(mdl.calc_mdl_rhs_wrapper_sundials,
        [t0, tf], y, method='admo')
    y = output.values.y[-1,:]
    return y


######
######
 # PBE Modifiers For Dissolution
######
######
@numba.njit
def check_smallers_than_mininal(y, ind, lmin, psd_id):
    x = ind.get_x(y, psd_id)
    i_last = np.where(x < lmin)[-1]
    if len(i_last) > 0:
        return i_last[-1]
    else:
        return -1

@numba.njit
def put_to_zero_smaller_than_minimal(y, ind, lmin, psd_id):
    x = ind.get_x(y)
    N = ind.get_N(y)
    i_rm = check_smallers_than_mininal(y, ind, lmin, psd_id)
    if i_rm > 0:
        x[0:i_rm + 1] = lmin
        N[0:i_rm + 1] = 0.0
    pass
@numba.njit
def remove_smallers_than_zero(y, ind, lmin, psd_id):
    i_rm = check_smallers_than_mininal(y, ind, lmin, psd_id)
    i_incr = i_rm + 1 if i_rm > 0 else 0
    ind.increment_additions(-i_incr, 1, psd_id)


######
######
 # Combine a Mesh (x, N) to another (x, N)
######
######
@numba.njit
def calc_boundary(x, lmin = 0.0):
    l = np.empty(x.shape[0] + 1)
    l[0] = lmin
    l[1:-1] = (x[1:] + x[0:-1])/2
    l[-1] = (x[-1] - l[-2]) + x[-1]
    return l

@numba.njit
def combine_mesh_simple_interp(xA, NA, xB, lmin = 0.0):
    lA = calc_boundary(xA, lmin)
    lB = calc_boundary(xB, lmin)

    nA = NA/(lA[1:] - lA[0:-1])

    # nB = np.empty_like(xB)
    nB = numba_utils.linear_inter_extrap_arr(xA, nA, xB)

    NB = nB * (lB[1:] - lB[0:-1])
    return NB

@numba.njit
def combine_mesh_boundary_based(xA, NA, xB, lmin = 0.0):
    lA = calc_boundary(xA, lmin)
    lB = calc_boundary(xB, lmin)
    NB = np.empty_like(xB)
    convert_to_given_mesh(lA, NA, lB, NB, 0)
    return NB


@numba.njit
def convert_to_given_mesh(lv, Nv, ln, Nn, ordConserve = 0):
    """This function converts a known mesh (lv, Nv) to a new mesh (ln, Nn)
    where Nn is calculated.
    This scheme allows only one property conservation
    See development in:
    /home/caio/Projects/Temporaries/dscpapermsmextended/src/finestrap-dynamic/dev_remeshing.py
    """
    if Nn.size != ln.size -1:
        print('Allocating memory for Nn to be ln.size - 1')
        Nn = np.empty(ln.size - 1)

    if ln[0] > lv[0]:
        raise ValueError('ln[0] > lv[0] -> Error to be handled')

    if ln[-1] < lv[-1]:
        raise ValueError('ln[-1] < lv[-1] -> Error to be handled')

    j_start = 0
    for i in np.arange(0, Nn.size):
        Naux = 0.0
        for j in np.arange(j_start, Nv.size):
            # if not (lv[j] < ln[i+1] and lv[j+1] > ln[i]):
                # continue
            if lv[j] > ln[i+1]:
                j_start = j - 1
                break
            if lv[j+1] < ln[i]:
                continue
            fracNum = min(lv[j+1], ln[i+1]) - max(lv[j], ln[i])
            fracDen = lv[j+1] - lv[j]
            frac = fracNum / fracDen
            x = 0.5*(lv[j]+lv[j+1]) # test
            #Naux += Nv[j] * x**ordConserve * frac
            Naux += Nv[j] * frac
        # xn = 0.5*(ln[i]+ln[i+1]) # test
        # Nn[i] = Naux / xn**ordConserve
        Nn[i] = Naux
    pass


