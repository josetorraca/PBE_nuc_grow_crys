import numpy as np
from scipy import integrate
import pbe_msm_solver
import pbe_rhs_funcs
import pbe_utils
import numba
from matplotlib import pyplot as plt
import os

TO_PLOT = False
pbe_utils.USE_NJIT = True

def initial_condition_square(Npts):
    lspan = np.linspace(0.0, 80.0, Npts + 1)
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = 10.0
    b = 20.0
    n0_fnc = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
    N_t0 = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        y, _ = \
        integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])

    return {
        'x': xspan,
        'N': N_t0
    }

def set_indexes_case(Npts, nt):
    ind = pbe_msm_solver.Indexes(Npts, nt - 1, 0, 0, 1)
    return ind

@pbe_utils.if_njit()
def calc_G():
    return 1.0e2

@pbe_utils.if_njit()
def calc_B():
    return 0.1

@pbe_utils.if_njit()
def calc_mdl_rhs(t, y, ind: pbe_msm_solver.Indexes):
    rhs = np.zeros_like(y)
    x = ind.get_x(y)
    N = ind.get_N(y)
    rhs_x = ind.get_x(rhs)
    rhs_N = ind.get_N(rhs)
    G = calc_G()
    B = calc_B()
    pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
    pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
    return rhs


def main():
    nt = 1001
    Npts = 1000
    ind = set_indexes_case(Npts, nt)
    psd0 = initial_condition_square(ind.Npts0)
    x0 = psd0['x']
    N0 = psd0['N']
    ymdl0 = np.array([])
    y0 = pbe_msm_solver.set_initial_states(x0, N0, ymdl0, ind)
    tspan = np.linspace(0.0, 1.0, nt)
    integration_func = pbe_msm_solver.create_integrate_nucl(calc_mdl_rhs)

    ii = 0
    while ii < 40:
        ind = set_indexes_case(Npts, nt)
        y0 = pbe_msm_solver.set_initial_states(x0, N0, ymdl0, ind)
        y = integration_func(tspan, y0, ind)
        ii += 1

    if TO_PLOT:
        x = ind.get_x(y)
        N = ind.get_N(y)
        plt.figure()
        plt.plot(x0, N0, '.-', label='ini')
        plt.plot(x, N, '.-', label='end')
        plt.legend()
    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




