import numpy as np
from scipy import integrate
from py_pbe_msm import pbe_msm_solver
from py_pbe_msm import pbe_rhs_funcs
from py_pbe_msm import pbe_utils
import numba
from matplotlib import pyplot as plt
import os

# pylint: disable=E1101

TO_PLOT = True
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
        quad_ret = \
        integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        y, _ = quad_ret
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])

    return xspan, N_t0

# if pbe_utils.USE_NJIT:
spec = [
    ('x0', numba.float64[:]),
    ('N0', numba.float64[:]),
    ('ind', pbe_msm_solver.Indexes.class_type.instance_type),
    # ('ymdl0', numba.float64[:])
]
# else:
    # spec = []
@numba.jitclass(spec)
# @pbe_utils.conditional_decorator(
#     lambda func: numba.jitclass(spec),
#     pbe_utils.USE_NJIT
# )
class MyModel():

    def __init__(self, Npts, nts, x0, N0):
        # self.x0, self.N0 = initial_condition_square(Npts)
        self.ind = self.set_indexes_case(Npts, nts)
        # self.ymdl0 = np.array([])

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 0, 1)
        return ind

    def calc_G(self):
        return 1.0e2

    def calc_B(self):
        return 0.1

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        G = self.calc_G()
        B = self.calc_B()
        pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        return rhs

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def before_simulation(self, y, tspan, i_t):
        self.ind.increment_additions(1, 0, 0)

    def after_simulation(self, y, tspan, i_t):
        pass


def main():
    nt = 1001
    Npts = 1000
    x0, N0 = initial_condition_square(Npts)
    mdl = MyModel(Npts, nt, x0, N0)
    ymdl0 = np.array([]) #mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind)
    tspan = np.linspace(0.0, 1.0, nt)
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        # pbe_utils.step_sundials
        pbe_utils.integrate_rkgill_numba_mdl
    )

    c = 0
    while c < 1:
        mdl = MyModel(Npts, nt, x0, N0)
        y = integration_func(tspan, y0, mdl)
        y = y[0, :]
        c += 1

    if TO_PLOT:
        x = mdl.ind.get_x(y, 0)
        N = mdl.ind.get_N(y, 0)
        plt.figure()
        plt.plot(x0, N0, '.-', label='ini')
        plt.plot(x, N, '.-', label='end')
        plt.legend()
    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




