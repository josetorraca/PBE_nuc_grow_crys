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
    lspan = np.linspace(0.0, 500.0, Npts + 1)
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = 250.0
    b = 300.0
    def n0_fnc(v, a, b):# = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
        if v < a or v > b:
            return 0.0
        else:
            return 0.0032*(300.0 - v)*(v - 250.0)
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
    ('ymdl0', numba.float64[:]),
    ('b', numba.float64), ('g', numba.float64), ('kb', numba.float64),
    ('kg', numba.float64), ('EbporR', numba.float64), ('EgporR', numba.float64),
    ('m_slv', numba.float64), ('rho_c', numba.float64), ('kv', numba.float64),
    ('S', numba.float64), ('mu3', numba.float64), ('T', numba.float64),
    ('C', numba.float64),
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
        self.ind = self.set_indexes_case(Npts, nts)
        self.b = 1.45
        self.g = 1.5
        self.kb = 285.01
        self.kg = 1.44e8
        self.EbporR = 7517.0
        self.EgporR = 4859.0
        self.m_slv = 27.0
        self.rho_c = 2.66e-12
        self.kv = 1.5

        # Variables
        self.S = 0.0
        self.mu3 = 0.0
        self.T = 20.0

        # States
        self.C = 0.1681

        self.ymdl0 = np.array([self.C])

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 1, 1)
        return ind

    def calc_G(self):
        return self.kg*np.exp(-self.EgporR/(self.T+273.15))*self.S**self.g

    def calc_B(self):
        return self.kb*np.exp(-self.EbporR/(self.T+273.15))*self.mu3*self.S**self.b

    def calc_sat(self):
        return 6.29e-2 + 2.46e-3*self.T - 7.14e-6*self.T**2

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        rhs_mdl = self.ind.get_mdl(rhs)

        self.C = self.ind.get_mdl(y)[0]
        Csat = self.calc_sat()
        self.S = (self.C-Csat)/Csat
        mu2 = np.sum(x**2 * N)
        self.mu3 = np.sum(x**3 * N)

        G = self.calc_G()
        B = self.calc_B()
        dCdt = -3*self.rho_c*self.kv*G*mu2

        pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        rhs_mdl[0] = dCdt
        return rhs

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def before_simulation(self, y, tspan, i_t):
        self.ind.increment_additions(1, 0, 0)

    def after_simulation(self, y, tspan, i_t):
        pass


def main():
    nt = 101
    Npts = 200
    x0, N0 = initial_condition_square(Npts)
    mdl = MyModel(Npts, nt, x0, N0)
    ymdl0 = mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind)
    tspan = np.linspace(0.0, 15.0*60.0, nt)
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
        print('C final = {}'.format(mdl.ind.get_mdl(y)[0]))
        # plt.figure()
        # plt.plot([tspan[0], tspan[-1]], [y0[-1]], '.-', label='ini')
        # plt.plot(x, N, '.-', label='end')
        # plt.legend()
    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




