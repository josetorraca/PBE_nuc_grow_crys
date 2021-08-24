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
    lmin = 1e-5
    lspan = np.linspace(lmin, 810.0, Npts + 1)
    # lspan = np.geomspace(lmin, 1500.0, Npts + 1)
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

    return lspan, xspan, N_t0

spec_mdl_out = [
    ('x0', numba.float64[:]),
    ('N0', numba.float64[:]),
    ('x1', numba.float64[:]),
    ('N1', numba.float64[:]),
    ('S', numba.float64[:]),
    ('C', numba.float64[:]),
    ('Csat', numba.float64[:]),
    ('T', numba.float64[:]),
]
@numba.jitclass(spec_mdl_out)
class ModelOutput():
    def __init__(self):
        pass

if os.getenv('NUMBA_DISABLE_JIT') == "1":
    ModelOutput.class_type = pbe_msm_solver.FakeNb()


spec = [
    ('x0', numba.float64[:]),
    ('N0', numba.float64[:]),
    ('ind', pbe_msm_solver.Indexes.class_type.instance_type),
    ('ymdl0', numba.float64[:]),
    ('b', numba.float64), ('g', numba.float64), ('kb', numba.float64),
    ('kg', numba.float64), ('EbporR', numba.float64), ('EgporR', numba.float64),
    ('m_slv', numba.float64), ('rho_c', numba.float64), ('kv', numba.float64),
    ('S', numba.float64[:]), ('mu3', numba.float64), ('T', numba.float64),
    ('C', numba.float64[:]),
    # ('mdl_output', numba.types.List(ModelOutput.class_type.instance_type)),
    ('lmin', numba.float64),
]
@numba.jitclass(spec)
class MyModel():

    def __init__(self, Npts, nts, x0, N0, x01, N01, lmin):
        self.b = 1.45
        self.g = 1.5
        self.kb = 285.01
        self.kg = 1.44e8
        self.EbporR = 7517.0
        self.EgporR = 4859.0
        self.m_slv = 27.0
        self.rho_c = 2.66e-12
        self.kv = 1.5
        self.lmin = lmin

        # States
        self.C = np.array([0.1681, 0.1681])

        # Variables
        self.S = np.empty_like(self.C)
        # mu3 = 0.0
        self.T = 20.0

        self.ymdl0 = self.C.copy()

        self.ind = self.set_indexes_case(Npts, nts)

        # Intermediaries storage
        # self.mdl_intermediaries = np.empty((nts, self.ind.n_tot))
        pass

    def set_indexes_case(self, Npts, nt):
        number_of_psd = 2
        n_el = [nt - 1, nt - 1]
        n_eu = [0, 0]
        n_mdls = self.ymdl0.shape[0]
        ind = pbe_msm_solver.Indexes(
            Npts, n_el, n_eu, n_mdls, number_of_psd
        )
        return ind

    def calc_Aggr(self, x1, x2, N1, N2):
        if x1 > 400.0 or x2 > 400.0:
            return 0.0
        else:
            return 1e-5

    def calc_G0(self):
        return self.kg*np.exp(-self.EgporR/(self.T+273.15))*self.S[0]**self.g

    def calc_B0(self, mu3):
        return self.kb*np.exp(-self.EbporR/(self.T+273.15))*mu3*self.S[0]**self.b

    def calc_G1(self):
        return 1.5*self.kg*np.exp(-self.EgporR/(self.T+273.15))*self.S[1]**self.g

    def calc_B1(self, mu3):
        return 0.2*self.kb*np.exp(-self.EbporR/(self.T+273.15))*mu3*self.S[1]**self.b


    def calc_sat(self):
        return 6.29e-2 + 2.46e-3*self.T - 7.14e-6*self.T**2

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x0 = self.ind.get_x(y, 0)
        N0 = self.ind.get_N(y, 0)
        rhs_x0 = self.ind.get_x(rhs, 0)
        rhs_N0 = self.ind.get_N(rhs, 0)
        NinOut = np.zeros_like(N0) #can be optimized to single allocation

        x1 = self.ind.get_x(y, 1)
        N1 = self.ind.get_N(y, 1)
        rhs_x1 = self.ind.get_x(rhs, 1)
        rhs_N1 = self.ind.get_N(rhs, 1)
        rhs_mdl = self.ind.get_mdl(rhs)

        self.C = self.ind.get_mdl(y)
        Csat = self.calc_sat()
        self.S[0] = (self.C[0]-Csat)/Csat
        self.S[1] = (self.C[1]-Csat)/Csat
        mu20 = np.sum(x0**2 * N0)
        mu30 = np.sum(x0**3 * N0)
        mu21 = np.sum(x1**2 * N1)
        mu31 = np.sum(x1**3 * N1)


        G0 = self.calc_G0()
        B0 = self.calc_B0(mu30)
        dCdt0 = -3*self.rho_c*self.kv*G0*mu20
        # pbe_rhs_funcs.numbers_nucl(x0, N0, rhs_N0, B0)
        pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x0, N0, rhs_N0, B0, NinOut, self)
        pbe_rhs_funcs.grid_const_G_half(x0, rhs_x0, G0)

        G1 = self.calc_G1()
        B1 = self.calc_B1(mu31)
        dCdt1 = -3*self.rho_c*self.kv*G1*mu21

        pbe_rhs_funcs.numbers_nucl(x1, N1, rhs_N1, B1)
        pbe_rhs_funcs.grid_const_G_half(x1, rhs_x1, G1)

        rhs_mdl[0] = dCdt0
        rhs_mdl[1] = dCdt1
        return rhs

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def before_simulation(self, y, tspan, i_t):
        self.ind.increment_additions(1, 0, 0)
        self.ind.increment_additions(1, 0, 1)

    def after_simulation(self, y, tspan, i_t):
        pass


def main():
    nt = 21
    Npts = 150
    lspn0, x0, N0 = initial_condition_square(Npts)
    lspn0, x01, N01 = initial_condition_square(round(Npts/2))
    mdl = MyModel([Npts, x01.shape[0]], nt, x0, N0, x01, N01, lspn0[0])
    ymdl0 = mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states(
        [(x0, N0), (x01, N01)], ymdl0, mdl.ind, lmin=lspn0[0]
    )
    tspan = np.linspace(0.0, 15.0*60.0, nt)
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        # pbe_utils.step_sundials
        pbe_utils.integrate_rkgill_numba_mdl
    )

    c = 0
    while c < 1:
        mdl = MyModel([Npts, x01.shape[0]], nt, x0, N0, x01, N01, lspn0[0])
        y = integration_func(tspan, y0, mdl, 1)
        y = y[-1, :]
        c += 1

    if TO_PLOT:
        x = mdl.ind.get_x(y, 0)
        N = mdl.ind.get_N(y, 0)
        x1 = mdl.ind.get_x(y, 1)
        N1 = mdl.ind.get_N(y, 1)
        plt.figure()
        plt.plot(x0, N0, '.-', label='ini-1')
        plt.plot(x, N, '.-', label='end')
        print('C final = {}'.format(mdl.ind.get_mdl(y)[0]))

        plt.figure()
        plt.plot(x01, N01, '.-', label='ini-2')
        plt.plot(x1, N1, '.-', label='ini-end')
    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




