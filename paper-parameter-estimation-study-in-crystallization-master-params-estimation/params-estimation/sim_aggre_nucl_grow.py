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

def n_norm_fnc(v, a, b):
    if v < a or v > b:
        return 0.0
    else:
        return 1.0/(b-a)

def n_desnorm_fnc(v, a, b, C):
    return n_norm_fnc(v, a, b) * C

def initial_condition_psd(Npts):
    # vspan = np.geomspace((1)**3*1e-12, (500.0)**3*1e-12, Npts + 1)
    vspan = np.linspace((1)**3*1e-12, (500.0)**3*1e-12, Npts + 1)
    xspan = (vspan[1:] + vspan[0:-1]) * 1/2.0
    a = (250.0**3)*1e-12
    b = (300.0**3)*1e-12
    C_vol = 264589.6361388702

    N_t0 = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        quad_ret = \
        integrate.quad(n_desnorm_fnc, vspan[i], vspan[i+1], args=(a,b, C_vol))
        y, _ = quad_ret
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (vspan[i+1] - vspan[i])

    # Check mass:
    rhoc_g_cm3 = 2.659999999999
    kv = 1.0
    mass_check = np.trapz(n_t0_num*xspan, xspan) * kv * rhoc_g_cm3
    assert np.isclose(mass_check, 15.0, 1e-2)

    return xspan, N_t0

def calc_mass_crystal(x, N, mdl):
    return np.sum(x**3*N) * mdl.kv * mdl.rho_c * mdl.m_slv

@numba.jitclass([
    ('x', numba.float64[:]),
    ('N', numba.float64[:]),
    ('agg_tot_num_rate', numba.float64),
    ('B', numba.float64),
    ]
)
class SimulationOutput():
    def __init__(self, x, N):
        self.x = x
        self.N = N
        # self.agg_tot_num_rate = agg_tot_num_rate
        # self.B = B
if os.getenv('NUMBA_DISABLE_JIT') == "1":
    SimulationOutput.class_type = pbe_msm_solver.FakeNb()

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
    ('C', numba.float64), ('mu1', numba.float64),
    ('out_span', numba.types.List(SimulationOutput.class_type.instance_type)),
    ('agg_tot_num_rate', numba.float64),

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

    def set_intermediaries(self, y, tspan, i_t):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        return SimulationOutput(x, N)

    def calc_Aggr(self, x1, x2, N1, N2):
        return -1
        if x1 > 200.0 or x2 > 200.0:
            return 0.0
        else:
            return 1e-5

    def calc_G_size_based(self):
        return self.kg*np.exp(-self.EgporR/(self.T+273.15))*self.S**self.g

    def calc_G(self, x):
        G = 3.0 * x**(2/3) * self.calc_G_size_based()
        return G

    def calc_B(self):
        return self.kb*np.exp(-self.EbporR/(self.T+273.15))*self.mu1*self.S**self.b

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
        # mu2 = np.sum(x**2 * N)
        self.mu3 = np.sum(x**3 * N)
        self.mu1 = np.sum(x * N)

        G = self.calc_G(x)
        B = self.calc_B()
        # dCdt = -3*self.rho_c*self.kv*G*mu2
        int_Gn_aux = np.sum(G * N)
        cryst_rate = -self.rho_c*self.kv*int_Gn_aux

        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, B, 0.0, self)

        # self.agg_tot_num_rate = np.sum(rhs_N)

        pbe_rhs_funcs.grid_G_first_half(x, rhs_x, G)
        rhs_mdl[0] = cryst_rate
        return rhs

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def before_simulation(self, y, tspan, i_t):
        self.ind.increment_additions(1, 0, 0)

        if i_t == 0:
            self.calc_mdl_rhs(0.0, y) #to initialize intermediaries values
            self.out_span = [self.set_intermediaries(y, tspan, i_t)]

    def after_simulation(self, y, tspan, i_t):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        self.out_span += [self.set_intermediaries(y, tspan, i_t)]
        # print(i_t)
        pass


def main():
    nt = 101
    Npts = 200
    x0, N0 = initial_condition_psd(Npts)
    mdl = MyModel(Npts, nt, x0, N0)
    ymdl0 = mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind)
    # tspan = np.linspace(0.0, 15.0*60.0, nt)
    tspan = np.linspace(0.0, 0.1, nt)
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

        plt.figure()
        plt.plot(x0**(1/3)*1e4, N0, '.-', label='ini')
        plt.plot(x**(1/3)*1e4, N, '.-', label='end')
        # plt.figure()
        # plt.plot([tspan[0], tspan[-1]], [y0[-1]], '.-', label='ini')
        # plt.plot(x, N, '.-', label='end')
        # plt.legend()
    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




