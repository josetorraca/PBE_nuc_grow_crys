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

@numba.jitclass([
    ('b', numba.float64),
    ('g', numba.float64),
    ('kb', numba.float64),
    ('kg', numba.float64),
    ('EbporR', numba.float64),
    ('EgporR', numba.float64),
    ('m_slv', numba.float64),
    ('rho_c', numba.float64),
    ('kv', numba.float64),
    ('lmin', numba.float64),
    ('T_initial', numba.float64),
    ('C_initial', numba.float64),
    ('k_aggr', numba.float64),
    ]
)
class ModelRawlings():

    def __init__(self):
        self.b = 1.45
        self.g = 1.5
        self.kb = 285.01
        self.kg = 1.44e8
        self.k_aggr = 1e-5
        self.EbporR = 7517.0
        self.EgporR = 4859.0
        self.m_slv = 27.0
        self.rho_c = 2.66e-12 #is g/um^3; 2.66 is g/cm^3
        self.kv = 1.0 #1.5
        self.lmin = 10

        # Variables
        # self.S = 0.0
        # self.mu3 = 0.0
        # self.T = 20.0

        # Initials
        self.T_initial = 20.0 #assumed constant
        self.C_initial = 0.1681
        pass

    def calc_G(self, T, S):
        return self.kg*np.exp(-self.EgporR/(T+273.15))*S**self.g

    def calc_B(self, T, S, vol_tot_particles):
        return self.kb*np.exp(-self.EbporR/(T+273.15))*vol_tot_particles*S**self.b

    def calc_sat(self, T):
        return 6.29e-2 + 2.46e-3*T - 7.14e-6*T**2

def initial_condition_psd(Npts):
    # lspan = np.linspace(1.0, (500.0), Npts + 1) #*1e-4 #um^3 to cm^3
    lspan = np.geomspace(1.0, (1200.0), Npts + 1) #*1e-4 #um^3 to cm^3
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = (250.0) #* 1e-4
    b = (300.0) #* 1e-4
    def n0_fnc(v, a, b):# = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
        if v < a or v > b:
            return 0.0
        else:
            # return 0.0032*(b - v)*(v - a)
            return 0.015*(b - v)*(v - a)
    N_t0 = np.empty(Npts)
    N_v = np.empty(Npts)
    vspan = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        quad_ret = \
        integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        y, _ = quad_ret
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])

        vspan[i] = xspan[i]**3
        N_v[i] = N_t0[i]

    # Checking
    mu0_l = np.sum(N_t0)
    mu0_v = np.sum(N_v)
    mu1_l = np.sum(N_t0 * xspan)
    mu1_v = np.sum(N_v * vspan)
    mu3_l = np.sum(N_t0 * xspan**3)

    rho_c = 2.66; kv = 1.0
    mass_l = kv*rho_c*1e-12*mu3_l
    mass_v = kv*rho_c*mu1_v*1e-12

    #ENTão n volumetric ficou em #/um^3 !

    return vspan, N_v, lspan, xspan
    # return xspan, N_t0


@numba.jitclass([
    ('x', numba.float64[:]),
    ('N', numba.float64[:]),
    ('agg_tot_num_rate', numba.float64),
    ('B', numba.float64),
    ('G', numba.float64[:]),
    ('S', numba.float64),
    ('C', numba.float64),
    ]
)
class SimulationOutput():
    def __init__(self, x, N, B, G, S, C):
        self.x = x
        self.N = N
        self.B = B
        self.G = G
        self.S = S
        self.C = C
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
    ('S', numba.float64), ('mu3', numba.float64), ('mu1', numba.float64), ('T', numba.float64),
    ('C', numba.float64),
    ('B', numba.float64),
    ('S', numba.float64),
    ('G', numba.float64[:]),
    ('agg_tot_num_rate', numba.float64),
    ('out_span', numba.types.List(SimulationOutput.class_type.instance_type)),
    ('base', ModelRawlings.class_type.instance_type), #CHANGE TO NEW DICT
]
# else:
    # spec = []
@numba.jitclass(spec)
# @pbe_utils.conditional_decorator(
#     lambda func: numba.jitclass(spec),
#     pbe_utils.USE_NJIT
# )
class MyModel():

    def __init__(self, Npts, nts, x0, N0, mdl_base):
        self.ind = self.set_indexes_case(Npts, nts)
        self.base = mdl_base
        # self.b = 1.45
        # self.g = 1.5
        # self.kb = 285.01
        # self.kg = 1.44e8
        # self.EbporR = 7517.0
        # self.EgporR = 4859.0
        # self.m_slv = 27.0
        # self.rho_c = 2.66e-12 #is g/um^3; 2.66 is g/cm^3
        # self.kv = 1.0 #1.5

        # Variables
        self.S = 0.0
        self.mu3 = 0.0
        self.T = self.base.T_initial

        # States
        self.C = self.base.C_initial

        self.ymdl0 = np.array([self.C])

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 1, 1)
        return ind

    def calc_G(self, x):
        # G = (3.0 * x**(2/3.0)) * self.calc_G_size_based()
        G = (3.0 * x**(2/3.0)) * self.base.calc_G(self.T, self.S) #self.calc_G_size_based()
        return G

    # def calc_G_size_based(self):
    #     if self.S > 0.0:
    #         G = self.kg*np.exp(-self.EgporR/(self.T+273.15))*self.S**self.g
    #     else:
    #         G = 0.0
    #     return G

    def calc_Aggr(self, x1, x2, N1, N2):
        # return -1 #1e-6
        return self.base.k_aggr
        # return -1
        # if x1 > 200.0 or x2 > 200.0:
        #     return 0.0
        # else:
        #     return 1e-5

    def calc_B(self):
        return self.base.calc_B(self.T, self.S, self.mu1)

    # def calc_B(self):
    #     if self.S > 0.0:
    #         B = self.kb*np.exp(-self.EbporR/(self.T+273.15))*self.mu1*self.S**self.b
    #     else:
    #         B = 0.0
    #     return B

    # def calc_sat(self):
    #     return 6.29e-2 + 2.46e-3*self.T - 7.14e-6*self.T**2

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        rhs_mdl = self.ind.get_mdl(rhs)

        self.C = self.ind.get_mdl(y)[0]
        Csat = self.base.calc_sat(self.T)
        self.S = (self.C-Csat)/Csat
        # mu2 = np.sum(x**2 * N)
        # self.mu3 = np.sum(x**3 * N)
        self.mu1 = np.sum(x * N)

        self.G = self.calc_G(x)
        self.B = self.calc_B()
        int_Gn_aux = np.sum(self.G  * N)
        cryst_rate = -self.base.rho_c*self.base.kv*int_Gn_aux

        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, self.B)
        if self.calc_Aggr(0,0,0,0) != -1:
            pbe_rhs_funcs.rhs_pbe_numbers_aggr_base(x, N, rhs_N, self)
        self.agg_tot_num_rate = np.sum(rhs_N)
        rhs_N[0] += self.B
        # pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        pbe_rhs_funcs.grid_G_first_half(x, rhs_x, self.G)
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
        pass

    def after_simulation(self, y, tspan, i_t):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        self.out_span += [self.set_intermediaries(y, tspan, i_t)]
        pass

    def set_intermediaries(self, y, tspan, i_t):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        return SimulationOutput(x, N, self.B, self.G, self.S, self.C)


def setup_simulation(Npts, nt, x0, N0, lmin):
    mdl_base = ModelRawlings()
    mdl = MyModel(Npts, nt, x0, N0, mdl_base)
    ymdl0 = mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind,
        lmin=lmin)
    # tspan = np.linspace(0.0, tf, nt)
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        # pbe_utils.step_sundials
        pbe_utils.integrate_rkgill_numba_mdl
    )

    # y = integration_func(tspan, y0, mdl)
    return mdl, integration_func, y0

def get_variables_profiles(mdl, tspan):
    xspan = [out.x for out in mdl.out_span]
    Nspan = [out.N for out in mdl.out_span]
    Sspan = [out.S for out in mdl.out_span]
    Cspan = [out.C for out in mdl.out_span]
    GMeanspan = np.array([np.mean(out.G) for out in mdl.out_span])
    mu0 = np.array([np.sum(out.N) for out in mdl.out_span])
    mu1 = np.array([np.sum(out.x*out.N) for out in mdl.out_span])
    return {
        't': tspan,
        'x': xspan,
        'N': Nspan,
        'S': Sspan,
        'C': Cspan,
        'GMean': GMeanspan,
        'mu0': mu0,
        'mu1': mu1,
    }

def plot_scalar_variables(mdl, d):
    tspan = d['t']
    plt.figure()
    plt.title('S')
    plt.plot(tspan, d['S'])
    plt.xlabel('time')

    plt.figure()
    plt.title('G - Mean')
    plt.plot(tspan, d['GMean'])
    plt.xlabel('time')

    plt.figure()
    plt.subplot(1,2,1)
    plt.title('$\mu_0$')
    plt.plot(tspan, d['mu0'])
    plt.subplot(1,2,2)
    plt.title('$\mu_1$')
    plt.plot(tspan, d['mu1'])
    return

def plot_psd(mdl, d):
    f = {}
    f['N-v'] = plt.figure()
    plt.title('Number vs volume')
    plt.plot(d['x'][0], d['N'][0], '.-', label='ini')
    plt.plot(d['x'][-1], d['N'][-1], '.-', label='end')
    plt.legend()
    print('C final = {}'.format(d['C'][-1]))

    f['N-l'] = plt.figure()
    plt.title('Number vs size (length)')
    plt.plot(d['x'][0]**(1/3), d['N'][0], '.-', label='ini')
    plt.plot(d['x'][-1]**(1/3), d['N'][-1], '.-', label='end')
    plt.legend()

    f['N_norm-l'] = plt.figure()
    plt.title('Number/(Total Number) vs size (length)')
    plt.plot(d['x'][0]**(1/3), d['N'][0]/np.sum(d['N'][0]), '.-', label='ini')
    plt.plot(d['x'][-1]**(1/3), d['N'][-1]/np.sum(d['N'][-1]), '.-', label='end')
    plt.legend()
    return f

def main():
    nt = 101
    Npts = 200
    x0, N0, lspan, xspan = initial_condition_psd(Npts)

    mdl, integration_func, y0 = setup_simulation(
        Npts, nt, x0, N0, lmin=lspan[0]**3
    )
    tspan = np.linspace(0.0, 10.0*60.0, nt)

    y = integration_func(tspan, y0, mdl)

    d = get_variables_profiles(mdl, tspan)

    if TO_PLOT:
        plot_scalar_variables(mdl, d)
        plot_psd(mdl, d)

    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




