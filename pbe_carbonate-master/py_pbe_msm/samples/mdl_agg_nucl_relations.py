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
pbe_utils.USE_NJIT = False


def initial_condition_square(Npts, lmin):
    lspan = np.geomspace(lmin, 1000.0, Npts + 1)
    # lspan = np.linspace(0.0, 200.0, Npts + 1)
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

    return lspan, xspan, N_t0

@numba.jitclass([
    ('x', numba.float64[:]),
    ('N', numba.float64[:]),
    ('agg_tot_num_rate', numba.float64),
    ('B', numba.float64),
    ]
)
class SimulationOutput():
    def __init__(self, x, N, agg_tot_num_rate, B):
        self.x = x
        self.N = N
        self.agg_tot_num_rate = agg_tot_num_rate
        self.B = B
if os.getenv('NUMBA_DISABLE_JIT') == "1":
    SimulationOutput.class_type = pbe_msm_solver.FakeNb()

spec = [
    ('x0', numba.float64[:]),
    ('N0', numba.float64[:]),
    ('ind', pbe_msm_solver.Indexes.class_type.instance_type),
    ('out_span', numba.types.List(SimulationOutput.class_type.instance_type)),
    ('agg_tot_num_rate', numba.float64),
    ('B', numba.float64),
    ('kb', numba.float64),
    ('kg', numba.float64),
    ('kaggr', numba.float64),

    # ('ymdl0', numba.float64[:])
]
@numba.jitclass(spec)
class MyModel():

    def __init__(self, Npts, nts, x0, N0, kb, kg, kaggr):
        self.ind = self.set_indexes_case(Npts, nts)

        self.kb = kb
        self.kg = kg
        self.kaggr = kaggr

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 0, 1)
        return ind

    def calc_G(self):
        return self.kg #1.0e1 #1.0e2

    def calc_B(self):
        return self.kb # 0.6  #0.1

    def calc_Aggr(self, x1, x2, N1, N2):
        return self.kaggr

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        NinOut = np.zeros_like(N)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        G = self.calc_G()
        self.B = self.calc_B()
        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, self.B, NinOut, self)
        if np.any(rhs_N > 0.0) or t >= 10.0-1e-3:
            db = 1
        pbe_rhs_funcs.rhs_pbe_numbers_aggr_base(x, N, rhs_N, self)
        self.agg_tot_num_rate = np.sum(rhs_N)
        rhs_N[0] += self.B
        pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        return rhs

    def calc_mdl_rhs_wrapper(self, t, y, yp):
        res = yp - self.calc_mdl_rhs(t, y)
        return res, 0

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def set_intermediaries(self, y, tspan, i_t):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        return SimulationOutput(x, N, self.agg_tot_num_rate, self.B)

    def before_simulation(self, y, tspan, i_t):
        # self.ind.increment_additions(1, 0, 0)

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
    nt = 51
    Npts = 20
    params = {'kb': 10.0, 'kg': 0.0, 'kaggr': 0.5, 'lmin': 1.0}
    l0, x0, N0 = initial_condition_square(Npts, params['lmin'])
    mdl = MyModel(Npts, nt, x0, N0, params['kb'], params['kg'], params['kaggr'])
    ymdl0 = np.array([]) #mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind, lmin=params['lmin'])
    tspan = np.linspace(0.0, 11.0, nt)

    #pbe_utils.integrate_rkgill_numba_mdl
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        pbe_utils.step_sundials
        # pbe_utils.integrate_rkgill_numba_mdl
    )

    # mdl = MyModel(Npts, nt, x0, N0)
    y = integration_func(tspan, y0, mdl)
    y = y[0, :]

    x_span = np.array([item.x for item in mdl.out_span])
    N_span = np.array([item.N for item in mdl.out_span])
    l_span = [np.hstack((l0[0], (x_i[1:] + x_i[0:-1])*0.5, x_i[-1] + (x_i[-1]-x_i[-2])*0.5 ))
        for x_i in x_span]
    n_span = [N_span[i] / (l_span[i][1:] - l_span[i][0:-1]) for i in range(len(x_span))]
    agg_tot_num_rate_span = np.array([item.agg_tot_num_rate for item in mdl.out_span])
    B_span = np.array([item.B for item in mdl.out_span])

    mu0 = [np.sum(item.N) for item in mdl.out_span]
    mu1 = [np.sum(item.x * item.N) for item in mdl.out_span]

    if TO_PLOT:
        x = mdl.ind.get_x(y, 0)
        N = mdl.ind.get_N(y, 0)
        plt.figure(figsize=(12,8))
        plt.title('Ini: mu0 = {}; mu1 = {} // Final: mu0 = {}; mu1 = {} '
            .format(np.sum(N0), np.sum(x0 * N0), np.sum(N), np.sum(x * N)))
        plt.plot(x0, N0, '.-', label='ini')
        plt.plot(x, N, '.-', label='end')
        plt.legend()

        plt.figure(figsize=(12,8))
        plt.plot(x_span[0], n_span[0], '.-', label='ini')
        plt.plot(x_span[-1], n_span[-1], '.-', label='end')
        plt.legend()

        # mu0 = np.sum()
        plt.figure(figsize=(12,8))
        plt.subplot(2,1,1)
        plt.title('$\mu_0$')
        plt.plot(tspan, mu0, 'o-')
        plt.subplot(2,1,2)
        plt.title('$\mu_1$')
        plt.plot(tspan, mu1, 'o-')

        plt.figure(figsize=(12,8))
        # plt.subplot(2,1,1)
        plt.title('Number Changing Mechanims')
        plt.ylabel('$-\dfrac{dN_{aggr,i}}{dt}$')
        plt.plot(tspan, (0.0-agg_tot_num_rate_span), 'o-', label='$-\dfrac{dN_{aggr,i}}{dt}$')
        plt.plot(tspan, B_span, 'd-', label='$B_0$')
        plt.plot(tspan, B_span + agg_tot_num_rate_span, 's-', label='net')
        plt.legend()

    # print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




