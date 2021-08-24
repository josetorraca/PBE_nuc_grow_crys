import numba
import numpy as np

from py_pbe_msm import pbe_msm_solver, pbe_rhs_funcs, pbe_utils
from utils import SimulationOutput, exponential_decay, log_normal_mod_behavior, log_normal
from utils_cm_toolbox import numericals

spec = [
    ('x0', numba.float64[:]),
    ('N0', numba.float64[:]),
    # ('rhs', numba.float64[:]),
    ('ind', pbe_msm_solver.Indexes.class_type.instance_type),
    ('out_span', numba.types.List(SimulationOutput.class_type.instance_type)),
    ('agg_tot_num_rate', numba.float64),
    ('B', numba.float64),
    ('kb', numba.float64),
    ('kg', numba.float64),
    ('kaggr', numba.float64),
    ('t', numba.float64),
    # ('ymdl0', numba.float64[:])

    ('kaggr_coefs', numba.float64[:]),
    ('t_spline', numba.float64[:]),
    ('h_spline', numba.float64[:]),
]
spec_aggre_varying = spec + [
    ('a_sigma', numba.float64),
    ('a_low_b', numba.float64),
    ('b_sigma', numba.float64),
    ('b_low_b', numba.float64),
]
@numba.jitclass(spec_aggre_varying)
class SimpleModelAggNucGrw():

    def __init__(self, Npts, nts, params_list):
        self.ind = self.set_indexes_case(Npts, nts)

        self.kb, self.b_sigma, self.b_low_b, \
            self.kaggr, self.a_sigma, self.a_low_b, \
            self.kg = params_list

        # self.rhs = np.zeros(self.ind.n_tot)

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 0, 1)
        return ind

    def calc_G(self, x):
        # return self.kg #1.0e1 #1.0e2 #REMEMBER THAT THIS G IS VOLUME BASED -> THUS NEED TO CHANGE TO THE OTHER WEUQATION
        G = 3.0 * x**(2/3) * self.kg
        return G

    def calc_B(self, t):
        r = exponential_decay(t, self.kb, self.b_low_b, self.b_sigma)
        # Ai = [self.kb, self.b_low_b]
        # xEi = [23.0, 50.0]
        # psi = [0.5, 0.1]
        # r = numericals.regularization_mult_func(t, 0.0, Ai, xEi, psi)
        # p = (2.0, 0.5, 1.0, 1e3, 23.0)
        # ini_ref = 30.0
        # kb, b_low_b, b_sigma = (ini_ref, ini_ref*1e-1, 15.0)
        # r = exponential_decay(t, kb, b_low_b, b_sigma)
        # r = 10.0
        # mean, sig, low_bound, amp, tcut = (2.0, 1.25, 0.0, 50e2, 0.0)
        # r = amp * log_normal((t + 1e-5), mean, sig) + low_bound
        # r = log_normal_mod_behavior(t, p)
        return r

    def calc_Aggr(self, x1, x2, N1, N2):
        # return -1
        if N1 < 1e-20 or N2 < 1e-20: return 0.0
        if x1 > (45.0)**3 or x2 > (45.0)**3: return 0.0
        # if x1 + x2 > (100.0)**3: return 0.0
        # r = exponential_decay(self.t, self.kaggr, self.a_low_b, self.a_sigma)
        r = numericals.b_spline_eval(self.t_spline, self.kaggr_coefs, self.h_spline, self.t)
        # Ai = [self.kaggr, self.a_low_b]
        # xEi = [25.0, 100.0]
        # psi = [0.5, 0.05]
        # r = numericals.regularization_mult_func(self.t, 0.0, Ai, xEi, psi)
        # mean, sig, low_bound, amp, tcut = (4.0, 0.5, 1e-2, 10.0, 0.0)
        # r = amp * log_normal((self.t + 1e-5), mean, sig) + low_bound
        # ini_ref = 5e-2
        # ka, a_low_b, a_sigma = (ini_ref, ini_ref*1e-1, 35.0)
        # r = exponential_decay(self.t, ka, a_low_b, a_sigma)
        # r = log_normal_mod_behavior(self.t, p)
        return r
        # return self.kaggr

    def initialize_kaggr_for_b_spline(self, t_spline, h_spline, coefs):
        self.kaggr_coefs = coefs
        self.t_spline = t_spline
        self.h_spline = h_spline
        return

    def calc_mdl_rhs(self, t, y):
        self.t = t
        rhs = np.zeros_like(y)
        # rhs = self.rhs
        # for i in range(len(rhs)):
        #     rhs[i] = 0.0
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        # NinOut = np.zeros_like(N)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        G = self.calc_G(x)
        self.B = self.calc_B(t)
        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, self.B, NinOut, self)
        if np.any(rhs_N > 0.0) or t > 1.0:
            db = 1
        if self.calc_Aggr(0,0,0,0) != -1:
            pbe_rhs_funcs.rhs_pbe_numbers_aggr_base(x, N, rhs_N, self)
        self.agg_tot_num_rate = np.sum(rhs_N)
        rhs_N[0] += self.B
        # pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        pbe_rhs_funcs.grid_G_first_half(x, rhs_x, G)
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
        if i_t % 10 == 0:
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

# @numba.njit()
# def nucl_dynamic_behavior(t, p):
#     mean, sig, low_bound = p
#     tcut = 23.0
#     if t < tcut:
#         r = 0.0
#     else:
#         r = log_normal((t-tcut), mean, sig) + low_bound
#     return r

# @numba.njit()
# def agg_dynamic_behavior(t, p):
#     mean, sig, low_bound = p
#     tcut = 25.0
#     if t < tcut:
#         r = 0.0
#     else:
#         r = log_normal((t-tcut), mean, sig) + low_bound
#     return r
