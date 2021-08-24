import os
import numba
import numpy as np
from py_pbe_msm import pbe_msm_solver

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

@numba.njit()
def exponential_decay(t, n_start, low_bound, sigma):
    n_max = n_start - low_bound
    r = n_max*np.exp(-(t/2.0/sigma)**2) + low_bound
    return r

@numba.njit()
def log_normal(x, mean, sig):
    t1 = 1.0/(sig*x*np.sqrt(2.0*np.pi))
    t2 = (np.log(x)-mean)**2 / (2.0*sig**2)
    r = np.exp(-(t2))
    return r

@numba.njit()
def log_normal_mod_behavior(t, p):
    mean, sig, low_bound, amp, tcut = p
    if t < tcut:
        r = 0.0
    else:
        r = amp * log_normal((t-tcut), mean, sig) + low_bound
    return r



# def retrieve_values_from_output(mdl, lmin):
#     x_span = np.array([item.x for item in mdl.out_span])
#     N_span = np.array([item.N for item in mdl.out_span])
#     l_span = [np.hstack((lmin, (x_i[1:] + x_i[0:-1])*0.5, x_i[-1] + (x_i[-1]-x_i[-2])*0.5 ))
#         for x_i in x_span]
#     n_span = [N_span[i] / (l_span[i][1:] - l_span[i][0:-1]) for i in range(len(x_span))]
#     agg_tot_num_rate_span = np.array([item.agg_tot_num_rate for item in mdl.out_span])
#     B_span = np.array([item.B for item in mdl.out_span])

#     mu0 = [np.sum(item.N) for item in mdl.out_span]
#     mu1 = [np.sum(item.x * item.N) for item in mdl.out_span]
#     return x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1

