import os

try:
    import eggs
except ImportError:
    pass

import numba
import numpy as np
# import dasslcy
# try:
#     from scikits.odes.odeint import odeint
#     SCIKITSODES_INSTALLED = True
# except ImportError:
#     SCIKITSODES_INSTALLED = False


# import pbe_funcs_numba
from . import pbe_utils

LIST_INT = numba.types.List(numba.int64)
spec = [
    ('Npts0', LIST_INT), #numba.int64[:]
    ('nel', LIST_INT),
    ('neu', LIST_INT),
    ('nmdl', numba.int64),
    ('npsds', numba.int64),
    ('n_pivot_tot', numba.int64[:]),
    # ('n_pivot_tot', numba.int64),
    ('ladd', LIST_INT),
    ('uadd', LIST_INT),
    ('nx0', numba.int64[:]),
    ('nx_last', numba.int64[:]),
    ('nN0', numba.int64[:]),
    ('nN_last', numba.int64[:]),
    ('n_tot', numba.int64),
]

# @pbe_utils.conditional_decorator(
#     lambda func: numba.jitclass(spec),
#     pbe_utils.USE_NJIT
# )
@numba.jitclass(spec)
class Indexes():
    def __init__(self, Npts, nel, neu, nmdl, npsds):
        self.Npts0 = Npts
        self.nel = nel
        self.neu = neu
        self.nmdl = nmdl
        self.npsds = npsds

        self.ladd = [0 for i in range(npsds)]
        self.uadd = [0 for i in range(npsds)]

        n_pivot_tot = np.array([self.nel[i] + self.Npts0[i] + self.neu[i] for i in range(npsds)])
        # self.nx0 = np.array([i*2*n_pivot_tot[i] + self.nel[i] for i in range(npsds)]) #0 + self.nel
        i_psd_last = 0
        nx0_aux = np.empty(npsds, dtype=np.int64)
        nx_last_aux = np.empty(npsds, dtype=np.int64)
        n_N_last_aux = np.empty(npsds, dtype=np.int64)
        for i in range(npsds):
            nx0_i = i_psd_last + self.nel[i]
            nx0_aux[i] = nx0_i
            nx_last_aux[i] = i_psd_last + n_pivot_tot[i]
            i_psd_last += 2*n_pivot_tot[i]
            n_N_last_aux[i] = i_psd_last
        self.nx0 = nx0_aux
        self.nx_last = nx_last_aux
        self.nN_last = n_N_last_aux
        # self.nx_last = np.array([n_pivot_tot[i]*(2*i + 1) for i in range(npsds)]) #self.nel + self.neu + self.Npts0
        self.nN0 = np.array([self.nx_last[i] + self.nel[i] for i in range(npsds)]) #self.nx_last + self.nel
        # self.nN_last = np.array([2*n_pivot_tot[i]*(i+1) for i in range(npsds)]) # self.nx_last + self.nel + self.neu + self.Npts0
        self.n_tot = 2*np.sum(n_pivot_tot) + nmdl # self.Npts0*2 + self.nel*2 + self.neu*2 + self.nmdl
        self.n_pivot_tot = n_pivot_tot

    def x_pick(self, psd_id = 0):
        return (self.nx0[psd_id] - self.ladd[psd_id],
            self.nx0[psd_id] + self.Npts0[psd_id] + self.uadd[psd_id]
        )

    def N_pick(self, psd_id = 0):
        return (self.nN0[psd_id] - self.ladd[psd_id],
            self.nN0[psd_id] + self.Npts0[psd_id] + self.uadd[psd_id]
        )

    def x_pick_tot(self, psd_id=0):
        return (self.nx0[psd_id] - self.nel[psd_id],
                self.nx_last[psd_id]
        )

    def get_x(self, y, psd_id = 0):
        p = self.x_pick(psd_id)
        return y[p[0]:p[1]]


    def get_N(self, y, psd_id = 0):
        p = self.N_pick(psd_id)
        return y[p[0]:p[1]]

    def get_mdl(self, y):
        p = (self.nN_last[-1], self.n_tot)
        return y[p[0]:p[1]]


    def increment_additions(self, incr_ladd, incr_uadd, psd_id = 0):
        self.ladd[psd_id] += incr_ladd
        self.uadd[psd_id] += incr_uadd

class FakeNb():
    instance_type = None
if os.getenv('NUMBA_DISABLE_JIT') == "1":
    Indexes.class_type = FakeNb()

#######
#######
# SOLVER
#######
#######

# TODO: add the multiple psd
def set_initial_states(psds0, y0_mdl, ind: Indexes,
        lmin = 0.0, lmax = 1e20):
    """
    Receives the initial psds as a list of tuples (x0, N0) for each PSD
    """

    y0 = np.zeros(ind.n_tot)
    for psd_id in range(len(psds0)):
        x0 = psds0[psd_id][0]
        N0 = psds0[psd_id][1]
        y0[ind.x_pick_tot(psd_id)[0]: ind.x_pick(psd_id)[0]] = \
            np.ones(ind.nel[psd_id]) * lmin
        y0[ind.x_pick(psd_id)[1]:ind.nx_last[psd_id]] = \
            np.ones(ind.neu[psd_id]) * lmax
        ind.get_x(y0, psd_id)[:] = x0
        ind.get_N(y0, psd_id)[:] = N0
    ind.get_mdl(y0)[:] = y0_mdl
    return y0

def create_integrate_nucl(mdl_rhs):

    step = pbe_utils.create_integrate_rkgill(mdl_rhs)

    @pbe_utils.if_njit()
    def integrate(tspan, y0, ind):
        y = y0.copy()
        # rhs = np.zeros_like(y)
        for i in range(len(tspan) - 1):
            ind.increment_additions(1, 0)
            y = step(tspan[i], tspan[i+1],
                    y, ind)
        return y

    return integrate

def create_integrate_nucl_class(step):

    #step = step_func #pbe_utils.integrate_rkgill_numba_mdl


    # @pbe_utils.if_njit()
    def integrate(tspan, y0, mdl, ret_array = False):
        y = y0.copy()
        if ret_array:
            y_full = np.empty((len(tspan), mdl.ind.n_tot))
            y_full[0,:] = y.copy()
        for i in range(len(tspan) - 1):
            mdl.before_simulation(y, tspan, i)
            y = step(tspan[i], tspan[i+1],
                    y, mdl)
            mdl.after_simulation(y, tspan, i)
            if ret_array:
                y_full[i+1,:] = y.copy()
        if ret_array:
            return y_full
        else:
            return np.atleast_2d(y)

    # if isinstance(step, numba.targets.registry.CPUDispatcher):
    #     integrate = numba.njit(integrate)

    return integrate

# def integrate_dasslcy(tspan, y0, mdl):
#     def step(t0, tf, y, mdl):
#         t, y, yp = dasslcy.solve(mdl.calc_mdl_rhs_wrapper, np.array([t0, tf]), y)
#         return y[-1]

#     y = y0.copy()
#     for i in range(len(tspan) - 1):
#         mdl.ind.increment_additions(1, 0)
#         y = step(tspan[i], tspan[i+1],
#                 y, mdl)
#     return y

# def integrate_sundials(tspan, y0, mdl, ret_array = False):
#     y = y0.copy()
#     for i in range(len(tspan) - 1):
#         mdl.before_simulation(y, tspan, i)
#         output = odeint(mdl.calc_mdl_rhs_wrapper_sundials,
#             [tspan[i], tspan[i+1]], y, method='admo')
#         y = output.values.y[-1,:]
#         mdl.after_simulation(y, tspan, i)
#     return y
