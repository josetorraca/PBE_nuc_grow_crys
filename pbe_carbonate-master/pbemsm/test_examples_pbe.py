import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
# import time
import numba
import pytest
import pbe_funcs_numba as pbe_funcs
import pbe_utils

## GLOBALS:
PLOT_FLAG = False

def initial_condition_square(Npts):
    lspan = np.linspace(0.0, 80.0, Npts + 1)
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = 10.0
    b = 20.0
    n0_fnc = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
    N_t0 = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        y, err = \
        integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])
    mu0_t0 = sum(N_t0)
    mu1_t0 = sum(N_t0 * xspan)

    return {
        'x': xspan,
        'N': N_t0
    }

@pytest.fixture()
def case_simple():
    Npts = 40
    G = 1.0
    B = 1.0
    ini_cond = initial_condition_square(Npts)
    return {
        'G': G,
        'B': B,
        'x0': ini_cond['x'],
        'N0': ini_cond['N'],
    }

def create_sim_grow_nucl(case_simple, fun_rhs_x, fun_rhs_N, jit = False):
    case = case_simple

    Npts = len(case['x0'])
    G = case['G']
    B = case['B']

    # if jit:
    fun_x = numba.njit(fun_rhs_x) if jit else fun_rhs_x
    fun_N = numba.njit(fun_rhs_N) if jit else fun_rhs_N

    @numba.njit
    def rhs_sys(t, y):
        rhs = np.zeros_like(y)
        x = y[0:Npts]
        N = y[Npts:]
        fun_x(x, rhs[0:Npts], G)
        fun_N(x, N, rhs[Npts:], B)
        return rhs

    return rhs_sys

def test_create_sim_grow_nucl(case_simple):
    case = case_simple
    x = case['x0']
    N = case['N0']
    dt = 1.0
    Npts = len(N)
    y = np.hstack((x, N))
    fun_rhs_x = pbe_funcs.create_pbe_grid_movement(jit=False)
    fun_rhs_N = pbe_funcs.create_rhs_pbe(jit=False)
    fun_rhs_sys_scoped = create_sim_grow_nucl(case, fun_rhs_x, fun_rhs_N, jit=True)
    assert callable(fun_rhs_sys_scoped)
    ynew = pbe_utils.integrate_rkgill(0.0, dt, y, fun_rhs_sys_scoped)
    assert ynew[Npts] == dt * case['B'] + N[0]
    assert ynew[0] == x[0] + case['G'] / 2.0 * dt


def create_sim_grow_nucl_aggr(case_simple, fun_rhs_x, fun_rhs_N, jit = False):
    case = case_simple

    Npts = len(case['x0'])
    G = case['G']
    B = case['B']

    fun_x = numba.njit(fun_rhs_x) if jit else fun_rhs_x
    fun_N = numba.njit(fun_rhs_N) if jit else fun_rhs_N

    @numba.njit
    def rhs_sys(t, y):
        rhs = np.zeros_like(y)
        x = y[0:Npts]
        N = y[Npts:]
        fun_x(x, rhs[0:Npts], G)
        fun_N(x, N, rhs[Npts:], B, None)
        return rhs

    return rhs_sys

def test_create_sim_aggr_no_nucl(case_simple):
    case = case_simple
    x = case['x0']
    N = case['N0']
    case['B'] = 0.0
    dt = 1.0
    Npts = len(N)
    y = np.hstack((x, N))
    aggr_func = lambda x1, x2, extr: 0.6
    fun_rhs_x = pbe_funcs.create_pbe_grid_movement(jit=True)
    fun_rhs_N = pbe_funcs.create_rhs_pbe(aggr_func, jit=True)
    fun_rhs_sys_scoped = create_sim_grow_nucl_aggr(case, fun_rhs_x, fun_rhs_N, jit=True)
    assert callable(fun_rhs_sys_scoped)
    ynew = pbe_utils.integrate_rkgill(0.0, dt, y, fun_rhs_sys_scoped)
    x_t = ynew[0:Npts]
    N_t = ynew[Npts:]
    mean_first_0 = np.mean(N[0:round(Npts/2)])
    mean_latters_0 = np.mean(N[round(Npts/2):])
    mean_firsts = np.mean(N_t[0:round(Npts/2)])
    mean_latters = np.mean(N_t[round(Npts/2):])
    assert(mean_firsts <= mean_first_0)
    assert(mean_latters >= mean_latters_0)
    assert ynew[0] == x[0] + case['G'] / 2.0 * dt
    if PLOT_FLAG:
        plt.plot(x, N, label='t0')
        plt.plot(x_t, N_t, label='tf')
        plt.show()

class GrowthNucleationMultiplesAdd():

    def __init__(self, case, fun_rhs_x, fun_rhs_N, jit = False):
        self.n_extra_low = case['n_extra_low']
        self.n_extra_upp = case['n_extra_upp']
        self.Npts0 = case['Npts']
        self.nx0 = 0 + self.n_extra_low
        self.nx_last = self.n_extra_low + self.n_extra_upp + self.Npts0
        self.nN0 = self.nx_last + self.n_extra_low
        self.nN_last = self.nx_last + self.n_extra_low + self.n_extra_upp + self.Npts0
        self.jit = jit
        self.fun_x = numba.njit(fun_rhs_x) if jit else fun_rhs_x
        self.fun_N = numba.njit(fun_rhs_N) if jit else fun_rhs_N
        self.ladd = 0
        self.uadd = 0

        # Problem specific:
        self.lmin = case['lmin']
        self.lmax = 1e20
        pass

    def calc_G(self, t, y):
        return 1.0e2

    def calc_B(self, t, y):
        return 1.0

    def x_pick(self):
        return (self.nx0 - self.ladd, self.nx0 + self.Npts0 + self.uadd)

    def N_pick(self):
        return (self.nN0 - self.ladd, self.nN0 + self.Npts0 + self.uadd)

    def get_x_active(self, y):
        xp = self.x_pick()
        return y[xp[0]:xp[1]]

    def get_N_active(self, y):
        Np = self.N_pick()
        return y[Np[0]:Np[1]]

    def increment_additions(self, incr_ladd, incr_uadd):
        self.ladd += incr_ladd
        self.uadd += incr_uadd

    # def integrate_multiples_addtion(t0, tf,)
    # @numba.njit
    def rhs_sys_with_adition(self, t, y):
        rhs = np.zeros_like(y)
        xp = self.x_pick()
        Np = self.N_pick()
        x = y[xp[0]:xp[1]]
        N = y[Np[0]:Np[1]]
        G = self.calc_G(t, y)
        B = self.calc_B(t, y)
        self.fun_x(x, rhs[xp[0]:xp[1]], G)
        self.fun_N(x, N, rhs[Np[0]:Np[1]], B, None)
        return rhs

    def sim_multiples_additions(self, tspan, y0, return_array = False,
            func_storage = None):

        if func_storage is None:
            func_storage = self.storage_as_list_of_actives
        if return_array:
            yFull = func_storage(y0, 0, tspan, None)
        y = y0.copy()
        for i in range(len(tspan) - 1):
            self.increment_additions(1, 0) #Forcing addition from t0
            y = pbe_utils.integrate_rkgill(tspan[i], tspan[i+1],
                    y, self.rhs_sys_with_adition)
            if return_array:
                yFull = func_storage(y, 0, tspan, yFull)

        if return_array:
            return yFull
        else:
            return y

    def storage_as_list_of_actives(self, y, i_t, tspan, yFull):
        if yFull == None:
            yFull = []
            yFull.append(y.copy())
        else:
            x_cp = self.get_x_active(y).copy()
            N_cp = self.get_N_active(y).copy()
            yFull.append([x_cp, N_cp])
        return yFull


    def fill_initial_vector(self, x0, N0):
        ny = self.n_extra_low*2 + self.n_extra_upp*2 + self.Npts0*2
        y = np.zeros(ny)
        xp = self.x_pick()
        Np = self.N_pick()
        y[0:xp[0]] = self.lmin
        y[xp[1]:self.nx_last] = self.lmax
        y[xp[0]:xp[1]] = x0
        y[Np[0]:Np[1]] = N0
        # print(y)
        return y

def test_create_sim_grow_nucl_multiple_addition_only_one_added():
    Npts = 4
    ic = initial_condition_square(4)
    case = {
        'Npts': Npts,
        'x0': ic['x'],
        'N0': ic['N'],
        'n_extra_low': 4,
        'n_extra_upp': 3,
        'tspan': np.linspace(0.0, 1.0, 2),
        'lmin': 0.0
    }
    x0 = case['x0']
    N0 = case['N0']
    dt = case['tspan'][1]
    Npts = len(N0)
    aggr_func = lambda x1, x2, extr: -1
    fun_rhs_x = pbe_funcs.create_pbe_grid_movement(jit=True)
    fun_rhs_N = pbe_funcs.create_rhs_pbe(aggr_func, jit=True)
    my_sys = GrowthNucleationMultiplesAdd(case, fun_rhs_x, fun_rhs_N, jit=True)
    y0 = my_sys.fill_initial_vector(x0, N0)
    ynew = my_sys.sim_multiples_additions(case['tspan'], y0)
    x_t = ynew[my_sys.x_pick()[0]: my_sys.x_pick()[1]]
    N_t = ynew[my_sys.N_pick()[0]: my_sys.N_pick()[1]]
    assert x_t[0] == case['lmin'] + my_sys.calc_G(0,0)/2 * dt
    assert x_t[0 + 1] == x0[0] + my_sys.calc_G(0,0) * dt
    assert N_t[0] == my_sys.calc_B(0,0) * dt
    assert N_t[0 + 1] == N0[0] + 0.0

def test_create_sim_grow_nucl_multiple_addition_various_added():
    Npts = 100
    n_extra_low = 50
    ic = initial_condition_square(Npts)
    case = {
        'Npts': Npts,
        'x0': ic['x'],
        'N0': ic['N'],
        'n_extra_low': n_extra_low,
        'n_extra_upp': 3,
        'tspan': np.linspace(0.0, 1.0, n_extra_low),
        'lmin': 0.0
    }
    x0 = case['x0']
    N0 = case['N0']
    dt = case['tspan'][1]
    Npts = len(N0)
    aggr_func = lambda x1, x2, extr: -1
    fun_rhs_x = pbe_funcs.create_pbe_grid_movement(jit=True)
    fun_rhs_N = pbe_funcs.create_rhs_pbe(aggr_func, jit=True)
    my_sys = GrowthNucleationMultiplesAdd(case, fun_rhs_x, fun_rhs_N, jit=True)
    y0 = my_sys.fill_initial_vector(x0, N0)
    ynew = my_sys.sim_multiples_additions(case['tspan'], y0, return_array=True)
    x_t = ynew[1][0]
    N_t = ynew[1][1]
    assert np.isclose(x_t[0], case['lmin'] + my_sys.calc_G(0,0)/2 * dt)
    assert np.isclose(x_t[0 + 1], x0[0] + my_sys.calc_G(0,0) * dt)
    assert np.isclose(N_t[0], my_sys.calc_B(0,0) * dt)
    assert np.isclose(N_t[0 + 1], N0[0] + 0.0)
    if PLOT_FLAG:
        plt.figure()
        plt.title('multiples bins add nucl and growth')
        plt.plot(x0, N0, '.-', label='initial')
        plt.plot(ynew[-1][0], ynew[-1][1], '.-', label='final')
        plt.show()


