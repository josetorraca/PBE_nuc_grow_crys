import numpy as np
# from matplotlib import pyplot as plt
from scipy import integrate
# import time
import numba
import pytest
import pbe_funcs_numba as pbe_funcs


## GLOBALS:
JIT_FLAG = False

@pytest.fixture()
def case1():
    return dict( G = 1.0,
                 B = 1.0,
                 Aggr = 1.0,
                 Npts = 40
    )

#@pytest.fixture()
def before_run_settings(case):
    G, B, Aggr, Npts = case['G'] , case['B'], case['Aggr'], case['Npts']
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

    arg1 = {}
    arg1['Npts'] = Npts
    arg1['lspan'] = lspan
    arg1['growth_fnc'] = lambda v: G
    arg1['nucl_fnc'] = lambda: B
    arg1['aggre_func'] = lambda xj, xk: Aggr
    arg1['t_END'] = 8e-3 #8e-3
    arg1['tspan'] = np.linspace(0.0, arg1['t_END'], 11)
    arg1['y0'] = np.hstack((xspan, N_t0))
    return arg1


def test_create_pbe_number_only_no_jit(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=None, jit=False)
    assert(callable(rhs_pbe_numbers))

def test_create_pbe_number_only_jit(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    Aggr_extras = None
    NinOut = None
    B0 = case1['B']
    rhs = np.empty_like(y)
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=None, jit=True)
    assert(callable(rhs_pbe_numbers))

def test_create_pbe_number_only_call_no_agg_no_inout(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    B0 = case1['B']
    rhs = np.empty_like(y)
    rhs_Ni = rhs[Npts:]
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=None, NinOut=False, jit=JIT_FLAG)
    assert(callable(rhs_pbe_numbers))
    rhs_pbe_numbers(x, N, rhs_Ni, B0)
    assert(rhs_Ni[0] == B0)

def test_create_pbe_number_call_no_agg_inout(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    B0 = case1['B']
    rhs = np.empty_like(y)
    rhs_Ni = rhs[Npts:]
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=None, NinOut=True, jit=JIT_FLAG)
    NinOut = np.zeros_like(N)
    NinOut[1] = 100.0
    rhs_pbe_numbers(x, N, rhs_Ni, B0, NinOut)
    assert(rhs_Ni[1] == NinOut[1])

def test_create_pbe_number_call_agg_no_inout(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    rhs = np.empty_like(y)
    rhs_Ni = rhs[Npts:]
    aggr_func = lambda x1, x2, extr: 0.6
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=aggr_func, NinOut=False, jit=JIT_FLAG)
    rhs_pbe_numbers(x, N, rhs_Ni, 0.0, None)
    mean_firsts = np.mean(rhs_Ni[0:round(Npts/2)])
    mean_latters = np.mean(rhs_Ni[round(Npts/2):])
    assert(mean_firsts <= 0.0)
    assert(mean_latters >= 0.0)

def test_create_pbe_number_call_agg_inout(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    rhs = np.empty_like(y)
    rhs_Ni = rhs[Npts:]
    aggr_func = lambda x1, x2, extr: 0.6
    NinOut = np.zeros_like(N)
    NinOut[1] = 100.0
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=aggr_func, NinOut=True, jit=JIT_FLAG)
    rhs_pbe_numbers(x, N, rhs_Ni, 0.0, NinOut, None)
    assert(rhs_Ni[1] == NinOut[1])

def test_create_pbe_number_call_agg_inout_ignore_aggr(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    rhs = np.empty_like(y)
    rhs_Ni = rhs[Npts:]
    aggr_func = lambda x1, x2, extr: -1
    rhs_pbe_numbers = pbe_funcs.create_rhs_pbe(aggr_func=aggr_func, NinOut=True, jit=JIT_FLAG)
    rhs_pbe_numbers(x, N, rhs_Ni, 0.0, np.zeros_like(N), None)
    #mean_firsts = np.mean(rhs_Ni[0:round(Npts/2)])
    #mean_latters = np.mean(rhs_Ni[round(Npts/2):])
    assert(rhs_Ni[0] == 0.0)

### GRID
##------------------------

def test_create_pbe_grid_movement_const(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    G = case1['G']
    y = args['y0']
    x = y[0:Npts]
    N = y[Npts:2*Npts]
    rhs = np.empty_like(y)
    rhs_x = rhs[0:Npts]
    rhs_x_func = pbe_funcs.create_pbe_grid_movement(as_array=False, jit=JIT_FLAG)
    assert(callable(rhs_x_func))
    rhs_x_func(x, rhs_x, G)
    assert(rhs_x[0] == G/2)
    assert(rhs_x[1] == G)

def test_create_pbe_grid_movement_array(case1):
    args = before_run_settings(case1)
    Npts = args['Npts']
    y = args['y0']
    x = y[0:Npts]
    G = np.ones_like(x) * (case1['G'])
    N = y[Npts:2*Npts]
    rhs = np.empty_like(y)
    rhs_x = rhs[0:Npts]
    rhs_x_func = pbe_funcs.create_pbe_grid_movement(as_array=True, jit=JIT_FLAG)
    assert(callable(rhs_x_func))
    rhs_x_func(x, rhs_x, G)
    assert(rhs_x[0] == G[0]/2)
    assert(rhs_x[1] == G[1])

# def


