

# from py_pbe_msm import pbe_msm_solver, pbe_rhs_funcs, pbe_utils
# from utils import exponential_decay, SimulationOutput
# import numba
# from utils import retrieve_values_from_output
import sys
sys.path.append('/home/caio/Projects/CarbonateDeposition/Repositories/psd-simulations-msm/vessel_carbonate/Analysis/')

import numpy as np
import matplotlib.pyplot as plt

import model
from aux_notebook_funcs import (create_dict_after_simulation)
from py_pbe_msm import pbe_msm_solver, pbe_utils
from scipy import integrate

from parameter_estimation_configs import CONFIG_PRE_SIM_V0 #, CONFIG_PRE_SIM_V1

CONFIG_PRE_SIM = CONFIG_PRE_SIM_V0

def setup_simulation_for_parameters(config):
    Npts, x0, N0 = initial_condition_no_seeds_geometric(config)

    mdl, y0 = create_model_with_initial_conditions(config, x0, N0)

    nt = config['nt']
    tspan = np.linspace(0.0, config['tf'], nt)
    return mdl, y0, tspan

def create_model_with_initial_conditions(config, x0, N0):
    Npts = len(x0)
    nt = config['nt']
    theta = config['theta']
    params_list = params_dict_to_list(theta)
    mdl = model.SimpleModelAggNucGrw(Npts, nt, params_list)
    ymdl0 = np.array([])
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind, lmin = config['lmin'])
    return mdl, y0

def initial_condition_no_seeds_geometric(config):
    Npts = config['Npts']
    lmin = config['lmin']
    lmax = config['lmax']
    l0 = np.geomspace(lmin, lmax, Npts + 1)
    x0 = (l0[1:] + l0[0:-1]) * 1/2.0
    N0 = np.zeros_like(x0)
    return Npts, x0, N0

def initial_condition_square(config):
    Npts = config['Npts']
    lmin = config['lmin']
    lmax = config['lmax']
    lspan = np.geomspace(lmin, lmax, Npts + 1)
    # lspan = np.linspace(lmin, lmax, Npts + 1)
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = (lmin)
    b = (6.0)**3
    mu0_Ref = 29.0
    n0_fnc = lambda v, a, b: 0 if v<a else (mu0_Ref/(b-a) if v<b else 0)
    # n0_fnc = lambda v, a, b: 0 if v<a else 2.0*mu0_Ref/9.0*(b**(5/2) - a**(5/2)) if v<b else 0
    N_t0 = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        y, _ = integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])

    return Npts, xspan, N_t0

def simulate(mdl, y0, tspan):
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        pbe_utils.step_sundials
    )
    integration_func(tspan, y0, mdl)

def params_dict_to_list(theta):
    return (
        theta['kb'], theta['sigma_b'], theta['min_b_factor'],
        theta['kaggr'], theta['sigma_a'], theta['min_a_factor'],
        theta['kg']
    )

def params_lmfit_to_list(theta):
    return (
        theta['kb'].value, theta['sigma_b'].value, theta['min_b_factor'].value,
        theta['kaggr'].value, theta['sigma_a'].value, theta['min_a_factor'].value,
        theta['kg'].value
    )

def b_spline_params_to_list(theta, n_coeffs):
    tt = []
    for i in range(n_coeffs):
        tt += [theta['kaggr{}'.format(i)].value]
    coefs = np.array(tt)
    return coefs

def params_lmfit_to_list_bspline(theta):
    t0 = [
        theta['kb'].value, theta['sigma_b'].value, theta['min_b_factor'].value,
    ]
    try:
        t_aggr = []
        i = 0
        while True:
            t_aggr += [theta['kaggr{}'.format(i)]]
    except KeyError:
        pass

    tlast = [theta['sigma_a'].value, theta['min_a_factor'].value, theta['kg'].value]

    return t0 + t_aggr + tlast

def plot_single_run(d):
    mean_size = np.array(d['mu1']) / np.array(d['mu0'])
    plt.figure()
    plt.plot(d['tspan'], d['mu0'])
    plt.figure()
    plt.plot(d['tspan'], d['mu1'])
    plt.figure()
    plt.plot(d['tspan'], mean_size)

def example_of_simulation(config):
    mdl, y0, tspan = setup_simulation_for_parameters(config)
    simulate(mdl, y0, tspan)
    d = create_dict_after_simulation(mdl, config, tspan)
    return d, mdl, tspan

if __name__ == "__main__":
    example_of_simulation(CONFIG_PRE_SIM)


    plt.show()
