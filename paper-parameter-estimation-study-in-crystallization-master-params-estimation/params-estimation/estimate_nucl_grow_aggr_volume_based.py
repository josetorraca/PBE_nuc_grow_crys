import os

import numba
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from lmfit import minimize, Parameters, report_fit

import sim_nucl_grow_aggr_volume_based as sim_mod
from py_pbe_msm import pbe_msm_solver, pbe_rhs_funcs, pbe_utils

TO_PLOT = True


class GorwthNuclSMOM():

    def __init__(self, params_mdl):
        self.base = params_mdl
        self.idx = {
            'C': 0,
            'mu_ini_s': 1,
            'mu_end_s': 5,
            'mu_ini_n': 5,
            'mu_end_n': 5+4,
            'size': 5+4,
        }
        self.rhs_gn_smom = np.empty(self.idx['size'])
        self.intermediaries = {
            'G': 0.0,
            'B': 0.0,
            'Csat': 0.0,
            'S': 0.0,
            'mass_cryst': 0.0,
        }
        self.j_orders = np.arange(0, 4)
        # self.base.lmin = 0.0 #FIXME
        pass

    def calc_mdl_rhs(self, t, y):
        idx = self.idx

        C = y[idx['C']]
        mus_s = y[idx['mu_ini_s']:idx['mu_end_s']]
        mus_n = y[idx['mu_ini_n']:idx['mu_end_n']]
        mus = mus_s + mus_n

        T = self.base.T_initial

        Csat = self.base.calc_sat(T)
        S = (C-Csat)/Csat
        G = self.base.calc_G(T, S)
        B = self.base.calc_B(T, S, mus[3])
        dCdt = -3*self.base.rho_c*self.base.kv*G*mus[2]
        self.rhs_gn_smom[idx['C']] = dCdt

        rhs_mus_s = self.rhs_gn_smom[idx['mu_ini_s']:idx['mu_end_s']]
        rhs_mus_s[0] = 0.0
        rhs_mus_s[1:] = self.j_orders[1:]*G*mus_s[0:-1]

        rhs_mus_n = self.rhs_gn_smom[idx['mu_ini_n']:idx['mu_end_n']]
        rhs_mus_n[0] = B
        rhs_mus_n[1:] = B*self.base.lmin**self.j_orders[1:] + \
                        self.j_orders[1:]*G*mus_n[0:-1]

        mass_cryst = mus[3] * self.base.rho_c * self.base.kv #CHECK IF Mass based
        mass_cryst_s = mus_s[3] * self.base.rho_c * self.base.kv #CHECK IF Mass based
        mass_cryst_n = mus_n[3] * self.base.rho_c * self.base.kv #CHECK IF Mass based
        self.intermediaries['G'] = G
        self.intermediaries['B'] = B
        self.intermediaries['Csat'] = Csat
        self.intermediaries['S'] = S
        self.intermediaries['mass_cryst'] = mass_cryst
        self.intermediaries['mass_cryst_s'] = mass_cryst_s
        self.intermediaries['mass_cryst_n'] = mass_cryst_n
        return self.rhs_gn_smom

    def get_intermediaries(self):
        return self.intermediaries

def simulate_gn_smom(mdl, tspan, y0):
    y = y0.copy()
    y_full = np.empty((len(tspan), y0.size))
    mdl.calc_mdl_rhs(tspan[0], y) #Initialize Intermediaries
    y_full[0,:] = y.copy()
    aux_interm = mdl.get_intermediaries()
    intermediaries = {key: np.empty(len(tspan)) for key in aux_interm}
    for key in intermediaries:
        intermediaries[key][0] = aux_interm[key]
    for i in range(len(tspan)-1):
        y = pbe_utils.integrate_rkgill_mdl(
            tspan[i], tspan[i+1], y, mdl)
        y_full[i+1,:] = y
        aux_interm = mdl.get_intermediaries()
        for key in intermediaries:
            intermediaries[key][i+1] = aux_interm[key]

    return y_full, intermediaries

def plot_smom_simulation(mdl, tspan, y_full, interm_full):

    C = y_full[:, mdl.idx['C']]
    mus, mus_s, mus_n = get_mus_from_y_full(y_full, mdl)
    Csat = interm_full['Csat']
    S = interm_full['S']
    G = interm_full['G']
    B = interm_full['B']
    mass_cryst = interm_full['mass_cryst']
    mass_cryst_s = interm_full['mass_cryst_s']
    mass_cryst_n = interm_full['mass_cryst_n']

    figs = {}
    figs['main'] = plt.figure(figsize=(15,8))
    plt.subplot(2,3,1)
    plt.plot(tspan, C, label='$C$')
    plt.plot(tspan, Csat, label='$C_{sat}$')
    plt.xlabel('time')
    plt.ylabel('C')
    plt.legend()

    plt.subplot(2,3,2)
    plt.plot(tspan, S, label='$S$')
    plt.legend()
    plt.ylabel('S')
    ax2 = plt.gca().twinx()
    ax2.plot(tspan, mass_cryst, label='$m_c$')
    plt.xlabel('time')
    ax2.set_ylabel('$m_c$')
    plt.legend()

    plt.subplot(2,3,3)
    plt.plot(tspan, G, label=r'$G$')
    plt.ylabel('G')
    plt.legend()
    ax2 = plt.gca().twinx()
    ax2.plot(tspan, B, label=r'$B_{0}$')
    plt.xlabel('time')
    ax2.set_ylabel('B')
    plt.legend()

    plt.subplot(2,3,4)
    plt.plot(tspan, mus[:,1]/mus[:,0], label=r'$L_{10}$')
    plt.ylabel(r'$L_{10}$')
    plt.xlabel('time')
    plt.legend()

    plt.subplot(2,3,5)
    plt.plot(tspan, mass_cryst, label=r'$m_c^{(tot)}$')
    plt.plot(tspan, mass_cryst_s, label=r'$m_c^{(s)}$')
    plt.plot(tspan, mass_cryst_n, label=r'$m_c^{(n)}$')
    plt.legend()
    # ax2 = plt.gca().twinx()
    # ax2.plot(tspan, mass_cryst_n/mass_cryst, label=r'$\frac{m_c^{(n)}}{m_c^{(tot)}}$')
    # plt.ylabel(r'$\frac{m_c^{(n)}}{m_c^{(tot)}}$')
    plt.xlabel('time')
    plt.legend()

    plt.subplot(2,3,6)
    plt.plot(tspan, mass_cryst_n/mass_cryst, label=r'$\frac{m_c^{(n)}}{m_c^{(tot)}}$')
    plt.ylabel(r'$\frac{m_c^{(n)}}{m_c^{(tot)}}$')
    plt.xlabel('time')
    plt.legend()


    figs['mus'] = plt.figure(figsize=(15,8))
    plt.subplot(2,2,1)
    plt.plot(tspan, mus[:,0], label=r'$\mu_0$')
    plt.plot(tspan, mus_s[:,0], label=r'$\mu_0^{(s)}$')
    plt.plot(tspan, mus_n[:,0], label=r'$\mu_0^{(n)}$')
    plt.ylabel(r'$\mu_0$')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(tspan, mus[:,1], label=r'$\mu_1$')
    plt.plot(tspan, mus_s[:,1], label=r'$\mu_1^{(s)}$')
    plt.plot(tspan, mus_n[:,1], label=r'$\mu_1^{(n)}$')
    plt.xlabel('time')
    plt.ylabel(r'$\mu_1$')
    plt.legend()

    plt.subplot(2,2,3)
    plt.plot(tspan, mus[:,2], label=r'$\mu_2$')
    plt.plot(tspan, mus_s[:,2], label=r'$\mu_2^{(s)}$')
    plt.plot(tspan, mus_n[:,2], label=r'$\mu_2^{(n)}$')
    plt.ylabel(r'$\mu_2$')
    plt.xlabel('time')
    plt.legend()

    plt.subplot(2,2,4)
    plt.plot(tspan, mus[:,3], label=r'$\mu_3$')
    plt.plot(tspan, mus_s[:,3], label=r'$\mu_3^{(s)}$')
    plt.plot(tspan, mus_n[:,3], label=r'$\mu_3^{(n)}$')
    plt.ylabel(r'$\mu_3$')
    plt.xlabel('time')
    plt.legend()

    return figs

def set_params_to_model(mdl, params):
    if 'kb' in params:
        mdl.base.kb = params['kb'].value
    if 'kb' in params:
        mdl.base.kg = params['kg'].value
    if 'k_aggr' in params:
        mdl.base.k_aggr = params['k_aggr'].value
    pass

def generate_test_data_grow_nucl_only(mdl, tspan, y0,
    params_ref, perc_error):

    set_params_to_model(mdl, params_ref)
    y_full, _ = simulate_gn_smom(mdl, tspan, y0)

    C_calc = y_full[:, mdl.idx['C']]
    mus, mus_s, mus_n = get_mus_from_y_full(y_full, mdl)
    mu0_calc = mus[:,0]

    rand_norm_c, sigma_C = create_normal_noise(C_calc, perc_error['C'])
    rand_norm_mu0, sigma_mu0 = create_normal_noise(mu0_calc, perc_error['mu0'])

    return {
        'C': C_calc + rand_norm_c,
        'mu0': mu0_calc + rand_norm_mu0,
        'sigma_C': sigma_C,
        'sigma_mu0': sigma_mu0,
    }

def create_normal_noise(var_calc, perc_error = 0.05):
    sigma = var_calc * perc_error
    rand_norm = np.random.normal(0.0, sigma, size=var_calc.shape)
    return rand_norm, sigma

def generate_test_data_grow_nucl_aggr(tspan, params_ref, perc_error=None):

    Npts = 200
    d, mdl = simulate_MSM_model_agg_nucl_grow(Npts, tspan, params_ref)

    C_calc = np.array(d['C'])
    mu0_calc = np.array(d['mu0'])

    rand_norm_c, sigma_C = create_normal_noise(C_calc, perc_error['C'])
    rand_norm_mu0, sigma_mu0 = create_normal_noise(mu0_calc, perc_error['mu0'])


    return {
        'C': C_calc + rand_norm_c,
        'mu0': mu0_calc + rand_norm_mu0,
        'sigma_C': sigma_C,
        'sigma_mu0': sigma_mu0,
    }

def simulate_MSM_model_agg_nucl_grow(Npts, tspan, params):

    x0, N0, lspan, xspan = sim_mod.initial_condition_psd(Npts)

    mdl, integration_func, y0 = sim_mod.setup_simulation(
        Npts, len(tspan), x0, N0, lmin=lspan[0]**3
    )

    set_params_to_model(mdl, params)

    y = integration_func(tspan, y0, mdl)

    d = sim_mod.get_variables_profiles(mdl, tspan)
    return d, mdl

def setup_estimate_smom_model(mdl, tspan):

    pars = Parameters()
    pars.add_many(#mdl.base.kg*1.5
        ('kg', mdl.base.kg, True, mdl.base.kg*0.0, None, None, None),
        ('kb', mdl.base.kb, True, mdl.base.kb*0.0, None, None, None),
    )
    return pars

def normalize_variable(x, xM, xm):
    return (x - xm) / (xM - xm)

def before_residual_calculation(params, mdl, tspan, y0):
    set_params_to_model(mdl, params)
    y_full, interm_full = simulate_gn_smom(mdl, tspan, y0)

    C_calc = y_full[:, mdl.idx['C']]
    mus, mus_s, mus_n = get_mus_from_y_full(y_full, mdl)
    mu0_calc = mus[:,0]
    return C_calc, mu0_calc

def obj_normalized(var_calc, var_exp):
    C_M = np.max(var_exp)
    C_m = np.min(var_exp)
    C_calc_norm = normalize_variable(var_calc, C_M, C_m)
    C_exp_norm = normalize_variable(var_exp, C_M, C_m)
    res_var = (C_calc_norm - C_exp_norm)
    return res_var

def residual_estimate_smom_model(params, mdl, tspan, y0, data):

    C_calc, mu0_calc = before_residual_calculation(params, mdl, tspan, y0)

    # resC = obj_normalized(C_calc, data['C'])
    # res_mu0 = obj_normalized(mu0_calc, data['mu0'])
    # return resC + res_mu0

    resC = obj_normalized(C_calc, data['C'])
    return resC

def run_estimate_smom_model(mdl, tspan, y0, params, data):

    args = (mdl, tspan, y0, data)
    # set_params_to_model(mdl, params)
    result_nelder = minimize(residual_estimate_smom_model, params, args=args,
        method='nelder')
    print('NELDER Intermediary Optimziation')
    report_fit(result_nelder)
    print('-------------------------------------------')
    result = minimize(residual_estimate_smom_model,
        result_nelder.params, args=args)

    return result

def plot_estimation(mdl, tspan, y_full, interm_full, data):

    C_exp = data['C']
    mu0_exp = data['mu0']

    C = y_full[:, mdl.idx['C']]
    mus, mus_s, mus_n = get_mus_from_y_full(y_full, mdl)
    mu0_calc = mus[:,0]

    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(tspan, C, '-k', label='$C$')
    plt.plot(tspan, C_exp, 'ok', label='$C_{exp}$')
    plt.xlabel('time')
    plt.ylabel('C')
    plt.legend()

    plt.subplot(2,2,2)
    plt.plot(tspan, mu0_calc, '-k', label=r'$\mu_0$')
    plt.plot(tspan, mu0_exp, 'ok', label=r'$\mu_{0,exp}$')
    plt.xlabel('time')
    plt.ylabel(r'$\mu_0$')
    plt.legend()

def get_mus_from_y_full(y_full, mdl):
    mus_s = y_full[:, mdl.idx['mu_ini_s']:mdl.idx['mu_end_s']]
    mus_n = y_full[:, mdl.idx['mu_ini_n']:mdl.idx['mu_end_n']]
    mus = mus_s + mus_n
    return mus, mus_s, mus_n


def main_only_growth_nucleation():
    nt = 101
    Npts = 2000
    _, N0, _, x0 = sim_mod.initial_condition_psd(Npts)
    mus_from_psd_initial = np.array(
        [np.sum(x0**o*N0) for o in range(4)]
    )
    mdl_base = sim_mod.ModelRawlings()
    mdl = GorwthNuclSMOM(mdl_base)
    y0 = np.zeros(mdl.idx['size'])
    y0[mdl.idx['C']] = mdl_base.C_initial
    y0[mdl.idx['mu_ini_s']:mdl.idx['mu_end_s']] = mus_from_psd_initial
    tspan = np.linspace(0.0, 10.0*60.0, nt)

    params_ref = setup_estimate_smom_model(mdl, tspan)

    params_data = params_ref.copy()
    # params_data['kb'].value *= 2e4

    perc_error = {'C': 0.01, 'mu0': 0.01}
    data = generate_test_data_grow_nucl_only(mdl, tspan, y0, params_data, perc_error=perc_error)

    y_full_real, interm_full_real = simulate_gn_smom(mdl, tspan, y0)
    figs = plot_smom_simulation(mdl, tspan, y_full_real, interm_full_real)
    [f.suptitle('Simulation from "Real" System ') for f in figs.values()]

    params_initial = params_ref.copy()
    params_initial['kg'].value *= 0.1
    params_initial['kb'].value *= 0.1
    params_initial['kb'].vary = True


    result_est = run_estimate_smom_model(mdl, tspan, y0, params_initial, data)
    params_est = result_est.params

    report_fit(result_est)
    print('FIT - Success:\n', result_est.success)
    print('FIT - Message:\n', result_est.message)

    set_params_to_model(mdl, params_est)
    y_full, interm_full = simulate_gn_smom(mdl, tspan, y0)

    plot_estimation(mdl, tspan, y_full, interm_full, data)


    return

def main_aggr_grow_nucl_as_data_source():
    nt = 101
    Npts = 2000
    _, N0, _, x0 = sim_mod.initial_condition_psd(Npts)
    mus_from_psd_initial = np.array(
        [np.sum(x0**o*N0) for o in range(4)]
    )
    mdl_base = sim_mod.ModelRawlings()
    mdl = GorwthNuclSMOM(mdl_base)
    y0 = np.zeros(mdl.idx['size'])
    y0[mdl.idx['C']] = mdl_base.C_initial
    y0[mdl.idx['mu_ini_s']:mdl.idx['mu_end_s']] = mus_from_psd_initial
    tspan = np.linspace(0.0, 10.0*60.0, nt)

    "Create Data"
    params_ref = setup_estimate_smom_model(mdl, tspan)
    perc_error = {'C': 0.01, 'mu0': 0.01}
    data = generate_test_data_grow_nucl_aggr(tspan, params_ref, perc_error=perc_error)

    "Create Parameters and Run Estimation"
    params_initial = params_ref.copy()
    params_initial['kg'].value *= 0.1
    params_initial['kb'].value *= 0.1
    params_initial['kb'].vary = True

    result_est = run_estimate_smom_model(mdl, tspan, y0, params_initial, data)
    params_est = result_est.params

    report_fit(result_est)
    print('FIT - Success:\n', result_est.success)
    print('FIT - Message:\n', result_est.message)

    "Plot Calculated from Model vs Experimental"
    set_params_to_model(mdl, params_est)
    y_full, interm_full = simulate_gn_smom(mdl, tspan, y0)
    plot_estimation(mdl, tspan, y_full, interm_full, data)

    "Simulating with MSM model without aggregation (as in the SMOM model)"
    params_msm_sim_aggr_zero = params_est.copy()
    params_msm_sim_aggr_zero.add(
        'k_aggr', value=0.0, vary=False
    )
    d, mdl_msm = simulate_MSM_model_agg_nucl_grow(
        200, tspan, params_msm_sim_aggr_zero
    )
    f = sim_mod.plot_psd(mdl_msm, d)
    f['N-l'].suptitle('PSD with estimated parameters')
    [plt.close(f_it.number) for key, f_it in f.items() if key != 'N-l']

    # "Real" Data:
    "Simulating REAL model - MSM with aggregation"
    d, _ = simulate_MSM_model_agg_nucl_grow(200, tspan, params_ref)
    f = sim_mod.plot_psd(mdl_msm, d)
    f['N-l'].suptitle('PSD with "real" parameters')
    [plt.close(f_it.number) for key, f_it in f.items() if key != 'N-l']

    return

if __name__ == '__main__':
    # main_only_growth_nucleation()
    main_aggr_grow_nucl_as_data_source()

    if TO_PLOT:
        plt.show()
