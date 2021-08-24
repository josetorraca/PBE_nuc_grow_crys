import os

import colorlover as cl
import numba
import numpy as np
import plotly.graph_objs as go
import plotly.offline as py
from plotly import tools
from scipy import integrate

from py_pbe_msm import pbe_msm_solver, pbe_rhs_funcs, pbe_utils

# import ptvsd
# # 5678 is the default attach port in the VS Code debug configurations
# print("Waiting for debugger attach")
# ptvsd.enable_attach(address=('localhost', 5678), redirect_output=True)
# ptvsd.wait_for_attach()
# breakpoint()

# Auxiliary functions
def iplot_show(fig):
    py.iplot(fig, show_link=False, config={'displaylogo': False})


###
# Generate charts for AGGREGATION Vs Nucleation
###
def generate_psd_aggre_nucl_fig(*args):
    tspan, x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1, \
        length_span, length_mean, length_n = args
    # size_span0 = np.array(x_span[0])**(1/3)
    # size_spanf = np.array(x_span[-1])**(1/3)

    greys = cl.scales['6']['seq']['Greys']
    greys = greys[-3:]
    t1 = go.Scattergl(x = length_span[0], y=N_span[0], mode = 'lines+markers', name = 'N-ini',
        line = dict(
        color = (greys[0])), marker = {'symbol':'circle'}
    )
    t12 = go.Scattergl(x = length_span[-1], y=N_span[-1], mode = 'lines+markers', name = 'N-end',
        line = dict(
        color = (greys[1])), marker = {'symbol':'triangle-up'}
    )
    t2 = go.Scattergl(x = length_span[0], y=length_n[0], mode = 'lines+markers', name = 'n-ini',
        line = dict(
        color = (greys[0])), marker = {'symbol':'circle'}
    )
    t22 = go.Scattergl(x = length_span[-1], y=length_n[-1], mode = 'lines+markers', name = 'n-end',
        line = dict(
        color = (greys[1])), marker = {'symbol':'triangle-up'}
    )
    fig = tools.make_subplots(rows=1, cols=2, subplot_titles=('Number', 'Density'), print_grid=False);
    fig.append_trace(t1, 1, 1)
    fig.append_trace(t12, 1, 1)
    fig.append_trace(t2, 1, 2)
    fig.append_trace(t22, 1, 2)
    fig['layout'].update(title= '$\\alpha$ PSD', xaxis={'showgrid': False}, xaxis2={'showgrid': False}) #height=600, width=800, title='i <3 annotations and subplots')
    return fig

def charts_psd_aggre_nucl(*args):
    fig = generate_psd_aggre_nucl_fig(*args)
    iplot_show(fig)

def charts_mus_rates_aggre_nucl(*args, return_fig = False):
    greys = cl.scales['6']['seq']['Greys']
    greys = greys[-3:]
    tspan, x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1, \
        length_span, length_mean, length_n = args
    # meanSize_conv = (np.array(mu1))**(1/3) / (np.array(mu0))
    size_span = [np.array(x)**(1/3) for x in x_span]
    N_span_np = [np.array(N) for N in N_span]
    meanSize_conv = [np.sum(s * N_span_np[i]) / mu0[i] for i, s in enumerate(size_span)]
    t11 = go.Scatter(x = tspan, y=mu0, mode = 'lines+markers', name = '$\mu_0$',
        line = dict(
        color = (greys[0])), marker = {'symbol':'circle'}
    )
    t21 = go.Scatter(x = tspan, y=meanSize_conv, mode = 'lines+markers', name = 'mean',
        line = dict(
        color = (greys[1])), marker = {'symbol':'triangle-up'}
    )
    t21a = go.Scatter(x=tspan, y=(0.0-agg_tot_num_rate_span), mode = 'lines+markers', name = '$R_{aggr}$',
        line = {'color':(greys[0])}, marker={'symbol':'circle'})
    t21b = go.Scatter(x=tspan, y=B_span, mode = 'lines+markers', name = '$B_0$',
        line = {'color':(greys[1])}, marker={'symbol':'triangle-up'})
    t21c = go.Scatter(x=tspan, y=B_span + agg_tot_num_rate_span, mode = 'lines+markers', name = 'net',
        line = {'color':(greys[2])}, marker={'symbol':'triangle-down'})

    fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {'rowspan': 2}], [{}, {}]],
        subplot_titles=('$\\mu_0$', 'Number Changing Mechanims', 'Mean Size'), print_grid=False)
    fig.append_trace(t11, 1, 1)
    fig.append_trace(t21, 2, 1)
    fig.append_trace(t21a, 1, 2)
    fig.append_trace(t21b, 1, 2)
    fig.append_trace(t21c, 1, 2)
    # fig.append_trace(t22, 1, 2)
    fig['layout'].update(title= 'Moments and Rates', xaxis={'showgrid': False}, xaxis2={'showgrid': False})
    if return_fig:
        return fig
    return go.FigureWidget(fig)

def multiple_runs_murs_rates_charts(d_span):
    greys = cl.scales['6']['seq']['Greys']
    greys10 = cl.interp(greys, len(d_span))
    # greys = greys[-3:]
    t11_span = []
    t21_span = []
    t12_span = []
    for i, d in enumerate(d_span):
        sol_args = dict_sol_to_tuple(d)

        tspan, x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1, *_ = sol_args
        meanSize = np.array(mu1) / np.array(mu0)
        t11 = go.Scattergl(x = tspan, y=mu0, mode = 'lines+markers', name = '$\mu_0$-{}'.format(i),
            line = dict(
            color = (greys10[i])), marker = {'symbol':'circle'}
        )
        t11_span += [t11]
        t21 = go.Scattergl(x = tspan, y=meanSize, mode = 'lines+markers', name = 'mean-{}'.format(i),
            line = dict(
            color = (greys10[i])), marker = {'symbol':'triangle-up'}
        )
        t21_span += [t21]
        t12aggr = go.Scattergl(x=tspan, y=(0.0-agg_tot_num_rate_span), mode = 'lines+markers', name = '$R_{aggr}$',
            line = {'color':(greys10[i])}, marker={'symbol':'circle'})

        t12_span += [t12aggr]

    t12b = go.Scattergl(x=tspan, y=B_span, mode = 'lines+markers', name = '$B_0$',
        line = {'color':(greys10[i])}, marker={'symbol':'triangle-up'})
    t12_span += [t12b]

    fig = tools.make_subplots(rows=2, cols=2, specs=[[{}, {'rowspan': 2}], [{}, {}]],
        subplot_titles=('$\\mu_0$', 'Number Changing Mechanims', '$\\mu_1 / \\mu_0$'), print_grid=False)

    for trace in t11_span:
        fig.append_trace(trace, 1, 1)
    for trace in t21_span:
        fig.append_trace(trace, 2, 1)
    for trace in t12_span:
        fig.append_trace(trace, 1, 2)
    # fig.append_trace(t11, 1, 1)
    # fig.append_trace(t21, 2, 1)
    # fig.append_trace(t12b, 1, 2)
    # fig.append_trace(t21, 1, 2)
    fig['layout'].update(title= 'Moments / Rates', xaxis={'showgrid': False}, xaxis2={'showgrid': False})
    fig['layout'].update(showlegend=False)
    iplot_show(fig)
    # return go.FigureWidget(fig)


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
    ('t', numba.float64),
    # ('ymdl0', numba.float64[:])
]
@numba.jitclass(spec)
class AggreConstNucleationModel():

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
        if np.any(rhs_N > 0.0) or t > 1.0:
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

@numba.njit()
def exponential_decay(t, n_start, low_bound, sigma):
    n_max = n_start - low_bound
    r = n_max*np.exp(-(t/2.0/sigma)**2) + low_bound
    return r

@numba.jitclass(spec)
class AggreVaryingNucleationModel():

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
        return self.kg

    def calc_B(self, t):
        low_b = 20.0
        sigma = 0.01
        r = exponential_decay(t, self.kb, low_b, sigma)
        return r

    def calc_Aggr(self, x1, x2, N1, N2):
        if N1 < 1e-20 or N2 < 1e-20: return 0.0
        return self.kaggr

    def calc_mdl_rhs(self, t, y):
        self.t = t
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        NinOut = np.zeros_like(N)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        G = self.calc_G()
        self.B = self.calc_B(t)
        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, self.B, NinOut, self)
        if np.any(rhs_N > 0.0) or t > 1.0:
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

spec_aggre_varying = spec + [
    ('a_sigma', numba.float64),
    ('a_low_b', numba.float64),
    ('b_sigma', numba.float64),
    ('b_low_b', numba.float64),
]
@numba.jitclass(spec_aggre_varying)
class VaryingAggreVaryingNucleationModel():

    def __init__(self, Npts, nts, x0, N0, kb, kg, kaggr):
        self.ind = self.set_indexes_case(Npts, nts)

        # self.kb = kb
        self.kb, self.b_sigma, self.b_low_b = kb
        self.kg = kg
        self.kaggr, self.a_sigma, self.a_low_b = kaggr
        # TODO: adjust to kaggr -> aggr_prams

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 0, 1)
        return ind

    def calc_G(self):
        return self.kg #1.0e1 #1.0e2

    def calc_B(self, t):
        r = exponential_decay(t, self.kb, self.b_low_b, self.b_sigma)
        return r

    def calc_Aggr(self, x1, x2, N1, N2):
        if N1 < 1e-20 or N2 < 1e-20: return 0.0
        r = exponential_decay(self.t, self.kaggr, self.a_low_b, self.a_sigma)
        return r
        # return self.kaggr

    def calc_mdl_rhs(self, t, y):
        self.t = t
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        NinOut = np.zeros_like(N)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        G = self.calc_G()
        self.B = self.calc_B(t)
        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, self.B, NinOut, self)
        if np.any(rhs_N > 0.0) or t > 1.0:
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

def builder_simple_VaryingAggreVaryingNucleationModel(kb, kg, kaggr):
    return VaryingAggreVaryingNucleationModel(0, 0, 0, 0, kb, kg, kaggr)


@numba.jitclass(spec_aggre_varying)
class VaryingAggreVaryingNucleationNonZeroGrowthModel():

    def __init__(self, Npts, nts, x0, N0, kb, kg, kaggr):
        self.ind = self.set_indexes_case(Npts, nts)

        # self.kb = kb
        self.kb, self.b_sigma, self.b_low_b = kb
        self.kg = kg
        self.kaggr, self.a_sigma, self.a_low_b = kaggr
        # TODO: adjust to kaggr -> aggr_prams

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 0, 1)
        return ind

    def calc_G(self):
        return self.kg #1.0e1 #1.0e2

    def calc_B(self, t):
        r = exponential_decay(t, self.kb, self.b_low_b, self.b_sigma)
        return r

    def calc_Aggr(self, x1, x2, N1, N2):
        if N1 < 1e-20 or N2 < 1e-20: return 0.0
        r = exponential_decay(self.t, self.kaggr, self.a_low_b, self.a_sigma)
        return r
        # return self.kaggr

    def calc_mdl_rhs(self, t, y):
        self.t = t
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        NinOut = np.zeros_like(N)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        G = self.calc_G()
        self.B = self.calc_B(t)
        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, self.B, NinOut, self)
        if np.any(rhs_N > 0.0) or t > 1.0:
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


def run_case_multiples_models(params, model_ref):
    """Varying Nucleation, constant Aggregation, constant growth, seed at 1"""
    l0, x0, N0 = initial_condition_square(params['Npts'], params['lmin'])
    mdl = model_ref(params['Npts'], params['nt'],
            x0, N0, params['kb'], params['kg'], params['kaggr'])
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], np.array([]), mdl.ind, lmin=params['lmin'])
    tspan = np.linspace(0.0, params['tf'], params['nt'])
    integration_func = pbe_msm_solver.create_integrate_nucl_class(pbe_utils.step_sundials)
    integration_func(tspan, y0, mdl)
    d = create_dict_after_simulation(mdl, params, tspan)
    return d

def create_dict_after_simulation(mdl, params, tspan):
    x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1 =\
        retrieve_values_from_output(mdl, params)

    size_span, mean_length, l_span_length, n_span_length = \
        retrieve_length_based_values(x_span, N_span, mu0, params)


    d = {
        'x_span': x_span, 'N_span': N_span, 'l_span': l_span,
        'n_span': n_span, 'agg_tot_num_rate_span': agg_tot_num_rate_span,
        'B_span': B_span, 'mu0': mu0, 'mu1': mu1, 'tspan': tspan,
        'length_span': size_span, 'length_mean': mean_length, 'length_n': n_span_length
    }
    return d

def setup_simulation_aggre_const_nucl(nt, Npts, params, mdl_class=AggreConstNucleationModel):
    l0, x0, N0 = initial_condition_square(Npts, params['lmin'])
    mdl = mdl_class(Npts, nt, x0, N0, params['kb'], params['kg'], params['kaggr'])
    ymdl0 = np.array([]) #mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind, lmin=params['lmin'])
    tspan = np.linspace(0.0, params['tf'], nt)
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        pbe_utils.step_sundials
    )
    def solve(mdl_):
        integration_func(tspan, y0, mdl_)
    return mdl, solve, tspan

def retrieve_values_from_output(mdl, params):
    x_span = np.array([item.x for item in mdl.out_span])
    N_span = np.array([item.N for item in mdl.out_span])
    l_span = [np.hstack((params['lmin'], (x_i[1:] + x_i[0:-1])*0.5, x_i[-1] + (x_i[-1]-x_i[-2])*0.5 ))
        for x_i in x_span]
    n_span = [N_span[i] / (l_span[i][1:] - l_span[i][0:-1]) for i in range(len(x_span))]

    mu0 = [np.sum(item.N) for item in mdl.out_span]
    mu1 = [np.sum(item.x * item.N) for item in mdl.out_span]

    agg_tot_num_rate_span = np.array([item.agg_tot_num_rate for item in mdl.out_span])
    B_span = np.array([item.B for item in mdl.out_span])


    return x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1

def retrieve_length_based_values(x_span, N_span, mu0_span, params):


    size_span = [x**(1/3) for x in x_span]
    mean_length = [np.sum(s * N_span[i]) / mu0_span[i] for i, s in enumerate(size_span)]

    l_span_length = [np.hstack((params['lmin'], (x_i[1:] + x_i[0:-1])*0.5, x_i[-1] + (x_i[-1]-x_i[-2])*0.5 ))
        for x_i in size_span]
    n_span_length = [N_span[i] / (l_span_length[i][1:] - l_span_length[i][0:-1]) for i in range(len(x_span))]

    return size_span, mean_length, l_span_length, n_span_length

def retrieve_values_from_output_as_tuple(mdl, params, tspan):
    x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1 = \
        retrieve_values_from_output(mdl, params)
    sol_args = (tspan, x_span, N_span, l_span, n_span, agg_tot_num_rate_span, B_span, mu0, mu1)
    return sol_args

def dict_sol_to_tuple(d):
    return (
        d['tspan'], d['x_span'], d['N_span'], d['l_span'], d['n_span'],
        d['agg_tot_num_rate_span'], d['B_span'], d['mu0'], d['mu1'],
        d['length_span'], d['length_mean'], d['length_n']
    )

def nucl_agg_behavior_with_params(mdl, tspan):
    nucl = np.array([mdl.calc_B(t) for t in tspan])
    aggr = np.zeros_like(tspan)
    for i, t in enumerate(tspan):
        mdl.t = t
        aggr[i] = mdl.calc_Aggr(0,0,1,1)
    return nucl, aggr

def plotly_chart_after_sim_agg_nucl_profiles(tspan, nucl, aggr):
    trace1 = go.Scatter(x=tspan,y=nucl,name='Nucl', mode = 'lines')
    trace2 = go.Scatter(x=tspan,y=aggr,name='Aggr',yaxis='y2', mode = 'lines')
    data_fig = [trace1, trace2]
    layout = go.Layout(
        title='Nucleation and Aggregation Profiles',
        yaxis={'title': 'Nucleation'},
        yaxis2={'title': 'Aggregation','titlefont':{'color': 'rgb(148, 103, 189)'},
                'tickfont':{'color': 'rgb(148, 103, 189)'}, 'overlaying': 'y', 'side': 'right'
        }
    )
    fig = go.Figure(data=data_fig, layout=layout)
    # aux.iplot_show(fig)
    return fig
