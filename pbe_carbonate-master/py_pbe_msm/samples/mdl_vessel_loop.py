import numpy as np
from scipy import integrate
from py_pbe_msm import pbe_msm_solver
from py_pbe_msm import pbe_rhs_funcs
from py_pbe_msm import pbe_utils
from py_pbe_msm import numba_utils
import numba
from matplotlib import pyplot as plt
import os

# pylint: disable=E1101

TO_PLOT = True
pbe_utils.USE_NJIT = True


def initial_condition_square(Npts):
    lspan = np.linspace(0.0, 500.0, Npts + 1)
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = 250.0
    b = 300.0
    def n0_fnc(v, a, b):# = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
        if v < a or v > b:
            return 0.0
        else:
            return 0.0032*(300.0 - v)*(v - 250.0)
    N_t0 = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        quad_ret = \
        integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        y, _ = quad_ret
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])

    return xspan, N_t0

@numba.njit
def calc_mass_crystal(x, N, mdl):
    return np.sum(x**3*N) * mdl.kv * mdl.rho_c * mdl.m_slv

@numba.jitclass([
    ('x', numba.float64[:]),
    ('N', numba.float64[:]),
    ('C', numba.float64),
    ('Csat', numba.float64),
    ('mCryst', numba.float64),
])
class PSD():
    def __init__(self, x, N, C, Csat, mdl):
        self.x = x
        self.N = N
        self.C = C
        self.Csat = Csat
        self.mCryst = calc_mass_crystal(x, N, mdl)
        pass
if os.getenv('NUMBA_DISABLE_JIT') == "1":
    PSD.class_type = pbe_msm_solver.FakeNb()

spec = [
    ('x0', numba.float64[:]),
    ('N0', numba.float64[:]),
    ('ind', pbe_msm_solver.Indexes.class_type.instance_type),
    ('ymdl0', numba.float64[:]),
    ('b', numba.float64), ('g', numba.float64), ('kb', numba.float64),
    ('kg', numba.float64), ('EbporR', numba.float64), ('EgporR', numba.float64),
    ('m_slv', numba.float64), ('rho_c', numba.float64), ('kv', numba.float64),
    ('S', numba.float64), ('mu3', numba.float64), ('T', numba.float64),
    ('C', numba.float64), ('Vdot', numba.float64), ('Vc', numba.float64),
    ('Csat', numba.float64),
    ('t_pipe', numba.float64),
    ('NinOut', numba.float64[:]),
    ('psd_full', numba.types.List(PSD.class_type.instance_type)),
    ('i_t_last', numba.int64),
    ('tspan', numba.float64[:]),
]
@numba.jitclass(spec)
class MyModel():
    """
    PSD in #/mu/g_slv

    """

    def __init__(self, Npts, tspan, x0, N0):
        nts = tspan.shape[0]
        self.ind = self.set_indexes_case(Npts, nts)
        self.b = 1.45
        self.g = 1.5
        self.kb = 285.01
        self.kg = 1.44e8
        self.EbporR = 7517.0
        self.EgporR = 4859.0
        self.m_slv = 27.0e3 #g
        self.rho_c = 2.66e-12
        self.kv = 1.5
        self.Vdot = 1.0 #mL/s
        Vpipe = 60.0 *(0.5)**2 * np.pi #mL
        tau = Vpipe / self.Vdot
        dt = (tspan[1] - tspan[0])
        self.t_pipe = int(tau / dt) * dt #0.0 #
        print('t_pipe is = ', self.t_pipe)
        # self.t_pipe = 0.
        # self.t_pipe = 5*dt #6*60.0

        # Variables
        self.S = 0.0
        self.mu3 = 0.0
        self.T = 20.0
        self.Csat = self.calc_sat()
        self.NinOut = np.zeros(self.ind.n_pivot_tot[0])

        # States
        self.C = 0.1681

        self.ymdl0 = np.array([self.C])

        # Full state history is needed for inteporlation and deadtime
        # self.psd_full = []
        self.i_t_last = 0

        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], 1, 1)
        return ind

    def calc_Aggr(self, x1, x2):
        return -1

    def calc_G(self):
        # return 0.0
        return self.kg*np.exp(-self.EgporR/(self.T+273.15))*self.S**self.g

    def calc_B(self):
        return self.kb*np.exp(-self.EbporR/(self.T+273.15))*self.mu3*self.S**self.b
        # return 0.0

    def calc_sat(self):
        return 6.29e-2 + 2.46e-3*self.T - 7.14e-6*self.T**2

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        ac_i = self.ind.x_pick(0)
        NinOut_ = self.NinOut[ac_i[0]:ac_i[1]]
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        rhs_mdl = self.ind.get_mdl(rhs)

        self.C = self.ind.get_mdl(y)[0]
        self.Csat = self.calc_sat()
        self.S = (self.C-self.Csat)/self.Csat
        mu2 = np.sum(x**2 * N)
        self.mu3 = np.sum(x**3 * N)

        G = self.calc_G()
        B = self.calc_B()

        ## In and Out from vessel:
        # Nin = np.zeros_like(NinOut_)
        # mSlvDot = self.Vdot * 1.0 #g/s
        # Nin = self.calc_NinOut_lag(t, x, N)
        CInLag = np.empty(1)
        self.calc_NinOut_lag(t, x, N, NinOut_, CInLag)
        # NinOut_[:] = mSlvDot/self.m_slv * (Nin[:] - N[:])

        mSlvDotOut = 1.0 * self.Vdot * self.C
        mSlvDotIn = 1.0 * self.Vdot * CInLag[0]
        dCdt = -3*self.rho_c*self.kv*G*mu2 + 1/self.m_slv*(mSlvDotIn - mSlvDotOut)


        pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, B, NinOut_, self)
        pbe_rhs_funcs.grid_const_G_half(x, rhs_x, G)
        rhs_mdl[0] = dCdt
        return rhs

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def before_simulation(self, y, tspan, i_t):
        self.ind.increment_additions(1, 0, 0)
        if i_t == 0:
            x = self.ind.get_x(y, 0)
            N = self.ind.get_N(y, 0)
            self.psd_full = [PSD(x, N, self.C, self.Csat, self)]
        if i_t == 0: self.tspan  = tspan.copy()

    def after_simulation(self, y, tspan, i_t):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        self.psd_full += [PSD(x, N, self.C, self.Csat, self)]
        self.i_t_last = i_t + 1
        pass

    def calc_NinOut_lag(self, t, x, N, NInOut, CInLag):
        if self.t_pipe < 1e-4:
            NInOut[:] = 0.0
            CInLag[0] = self.C
            return
        mSlvDot = self.Vdot * 1.0 #g/s
        t_lagged = t - self.t_pipe
        if t_lagged <= 0.0:
            NInOut[:] = 0.0
            CInLag[0] = self.C
            return #initially there is particles as in vessel

        t_history = self.tspan[0:self.i_t_last+1]
        psd_history = self.psd_full[0:self.i_t_last+1]
        idx_found = np.searchsorted(t_history, t_lagged)
        if idx_found == 0:
            NInOut[:] = 0.0
            CInLag[0] = self.C
            return
        # n_interp_max_each_side = 4
        # if t_history.shape[0] < 2*n_interp_max_each_side:
        #     t_2interp = t_history[:]
        #     psd_2interp = psd_history
        # elif idx_found < n_interp_max_each_side:
        #     t_2interp = t_history[0:idx_found+n_interp_max_each_side]
        #     psd_2interp = psd_2interp[0:idx_found+n_interp_max_each_side]
        # elif self.i_t_last - idx_found < n_interp_max_each_side:
        #     t_2interp = t_history[idx_found-n_interp_max_each_side:]
        #     psd_2interp = psd_2interp[idx_found-n_interp_max_each_side:]

        t_intp = t_history[idx_found - 1]
        psd_intp = psd_history[idx_found - 1]
        xA = psd_intp.x
        NA = psd_intp.N
        NB = pbe_utils.combine_mesh_simple_interp(xA, NA, x, 0.0)
        # NB = pbe_utils.combine_mesh_boundary_based(xA, NA, x, 0.0)
        NInOut[:] = mSlvDot/self.m_slv * (NB - N)
        CInLag[0] = psd_intp.C
        # NInOut[:] = 0.0
        return


def main():
    nt = 5001
    tspan = np.linspace(0.0, 180.0*60.0, nt)
    Npts = 2000
    x0, N0 = initial_condition_square(Npts)
    mdl = MyModel(Npts, tspan, x0, N0)
    ymdl0 = mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind)
    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        # pbe_utils.step_sundials
        pbe_utils.integrate_rkgill_numba_mdl
    )

    c = 0
    while c < 1:
        mdl = MyModel(Npts, tspan, x0, N0)
        y = integration_func(tspan, y0, mdl)
        y = y[0, :]
        c += 1

    if TO_PLOT:
        x = mdl.ind.get_x(y, 0)
        N = mdl.ind.get_N(y, 0)
        mCIni = calc_mass_crystal(x0, N0, mdl)
        mCEnd = calc_mass_crystal(x, N, mdl)
        Csp = np.array([psd.C for psd in mdl.psd_full] )
        Csatsp = np.array([psd.Csat for psd in mdl.psd_full] )
        mCrystsp = np.array([psd.mCryst for psd in mdl.psd_full] )
        delta_mass_liq_solute = (Csp[-1] - Csp[0])*mdl.m_slv
        plt.figure()
        plt.plot(x0, N0, '.-', label='ini')
        plt.plot(x, N, '.-', label='end')
        plt.figure()
        plt.plot(tspan, Csp, '-k')
        plt.plot(tspan, Csatsp, ':k')
        plt.title('Concentration')
        plt.figure()
        plt.plot(tspan, mCrystsp, '-k')
        plt.title('Mass of Crystals')

        print('C final = {}'.format(mdl.ind.get_mdl(y)[0]))
        print('mass cryst ini = {}'.format(mCIni))
        print('mass cryst end = {}'.format(mCEnd))
        print('delta mass cryst = {}'.format(mCEnd - mCIni))
        print('delta mass liquid = {}'.format(delta_mass_liq_solute))
        # plt.figure()
        # plt.plot([tspan[0], tspan[-1]], [y0[-1]], '.-', label='ini')
        # plt.plot(x, N, '.-', label='end')
        # plt.legend()
    print('Simulation done.')

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()




