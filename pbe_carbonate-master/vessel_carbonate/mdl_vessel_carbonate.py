import os
import sys
import warnings
import time
import autoinit

import numba
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
from thermo import Chemical

from py_pbe_msm import pbe_msm_solver, pbe_rhs_funcs, pbe_utils
# import calciumcarbonate_supersaturation_module as carbonate_eq
# import calciumcarbonate_equilibrium as carbonate_eq #LAST
import calciumcarbonate_thermodynamics_mod as carbonate_eq #MOD
# import data_access_object
import h5py_utils
import h5py_manager

# pylint: disable=E1101

# np.seterr(under='raise')
TO_PLOT = True
pbe_utils.USE_NJIT = True

def initial_condition_square(Npts, V):
    """
    See VesselCarbonateDeposition.ipynb for a0 definition
    """
    # lspan = np.linspace(1e-5, 2e-5, Npts + 1)
    # size is in cm3 -> independent coordinate as volume
    # lspan = np.linspace(((1e-3)*1e-4)**(3), (5*1e-4)**(3), Npts + 1)
    lspan = np.geomspace(((1e-3)*1e-4)**(3), (5*1e-4)**(3), Npts + 1)
    xspan = (lspan[1:] + lspan[0:-1]) * 1/2.0
    a = 250.0
    b = 300.0
    a0 = 0.0 #5.741626794258374e-12 # TO GIVE 5.0 grams (FIX this)
    # def n0_fnc(v, a, b):# = lambda v, a, b: 0 if v<a else (1/(b-a) if v<b else 0)
    #     if v < a or v > b:
    #         return 0.0
    #     else:
    #         return a0*(300.0 - v)*(v - 250.0) / V
    def n0_fnc(v,a,b):
        return 0.0# /V NOT VOLUMETRIC BASED ANYMORE (CHECK!!)
    N_t0 = np.empty(Npts)
    n_t0_num = np.empty_like(N_t0)
    for i in np.arange(0, Npts):
        quad_ret = \
        integrate.quad(n0_fnc, lspan[i], lspan[i+1], args=(a,b))
        y, _ = quad_ret
        N_t0[i] = y
        n_t0_num[i] = N_t0[i] / (lspan[i+1] - lspan[i])

    return xspan, N_t0, lspan



@numba.jitclass([
    ('M_C', numba.float64),
    ('M_Na', numba.float64),
    ('M_NaHCO3', numba.float64),
    ('M_CaCO3', numba.float64),
    ('M_CaCl2', numba.float64),
    ('M_Ca', numba.float64),
    ('M_Cl', numba.float64),
    ('rho_c', numba.float64)
])
class SystemPhysicoChemicalParameters():

    def __init__(self, M_C, M_Na, M_NaHCO3, M_CaCO3, M_CaCl2, M_Ca, M_Cl, rho_c):
        self.M_C = M_C
        self.M_Na = M_Na
        self.M_NaHCO3 = M_NaHCO3
        self.M_CaCO3 = M_CaCO3
        self.M_CaCl2 = M_CaCl2
        self.M_Ca = M_Ca
        self.M_Cl = M_Cl
        self.rho_c = rho_c
        pass

@numba.jitclass([
    ('Qin', numba.float64),
    ('C_NaHCO3_initial', numba.float64),
    ('C_CaCl_in', numba.float64),
    ('V_initial', numba.float64),
    ('T', numba.float64),
])
class OperationalParameters():

    def __init__(self):
        self.Qin = 1.75 # cm^3/min
        self.C_NaHCO3_initial = 1.2275e-3 #g/cm3
        self.C_CaCl_in = 0.7275e-3 #g/cm3
        self.V_initial = 600.0 #cm^3
        self.T = 25.0 #Celsius CHANGE TEMPERATURE WAS 20!
        return

def create_physicoChemicalParameters():
    return SystemPhysicoChemicalParameters(
         Chemical('C').MW, Chemical('Na').MW,
         Chemical('NaHCO3').MW, Chemical('CaCO3').MW,
         Chemical('CaCl2').MW, Chemical('Ca').MW,
         Chemical('Cl').MW, Chemical('CaCO3').rho * 1e-3,
    )

spec_output = [
    ('x', numba.float64[:]),
    ('N', numba.float64[:]),
    ('mCa', numba.float64),
    ('mC', numba.float64),
    ('mNa', numba.float64),
    ('mCl', numba.float64),
    ('V', numba.float64),
    ('S', numba.float64),
    ('B', numba.float64),
    ('Ksp', numba.float64),
    ('massCrystal', numba.float64),
    ('mCaCl2_added', numba.float64),
    ('IAP', numba.float64),
    ('pH', numba.float64),
    ('I', numba.float64),
    #('conduc', numba.float64),
    ('sigma', numba.float64),
]
@numba.jitclass(spec_output)
class SimulationOutput(): #metaclass=autoinit.AutoInit
    def __init__(self, x, N, mCa, mC, mNa, mCl,
            V, S, Ksp, massCrystal, mCaCl2_added,
            IAP, pH, I, sigma, B):
        self.x = x; self.N = N; self.mCa = mCa
        self.mC = mC
        self.mNa = mNa; self.mCl = mCl; self.V = V
        self.S = S; self.Ksp = Ksp; self.massCrystal = massCrystal
        self.mCaCl2_added = mCaCl2_added
        self.IAP = IAP; self.pH = pH; self.I = I
        self.sigma = sigma
        self.B = B
        pass

if os.getenv('NUMBA_DISABLE_JIT') == "1":
    SimulationOutput.class_type = pbe_msm_solver.FakeNb()
    SystemPhysicoChemicalParameters.class_type = pbe_msm_solver.FakeNb()

spec = [
    # Doubles
    ('Qin', numba.float64),
    ('C_NaHCO3_initial', numba.float64),
    ('C_CaCl_in', numba.float64),
    ('V_initial', numba.float64),
    ('T', numba.float64),
    ('rpm', numba.float64),
    ('final_addition_time', numba.float64),
    ('rho_w', numba.float64),
    ('kv', numba.float64),
    ('b', numba.float64),
    ('g', numba.float64),
    ('kb', numba.float64),
    ('kg', numba.float64),
    ('S', numba.float64),
    ('B', numba.float64),
    ('mu3', numba.float64),
    ('Ksp', numba.float64),
    ('IAP', numba.float64),
    ('mCa', numba.float64),
    ('mC', numba.float64),
    ('mNa', numba.float64),
    ('mCl', numba.float64),
    ('mCaCl_addded', numba.float64),
    ('V', numba.float64),
    ('mCl', numba.float64),
    ('pH', numba.float64),
    ('I', numba.float64),
    ('conduc', numba.float64),
    ('sigma', numba.float64),
    ('conc_vol_particles', numba.float64),
    ('kbsec', numba.float64),
    ('agg_regular_val', numba.float64),

    # Integers
    ('multiplier_bin_addition', numba.int64),

    # Arrays
    ('ymdl0', numba.float64[:]),
    ('x_guess', numba.float64[:]),
    #('G', numba.float64[:]),

    # List
    ('out_span', numba.types.List(SimulationOutput.class_type.instance_type)),


    # Classes
    ('pp', SystemPhysicoChemicalParameters.class_type.instance_type),
    ('ind', pbe_msm_solver.Indexes.class_type.instance_type),

    # Experimental
    ('t', numba.float64),

]
@numba.jitclass(spec)
class MyModel():

    def __init__(self, Npts, nts, pp, op, extras):

        # Operational Parameters:
        # self.Qin = 1.75 # cm^3/min
        # self.C_NaHCO3_initial = 1.2275e-3 #g/cm3
        # self.C_CaCl_in = 0.7275e-3 #g/cm3
        # self.V_initial = 600.0 #cm^3
        self.Qin = op.Qin
        self.C_NaHCO3_initial = op.C_NaHCO3_initial
        self.C_CaCl_in = op.C_CaCl_in
        self.V_initial = op.V_initial
        self.T = op.T

        # self.T = 25+273.15 # K
        self.rpm = 350 #rpm
        self.final_addition_time = 56.0 #min

        ## Physico-chemical Parameters
        self.pp = pp
        self.rho_w = 1.0 #- #FIX
        # self.pp.rho_c = 2.0 #- #FIX
        self.kv = 1.0 #- #FIX

        # Kinetics parameters (Better until now)
        self.b = 1.45
        self.g = 1.5
        self.kb = 12e4
        self.kbsec = 0.0
        self.kg = 0.1e-3 #0.05e-4 #2e-3 #1.0e-5

        # Variables
        self.S = 0.0
        self.mu3 = 0.0
        self.Ksp = -999.99
        self.IAP = -999.99

        # Initial States #Require fix for CO2 from air equilibrium at initial time!
        self.mCa = 0.0

        # self.mC = extras[0] + extras[2] #Wait until talk with Elvis to check how to integrate the equilibrium properly.
        self.mC = (extras[1]) * self.pp.M_C * (self.V_initial*1e-3)

        # self.mC = (self.pp.M_C/self.pp.M_NaHCO3) * self.C_NaHCO3_initial * self.V_initial
        self.mNa = (self.pp.M_Na/self.pp.M_NaHCO3) * self.C_NaHCO3_initial * self.V_initial
        self.mCl = 0.0
        self.V = self.V_initial
        self.mCaCl_addded = 0.0

        self.ymdl0 = np.array([
            self.mCa, self.mC, self.mNa, self.mCl, self.V,
            self.mCaCl_addded
            ])

        # For visualization
        # self.out_span = []# it will be create later on!
        # self.equilibrium = carbonate_eq.CalciumCarbonateReaction()
        self.x_guess = np.array([]) #np.ones(13) * -0.1 #carbonate_eq.x_guess

        ## Indices
        self.ind = self.set_indexes_case(Npts, nts)

        self.multiplier_bin_addition = 1

        self.agg_regular_val = -1.0 #FIX
        pass

    def set_indexes_case(self, Npts, nt):
        ind = pbe_msm_solver.Indexes([Npts], [nt - 1], [0], len(self.ymdl0), 1)
        return ind

    def calc_G_size_based(self):
        if self.S < 1.0:
            return 0.0
        # return self.kg*(self.S - 1.0)**self.g

        # Verdoes 92
        # return (2.4e-12)*(self.S - 1.0)**(1.8) * 1e2 * 60.0 # to cm/min

        # # Verdoes 92 - Modified *1e2
        # return (2.4e-12*1e2)*(self.S - 1.0)**(1.8) * 1e2 * 60.0 # to cm/min

        # Reis(2018) and Brečević, (2007)
        I = self.I
        logkb = -0.275 + 0.228*(np.sqrt(I)/(1.0+np.sqrt(I)) - 0.3*I)
        kb = 10**(logkb)
        G = kb * (self.S - 1)**2
        return G * 1e-7 * 60 #nm/s -> cm/min

    def calc_G(self, x):
        G = 3.0 * x**(2/3) * self.calc_G_size_based()
        return G

    def calc_B(self):
        if self.S < 1.0:
            return 0.0
        # return self.kb*(self.S - 1.0)**self.b #* self.mu3 #self.mu3*
        # if self.S > 1.2:
        #     dd = 1
        # Bprim = self.kb*(self.S - 1.0)**self.b
        # Bsec = self.kbsec*(self.S - 1.0)**self.b * self.conc_vol_particles
        #B = Bprim + Bsec
        # B = Bprim

        # # Reis (2018)
        # D = 8.67e-10
        # eps = 7.62e-10
        # beta = 16.76
        # sigma = 0.0068 #N/m
        # vol_molar = 6.132e-29 #m^3
        # kboltzmann = 1.38064852E-23 #m2 kg s-2 K-1

        # A = D/eps**5/self.S**(5/3)
        # T_K = self.T + 273.15
        # B = A * np.exp(- (beta * sigma**3 * vol_molar**2) / (2.3*kboltzmann**3*T_K**3*(np.log(self.S))**2))
        # B *= 60.0 #*1e-12 #to min

        # # Verdoes 92
        Eb = 12.8
        Ks = 1.4e18
        S = self.S
        B = Ks * S * np.exp(-Eb/(np.log(S))**2)
        B *= (self.V*1e-6) * 60.0 #to #/min

        return B

    def calc_Aggr(self, x1, x2, N1, N2): #Modify to pass N also?
        return -1
        # # if self.t > 140.0: return 1e-6
        bound_aggr_up = (500.0)**3*1e-12 #1e-7
        bound_aggr_low = -1 #(1.00)**3*1e-12
        if x1 > bound_aggr_up or x2 > bound_aggr_up:
            return 0.0
        elif x1 < bound_aggr_low or x2 < bound_aggr_low:
            return 0.0
        if N1 < 1e-20 or N2 < 1e-20:
            return 0.0
        # return self.agg_regular_val
        return 1e-11
            #  return 1e-7 #1e-5 #1e6

    # def calc_sat(self):
    #     return 6.29e-2 + 2.46e-3*self.T - 7.14e-6*self.T**2

    def equilibrium_calcite_constant(self, T):
        T_K = T + 273.15
        K_sp = -171.906 - 0.0779*T_K + 2839.319/T_K + 71.595*np.log10(T_K)
        return K_sp

    def calc_molalities(self, m_slv):
        """
        Remove: Molalities are not used
        """
        cCa = self.mCa / m_slv
        cC = self.mC / m_slv
        cNa = self.mNa / m_slv
        cCl = self.mCl / m_slv
        return cCa, cC, cNa, cCl

    def calc_molar_concentations(self):
        """
        From mass [g] to Molar Concentration [mol/L]
        """
        cCa = (self.mCa/ self.pp.M_Ca) / (self.V*1e-3)
        cC = (self.mC/ self.pp.M_C) / (self.V*1e-3)
        cNa = (self.mNa/ self.pp.M_Na) / (self.V*1e-3)
        cCl = (self.mCl/ self.pp.M_Cl) / (self.V*1e-3)
        return cCa, cC, cNa, cCl


    def calc_mdl_rhs_dummy(self, t, y):
        rhs = np.zeros_like(y)
        return rhs

    def calc_mdl_rhs(self, t, y):
        rhs = np.zeros_like(y)
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        rhs_x = self.ind.get_x(rhs, 0)
        rhs_N = self.ind.get_N(rhs, 0)
        rhs_mdl = self.ind.get_mdl(rhs)

        if np.any(x < 0.0):
            print('negative x ???!?')
            dbug = 1
        self.t = t

        self.mCa = self.ind.get_mdl(y)[0]
        self.mC = self.ind.get_mdl(y)[1]
        self.mNa = self.ind.get_mdl(y)[2]
        self.mCl = self.ind.get_mdl(y)[3]
        self.V = self.ind.get_mdl(y)[4]
        self.mCaCl_addded = self.ind.get_mdl(y)[5]

        ## Auxiliaries
        # rho_mix = self.rho_w
        # m_slv = rho_mix * self.V
        # cCa = self.mCa / m_slv
        # cC = self.mC / m_slv
        # cNa = self.mNa / m_slv
        # cCl = self.mCl / m_slv
        cCa, cC, cNa, cCl = self.calc_molar_concentations()
        C_Ca_in = self.pp.M_Ca/self.pp.M_CaCl2 * self.C_CaCl_in
        C_Cl_in = 2*self.pp.M_Cl/self.pp.M_CaCl2 * self.C_CaCl_in

        self.conc_vol_particles = np.sum(x*N)

        # self.S = 1.0 #From Elvis Module
        # self.equilibrium.solve(cC, cCa, cNa, cCl) #cCT,cCaT,cNaT,cClT
        # if self.S - 1 > 0.0:
        #     x_guess = self.x_sol_eq
        # else:
        # x_guess = np.ones(13) * -0.1
        # x_sol_eq, pH, sigma, Ksp, S, IAP, I, cond_ideal, conduc = carbonate_eq.solve(
            # cC, cCa, cNa, cCl, self.x_guess
        # )
        x_sol_eq, pH, sigma, Ksp, S, IAP, I, cond_ideal, conduc, dic_eq = \
            carbonate_eq.equilibrium_closed(cC, cCa, cNa, cCl, self.T + 273.15, self.x_guess)
        ## self.S = (self.equilibrium.S)
        ## self.S = ((self.equilibrium.S)**2)**(1/2)
        # self.KspElvis = self.equilibrium.Ksp
        # self.IAP = self.equilibrium.IAP
        self.x_guess = x_sol_eq
        self.pH = pH
        self.sigma = conduc # New calculation for conductivity
        self.I = I
        self.Ksp = 10**(self.equilibrium_calcite_constant(self.T))
        self.IAP = IAP
        self.S = (self.IAP / self.Ksp)**0.5 #MODIFIED HERE: Verdoes (Growth rate) also use the sqrt root
        if self.S - 1 > 0.0:
            a = 1
        G = self.calc_G(x)
        self.B = self.calc_B()

        if self.S > 1.5: #just to get values for testing equilibrium
            db = 1

        int_Gn_aux = np.sum(G * N)

        cryst_rate = -self.pp.rho_c*self.kv*int_Gn_aux

        # if t > 56.0:
        #     self.Qin = 0.0

        ratio_Ca_crystal = self.pp.M_Ca / self.pp.M_CaCO3
        ratio_C_crystal = self.pp.M_C / self.pp.M_CaCO3

        dmCadt = self.Qin*C_Ca_in + ratio_Ca_crystal*cryst_rate
        dmCdt = ratio_C_crystal*cryst_rate
        dmNadt = 0.0
        dmCldt = self.Qin*C_Cl_in
        dVdt = self.Qin
        dmCaCl_adddeddt = self.Qin*self.C_CaCl_in

        # pbe_rhs_funcs.rhs_pbe_numbers_aggr_base(x, N, rhs_N, self)
        pbe_rhs_funcs.rhs_pbe_numbers_agg_inout(x, N, rhs_N, self.B, 0.0, self)
        # pbe_rhs_funcs.numbers_nucl(x, N, rhs_N, B)
        # rhs_N[0] += B
        pbe_rhs_funcs.grid_G_first_half(x, rhs_x, G)
        rhs_mdl[0] = dmCadt
        rhs_mdl[1] = dmCdt
        rhs_mdl[2] = dmNadt
        rhs_mdl[3] = dmCldt
        rhs_mdl[4] = dVdt
        rhs_mdl[5] = dmCaCl_adddeddt
        return rhs

    def calc_mdl_rhs_wrapper_sundials(self, t, y, ydot):
        ydot[:] = self.calc_mdl_rhs(t, y)
        pass

    def get_intermediaries(self, y):
        x = self.ind.get_x(y, 0)
        N = self.ind.get_N(y, 0)
        massCrystal = np.sum(x * N) * self.kv * self.pp.rho_c #* self.V NOT Volumetric Based!
        mCaCl2_added = self.mCaCl_addded
        # return [(
        #     x, N, self.mCa, self.mC, self.mNa, self.mCl, self.V,
        #     self.S, self.Ksp, massCrystal, mCaCl2_added, self.IAP
        # )]
        return SimulationOutput(
            x, N, self.mCa, self.mC, self.mNa, self.mCl, self.V,
            self.S, self.Ksp, massCrystal, mCaCl2_added, self.IAP,
            self.pH, self.I, self.sigma, self.B
        )

    def before_simulation(self, y, tspan, i_t):
        if i_t % self.multiplier_bin_addition == 0 and self.S > 0.9:
            self.ind.increment_additions(1, 0, 0)
        if i_t == 0:
            self.calc_mdl_rhs(0.0, y) #to initialize intermediaries values
            self.out_span = [self.get_intermediaries(y)]
        # if np.isclose(tspan[i_t], 56.0):
        if tspan[i_t] > 56.0: #TODO: ADJUST TO LINEAR FILLIGIN EQUATION
            self.Qin = 0.0

    def after_simulation(self, y, tspan, i_t):
        self.out_span += [self.get_intermediaries(y)]
        # print('iteration = {}'.format(i_t), end='\n', flush=True)
        # sys.stdout.flush()
        # print('iteration = ' + (i_t))
        print(i_t, tspan[i_t])
        pass



# def set_physico_chemical_parameters_from_db(mdl: MyModel):
#     mdl.M_C = Chemical('C').MW
#     mdl.M_Na = Chemical('Na').MW
#     mdl.M_NaHCO3 = Chemical('NaHCO3').MW
#     mdl.M_CaCO3 = Chemical('CaCO3').MW
#     mdl.M_CaCl2 = Chemical('CaCl2').MW
#     mdl.M_Ca = Chemical('Ca').MW
#     mdl.M_Cl = Chemical('Cl').MW

def calculate_maximum_conditions(mdl: MyModel):
    mCaCl2_added = mdl.Qin * mdl.C_CaCl_in * mdl.final_addition_time
    nCaCl2 = mCaCl2_added / mdl.pp.M_CaCl2
    mCaCO3 = nCaCl2 * mdl.pp.M_CaCO3
    mCl_added = (2*mdl.pp.M_Cl / mdl.pp.M_CaCl2) * mCaCl2_added
    mCa_added = (mdl.pp.M_Ca / mdl.pp.M_CaCl2) * mCaCl2_added
    mC = (mdl.pp.M_C / mdl.pp.M_NaHCO3) * mdl.C_NaHCO3_initial * mdl.V_initial
    mNa = (mdl.pp.M_Na / mdl.pp.M_NaHCO3) * mdl.C_NaHCO3_initial * mdl.V_initial
    Vfinal = mdl.Qin * mdl.final_addition_time + mdl.V_initial
    return {
        'CaCO3-max': mCaCO3,
        'Cl-max': mCl_added,
        'Ca-max': mCa_added,
        'C-max': mC,
        'Na-max': mNa,
        'mass-CaCl2-added': mCaCl2_added,
        'Vfinal': Vfinal,
    }


def main():
    h5_fname = 'outtest.h5'
    simulation_note = '''
Simulation for PSD with Aggregation, Nucleation and Growth.
Goals:
- [] Achieve the higher nucleation around 25min
- [] Fast dropping in Number due to strong aggregation
- [] Around 150min, number get small and nucleation starts to outpass aggregation
'''
    nt = 400
    Npts = 101
    pp = create_physicoChemicalParameters()
    op = OperationalParameters()

    # Initial Solution Equilibrium:
        # Adding CO2 equilibrium at initial time with NaHCO3
    cNaHCO3_initial_molar = (op.C_NaHCO3_initial*1e3) / pp.M_NaHCO3
    # dIC_NaHCO3, pH_NaHCO3, saltIC = carbonate_eq.solve_NaHCO3_equilibrium(cNaHCO3_initial_molar, op.T + 273.15)
    pH_NaHCO3, dIC_NaHCO3, I_NaHCO3, _ = carbonate_eq.calc_NaHCO3_equilibrium_open(
        cNaHCO3_initial_molar, op.T + 273.15
    )
    extras = (pH_NaHCO3, dIC_NaHCO3)

    mdl = MyModel(Npts, nt, pp, op, extras)
    # set_physico_chemical_parameters_from_db(mdl)
    maxConditions = calculate_maximum_conditions(mdl)
    x0, N0, l0 = initial_condition_square(Npts, mdl.V)
    ymdl0 = mdl.ymdl0
    y0 = pbe_msm_solver.set_initial_states([(x0, N0)], ymdl0, mdl.ind, lmin=l0[0])
    # n_add_points = np.floor(nt/2)
    n_add_points = 400
    # tspan = np.hstack((np.linspace(0.0, 56.0, n_add_points), np.linspace(56.01, 280.0, nt - n_add_points) ))
    # tspan = np.hstack((np.linspace(0.0, 56.0, n_add_points), np.linspace(56.1, 280.0, nt - n_add_points) ))
    tspan = np.linspace(0.0, 60.0, nt)
    # tspan = np.hstack((np.linspace(0.0, 100.0, n_add_points), np.linspace(100.1, 280.0, nt - n_add_points) ))
    # tspan = np.ascontiguousarray(tspan)

    integration_string = 'sundials'
    # integration_string = 'rkgill'
    integration_step = pbe_utils.integrate_rkgill_numba_mdl \
        if integration_string == 'rkgill' else pbe_utils.step_sundials

    integration_func = pbe_msm_solver.create_integrate_nucl_class(
        # # pbe_utils.step_sundials
        # pbe_utils.integrate_rkgill_numba_mdl,
        integration_step
    )

    # c = 0
    # while c < 1:
    #     mdl = MyModel(Npts, nt)
    time_started_sim = time.time()
    y = integration_func(tspan, y0, mdl)
    y = y[0, :]

    # data_access_object.insert_simulation_run(
    #     tspan, mdl, integration_string, simulation_note
    # )
    h5py_manager.save_data_to_h5_file(
        h5_fname, simulation_note, mdl, tspan, integration_string
    )
        # c += 1
#  x, N, mCa, mC, mNa, mCl,
#             V, S, Ksp, massCrystal, mCaCl2_added,
#             IAP, pH, I, sigma
    if TO_PLOT:
        x = mdl.ind.get_x(y, 0)
        N = mdl.ind.get_N(y, 0)
        out_span = mdl.out_span
        x_span = [out.x for out in out_span]
        N_span = [out.N for out in out_span]

        m_Ca = np.array([out.mCa for out in out_span])
        m_C = np.array([out.mC for out in out_span])
        m_Na = np.array([out.mNa for out in out_span])
        m_Cl = np.array([out.mCl for out in out_span])
        V = np.array([out.V for out in out_span])
        S = np.array([out.S for out in out_span])
        Ksp = np.array([out.Ksp for out in out_span])
        massCryst = np.array([out.massCrystal for out in out_span])
        mu0 = np.array([np.sum(out.N) for out in out_span])
        mu1 = np.array([np.sum(out.x*out.N) for out in out_span])
        mCaCl2_added = np.array([out.mCaCl2_added for out in out_span])
        IAP = np.array([out.IAP for out in out_span])
        pH = np.array([out.pH for out in out_span])
        I = np.array([out.I for out in out_span])
        sigma = np.array([out.sigma for out in out_span])
        #G = np.array([out.G for out in out_span])
        B = np.array([out.B for out in out_span])
        mCaCO3_if_preciptated = mCaCl2_added/mdl.pp.M_CaCl2 * mdl.pp.M_CaCO3
        VolCrystmean = mu1 / mu0

        x_length_span = [(out.x)**(1/3) for out in out_span]
        mu1_length = np.array([np.sum(x_length*out.N) for out, x_length in zip(out_span, x_length_span)])
        mu2_length = np.array([np.sum(x_length**2*out.N) for out, x_length in zip(out_span, x_length_span)])
        mu3_length = np.array([np.sum(x_length**3*out.N) for out, x_length in zip(out_span, x_length_span)])
        Lmean = (mu1_length / mu0) * 1e4 #cmto mum

        # Lmean = (VolCrystmean)**(1/3) * 1e4
        numberOfParticles = mu0 #* V
        x_len_micrometro = x**(1/3) * 1e4 #cm to mum
        x0_micrometro3 = x0 * 1e12 #cm^3 to mum
        x_micrometro3 = x * 1e12 #cm^3 to mum

        # Measured range > 0.55um
        MIN_SIZE_MEASURED = 0.55 * 1e-4 #to cm
        mu0_len_measured = np.array([np.sum(out.N[x_length > MIN_SIZE_MEASURED]) for out, x_length in zip(out_span, x_length_span)])
        mu1_len_measured = np.array([np.sum(x_length[x_length > MIN_SIZE_MEASURED]    * out.N[x_length > MIN_SIZE_MEASURED]) for out, x_length in zip(out_span, x_length_span)])
        mu2_len_measured = np.array([np.sum(x_length[x_length > MIN_SIZE_MEASURED]**2 * out.N[x_length > MIN_SIZE_MEASURED]) for out, x_length in zip(out_span, x_length_span)])
        mu3_len_measured = np.array([np.sum(x_length[x_length > MIN_SIZE_MEASURED]**3 * out.N[x_length > MIN_SIZE_MEASURED]) for out, x_length in zip(out_span, x_length_span)])

        m_Carbone_tot = m_C + mdl.pp.M_C/mdl.pp.M_CaCO3 * massCryst
        m_Calcium_tot = m_Ca + mdl.pp.M_Ca/mdl.pp.M_CaCO3 * massCryst
        m_Ca_added = mdl.pp.M_Ca/mdl.pp.M_CaCl2 * mCaCl2_added

        plt.figure(figsize=(16,12))
        plt.subplot(2,3,1)
        plt.title('PSD - $N_i$')
        plt.plot(x0, N0, '.-', label='ini')
        plt.plot(x, N, '.-', label='end')
        plt.xlabel('x [cm^3]')

        plt.subplot(2,3,2)
        plt.title('PSD - $n_i$')
        l = np.hstack((l0[0], (x[1:] + x[0:-1])*0.5, x[-1] + (x[-1]-x[-2])*0.5 ))
        n = N / (l[1:] - l[0:-1])
        plt.plot(x, n, '.-', label='end')
        plt.xlabel('x [cm^3]')

        plt.subplot(2,3,3)
        plt.title('$B_0(t)$')
        plt.plot(tspan, B, '.-', label='B')
        plt.xlabel('time [min]')

        plt.subplot(2,3,4)
        plt.title('Mean size')
        plt.plot(tspan, Lmean, '.-', label='size')
        plt.xlabel('time [min]')
        plt.ylabel('linear size [$\mu$m]')
        ax2 = plt.gca().twinx()
        ax2.plot(tspan, VolCrystmean*1e12, 'v-', label='vol')
        plt.ylabel('volumetric size [$\mu m^3$]')
        plt.legend()
        plt.subplot(2,3,5)

        plt.title('Number of Particles')
        plt.plot(tspan, numberOfParticles, '.-')
        plt.xlabel('time [min]')
        plt.subplot(2,3,6)
        plt.title('PSD - $N_i$ - Length based')
        plt.plot(x_len_micrometro, N, '.-', label='end')
        plt.xlabel('length [$\mu$m]')

        print('C final = {}'.format(mdl.ind.get_mdl(y)[0]))

        plt.figure()
        plt.title('Crystal mass')
        plt.xlabel('t [min]')
        plt.ylabel('$m_{crystal}$ [g]')
        plt.plot(tspan, massCryst, '.-')
        plt.plot([tspan[0], tspan[-1]], np.ones(2)*maxConditions['CaCO3-max'], '--', label='CaCO3-max')
        plt.plot(tspan, mCaCl2_added, ':', label='CaCl2 added')
        plt.plot(tspan, mCaCO3_if_preciptated, ':', label='CaCO3 max sim')
        plt.legend()

        plt.figure()
        plt.title('Species masses')
        plt.ylabel('m [g]')
        plt.xlabel('t [min]')
        plt.plot(tspan, m_Ca, '.-',  label='Ca')
        plt.plot(tspan, m_C, '.-',  label='C')
        # plt.plot(tspan, m_Na, label='Na')
        plt.plot(tspan, m_Cl, '.-',  label='Cl')
        plt.plot([tspan[0], tspan[-1]], np.ones(2)*maxConditions['Cl-max'], '--', label='Cl-max')
        plt.legend()

        plt.figure()
        plt.title('Vessel volume')
        plt.xlabel('t [min]')
        plt.ylabel('$V(t)$ [$cm^3$]')
        plt.plot(tspan, V, '.-')

        plt.figure()
        plt.suptitle('Check consistency')
        plt.subplot(3,1,1)
        plt.plot(tspan, m_Carbone_tot, '.-')
        plt.xlabel('t [min]')
        plt.ylabel('m_C global [g]')
        plt.subplot(3,1,2)
        plt.plot(tspan, m_Carbone_tot/m_Carbone_tot[0], '.-')
        plt.xlabel('t [min]')
        plt.ylabel('m_C global ratio')
        plt.subplot(3,1,3)
        plt.plot(tspan, m_Calcium_tot - m_Ca_added, '.-')
        plt.xlabel('t [min]')
        plt.ylabel('m_Ca global [g] - $m_{Ca}^{added}$')

        plt.figure(figsize=(8,12))
        plt.title('Supersaturation and IAP vs Ksp')
        plt.subplot(2,1,1)
        plt.ylabel('S')
        plt.xlabel('t [min]')
        plt.plot(tspan, S, '.-', label='S')
        plt.subplot(2,1,2)
        plt.plot(tspan, Ksp, '.-', label='$K_{sp}$')
        plt.plot(tspan, IAP, '.-', label='$IAP$')
        plt.legend()
        plt.xlabel('t [min]')
        plt.ylabel('IAP and Ksp [-]')

        plt.figure(figsize=(8,12))
        plt.subplot(3,1,1)
        plt.ylabel('pH')
        plt.xlabel('t [min]')
        plt.plot(tspan, pH, '.-', label='pH')
        plt.subplot(3,1,2)
        plt.ylabel('I')
        plt.xlabel('t [min]')
        plt.plot(tspan, I, '.-', label='I')
        plt.subplot(3,1,3)
        plt.ylabel('sigma')
        plt.xlabel('t [min]')
        plt.plot(tspan, sigma, '.-', label='sigma')

        plt.figure(figsize=(8,14))
        plt.suptitle('Moments - size as volume')
        plt.subplot(3,1,1)
        plt.title('$\mu_0(t)$')
        plt.plot(tspan, mu0, '.-', label='mu0')
        plt.xlabel('time [min]')
        plt.subplot(3,1,2)
        plt.title('$\mu_1(t)$')
        plt.plot(tspan, mu1, '.-', label='mu1')
        plt.xlabel('time [min]')

        plt.figure(figsize=(8,14))
        plt.suptitle('Moments - size as length')
        plt.subplot(3,1,1)
        plt.title('$\mu_1(t)$ (LENGTH-BASED)')
        plt.plot(tspan, mu1_length, '.-', label='mu1-length')
        plt.xlabel('time [min]')
        plt.subplot(3,1,2)
        plt.title('$\mu_2(t)$ (LENGTH-BASED)')
        plt.plot(tspan, mu2_length, '.-', label='mu2-length')
        plt.xlabel('time [min]')
        plt.subplot(3,1,3)
        plt.title('$\mu_3(t)$ (LENGTH-BASED)')
        plt.plot(tspan, mu3_length, '.-', label='mu3-length')
        plt.xlabel('time [min]')

        plt.figure(figsize=(8,14))
        plt.suptitle('Aspect of turbidity')
        plt.subplot(3,1,1)
        plt.title('$\mu_2(t)$ / V (LENGTH-BASED)')
        plt.plot(tspan, mu2_length/V, '.-', label='mu2-length')

        plt.figure(figsize=(12,14))
        plt.suptitle('Measured Moments (Length based)')
        plt.subplot(2,2,1)
        plt.title('$\mu_0(t)$')
        plt.plot(tspan, mu0_len_measured, '.-', label='mu0-length')
        plt.xlabel('time [min]')
        plt.subplot(2,2,2)
        plt.title('$\mu_1(t)$')
        plt.plot(tspan, mu1_len_measured, '.-', label='mu1-length')
        plt.xlabel('time [min]')
        plt.subplot(2,2,3)
        plt.title('$\mu_2(t)$')
        plt.plot(tspan, mu2_len_measured, '.-', label='mu2-length')
        plt.xlabel('time [min]')
        plt.subplot(2,2,4)
        plt.title('$\mu_3(t)$')
        plt.plot(tspan, mu3_len_measured, '.-', label='mu3-length')
        plt.xlabel('time [min]')

        from mpl_toolkits.mplot3d import Axes3D # pylint: disable=W0612
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        lst_indx = list(range(0,len(tspan),100))
        lst_indx = lst_indx if lst_indx[-1] == len(tspan)-1 else lst_indx + [len(tspan)-1]
        for i_t in lst_indx:
            ax.scatter(x_span[i_t], np.ones(len(x_span[i_t]))*tspan[i_t], N_span[i_t])
        ax.set_zlim(0.0, 5.0)
        # ax.set_xlim(0.0, 10e-6)
        # plt.figure()
        # plt.plot([tspan[0], tspan[-1]], [y0[-1]], '.-', label='ini')
        # plt.plot(x, N, '.-', label='end')
        # plt.legend()
    print('Simulation done:')
    print('\tFinal time = {}'.format(tspan[-1]))
    print('\tCPU time = {}'.format(time.time() - time_started_sim))

if __name__ == '__main__':
    main()

    if TO_PLOT:
        plt.show()
