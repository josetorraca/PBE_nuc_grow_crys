import numpy as np
import numba
from scipy import optimize
import utils_for_numba

"""
Author: Caio Curitiba Marcellos
Date: 15/01/2019
Modified from calciumcarbonate_supersaturation_module.py:
    Author: Elvis
    Title: Module for Calcium Carbonate SUpersaturation Calculation

## Conductivity:

- Table of values: http://www.aqion.de/site/194
- http://www.aqion.de/site/77#Nernst-Einstein (Used reference)
- https://www.hydrochemistry.eu/exmpls/sc.html
- Handbook of Chemistry and Physic pag. 940


Goal: Adjust the calculation for compilation with the nukmba package
"""

logK_H = -1.464
logK_a1 = -6.363
logK_a2 = -10.329
logK_w = -13.997
logK_sp = -8.48
logK_CaH = 1.26
logK_CaC = 3.15
logK_CaOH = 1.3
logK_NaOH = -14.18
logK_NaCO3minus = 1.27
logK_NaHCO3 = -0.25
                                                                  #caoh(unk)
conductivity_molar_zero = np.array([349.6, 197.9, 0.0, 143.5, 44.3, 119.1, 19.0, 19.0, 0.0, 50.0, 0.0, 22.0, 0.0, 76.2])
charge_species = np.array([1, 1, 0, 2, 1, 2, 1, 1, 0, 1, 0, 1, 0, 1], dtype=int)

IDX_Hp      =0
IDX_OHm     =1
IDX_CO2     =2
IDX_CO3mm   =3
IDX_HCO3m   =4
IDX_Capp    =5
IDX_CaOHp   =6
IDX_CaHCO3p =7
IDX_CaCO3aq =8
IDX_Nap     =9
IDX_NaOH    =10
IDX_NaCO3m  =11
IDX_NaHCO3  =12
IDX_Clm     =13

"""
OBS:
x: vector of solution
    size = 13
    Hp, OHm, CO2, CO3mm, HCO3m, Capp, CaOHp, CaHCO3p, CaCO3aq, Nap, NaOH, NaCO3m, NaHCO3
    Thus following the IDX variables up to IDX_NaHCO3
c: vector of logarithmic molalities (CHECK)
    size = 14
    Hp, OHm, CO2, CO3mm, HCO3m, Capp, CaOHp, CaHCO3p, CaCO3aq, Nap, NaOH, NaCO3m, NaHCO3, Clm
    Thus following the IDX variables up to the last one: IDX_Clm
"""

x_guess = np.ones(13) * -0.1

@numba.njit()
def calculate_loggamma(m):
    b = 0.1; A = 0.5
    loggamma_0 = b*m
    loggamma_1Plus = loggamma_1Minus = -A*(1)*(np.sqrt(m)/(1+np.sqrt(m))-0.2*m)
    loggamma_2Plus = loggamma_2Minus = 4*loggamma_1Plus
    return loggamma_0, loggamma_1Plus, loggamma_1Minus, loggamma_2Plus, loggamma_2Minus

@numba.njit()
def ionic_strength(c):
    return 0.5*(4*np.power(10,c[IDX_Capp])+np.power(10,c[IDX_CaHCO3p])\
            +np.power(10,c[IDX_CaOHp])+np.power(10,c[IDX_Hp])\
            +4*np.power(10,c[IDX_CO3mm])+np.power(10,c[IDX_HCO3m])\
            +np.power(10,c[IDX_OHm]) + np.power(10,c[IDX_Nap]) \
            + np.power(10,c[IDX_Clm]) + np.power(10,c[IDX_NaCO3m]) )

@numba.njit()
def sodiumBicarbonateMixture_residual_function(x, args):

    cCT, cCaT, cNaT, cClT = args
    c = np.empty(14)
    c[0:-1] = x
    c[-1] = np.log10(cClT)

    m = ionic_strength(c)
    loggamma_0, loggamma_1Plus, loggamma_1Minus, loggamma_2Plus, loggamma_2Minus \
        = calculate_loggamma(m)
    reactions = np.empty(13)
    reactions[0] = c[IDX_HCO3m] + c[IDX_Hp] - (logK_a1 + loggamma_0 - loggamma_1Plus - loggamma_1Minus) - c[IDX_CO2]
    reactions[1] = c[IDX_CO3mm] + c[IDX_Hp] - (logK_a2 - loggamma_2Minus) - c[IDX_HCO3m]
    reactions[2] = c[IDX_OHm] + c[IDX_Hp] - (logK_w + loggamma_0 - loggamma_1Plus - loggamma_1Minus)

    # Calcium reactions
    reactions[3] = c[IDX_CaHCO3p] - (logK_CaH + loggamma_2Plus) - c[IDX_Capp] - c[IDX_HCO3m]
    reactions[4] = c[IDX_CaCO3aq] - (logK_CaC - loggamma_0 + loggamma_2Plus + loggamma_2Minus) - c[IDX_Capp] - c[IDX_CO3mm]
    reactions[5] = c[IDX_CaOHp] - (logK_CaOH + loggamma_2Plus) - c[IDX_Capp] - c[IDX_OHm]

    # Charge conservation
    reactions[6] = np.power(10,c[IDX_Hp]) + 2*np.power(10,c[IDX_Capp]) \
        + np.power(10,c[IDX_CaHCO3p]) + np.power(10,c[IDX_CaOHp]) - np.power(10,c[IDX_HCO3m]) \
                - 2*np.power(10,c[IDX_CO3mm]) - np.power(10,c[IDX_OHm]) + np.power(10,c[IDX_Nap]) \
                - np.power(10,c[IDX_Clm]) - np.power(10,c[IDX_NaCO3m])

    # Total calcium concentration
    reactions[7] = cCaT - np.power(10,c[IDX_Capp]) - np.power(10,c[IDX_CaHCO3p]) \
        - np.power(10,c[IDX_CaCO3aq]) - np.power(10,c[IDX_CaOHp])

    # Total carbon concentration
    reactions[8] =  cCT - np.power(10,c[IDX_CO2]) - np.power(10,c[IDX_CO3mm]) \
        - np.power(10,c[IDX_HCO3m])- np.power(10,c[IDX_CaHCO3p])- np.power(10,c[IDX_CaCO3aq]) \
        - np.power(10,c[IDX_NaCO3m])- np.power(10,c[IDX_NaHCO3])

    # Sodium reactions
    reactions[9] =  c[IDX_Nap] + logK_NaOH - c[IDX_NaOH] - c[IDX_Hp]
    reactions[10] =  c[IDX_Nap] + c[IDX_CO3mm] + (logK_NaCO3minus + loggamma_2Minus) - c[IDX_NaCO3m]
    reactions[11] =  c[IDX_Nap] + c[IDX_HCO3m] + (logK_NaHCO3 - loggamma_0 + loggamma_1Plus + loggamma_1Minus) - c[IDX_NaHCO3]
    reactions[12] = cNaT - np.power(10,c[IDX_Nap]) - np.power(10,c[IDX_NaCO3m]) - \
        np.power(10,c[IDX_NaOH])- np.power(10,c[IDX_NaHCO3])
    return reactions

@numba.njit()
def numba_jacobian(x, args):
    return utils_for_numba.numeric_jacobian(
        sodiumBicarbonateMixture_residual_function,
        x, 1e-8, args
    )

@numba.njit()
def solution_conductivity_ideal(conc_vals):
    return np.sum(conductivity_molar_zero * conc_vals)

@numba.njit()
def solution_conductivity(I, gamma, conc_vals):
    ret = 0.0
    for i in range(len(conc_vals)):
        if charge_species[i] == 0:
            continue
        if I < 0.36*charge_species[i]:
            alpha = 0.6*np.sqrt(charge_species[i])
        else:
            alpha = np.sqrt(I) * charge_species[i]
        aux = conductivity_molar_zero[i] * gamma[i]**alpha * conc_vals[i]
        ret += aux
    return ret

@numba.njit()
def solve(cCT, cCaT, cNaT, cClT, x_guess):
    args = (cCT, cCaT, cNaT, cClT)
    # jacob = utils_for_numba.create_jacobian(sodiumBicarbonateMixture_residual_function)
    x, counter_iteractions = utils_for_numba.root_finding_newton(
        sodiumBicarbonateMixture_residual_function,
        calc_jacobian_carbonate_equilibrium,
        x_guess,
        1e-7,
        200,
        args
    )

    c = np.empty(14)
    c[0:-1] = x
    c[-1] = np.log10(cClT)
    m = ionic_strength(c)
    loggamma_0, loggamma_1Plus, loggamma_1Minus, loggamma_2Plus, loggamma_2Minus \
        = calculate_loggamma(m)
    g0 = loggamma_0
    g1 = loggamma_1Plus
    g2 = loggamma_2Plus
    pH = pH = -c[IDX_Hp]-loggamma_1Plus
    I = m
    sigma = 6.2e4*I
    coefB = np.power(10,loggamma_2Minus)
    coefA = np.power(10,loggamma_2Plus)
    Ksp = np.power(10,logK_sp)
    cBion = np.power(10,c[IDX_CO3mm])
    cAion = np.power(10,c[IDX_Capp])
    S = np.power(10,c[IDX_Capp]+loggamma_2Plus)*np.power(10,c[IDX_CO3mm]+loggamma_2Minus)/np.power(10.0,logK_sp)
    IAP = np.power(10,c[IDX_Capp]+loggamma_2Plus)*np.power(10,c[IDX_CO3mm]+loggamma_2Minus)

    concentration_values = np.power(10, c) / 1e3 #mol/cm^3
    conduc_ideal = solution_conductivity_ideal(concentration_values) * 1e6 #muS / cm
    gamma = 10**(np.array([g1, g1, g0, g2, g1, g2, g1, g1, g0, g1, g0, g1, g0, g1]))
    conduc = solution_conductivity(I, gamma, concentration_values) * 1e6 #muS / cm

    return x, pH, sigma, Ksp, S, IAP, I, conduc_ideal, conduc

#@numba.njit
def dieletricconstant_water(TK):
    # for TK: 273-372
    return (0.24921e3 - 0.79069*TK + 0.72997e-3*TK**2)

#@numba.njit
def density_water(TK):
    # for TK: 273-372
    return (0.183652 + 0.00724987*TK - 0.203449e-4*TK**2 + 1.73702e-8*TK**3)

logPCO2 = -3.455

#@numba.njit
def calculate_log10_equilibrium_constant(TK):

    # Calculate the log10 of the equilibrium constants
    # Nordstrom, D. K., Plummer, L. N., Langmuir, D., Busenberg, E., May, H. M., Jones, B. F., & Parkhurst, D. L. (1990). Revised Chemical Equilibrium Data for Major Water—Mineral Reactions and Their Limitations (pp. 398–413). https://doi.org/10.1021/bk-1990-0416.ch031
    logK = {'NaOH': -14.18,'NaCO3': 1.27, 'NaHCO3':-0.25, 'Na2CO3': 0.672, 'CaOH': -12.78, 'NaCl': -1.602, 'HCl': -6.100}
    logK['H'] = 108.3865 + 0.01985076*TK - 6919.53/TK - 40.45154*np.log10(TK) + 669365.0/(TK**2)
    logK['a1'] = -356.3094 - 0.06091964*TK + 21834.37/TK + 126.8339*np.log10(TK) -1684915/(TK**2)
    logK['a2'] = -107.8871 - 0.03252849*TK + 5151.79/TK + 38.92561*np.log10(TK) -563713.9/(TK**2)
    logK['w'] = -283.9710 - 0.05069842*TK + 13323.00/TK + 102.24447*np.log10(TK) -1119669/(TK**2)
    logK['CaHCO3+'] = 1209.120 + 0.31294*TK - 34765.05/TK - 478.782*np.log10(TK)
    logK['CaCO3'] = -1228.732 -0.29944*TK + 35512.75/TK + 485.818*np.log10(TK)
    # Partial pressure of CO2 in air
    logPCO2 = -3.455
    # calcite solubility product
    logK['calcite'] = -171.9065 - 0.077993*TK + 2839.319/TK + 71.595*np.log10(TK)
    logK['vaterite'] = -172.1295 - 0.077993*TK + 3074.688/TK + 71.595*np.log10(TK)
    logK['aragonite'] =-171.9773 - 0.077993*TK + 2903.293/TK + 71.595*np.log10(TK)
    return logK

# model for activity coefficient
def ion_activity(I,TK, c):
    loggamma = {'H+': 0.}
    #Truesdell, A. H., & Jones, B. F. (1974). WATEQ, A computer program for calculating chemical equilibria of natural waters. Jour. Research U.S. Geol. Survey, 2(2), 233–248.
    epsilon = dieletricconstant_water(TK)
    rho = density_water(TK)
    A = 1.82483e6*np.sqrt(rho)/np.power(epsilon*TK,1.5) # (L/mol)^1/2
    B = 50.2916*np.sqrt(rho/(epsilon*TK)) # Angstrom^-1 . (L/mol)^1/2

    loggamma['H+'] = -A*1*np.sqrt(I)/(1+B*9.0*np.sqrt(I))
    loggamma['OH-'] = -A*1*np.sqrt(I)/(1+B*3.5*np.sqrt(I))
    loggamma['Ca++'] = -A*4*np.sqrt(I)/(1+B*5.0*np.sqrt(I))+0.165*I
    loggamma['HCO3-'] = loggamma['NaCO3-'] = -A*1*np.sqrt(I)/(1+B*5.4*np.sqrt(I))
    loggamma['CaOH+'] = loggamma['CaHCO3+'] = -A*1*np.sqrt(I)/(1+B*6.0*np.sqrt(I))
    loggamma['CO3--'] = -A*4*np.sqrt(I)/(1+B*5.4*np.sqrt(I))
    loggamma['Na+'] = -A*1*np.sqrt(I)/(1+B*4.0*np.sqrt(I))+0.075*I
    loggamma['Cl-'] = -A*1*np.sqrt(I)/(1+B*3.5*np.sqrt(I))+0.015*I
    loggamma['CaCO3'] = loggamma['NaHCO3'] = loggamma['Na2CO3'] = loggamma['NaOH'] = loggamma['CO2'] = loggamma['NaCl'] = loggamma['HCl']= -0.5*I

    m = 0.0
    for key,val in c.items():
        m = m + np.power(10,val)
    if m < 1.0:
        loggamma['H2O'] = np.log10(1-0.017*m)
    else: loggamma['H2O'] = -0.5*I
    return loggamma

def NaHCO3_equilibrium(x,cNaHCO3, TK):
    c = {}
    c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]
    c['HCO3-'] = x[4]; c['Na+'] = x[5]; c['NaOH'] = x[6]; c['NaCO3-'] = x[7]
    c['NaHCO3'] = x[8]; c['Na2CO3'] = x[9]
    cNaT = cNaHCO3 #This came from sodium carbonate
    # ionic strength
    I = 0.5*(np.power(10,c['H+'])\
            +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
            +np.power(10,c['OH-']) + np.power(10,c['Na+']) + np.power(10,c['NaCO3-']) )
    loggamma = ion_activity(I,TK, c)
    logK = calculate_log10_equilibrium_constant(TK)
    # carbonate-CO2 equilibrium
    Reaction = [None]*10
    Reaction[0] = (c['CO2']+loggamma['CO2']) - logPCO2 - logK['H']
    Reaction[1] = (c['HCO3-']+loggamma['HCO3-']) + (c['H+']+loggamma['H+']) - logK['a1']- (c['CO2']+loggamma['CO2']) - loggamma['H2O']
    Reaction[2] = (c['CO3--']+loggamma['CO3--']) + (c['H+']+loggamma['H+']) - logK['a2'] - (c['HCO3-']+loggamma['HCO3-'])
    Reaction[3] = (c['OH-']+loggamma['OH-']) + (c['H+']+loggamma['H+']) - logK['w'] - loggamma['H2O']
    # Sodium reactions
    Reaction[4] = (c['Na+']+loggamma['Na+']) + loggamma['H2O'] + logK['NaOH'] - (c['H+']+loggamma['H+']) - (c['NaOH']+loggamma['NaOH'])
    Reaction[5] = (c['Na+']+loggamma['Na+']) + (c['CO3--']+loggamma['CO3--']) + logK['NaCO3'] - (c['NaCO3-']+loggamma['NaCO3-'])
    Reaction[6] = (c['Na+']+loggamma['Na+']) + (c['HCO3-']+loggamma['HCO3-']) + logK['NaHCO3'] - (c['NaHCO3']+loggamma['NaHCO3'])
    Reaction[7] = 2*(c['Na+']+loggamma['Na+']) + (c['CO3--']+loggamma['CO3--']) + logK['Na2CO3'] - (c['Na2CO3']+loggamma['Na2CO3'])
    Reaction[8] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- \
    np.power(10,c['NaOH'])- np.power(10,c['NaHCO3']) - np.power(10,c['Na2CO3'])
    # Charge Conservation
    Reaction[9] = np.power(10,c['H+']) + np.power(10,c['Na+']) - np.power(10,c['HCO3-'])\
    - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) - np.power(10,c['NaCO3-'])
    return Reaction

def solve_NaHCO3_equilibrium(cNaHCO3, TK):
    """NaHCO3 equilibrium """
    solNaHCO3 = optimize.root(NaHCO3_equilibrium, [-1.0] * 10, args=(cNaHCO3, TK), method='hybr')
    # sigmaNaHCO3 = 6.2e4*(0.5*(np.power(10,c['H+'])\
    #             +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
    #             +np.power(10,c['OH-']) + np.power(10,c['Na+']) + np.power(10,c['NaCO3-']) ))
    x = solNaHCO3.x
    c = {}
    # c = {'H+': solNaHCO3.x[0], 'OH-': x[1], 'CO2': x[2], 'CO3--': x[3], 'HCO3-': x[4], 'NaCO3-': x[7]}
    c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]
    c['HCO3-'] = x[4]; c['Na+'] = x[5]; c['NaOH'] = x[6]; c['NaCO3-'] = x[7]
    c['NaHCO3'] = x[8]; c['Na2CO3'] = x[9]
    c['Na+'] = x[5]
    I = 0.5*(np.power(10,c['H+'])\
            +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
            +np.power(10,c['OH-']) + np.power(10,c['Na+']) + np.power(10,c['NaCO3-']) )
    loggamma = ion_activity(I, TK, c)
    pHNaHCO3 = -c['H+']-loggamma['H+']

    conc = 10**x #molar concentration
    DICNaHCO3 = conc[2] + conc[3] + conc[4] + conc[7]
    saltIC = conc[8] + conc[9]

    DICNaHCO3 = np.power(10,c['CO2'])+np.power(10,c['CO3--'])+ np.power(10,c['HCO3-']) + np.power(10,c['NaCO3-'])
    return DICNaHCO3, pHNaHCO3, saltIC



@numba.njit()
def calc_jacobian_carbonate_equilibrium(x, args):
    c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12 = x
    # c13 = np.log10(args[-1])
    c13 = np.log10(args[3])
    # print(args[3])

    log10_cst = np.log(10)
    K1 = ((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9))
    K2 = np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9)
    J = [
    [
    (0.05*10**c0*log10_cst - 0.25*10**c0*log10_cst/K1 + 0.25*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.05*10**c1*log10_cst - 0.25*10**c1*log10_cst/K1 + 0.25*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-1),
    (0.2*10**c3*log10_cst - 1.0*10**c3*log10_cst/K1 + 1.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c4*log10_cst - 0.25*10**c4*log10_cst/K1 + 0.25*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.2*10**c5*log10_cst - 1.0*10**c5*log10_cst/K1 + 1.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c6*log10_cst - 0.25*10**c6*log10_cst/K1 + 0.25*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c7*log10_cst - 0.25*10**c7*log10_cst/K1 + 0.25*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.05*10**c9*log10_cst - 0.25*10**c9*log10_cst/K1 + 0.25*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.05*10**c11*log10_cst - 0.25*10**c11*log10_cst/K1 + 0.25*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    ],
    [
    (0.2*10**c0*log10_cst - 0.5*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.2*10**c1*log10_cst - 0.5*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.8*10**c3*log10_cst - 2.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 2.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.2*10**c4*log10_cst - 0.5*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (0.8*10**c5*log10_cst - 2.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 2.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.2*10**c6*log10_cst - 0.5*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.2*10**c7*log10_cst - 0.5*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.2*10**c9*log10_cst - 0.5*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.2*10**c11*log10_cst - 0.5*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    ],
    [
    (0.05*10**c0*log10_cst - 0.25*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.05*10**c1*log10_cst - 0.25*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0),
    (0.2*10**c3*log10_cst - 1.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 1.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c4*log10_cst - 0.25*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.2*10**c5*log10_cst - 1.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 1.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c6*log10_cst - 0.25*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c7*log10_cst - 0.25*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.05*10**c9*log10_cst - 0.25*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.05*10**c11*log10_cst - 0.25*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    ],
    [
    (-0.2*10**c0*log10_cst + 0.5*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.2*10**c1*log10_cst + 0.5*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (-0.8*10**c3*log10_cst + 2.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 2.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.2*10**c4*log10_cst + 0.5*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (-0.8*10**c5*log10_cst + 2.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 2.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (-0.2*10**c6*log10_cst + 0.5*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.2*10**c7*log10_cst + 0.5*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0),
    (-0.2*10**c9*log10_cst + 0.5*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (-0.2*10**c11*log10_cst + 0.5*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    ],
    [
    (-0.35*10**c0*log10_cst + 1.0*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.35*10**c1*log10_cst + 1.0*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (-1.4*10**c3*log10_cst + 4.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 4.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (-0.35*10**c4*log10_cst + 1.0*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-1.4*10**c5*log10_cst + 4.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 4.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (-0.35*10**c6*log10_cst + 1.0*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.35*10**c7*log10_cst + 1.0*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (1),
    (-0.35*10**c9*log10_cst + 1.0*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (-0.35*10**c11*log10_cst + 1.0*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 1.0*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    ],
    [
    (-0.2*10**c0*log10_cst + 0.5*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.2*10**c1*log10_cst + 0.5*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (0),
    (-0.8*10**c3*log10_cst + 2.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 2.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.2*10**c4*log10_cst + 0.5*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-0.8*10**c5*log10_cst + 2.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 2.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (-0.2*10**c6*log10_cst + 0.5*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (-0.2*10**c7*log10_cst + 0.5*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (-0.2*10**c9*log10_cst + 0.5*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (-0.2*10**c11*log10_cst + 0.5*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) - 0.5*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    ],
    [
    (10**c0*log10_cst),
    (-10**c1*log10_cst),
    (0),
    (-2*10**c3*log10_cst),
    (-10**c4*log10_cst),
    (2*10**c5*log10_cst),
    (10**c6*log10_cst),
    (10**c7*log10_cst),
    (0),
    (10**c9*log10_cst),
    (0),
    (-10**c11*log10_cst),
    (0),
    ],
    [
    (0),
    (0),
    (0),
    (0),
    (0),
    (-10**c5*log10_cst),
    (-10**c6*log10_cst),
    (-10**c7*log10_cst),
    (-10**c8*log10_cst),
    (0),
    (0),
    (0),
    (0),
    ],
    [
    (0),
    (0),
    (-10**c2*log10_cst),
    (-10**c3*log10_cst),
    (-10**c4*log10_cst),
    (0),
    (0),
    (-10**c7*log10_cst),
    (-10**c8*log10_cst),
    (0),
    (0),
    (-10**c11*log10_cst),
    (-10**c12*log10_cst),
    ],
    [
    (-1),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (1),
    (-1),
    (0.0),
    (0.0),
    ],
    [
    (0.2*10**c0*log10_cst - 0.5*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.2*10**c1*log10_cst - 0.5*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.8*10**c3*log10_cst - 2.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 2.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.2*10**c4*log10_cst - 0.5*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.8*10**c5*log10_cst - 2.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 2.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.2*10**c6*log10_cst - 0.5*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.2*10**c7*log10_cst - 0.5*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.2*10**c9*log10_cst - 0.5*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0),
    (0.2*10**c11*log10_cst - 0.5*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.5*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 - 1),
    (0),
    ],
    [
    (0.05*10**c0*log10_cst - 0.25*10**c0*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c0*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c1*log10_cst - 0.25*10**c1*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c1*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.2*10**c3*log10_cst - 1.0*10**c3*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 1.0*10**c3*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c4*log10_cst - 0.25*10**c4*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c4*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0.2*10**c5*log10_cst - 1.0*10**c5*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 1.0*10**c5*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c6*log10_cst - 0.25*10**c6*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c6*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0.05*10**c7*log10_cst - 0.25*10**c7*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c7*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (0),
    (0.05*10**c9*log10_cst - 0.25*10**c9*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c9*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2 + 1),
    (0),
    (0.05*10**c11*log10_cst - 0.25*10**c11*log10_cst/((np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)*K2) + 0.25*10**c11*log10_cst/(np.sqrt(0.5*10**c0 + 0.5*10**c1 + 0.5*10**c11 + 0.5*10**c13 + 2.0*10**c3 + 0.5*10**c4 + 2.0*10**c5 + 0.5*10**c6 + 0.5*10**c7 + 0.5*10**c9) + 1)**2),
    (-1),
    ],
    [
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (0.0),
    (-10**c9*log10_cst),
    (-10**c10*log10_cst),
    (-10**c11*log10_cst),
    (-10**c12*log10_cst),
    ],
    ]
    # J = np.ones((13,13))
    return np.array(J)



def main():

    tuple_cC_cCa_cNa_cCl =  (0.014148761431388462, 0.0002063963961295059, 0.0141497213599971, 0.00041471264947629455)
    # x = x_guess
    #x = solve(*tuple_cC_cCa_cNa_cCl, x_guess)

    # cClT = tuple_cC_cCa_cNa_cCl[-1]
    # c = np.hstack((x_guess, np.asarray(np.log10(cClT))))
    # J = calc_jacobian_carbonate_equilibrium(x_guess, np.array(tuple_cC_cCa_cNa_cCl))

    cNaHCO3 = 1.2275 / 84.007 #mol/L
    TK = 25 + 273.15
    DIC = solve_NaHCO3_equilibrium(cNaHCO3, TK)
    cC_former = cNaHCO3

    # Jnum = numba_jacobian(x_guess, np.array(tuple_cC_cCa_cNa_cCl))
    # print(J - Jnum)
    return

if __name__ == "__main__":
    main()
