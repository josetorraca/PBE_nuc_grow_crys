#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

def dieletricconstant_water(TK):
    # for TK: 273-372
    return (0.24921e3 - 0.79069*TK + 0.72997e-3*TK**2)

def density_water(TK):
    # for TK: 273-372
    return (0.183652 + 0.00724987*TK - 0.203449e-4*TK**2 + 1.73702e-8*TK**3)

def equilibrium(cNaHCO3, cCaCl2, QCaCl2, QNaHCO3, T):
    #########################################################################
    # MAIN PROGRAM
    ##########################################################################
    # Define the log10 of concentration of each element
    logc = {'H+': 0., 'OH-': 0., 'CO2': 0., 'CO3--': 0., 'HCO3-': 0., 'Na+': 0., 'Cl-': 0., 'Ca++': 0., 'CaHCO3+': 0., 'CaOH+': 0., 'NaCl': 0., 'HCl': 0., 'Na2CO3': 0.}
    TK = T + 273.16
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

    # model for activity coefficient
    loggamma = {'H+': 0.}
    def ion_activity(I,TK):
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
        loggamma['CaCO3'] = 0.5*loggamma['Ca++'] + 0.5*loggamma['CO3--']
        loggamma['NaHCO3'] = 0.5*loggamma['Na+'] + 0.5*loggamma['HCO3-']
        loggamma['Na2CO3'] = (1.0/3.0)*loggamma['Na+'] + (2.0/3.0)*loggamma['CO3--']
        loggamma['NaOH'] = 0.5*loggamma['Na+'] + 0.5*loggamma['OH-']
        loggamma['CO2'] =  -0.5*I
        loggamma['NaCl'] = 0.5*loggamma['Na+'] + 0.5*loggamma['Cl-']
        loggamma['HCl'] = 0.5*loggamma['H+'] + 0.5*loggamma['Cl-']

        if I < 1.0:
             loggamma['H2O'] = np.log10(1-0.017*I)
        else: loggamma['H2O'] = -0.5*I
        return

    # The aqueous NaHCO3 equilibrium reactions for an open system
    def NaHCO3_equilibrium(x,cNaHCO3):
        logc['H+'] = x[0]; logc['OH-'] = x[1]; logc['CO2'] = x[2]; logc['CO3--'] = x[3]; \
        logc['HCO3-'] = x[4]; logc['Na+'] = x[5]; logc['NaOH'] = x[6]; logc['NaCO3-'] = x[7];\
        logc['NaHCO3'] = x[8]; logc['Na2CO3'] = x[9]
        cNaT = cNaHCO3 #This came from sodium carbonate
        # ionic strength
        I = 0.5*(np.power(10,logc['H+'])\
                +4*np.power(10,logc['CO3--'])+np.power(10,logc['HCO3-'])\
                +np.power(10,logc['OH-']) + np.power(10,logc['Na+']) + np.power(10,logc['NaCO3-']) )
        ion_activity(I,TK)
        # carbonate-CO2 equilibrium
        Reaction = [None]*10
        Reaction[0] = (logc['CO2']+loggamma['CO2']) - logPCO2 - logK['H']
        Reaction[1] = (logc['HCO3-']+loggamma['HCO3-']) + (logc['H+']+loggamma['H+']) - logK['a1']- (logc['CO2']+loggamma['CO2']) - loggamma['H2O']
        Reaction[2] = (logc['CO3--']+loggamma['CO3--']) + (logc['H+']+loggamma['H+']) - logK['a2'] - (logc['HCO3-']+loggamma['HCO3-'])
        Reaction[3] = (logc['OH-']+loggamma['OH-']) + (logc['H+']+loggamma['H+']) - logK['w'] - loggamma['H2O']
        # Sodium reactions
        Reaction[4] = (logc['Na+']+loggamma['Na+']) + loggamma['H2O'] + logK['NaOH'] - (logc['H+']+loggamma['H+']) - (logc['NaOH']+loggamma['NaOH'])
        Reaction[5] = (logc['Na+']+loggamma['Na+']) + (logc['CO3--']+loggamma['CO3--']) + logK['NaCO3'] - (logc['NaCO3-']+loggamma['NaCO3-'])
        Reaction[6] = (logc['Na+']+loggamma['Na+']) + (logc['HCO3-']+loggamma['HCO3-']) + logK['NaHCO3'] - (logc['NaHCO3']+loggamma['NaHCO3'])
        Reaction[7] = 2*(logc['Na+']+loggamma['Na+']) + (logc['CO3--']+loggamma['CO3--']) + logK['Na2CO3'] - (logc['Na2CO3']+loggamma['Na2CO3'])
        Reaction[8] = cNaT - np.power(10,logc['Na+']) - np.power(10,logc['NaCO3-'])- \
        np.power(10,logc['NaOH'])- np.power(10,logc['NaHCO3']) - np.power(10,logc['Na2CO3'])
        # Charge Conservation
        Reaction[9] = np.power(10,logc['H+']) + np.power(10,logc['Na+']) - np.power(10,logc['HCO3-'])\
        - 2*np.power(10,logc['CO3--']) - np.power(10,logc['OH-']) - np.power(10,logc['NaCO3-'])
        return Reaction

    # The aqueous CaCl2 equilibrium reactions for an open system
    def CaCl2_equilibrium(x,cCaCl2):
        logc['H+'] = x[0]; logc['OH-'] = x[1]; logc['CO2'] = x[2]; logc['CO3--'] = x[3]; \
        logc['HCO3-'] = x[4]; logc['CaOH+'] = x[5]; logc['CaHCO3+'] = x[6];\
        logc['CaCO3'] = x[7]; logc['Ca++'] = x[8]; logc['Cl-'] = x[9]; logc['HCl'] = x[10]
        cClT = 2*cCaCl2; cCaT = cCaCl2
        # calculate the ionic strength
        I = 0.5*(np.power(10,logc['H+'])\
                +4*np.power(10,logc['CO3--'])+np.power(10,logc['HCO3-'])\
                +np.power(10,logc['OH-']) + np.power(10,logc['Cl-']) + 4*np.power(10,logc['Ca++'])\
                +np.power(10,logc['CaHCO3+']) + np.power(10,logc['CaOH+']) )
        ion_activity(I,TK)
        # carbonate-CO2 equilibrium
        Reaction = [None]*11
        Reaction[0] = (logc['CO2']+loggamma['CO2']) - logPCO2 - logK['H']
        Reaction[1] = (logc['HCO3-']+loggamma['HCO3-']) + (logc['H+']+loggamma['H+']) - logK['a1']- (logc['CO2']+loggamma['CO2']) - loggamma['H2O']
        Reaction[2] = (logc['CO3--']+loggamma['CO3--']) + (logc['H+']+loggamma['H+']) - logK['a2'] - (logc['HCO3-']+loggamma['HCO3-'])
        Reaction[3] = (logc['OH-']+loggamma['OH-']) + (logc['H+']+loggamma['H+']) - logK['w'] - loggamma['H2O']
        # Calcium reactions
        Reaction[4] = (logc['CaHCO3+']+loggamma['CaHCO3+'])  - logK['CaHCO3+']- (logc['Ca++']+loggamma['Ca++']) - (logc['HCO3-']+loggamma['HCO3-'])
        Reaction[5] = (logc['CaCO3']+loggamma['CaCO3'])- logK['CaCO3'] - (logc['Ca++']+loggamma['Ca++']) - (logc['CO3--']+loggamma['CO3--'])
        Reaction[6] = (logc['Ca++']+loggamma['Ca++']) + loggamma['H2O'] + logK['CaOH'] - (logc['CaOH+']+loggamma['CaOH+']) - (logc['H+']+loggamma['H+'])
        Reaction[7] = cCaT - np.power(10,logc['Ca++']) - np.power(10,logc['CaOH+']) -np.power(10,logc['CaHCO3+']) - np.power(10,logc['CaCO3'])
        # Chrolide reactions
        Reaction[8] = (logc['H+']+loggamma['H+']) +  (logc['Cl-']+loggamma['Cl-']) - logK['HCl'] - (logc['HCl']+loggamma['HCl'])
        Reaction[9] = cClT - np.power(10,logc['Cl-']) - np.power(10,logc['HCl'])
        # Charge conservation
        Reaction[10] = np.power(10,logc['H+']) +2*np.power(10,logc['Ca++'])- np.power(10,logc['Cl-']) - np.power(10,logc['HCO3-'])\
        - 2*np.power(10,logc['CO3--']) - np.power(10,logc['OH-']) + np.power(10,logc['CaHCO3+']) \
        + np.power(10,logc['CaOH+'])
        return Reaction

    # The NaHCO3+CaCl2 -> CaCO3 system of chemical equilibrium equations to solve
    def CaCO3_mixture(x,cCaT,DIC,cNaT,cClT):
        logc['H+'] = x[0]; logc['OH-'] = x[1]; logc['CO2'] = x[2]; logc['CO3--'] = x[3]; \
        logc['HCO3-'] = x[4]; logc['Ca++'] = x[5]; logc['CaOH+'] = x[6]; logc['CaHCO3+'] = x[7];\
        logc['CaCO3'] = x[8]; logc['Na+'] = x[9]; logc['NaOH'] = x[10]; logc['NaCO3-'] = x[11]; \
        logc['NaHCO3'] = x[12]; logc['Cl-'] = x[13]; logc['NaCl'] = x[14]; logc['HCl'] = x[15]
        # Calculate the ionic strength
        I = 0.5*(4*np.power(10,logc['Ca++'])+np.power(10,logc['CaHCO3+'])\
                +np.power(10,logc['CaOH+'])+np.power(10,logc['H+'])\
                +4*np.power(10,logc['CO3--'])+np.power(10,logc['HCO3-'])\
                +np.power(10,logc['OH-']) + np.power(10,logc['Na+']) \
                + np.power(10,logc['Cl-']) + np.power(10,logc['NaCO3-']) )
        ion_activity(I,TK)
        # Here we go, to define the reactions to carbonate-CO2 equilibrium
        Reaction = [None]*16
        Reaction[0] = (logc['HCO3-']+loggamma['HCO3-']) + (logc['H+']+loggamma['H+']) - logK['a1']- (logc['CO2']+loggamma['CO2']) - loggamma['H2O']
        Reaction[1] = (logc['CO3--']+loggamma['CO3--']) + (logc['H+']+loggamma['H+']) - logK['a2'] - (logc['HCO3-']+loggamma['HCO3-'])
        Reaction[2] = (logc['OH-']+loggamma['OH-']) + (logc['H+']+loggamma['H+']) - logK['w'] - loggamma['H2O']
        # Calcium reactions
        Reaction[3] = (logc['CaHCO3+']+loggamma['CaHCO3+'])  - logK['CaHCO3+']- (logc['Ca++']+loggamma['Ca++']) - (logc['HCO3-']+loggamma['HCO3-'])
        Reaction[4] = (logc['CaCO3']+loggamma['CaCO3'])- logK['CaCO3'] - (logc['Ca++']+loggamma['Ca++']) - (logc['CO3--']+loggamma['CO3--'])
        Reaction[5] = (logc['Ca++']+loggamma['Ca++']) + loggamma['H2O'] + logK['CaOH'] - (logc['CaOH+']+loggamma['CaOH+']) - (logc['H+']+loggamma['H+'])
        Reaction[6] = cCaT - np.power(10,logc['Ca++']) - np.power(10,logc['CaOH+']) -np.power(10,logc['CaHCO3+']) - np.power(10,logc['CaCO3'])
        # Charge conservation
        Reaction[7] = np.power(10,logc['H+']) + 2*np.power(10,logc['Ca++']) \
        + np.power(10,logc['CaHCO3+']) + np.power(10,logc['CaOH+']) - np.power(10,logc['HCO3-'])\
        - 2*np.power(10,logc['CO3--']) - np.power(10,logc['OH-']) + np.power(10,logc['Na+']) \
        - np.power(10,logc['Cl-']) - np.power(10,logc['NaCO3-'])
        # Total carbon concentration
        Reaction[8] =  DIC - np.power(10,logc['CO2']) - np.power(10,logc['CO3--']) \
        - np.power(10,logc['HCO3-'])- np.power(10,logc['CaHCO3+'])- np.power(10,logc['CaCO3'])\
        - np.power(10,logc['NaCO3-'])- np.power(10,logc['NaHCO3'])
        # Chrolide reactions
        Reaction[9] = (logc['Na+']+loggamma['Na+']) + (logc['Cl-']+loggamma['Cl-']) + logK['NaCl'] - (logc['NaCl']+loggamma['NaCl'])
        Reaction[10] = (logc['H+']+loggamma['H+']) + (logc['Cl-']+loggamma['Cl-']) + logK['NaCl'] - (logc['HCl']+loggamma['HCl'])
        Reaction[11] = cClT - np.power(10,logc['Cl-']) - np.power(10,logc['NaCl']) - np.power(10,logc['HCl'])
        # Sodium reactions
        Reaction[12] =  (logc['Na+']+loggamma['Na+']) + loggamma['H2O'] + logK['NaOH'] - (logc['H+']+loggamma['H+']) - (logc['NaOH']+loggamma['NaOH'])
        Reaction[13] = (logc['Na+']+loggamma['Na+']) + (logc['CO3--']+loggamma['CO3--']) + logK['NaCO3'] - (logc['NaCO3-']+loggamma['NaCO3-'])
        Reaction[14] = (logc['Na+']+loggamma['Na+']) + (logc['HCO3-']+loggamma['HCO3-']) + logK['NaHCO3'] - (logc['NaHCO3']+loggamma['NaHCO3'])
        Reaction[15] = cNaT - np.power(10,logc['Na+']) - np.power(10,logc['NaCO3-'])- \
        np.power(10,logc['NaOH'])- np.power(10,logc['NaHCO3'])- np.power(10,logc['NaCl'])
        return Reaction


    ########## NaHCO3 equilibrium  ##########
    solNaHCO3 = optimize.root(NaHCO3_equilibrium, [-1.0] * 10, args=cNaHCO3, method='hybr')
    sigmaNaHCO3 = 6.2e4*(0.5*(np.power(10,logc['H+'])\
                +4*np.power(10,logc['CO3--'])+np.power(10,logc['HCO3-'])\
                +np.power(10,logc['OH-']) + np.power(10,logc['Na+']) + np.power(10,logc['NaCO3-']) ))
    pHNaHCO3 = -logc['H+']-loggamma['H+']
    DICNaHCO3 = np.power(10,logc['CO2'])+np.power(10,logc['CO3--'])+ np.power(10,logc['HCO3-']) + np.power(10,logc['NaCO3-'])

    ########## CaCl2 equilibrium  ##########
    solCaCl2 = optimize.root(CaCl2_equilibrium, [-1.0] * 11, args=cCaCl2, method='hybr')
    pHCaCl2 = -logc['H+']-loggamma['H+']
    sigmaCaCl2 = 6.2e4*(0.5*(np.power(10,logc['H+'])\
                +4*np.power(10,logc['CO3--'])+np.power(10,logc['HCO3-'])\
                +np.power(10,logc['OH-']) + np.power(10,logc['Cl-']) + 4*np.power(10,logc['Ca++'])\
                +np.power(10,logc['CaHCO3+']) + np.power(10,logc['CaOH+']) ))
    DICCaCl2 = np.power(10,logc['CO2'])+np.power(10,logc['CO3--'])+ np.power(10,logc['HCO3-']) +np.power(10,logc['CaHCO3+']) + np.power(10,logc['CaCO3'])

    ########## NaHCO3 + CaCl2 -> CaCO3 mixture  ##########
    # Agora faremos a mistura das duas solucoes
    Qtot = QCaCl2 + QNaHCO3
    cCaT = cCaCl2*QCaCl2/Qtot
    cClT = 2*cCaT
    cNaT = cNaHCO3*QNaHCO3/Qtot
    DIC = DICNaHCO3*QNaHCO3/Qtot + DICCaCl2*QCaCl2/Qtot

    solmix = optimize.root(CaCO3_mixture, [-1.0] * 16, args=(cCaT,DIC,cNaT,cClT), method='hybr')
    # Calculating the supersaturation index for calcite
    pHmix = -logc['H+']-loggamma['H+']
    S = np.power(10,logc['Ca++']+loggamma['Ca++'])*np.power(10,logc['CO3--']+loggamma['CO3--'])/np.power(10.0,logK['calcite'])
    IS = np.log10(S)
    print('NaHCO3 root solver success = {}'.format(solNaHCO3.success))
    print('CaCl2 root solver success = {}'.format(solCaCl2.success))
    print('Mix root solver success = {}'.format(solmix.success))

    results = [pHNaHCO3, sigmaNaHCO3, pHCaCl2, sigmaCaCl2, IS, pHmix]
    return results

def plot_activityproduct(T,IS):
    TC = np.arange(0,90,0.1)
    TK = TC + 273.16

    logKcal = -171.9065 - 0.077993*TK + 2839.319/TK + 71.595*np.log10(TK)
    logKvat = -172.1295 - 0.077993*TK + 3074.688/TK + 71.595*np.log10(TK)
    logKara =-171.9773 - 0.077993*TK + 2903.293/TK + 71.595*np.log10(TK)
    logKACChyd = 1247.0/TK - 10.224 # hydrated ACC (Clarkson, J. R., Price, T. J., & Adams, C. J. (1992). Role of metastable phases in the spontaneous precipitation of calcium carbonate. Journal of the Chemical Society, Faraday Transactions, 88(2), 243–249. https://doi.org/10.1039/FT9928800243)

    # calculated
    tK = T + 273.16
    logKcalcnow = -171.9065 - 0.077993*tK + 2839.319/tK + 71.595*np.log10(tK)
    logaCaaCO3 = IS + logKcalcnow
    fig = plt.figure()
    plt.plot(TC,logKcal, label='calcite')
    plt.plot(TC,logKara, label='aragonite')
    plt.plot(TC,logKvat, label='vaterite')
    plt.plot(TC,logKACChyd, '--', color='silver', label='hyd ACC')
    plt.plot(T, logaCaaCO3, 'ro', label='calculated')
    plt.legend()
    plt.xlim(0,90)
    plt.xlabel("Temperature (Celsius)")
    plt.ylabel("$\log{(a_{Ca}a_{CO_3})}$")
    plt.show()
    return

if __name__== "__main__":
    ################ Input ##############################
    ## concentracao de cada solução
    cNaHCO3 = 0.015 #mol/L
    cCaCl2 = 0.005 #mol/L
    ## vazao de cada solução
    QCaCl2 = 150 #L/h
    QNaHCO3 = 150 #L/h
    ## temperatura
    T = 25 #Celsius
    ################ Calculating ##############################
    [pHNaHCO3, sigmaNaHCO3, pHCaCl2, sigmaCaCl2, IS, pHmix] = equilibrium(cNaHCO3, cCaCl2, QCaCl2, QNaHCO3, T)
    ################ Output ##############################
    print("The output for NaHCO3 equilibrium:")
    print("pHNaHCO3= ", pHNaHCO3)
    print("sigmaNaHCO3= ", sigmaNaHCO3, " muS/cm")
    print("The output for CaCl2 equilibrium:")
    print("pHCaCl2= ", pHCaCl2)
    print("sigmaCaCl2= ", sigmaCaCl2, "muS/cm")
    print("The output for NaHCO3 + CaCl2 -> CaCO3 mixture :")
    print("IS= ", IS)
    print("pH= ", pHmix)
    ################ Plotting ##############################
    fig = plot_activityproduct(T,IS)
