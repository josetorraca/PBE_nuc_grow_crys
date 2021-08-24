import numpy as np
from scipy import optimize

class CalciumCarbonateReaction():

    logK = {'H': -1.464, 'a1': -6.363, 'a2': -10.329, 'w': -13.997, 'sp': -8.48, \
    'CaH': 1.26, 'CaC': 3.15, 'CaOH': 1.3, 'NaOH': -14.18, 'NaCO3-': 1.27, 'NaHCO3':-0.25 }

    def __init__(self):
        self.c = {'H+': 0., 'OH-': 0., 'CO2': 0., 'CO3--': 0., 'HCO3-': 0., 'Ca++': 0., \
        'CaOH+': 0., 'CaHCO3+': 0., 'CaCO3(aq)': 0., 'Na+': 0., 'NaOH': 0.,'NaCO3-': 0., 'Cl-': 0 }
        self.loggamma = {'0': 0., '+': 0., '-': 0., '++': 0., '--': 0.}
        self.guess = [-0.1] * 13

    def ionic_strength(self, c):
        return 0.5*(4*np.power(10,c['Ca++'])+np.power(10,c['CaHCO3+'])\
                +np.power(10,c['CaOH+'])+np.power(10,c['H+'])\
                +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
                +np.power(10,c['OH-']) + np.power(10,c['Na+']) \
                + np.power(10,c['Cl-']) + np.power(10,c['NaCO3-']) )

    def calculate_loggamma(self, m):
        b = 0.1; A = 0.5
        loggamma = self.loggamma
        loggamma['0'] = b*m
        loggamma['+'] = loggamma['-'] = -A*(1)*(np.sqrt(m)/(1+np.sqrt(m))-0.2*m)
        loggamma['++'] = loggamma['--'] = 4*loggamma['+']
        return

    def SodiumCarbonateMixture(self,x,cCT,cCaT,cNaT,cClT):
        c = self.c
        loggamma = self.loggamma
        logK = self.logK

        c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
        c['HCO3-'] = x[4]; c['Ca++'] = x[5]; c['CaOH+'] = x[6]; c['CaHCO3+'] = x[7];\
        c['CaCO3(aq)'] = x[8]; c['Na+'] = x[9]; c['NaOH'] = x[10]; c['NaCO3-'] = x[11]; \
        c['NaHCO3'] = x[12];
        c['Cl-'] = np.log10(cClT);
        DIC = DICNa2CO3 + DICCaCl2

        m = self.ionic_strength(c)
        self.calculate_loggamma(m)
        # Here we go, to define the reactions to carbonate-CO2 equilibrium
        Reaction = [None]*13
        Reaction[0] = c['HCO3-'] + c['H+'] - (logK['a1']+loggamma['0'] -loggamma['+']-loggamma['-'])- c['CO2']
        Reaction[1] = c['CO3--'] + c['H+'] - (logK['a2']-loggamma['--']) - c['HCO3-']
        Reaction[2] = c['OH-'] + c['H+'] - (logK['w'] + loggamma['0'] - loggamma['+']-loggamma['-'])
        # Calcium reactions
        Reaction[3] = c['CaHCO3+'] - (logK['CaH']+loggamma['++']) - c['Ca++'] - c['HCO3-']
        Reaction[4] = c['CaCO3(aq)'] - (logK['CaC']-loggamma['0']+loggamma['++']+loggamma['--']) - c['Ca++'] - c['CO3--']
        Reaction[5] = c['CaOH+'] - (logK['CaOH']+loggamma['++']) - c['Ca++'] - c['OH-']
        # Charge conservation
        Reaction[6] = np.power(10,c['H+']) + 2*np.power(10,c['Ca++']) \
        + np.power(10,c['CaHCO3+']) + np.power(10,c['CaOH+']) - np.power(10,c['HCO3-'])\
         - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) + np.power(10,c['Na+']) \
         - np.power(10,c['Cl-']) - np.power(10,c['NaCO3-'])
        # Total calcium concentration
        Reaction[7] = cCaT - np.power(10,c['Ca++']) - np.power(10,c['CaHCO3+']) \
        - np.power(10,c['CaCO3(aq)']) - np.power(10,c['CaOH+'])
        # Total carbon concentration
        Reaction[8] =  cCT - np.power(10,c['CO2']) - np.power(10,c['CO3--']) \
        - np.power(10,c['HCO3-'])- np.power(10,c['CaHCO3+'])- np.power(10,c['CaCO3(aq)'])\
        - np.power(10,c['NaCO3-'])- np.power(10,c['NaHCO3'])
        # Sodium reactions
        Reaction[9] =  c['Na+'] + logK['NaOH'] - c['NaOH'] - c['H+']
        Reaction[10] =  c['Na+'] + c['CO3--'] + (logK['NaCO3-']+loggamma['--']) - c['NaCO3-']
        Reaction[11] =  c['Na+'] + c['HCO3-'] + (logK['NaHCO3']-loggamma['0'] +loggamma['+']+loggamma['-']) - c['NaHCO3']
        Reaction[12] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- \
            np.power(10,c['NaOH'])- np.power(10,c['NaHCO3'])

        return Reaction

    def SodiumBicarbonateMixture(self,x,cCT,cCaT,cNaT,cClT):
        c = self.c
        loggamma = self.loggamma
        logK = self.logK

        c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
        c['HCO3-'] = x[4]; c['Ca++'] = x[5]; c['CaOH+'] = x[6]; c['CaHCO3+'] = x[7];\
        c['CaCO3(aq)'] = x[8]; c['Na+'] = x[9]; c['NaOH'] = x[10]; c['NaCO3-'] = x[11]; \
        c['NaHCO3'] = x[12];
        c['Cl-'] = np.log10(cClT);

        m = self.ionic_strength(c)
        self.calculate_loggamma(m)
        # Here we go, to define the reactions to carbonate-CO2 equilibrium
        Reaction = [None]*13
        Reaction[0] = c['HCO3-'] + c['H+'] - (logK['a1']+loggamma['0'] -loggamma['+']-loggamma['-'])- c['CO2']
        Reaction[1] = c['CO3--'] + c['H+'] - (logK['a2']-loggamma['--']) - c['HCO3-']
        Reaction[2] = c['OH-'] + c['H+'] - (logK['w'] + loggamma['0'] - loggamma['+']-loggamma['-'])
        # Calcium reactions
        Reaction[3] = c['CaHCO3+'] - (logK['CaH']+loggamma['++']) - c['Ca++'] - c['HCO3-']
        Reaction[4] = c['CaCO3(aq)'] - (logK['CaC']-loggamma['0']+loggamma['++']+loggamma['--']) - c['Ca++'] - c['CO3--']
        Reaction[5] = c['CaOH+'] - (logK['CaOH']+loggamma['++']) - c['Ca++'] - c['OH-']
        # Charge conservation
        Reaction[6] = np.power(10,c['H+']) + 2*np.power(10,c['Ca++']) \
        + np.power(10,c['CaHCO3+']) + np.power(10,c['CaOH+']) - np.power(10,c['HCO3-'])\
         - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) + np.power(10,c['Na+']) \
         - np.power(10,c['Cl-']) - np.power(10,c['NaCO3-'])
        # Total calcium concentration
        Reaction[7] = cCaT - np.power(10,c['Ca++']) - np.power(10,c['CaHCO3+']) \
        - np.power(10,c['CaCO3(aq)']) - np.power(10,c['CaOH+'])
        # Total carbon concentration
        Reaction[8] =  cCT - np.power(10,c['CO2']) - np.power(10,c['CO3--']) \
        - np.power(10,c['HCO3-'])- np.power(10,c['CaHCO3+'])- np.power(10,c['CaCO3(aq)'])\
        - np.power(10,c['NaCO3-'])- np.power(10,c['NaHCO3'])
        # Sodium reactions
        Reaction[9] =  c['Na+'] + logK['NaOH'] - c['NaOH'] - c['H+']
        Reaction[10] =  c['Na+'] + c['CO3--'] + (logK['NaCO3-']+loggamma['--']) - c['NaCO3-']
        Reaction[11] =  c['Na+'] + c['HCO3-'] + (logK['NaHCO3']-loggamma['0'] +loggamma['+']+loggamma['-']) - c['NaHCO3']
        Reaction[12] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- \
            np.power(10,c['NaOH'])- np.power(10,c['NaHCO3'])

        return Reaction

    def solve(self, cCT,cCaT,cNaT,cClT, guess=None, mix="SodiumBicarbonate"):
        if guess == None:
            guess = self.guess
        mix_dict = {"SodiumBicarbonate": self.SodiumBicarbonateMixture, "SodiumCarbonate": self.SodiumCarbonateMixture}
        fun = mix_dict[mix]
        self.sol = optimize.root(fun, guess, args=(cCT,cCaT,cNaT,cClT), method='hybr')
        x = self.sol['x']
        c = self.c
        c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
        c['HCO3-'] = x[4]; c['Ca++'] = x[5]; c['CaOH+'] = x[6]; c['CaHCO3+'] = x[7];\
        c['CaCO3(aq)'] = x[8]; c['Na+'] = x[9]; c['NaOH'] = x[10]; c['NaCO3-'] = x[11]; \
        c['NaHCO3'] = x[12];
        m = self.ionic_strength(c)
        self.calculate_loggamma(m)
        loggamma = self.loggamma
        logK = self.logK
        self.pH = -c['H+']-loggamma['+']
        self.I = self.ionic_strength(c)
        self.sigma = 6.2e4*self.I
        self.coefB = np.power(10,loggamma['--'])
        self.coefA = np.power(10,loggamma['++'])
        self.Ksp = np.power(10,logK['sp'])
        self.cBion = np.power(10,c['CO3--'])
        self.cAion = np.power(10,c['Ca++'])
        self.S = np.power(10,c['Ca++']+loggamma['++'])*np.power(10,c['CO3--']+loggamma['--'])/np.power(10.0,logK['sp'])
        # CFCM Mod:
        self.IAP = np.power(10,c['Ca++']+loggamma['++'])*np.power(10,c['CO3--']+loggamma['--'])
