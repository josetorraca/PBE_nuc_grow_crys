import numpy as np
from scipy import optimize

logK = {'HO2': -2.886, 'HCO2': -1.464, 'a1': -6.363, 'a2': -10.329, 'w': -13.997, 'NaOH': -0.183,\
 'NaCO3-': 1.27, 'NaHCO3':-0.25 }
c = {'H+': 0., 'OH-': 0., 'CO2': 0., 'CO3--': 0., 'HCO3-': 0., 'Na+': 0.}

logO2 = -0.67

def ionic_strength(c):
    return 0.5*(np.power(10,c['H+'])\
            +4*np.power(10,c['CO3--'])+np.power(10,c['HCO3-'])\
            +np.power(10,c['OH-']) + np.power(10,c['Na+']) + np.power(10,c['NaCO3-']) )

loggamma = {'0': 0., '+': 0., '-': 0., '++': 0., '--': 0.}

def calculate_loggamma(m):
    b = 0.1; A = 0.5
    loggamma['0'] = b*m
    loggamma['+'] = loggamma['-'] = -A*(1)*(np.sqrt(m)/(1+np.sqrt(m))-0.2*m)
    loggamma['++'] = loggamma['--'] = 4*loggamma['+']
    return

def equilibrium_fromPCO2(x,logPCO2,cNaHCO3):
    c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
    c['HCO3-'] = x[4]; c['Na+'] = x[5]; c['NaOH'] = x[6]; c['NaCO3-'] = x[7];\
    c['NaHCO3'] = x[8]; c['O2'] = x[9];
    cNaT = cNaHCO3; #This came from sodium carbonate

    m = ionic_strength(c)
    calculate_loggamma(m)
    # Here we go, to define the reactions to carbonate-CO2 equilibrium
    Reaction = [None]*10
    Reaction[0] = c['O2'] - logO2 - (logK['HO2']-loggamma['0'])
    Reaction[1] = c['CO2'] - logPCO2 - (logK['HCO2']-loggamma['0'])
    Reaction[2] = c['HCO3-'] + c['H+'] - (logK['a1']+loggamma['0'] -loggamma['+']-loggamma['-'])- c['CO2']
    Reaction[3] = c['CO3--'] + c['H+'] - (logK['a2']-loggamma['--']) - c['HCO3-']
    Reaction[4] = c['OH-'] + c['H+'] - (logK['w'] + loggamma['0'] - loggamma['+']-loggamma['-'])
    # Sodium reactions
    Reaction[5] =  c['Na+'] + loggamma['+'] + logK['NaOH'] + c['OH-']+ loggamma['-'] - c['NaOH'] - loggamma['0']
    Reaction[6] =  c['Na+'] + c['CO3--'] + (logK['NaCO3-']+loggamma['--']) - c['NaCO3-']
    Reaction[7] =  c['Na+'] + c['HCO3-'] + (logK['NaHCO3']-loggamma['0'] +loggamma['+']+loggamma['-']) - c['NaHCO3']
    Reaction[8] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- \
    np.power(10,c['NaOH'])- np.power(10,c['NaHCO3'])
    # Charge Conservation
    Reaction[9] = np.power(10,c['H+']) + np.power(10,c['Na+']) - np.power(10,c['HCO3-'])\
     - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) - np.power(10,c['NaCO3-'])
    return Reaction

def equilibrium_fromDIC(x,DIC,cNaHCO3):
    c['H+'] = x[0]; c['OH-'] = x[1]; c['CO2'] = x[2]; c['CO3--'] = x[3]; \
    c['HCO3-'] = x[4]; c['Na+'] = x[5]; c['NaOH'] = x[6]; c['NaCO3-'] = x[7];\
    c['NaHCO3'] = x[8]; c['O2'] = x[8];
    cNaT = cNaHCO3; #This came from sodium carbonate

    m = ionic_strength(c)
    calculate_loggamma(m)
    # Here we go, to define the reactions to carbonate-CO2 equilibrium
    Reaction = [None]*10
    # Total carbon concentration
    Reaction[0] = c['O2'] - logO2 - (logK['HO2']-loggamma['0'])
    Reaction[1] =  DIC - np.power(10,c['CO2']) - np.power(10,c['CO3--']) \
    - np.power(10,c['HCO3-'])- np.power(10,c['NaCO3-'])
    Reaction[2] = c['HCO3-'] + c['H+'] - (logK['a1']+loggamma['0'] -loggamma['+']-loggamma['-'])- c['CO2']
    Reaction[3] = c['CO3--'] + c['H+'] - (logK['a2']-loggamma['--']) - c['HCO3-']
    Reaction[4] = c['OH-'] + c['H+'] - (logK['w'] + loggamma['0'] - loggamma['+']-loggamma['-'])
    # Sodium reactions
    Reaction[5] =  c['Na+'] + loggamma['+'] + logK['NaOH'] + c['OH-']+ loggamma['-'] - c['NaOH'] - loggamma['0']
    Reaction[6] =  c['Na+'] + c['CO3--'] + (logK['NaCO3-']+loggamma['--']) - c['NaCO3-']
    Reaction[7] =  c['Na+'] + c['HCO3-'] + (logK['NaHCO3']-loggamma['0'] +loggamma['+']+loggamma['-']) - c['NaHCO3']
    Reaction[8] = cNaT - np.power(10,c['Na+']) - np.power(10,c['NaCO3-'])- np.power(10,c['NaOH'])- np.power(10,c['NaHCO3'])
    # Charge Conservation
    Reaction[9] = np.power(10,c['H+']) + np.power(10,c['Na+']) - np.power(10,c['HCO3-'])\
     - 2*np.power(10,c['CO3--']) - np.power(10,c['OH-']) + np.power(10,c['NaCO3-'])
    return Reaction

print("The input parameters:")
logPCO2 = -3.455
cNaHCO3 = 1.2275/84.007
print("logPCO2= ", logPCO2)
print("[NaHCO3]= ", cNaHCO3)

sol = optimize.root(equilibrium_fromPCO2, [-1.0] * 10, args=(logPCO2,cNaHCO3), method='hybr')
I = ionic_strength(c)
sigma = 6.2e4*I

print("The output of concentrations (mol/L)")
for item in c:
     print(item+"\t", np.power(10,c[item]))

print("DIC= ", np.power(10,c['CO2'])+np.power(10,c['CO3--'])+ np.power(10,c['HCO3-']) + np.power(10,c['NaCO3-']))
print("pH= ", -c['H+']-loggamma['+'])
print("sigma= ", sigma, " muS/cm")

# # Variando o DIC
# DICarray = np.arange(0.0,0.03,0.0001)
#
# f1=open('sodiumbicarbonate-conductivityandpH-fromDIC.dat', 'w+')
# f1.write('# DIC (mol/L)\t pH\t Conductivity (mS/cm)\n')
#
# for j in range(len(DICarray)):
#     DIC = DICarray[j]
#     sol = optimize.root(equilibrium_fromDIC, [-1.0] * 10, args=(DIC,cNaHCO3), method='hybr')
#     pH = -sol.x[0]-loggamma['+']
#     I = ionic_strength(c)
#     sigma = 6.2e4*I
#     f1.write('%2.3g \t %2.3g \t %2.3f\n' % (DIC,pH,sigma))
#
# f1.close()
