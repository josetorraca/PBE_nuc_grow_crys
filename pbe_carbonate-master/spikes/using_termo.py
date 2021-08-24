import thermo
# from thermo.chemical import Mixture

# vodka = Mixture(['water', 'ethanol'], Vfls=[.6, .4], T=300, P=1E5)
# print(vodka.Prl,vodka.Prg)

from thermo.chemical import Chemical
tol = Chemical('toluene')

print(tol.Tm, tol.Tb, tol.Tc)
