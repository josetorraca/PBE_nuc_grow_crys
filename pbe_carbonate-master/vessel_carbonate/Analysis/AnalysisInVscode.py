#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataSciece.changeDirOnImportExport setting
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'vessel_carbonate/Analysis'))
	print(os.getcwd())
except:
	pass
#%% [markdown]
# ## Equilibrium Analysis for the $CaCl_2$ and $NaHCO_3$ System

#%%
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('env', 'NUMBA_DISABLE_JIT=1')
import numpy as np
import plotly.offline as py
import plotly.graph_objs as go
import sys
sys.path.append('/home/caio/Projects/CarbonateDeposition/Repositories/psd-simulations-msm/vessel_carbonate/')
import calciumcarbonate_supersaturation_module as carbonate_eq
import mdl_vessel_carbonate


#%%
py.init_notebook_mode(connected=True)


#%%
pp = mdl_vessel_carbonate.SystemPhysicoChemicalParameters()
mdl = mdl_vessel_carbonate.MyModel(100, 101, pp)
max_cond = mdl_vessel_carbonate.calculate_maximum_conditions(mdl)


#%%
# Maximum species after complete addition
m_slv = mdl.rho_w * max_cond['Vfinal']
# cCa = max_cond['Ca-max'] / m_slv
# cC = max_cond['C-max'] / m_slv
# cNa = max_cond['Na-max'] / m_slv
# cCl = max_cond['Cl-max'] / m_slv
VinL = max_cond['Vfinal'] * 1e-3
cCa = (max_cond['Ca-max']/pp.M_Ca) / VinL
cC = (max_cond['C-max']/pp.M_C) / VinL
cNa = (max_cond['Na-max']/pp.M_Na) / VinL
cCl = (max_cond['Cl-max']/pp.M_Cl) / VinL
eQ = carbonate_eq.CalciumCarbonateReaction()
eQ.solve(cC, cCa, cNa, cCl)


#%%
max_cond


#%%
eQ.I /4


#%%
eQ.IAP


#%%
eQ.Ksp
