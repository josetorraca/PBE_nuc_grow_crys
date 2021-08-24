import os
import sys
import pandas as pd
import numpy as np

# sys.path.append('/home/caio/Projects/CarbonateDeposition/Repositories/psd-simulations-msm/vessel_carbonate/')

DATA_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data/original-torraca-msc/')
DATA_FOLDER = os.path.abspath(DATA_FOLDER)
COL_NAMES = ['time', '1', '2', '3']
VOLUME_MEASURED_QICPIC_ML = 80.0
FINAL_ADDITION_TIME = 56.0
VOLUMETRIC_FLOW_IN = 1.75
INITIAL_VOLUME = 600.0

def load_data():

    df_condutivity = pd.read_excel(DATA_FOLDER + '/Cond_TB.xlsx', usecols=3, nrows=18141,
                names=COL_NAMES)
    df_ph = pd.read_excel(DATA_FOLDER + '/ph_TB.xlsx', usecols=3, nrows=18141, names=COL_NAMES)
    df_dm = pd.read_excel(DATA_FOLDER + '/dm_TB.xlsx', usecols=3, nrows=116, names=COL_NAMES)
    df_num = pd.read_excel(DATA_FOLDER + '/Cpnum_TB.xlsx', usecols=3, nrows=120, names=COL_NAMES)


    # Adjusted Number by Volume
    df_num_adj = df_num.copy()
    df_num_adj.iloc[:,1:] = df_num_adj.iloc[:,1:] / VOLUME_MEASURED_QICPIC_ML

    # Total Number of Particles
    df_num_total = df_num_adj.copy()
    V_aux = df_num_total['time'].map(lambda x: linear_filling_equation(x))
    df_num_total.iloc[:,1:] = df_num_total.iloc[:,1:].multiply(V_aux, axis=0)
    d = {
        'ph': df_condutivity,
        'mean-d': df_dm,
        'num-measured': df_num,
        'num-total': df_num_total,
    }
    return d

def load_data_from_exp_i_as_numpy(i: int):
    d = load_data()
    d_conv = {key: item[['time', '{}'.format(i)]].dropna().values for key,item in d.items()}
    # d_conv = {key: item[['time', '{}'.format(i)]].values for key,item in d.items()}
    # for key, val in d_conv.items():
    #     i_cut = np.where(val[:,0] > tcut)[0][0]
    #     d_conv[key] = val[i_cut:, :]

    return d_conv

def remove_initial_points_and_normalize_data(d: dict, tcut_low, tcut_upp):
    for key, val in d.items():
        i_cut_l = np.where(val[:,0] >= tcut_low)[0][0]
        i_cut_u = np.where(val[:,0] > tcut_upp)[0][0]
        d[key] = np.hstack((val[i_cut_l:i_cut_u, [0]] - val[i_cut_l, 0], val[i_cut_l:i_cut_u, [1]]))
    # return d

# def clear_initial_points(d, t_cut = 18.15):
#     d_conv = {key: item[['time', '{}'.format(i)]].values for key,item in d.items()}



#     return

def linear_filling_equation(t: 'min'):
    if  t <= FINAL_ADDITION_TIME:
        V = INITIAL_VOLUME + VOLUMETRIC_FLOW_IN * t
    else:
        V = INITIAL_VOLUME + VOLUMETRIC_FLOW_IN * FINAL_ADDITION_TIME
    return V

if __name__ == "__main__":
    d = load_data_from_exp_i_as_numpy(2)


    pass
