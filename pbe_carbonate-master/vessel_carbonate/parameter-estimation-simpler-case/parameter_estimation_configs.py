import copy

CONFIG_PRE_SIM_V0 = {
    'Npts': 51,
    'nt': 301,
    'lmin': (1.0)**3, #(mu)**3 [mu diameter to mum3] #FAKING LARGER SIZE SINCE NOT INCLUDING GROWTH
    'lmax': (1e2)**3, #1e4
    'tf': 200.0,
    'theta': {
        'kb': 5.97528,
        'sigma_b': 7.84134114,
        'min_b_factor': 0.266593731,
        'kg': 0.156515,  # 1e1, #1e-3, #USING GROWTH BUT NOT ADDING CELLS
        'kaggr': 0.02079273,
        'sigma_a': 58.3172646,
        'min_a_factor': 1.3547e-04,
        },
    'bounds': {
        'kb': [1.0, 60.0],
        'sigma_b': [5.0, 60.0],
        'min_b_factor': [0.0, 0.75],
        'kg': [0.0, 30.0],
        'kaggr': [0.0, 1.0],
        'sigma_a': [20.0, 70.0],
        'min_a_factor': [0.00, 0.75],
    },
    'flag-estimate': {
        'kb': True,
        'sigma_b': True,
        'min_b_factor': True,
        'kg': True,
        'kaggr': False,
        'sigma_a': False,
        'min_a_factor': False,
    }
}

# CONFIG_PRE_SIM_V1 = {
#     'Npts': 51,
#     'nt': 301,
#     'lmin': (1.0)**3, #(mu)**3 [mu diameter to mum3] #FAKING LARGER SIZE SINCE NOT INCLUDING GROWTH
#     'lmax': (1e2)**3, #1e4
#     'tf': 200.0,
#     'theta': {
#         'kb': 5.01371,
#         'sigma_b': 20.794137,
#         'min_b_factor': 0.232783,
#         'kg': 17.983119,  # 1e1, #1e-3, #USING GROWTH BUT NOT ADDING CELLS
#         'kaggr': 0.02079273,
#         'sigma_a': 58.3172646,
#         'min_a_factor': 1.3547e-04,
#         },
#     'bounds': {
#         'kb': [1.0, 60.0],
#         'sigma_b': [5.0, 60.0],
#         'min_b_factor': [0.0, 0.75],
#         'kg': [0.0, 30.0],
#         'kaggr': [0.0, 1.0],
#         'sigma_a': [20.0, 70.0],
#         'min_a_factor': [0.00, 0.75],
#     },
#     'flag-estimate': {
#         'kb': False,
#         'sigma_b': False,
#         'min_b_factor': False,
#         'kg': False,
#         'kaggr': False,
#         'sigma_a': False,
#         'min_a_factor': False,
#     }
# }

def generate_config_v01_aggregation_as_b_spline(config_ref):


    guess_kaggr = [
        0.00710613,
        0.01006669,
        0.00258859,
        0.00541095,
        0.02003922,
        0.05028381,
        0.01081968,
        1.5512e-04,
        0.00169209,
        0.01140761
    ]
    config = copy.deepcopy(config_ref)
    npts_bspline = 10
    config['spline-coeffs-size'] = npts_bspline
    config['flag-estimate']['kaggr'] = False
    config['flag-estimate']['sigma_a'] = False
    config['flag-estimate']['min_a_factor'] = False
    for i in range(npts_bspline):
        config['theta']['kaggr{}'.format(i)] = guess_kaggr[i]
        config['bounds']['kaggr{}'.format(i)] = [0.0, 1.0]
        config['flag-estimate']['kaggr{}'.format(i)] = True

    return config

if __name__ == "__main__":
    generate_config_v01_aggregation_as_b_spline(CONFIG_PRE_SIM_V0)

#  Former:
        # 'kb': 10.0,
        # 'sigma_b': 30.0,
        # 'min_b_factor': 1e-1,
        # 'kg': 10.0,  # 1e1, #1e-3, #USING GROWTH BUT NOT ADDING CELLS
        # 'kaggr': 20.0,
        # 'sigma_a': 50.0,
        # 'min_a_factor': 5.0,

## Note que está zerando agregacao no final praticamente, porem agregacao fica forte devido a sua curva, testar outra curva?

# 1000 fevals estimation with Nelder:
    # sigma_b:       21.0737294 (init= 30)
    # min_a_factor:  1.3547e-04 (init=0.75)
    # kaggr:         0.02079273 (init=1)
    # sigma_a:       58.3172646 (init=50)
    # kg:            18.0528499 (init=10)
    # kb:            6.37879629 (init=10)
    # min_b_factor:  0.23204331

# Idea: Estimation for linear-piecewise trajectories? Search for linda book on dynamics?

#04/02/2019 - Estimation with B-spline for aggregation
# Report saved in Analysis/Reports/bspline-aggre-decay-nucl-04-02.html
# Iteration = 690[[Fit Statistics]]
#     # fitting method   = Nelder-Mead
#     # function evals   = 689
#     # data points      = 1
#     # variables        = 14
#     chi-square         = 121992.039
#     reduced chi-square = 121992.039
#     Akaike info crit   = 39.7117111
#     Bayesian info crit = 11.7117111
# [[Variables]]
#     kb:            5.02768403 (init = 6.378796)
#     sigma_b:       15.4578858 (init = 21.07373)
#     min_b_factor:  0.38309546 (init = 0.2320433)
#     kg:            17.0953546 (init = 18.05285)
#     kaggr:         0.02079273 (fixed)
#     sigma_a:       58.31726 (fixed)
#     min_a_factor:  0.00013547 (fixed)
#     kaggr0:        1.3910e-04 (init = 0.02)
#     kaggr1:        0.01318632 (init = 0.02)
#     kaggr2:        0.01375253 (init = 0.02)
#     kaggr3:        0.00564992 (init = 0.02)
#     kaggr4:        0.01339338 (init = 0.02)
#     kaggr5:        0.02455551 (init = 0.02)
#     kaggr6:        0.01729478 (init = 0.02)
#     kaggr7:        0.00303656 (init = 0.02)
#     kaggr8:        0.00335025 (init = 0.02)
#     kaggr9:        0.00300579 (init = 0.02)
# Elapsed time = 349.3424042224884 min

#05/02/2019 - Estimation with B-spline for aggregation and COBYLA
# Report saved in Analysis/Reports/bspline-aggre-decay-nucl-05-02-0840.html
# Iteration = 801[[Fit Statistics]]
#     # fitting method   = COBYLA
#     # function evals   = 800
#     # data points      = 1
#     # variables        = 14
#     chi-square         = 111498.719
#     reduced chi-square = 111498.719
#     Akaike info crit   = 39.6217684
#     Bayesian info crit = 11.6217684
# [[Variables]]
#     kb:            5.05778334 (init = 6.378796)
#     sigma_b:       13.6922688 (init = 21.07373)
#     min_b_factor:  0.28525387 (init = 0.2320433)
#     kg:            16.5810520 (init = 18.05285)
#     kaggr:         0.02079273 (fixed)
#     sigma_a:       58.31726 (fixed)
#     min_a_factor:  0.00013547 (fixed)
#     kaggr0:        9.6968e-05 (init = 0.0001391)
#     kaggr1:        0.01343628 (init = 0.01318632)
#     kaggr2:        0.01002965 (init = 0.01375253)
#     kaggr3:        0.00441196 (init = 0.01318632)
#     kaggr4:        0.01096159 (init = 0.01339338)
#     kaggr5:        0.02126112 (init = 0.02455551)
#     kaggr6:        0.01165331 (init = 0.01729478)
#     kaggr7:        6.2299e-04 (init = 0.00303656)
#     kaggr8:        0.00294249 (init = 0.00335025)
#     kaggr9:        5.4227e-06 (init = 0.00300579)
# Elapsed time = 370.9336193641027 min

#05/02/2019 - Estimation with B-spline for aggregation Equal weight sum_mu0/70.0 + sum_mean_size/275.0
# Report saved in Analysis/Reports/bspline-aggre-decay-nucl-05-02-1600-weigh-equal.html
# Iteration = 294[[Fit Statistics]]
#     # fitting method   = Nelder-Mead
#     # function evals   = 293
#     # data points      = 1
#     # variables        = 14
#     chi-square         = 4.32690919
#     reduced chi-square = 4.32690919
#     Akaike info crit   = 29.4648535
#     Bayesian info crit = 1.46485347
# [[Variables]]
#     kb:            5.01371967 (init = 6.378796)
#     sigma_b:       20.7941374 (init = 21.07373)
#     min_b_factor:  0.23278347 (init = 0.2320433)
#     kg:            17.9831172 (init = 18.05285)
#     kaggr:         0.02079273 (fixed)
#     sigma_a:       58.31726 (fixed)
#     min_a_factor:  0.00013547 (fixed)
#     kaggr0:        2.7371e-04 (init = 9.6968e-05)
#     kaggr1:        0.01278016 (init = 0.01343628)
#     kaggr2:        0.02201907 (init = 0.01002965)
#     kaggr3:        0.01036892 (init = 0.00441196)
#     kaggr4:        0.01100872 (init = 0.01096159)
#     kaggr5:        0.01306915 (init = 0.02126112)
#     kaggr6:        0.00773777 (init = 0.01165331)
#     kaggr7:        4.8625e-04 (init = 0.00062299)
#     kaggr8:        0.00242796 (init = 0.00294249)
#     kaggr9:        9.4051e-06 (init = 5.4227e-06)
# Elapsed time = 154.00855251550675 min

#05/02/2019 - Estimation with B-spline for aggregation unequal weight sum_mu0/70.0 + sum_mean_size;
# x1 + x2 > 100 ? return 0.0 in aggregation
# Estimation only aggrgation kernels coefficients
# FAILED! aggregation higher than size max in mesh!
# Report saved in Analysis/Reports/xxx.html
# Iteration = 1064[[Fit Statistics]]
#     # fitting method   = Nelder-Mead
#     # function evals   = 1063
#     # data points      = 1
#     # variables        = 14
#     chi-square         = 4590.65036
#     reduced chi-square = 4590.65036
#     Akaike info crit   = 36.4317770
#     Bayesian info crit = 8.43177698
# [[Variables]]
#     kb:            4.58538464 (init = 6.378796)
#     sigma_b:       10.2166977 (init = 21.07373)
#     min_b_factor:  0.30014367 (init = 0.2320433)
#     kg:            16.7445119 (init = 18.05285)
#     kaggr:         0.02079273 (fixed)
#     sigma_a:       58.31726 (fixed)
#     min_a_factor:  0.00013547 (fixed)
#     kaggr0:        7.9317e-04 (init = 0.00027371)
#     kaggr1:        0.00965561 (init = 0.01278016)
#     kaggr2:        0.00487594 (init = 0.02201907)
#     kaggr3:        0.00706746 (init = 0.01036892)
#     kaggr4:        0.02996010 (init = 0.01100872)
#     kaggr5:        0.13208512 (init = 0.01306915)
#     kaggr6:        0.59272258 (init = 0.00773777)
#     kaggr7:        0.04426110 (init = 0.00048625)
#     kaggr8:        2.4077e-06 (init = 0.00242796)
#     kaggr9:        0.00191244 (init = 9.4051e-06)
# Elapsed time = 723.4098168452581 min


#05/02/2019 - Estimation with B-spline for aggregation
# weight: sum_mu0/(5.0**2) + sum_mean_size/(4.0**2)
# Iteration = 1606[[Fit Statistics]]
#     # fitting method   = Nelder-Mead
#     # function evals   = 1605
#     # data points      = 1
#     # variables        = 14
#     chi-square         = 807.090690
#     reduced chi-square = 807.090690
#     Akaike info crit   = 34.6934360
#     Bayesian info crit = 6.69343604
# [[Variables]]
#     kb:            5.97528981 (init = 6.378796)
#     sigma_b:       7.84134110 (init = 21.07373)
#     min_b_factor:  0.26659370 (init = 0.2320433)
#     kg:            0.15651592 (init = 0.5)
#     kaggr:         0.02079273 (fixed)
#     sigma_a:       58.31726 (fixed)
#     min_a_factor:  0.00013547 (fixed)
#     kaggr0:        0.00710613 (init = 0.00027371)
#     kaggr1:        0.01006669 (init = 0.01278016)
#     kaggr2:        0.00258859 (init = 0.02201907)
#     kaggr3:        0.00541095 (init = 0.01036892)
#     kaggr4:        0.02003922 (init = 0.01100872)
#     kaggr5:        0.05028381 (init = 0.01306915)
#     kaggr6:        0.01081968 (init = 0.00773777)
#     kaggr7:        1.5512e-04 (init = 0.00048625)
#     kaggr8:        0.00169209 (init = 0.00242796)
#     kaggr9:        0.01140761 (init = 9.4051e-06)
# Elapsed time = 241.1527060985565 min


#05/02/2019 - Estimation with B-spline for aggregation
# weight: sum_mu0/(5.0**2) + sum_mean_size/(4.0**2)
# Report saved at: bspline-aggre-decay-nucl-08-02-0800-weigh-unequal-better-estimation.html
# Iteration = 2501[[Fit Statistics]]
#     # fitting method   = COBYLA
#     # function evals   = 2500
#     # data points      = 1
#     # variables        = 14
#     chi-square         = 569.455742
#     reduced chi-square = 569.455742
#     Akaike info crit   = 34.3446811
#     Bayesian info crit = 6.34468107
# [[Variables]]
#     kb:            5.32515848 (init = 5.97528)
#     sigma_b:       6.87702820 (init = 7.841341)
#     min_b_factor:  0.28780743 (init = 0.2665937)
#     kg:            0.15473462 (init = 0.156515)
#     kaggr:         0.02079273 (fixed)
#     sigma_a:       58.31726 (fixed)
#     min_a_factor:  0.00013547 (fixed)
#     kaggr0:        0.00594052 (init = 0.00710613)
#     kaggr1:        0.00773895 (init = 0.01006669)
#     kaggr2:        0.00196840 (init = 0.00258859)
#     kaggr3:        0.00858791 (init = 0.00541095)
#     kaggr4:        0.00855946 (init = 0.02003922)
#     kaggr5:        0.04344752 (init = 0.05028381)
#     kaggr6:        0.01490301 (init = 0.01081968)
#     kaggr7:        3.6720e-05 (init = 0.00015512)
#     kaggr8:        8.2478e-04 (init = 0.00169209)
#     kaggr9:        0.00907569 (init = 0.01140761)
# Elapsed time = 363.60723414023715 min
