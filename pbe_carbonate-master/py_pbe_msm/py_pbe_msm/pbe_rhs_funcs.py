import warnings
import numpy as np
import numba
from . import numba_utils

# np.seterr(under='raise', over='raise')
# warnings.filterwarnings('error')

@numba.njit(cache=True)
def numbers_nucl(x, N, rhs_Ni, B0):
    """
    Assumes that rhs_Ni is initialized at Zero!
    """
    rhs_Ni[0] = B0
    return

@numba.njit(cache=True)
def grid_const_G_half(x, rhs_x, G):
    rhs_x[0] = G/2.0
    rhs_x[1:] = G
    return

@numba.njit(cache=True)
def grid_G_first_half(x, rhs_x, G):
    rhs_x[0] = G[0]/2.0
    rhs_x[1:] = G[1:]
    return

@numba.njit(cache=True)
def rhs_pbe_numbers_agg_inout(x, N, rhs_Ni, B0, NinOut, mdl):
    if mdl.calc_Aggr(0,0, 0, 0) != -1:
        rhs_pbe_numbers_aggr_base(x, N, rhs_Ni, mdl)
    rhs_Ni[0] += B0
    rhs_Ni += NinOut
    return

@numba.njit(cache=True)
# @numba.njit(parallel=True)
def rhs_pbe_numbers_aggr_base(x, N, rhs_Ni, mdl):
    Npts = N.shape[0]

    # First Term in Eq. 29: (i=Npts?)
    # for i in numba.prange(1, Npts): #np.arange(1, Npts): #does not apply for first bin this term
    for i in np.arange(1, Npts): #does not apply for first bin this term
        # if (np.isclose(x[i],x[i-1])):
        if abs(x[i] - x[i-1]) < 1e-30:
            print(-3, i, x[i])
            # print('possible bug1!')
            continue
        if i < Npts - 1:
            if abs(x[i+1] - x[i]) < 1e-30:
                print(-2)
                continue
        if i == Npts - 1:
            xiplus1 = 1e20 #is that right?
        else:
            xiplus1 = x[i+1]
        term = 0.0
        for j in np.arange(0, i + 1):
            if N[j] == 0.0: #CHECK THIS LATER
                continue

            # Testing for Binary Search to get k:
            # k_bn_lb = np.searchsorted(x[0:j+1], x[i - 1] - x[j])
            # k_bn_ub = np.searchsorted(x[k_bn_lb + 1: j+1], xiplus1 - x[j])
            # k_bn_ub = k_bn_lb + k_bn_ub
            # for k_inner in range(k_bn_lb, k_bn_ub):
            #     nu = x[j] + x[k_inner]
            #     if (nu <= x[i]):
            #         eta = (x[i-1] - nu)/(x[i-1]-x[i])
            #     else: # (nu >= x[i]):
            #         eta = (xiplus1 - nu)/(xiplus1-x[i])
            #     aux1 = (1.0-1.0/2.0*numba_utils.delta_kron(j,k_inner))*eta
            #     qjk = mdl.calc_Aggr(x[j], x[k_inner], N[j], N[k_inner])
            #     t1 = aux1*qjk*N[j]*N[k_inner]
            #     term += t1

            # if x[j] < x[i-1] - x[j]: #30s to 50s increase!
            #     continue
            # if i < Npts:
            #     if x[0] > x[i+1] - x[j]:
            #         continue
            for k in np.arange(0, j + 1):
                # if N[k] == 0.0: #CHECK THIS LATER
                if N[k] < 1e-30: #CHECK THIS LATER
                    continue

                t1 = 0.0
                nu = x[j] + x[k]
                # if nu > x[-1]:
                #     print('Outside range!')
                if (nu >= x[i-1]) and (nu <= xiplus1):
                    if (nu <= x[i]):
                        eta = (x[i-1] - nu)/(x[i-1]-x[i])
                    else: # (nu >= x[i]):
                        eta = (xiplus1 - nu)/(xiplus1-x[i])
                    aux1 = (1.0-1.0/2.0*numba_utils.delta_kron(j,k))*eta
                    qjk = mdl.calc_Aggr(x[j], x[k], N[j], N[k])
                    # try:
                    #     t1 = aux1*qjk*N[j]*N[k]
                    # except (Warning, FloatingPointError):
                    #     t1 = 0.0
                    # except (OverflowError):
                    #     a = 1
                    #     print('aow')
                        # print(e)
                    t1 = aux1*qjk*N[j]*N[k]
                # To Improve Speed:
                elif nu > xiplus1:
                    break #break k loop since it is greater than x[i+1]

                term += t1 #it is set to zero at k iteration
        rhs_Ni[i] = term

    # Second Term in Eq. 29:
    # for i in numba.prange(0, Npts):
    for i in np.arange(0, Npts):
        if N[i] == 0.0: #CHECK THIS LATER
            continue
        term = 0.0
        for k in np.arange(0, Npts): #0->1
            t2 = mdl.calc_Aggr(x[i],x[k], N[i], N[k])*N[k]
            term += t2
        rhs_Ni[i] += - N[i] * term
        # try:
        #     rhs_Ni[i] += -N[i] * term
        # except (Warning, FloatingPointError):
        #     rhs_Ni[i] += 0.0

    return

@numba.njit(cache=True)
def rhs_pbe_numbers_aggr_base_ord(x, N, rhs_Ni, mdl, o0 = 0, o1 = 1):
    Npts = N.shape[0]

    # First Term in Eq. 29: (i=Npts?)
    for i in np.arange(1, Npts): #does not apply for first bin this term
        # if (np.isclose(x[i],x[i-1])):
        if abs(x[i] - x[i-1]) < 1e-5:
            continue
        if i == Npts - 1:
            xiplus1 = 1e20 #is that right?
        else:
            xiplus1 = x[i+1]
        term = 0.0
        for j in np.arange(0, i + 1):
            for k in np.arange(0, j + 1):
                t1 = 0.
                nu = x[j] + x[k]
                if (nu >= x[i-1]) and (nu <= xiplus1):
                    if (nu <= x[i]):
                        eta = (nu**o0*x[i-1]**o1 - nu**o1*x[i-1]**o0) \
                            /(x[i]**o0*x[i-1]**o1 - x[i]**o1*x[i-1]**o0)
                    else: # (nu >= x[i]):
                        eta = (nu**o0*xiplus1**o1 - nu**o1*xiplus1**o0) \
                            /(x[i]**o0*xiplus1**o1 - x[i]**o1*xiplus1**o0)
                    aux1 = (1.0-1.0/2.0*numba_utils.delta_kron(j,k))*eta
                    qjk = mdl.calc_Aggr(x[j], x[k])
                    t1 = aux1*qjk*N[j]*N[k]

                term += t1 #it is set to zero at k iteration
        rhs_Ni[i] = term

    # Second Term in Eq. 29:
    for i in np.arange(0, Npts):
        term = 0.0
        for k in np.arange(1, Npts):
            t2 = mdl.calc_Aggr(x[i],x[k])*N[k]
            term += t2
        rhs_Ni[i] += -N[i] * term

    return

