import numpy as np
# from matplotlib import pyplot as plt
# from scipy import integrate
# import time
import numba

#TODO: Jit has to be done in the outter function!
# Thus, there is no need to JIT option in those functions!

## ---------------------------------
## NUMBER RIGHT HAND SIDE EQUATIONS
## ---------------------------------

def create_rhs_pbe(aggr_func = None, NinOut=False, jit=False):

    if aggr_func is not None and jit:
        aggr_func = numba.jit(aggr_func, nopython=True)

    if jit:
        delta_kron_ = numba.jit(delta_kron, nopython=True)
    else:
        delta_kron_ = delta_kron

    # @numba.njit
    def rhs_pbe_numbers_aggr_base(x, N, rhs_Ni, Aggr_extras):
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
                            eta = (x[i-1] - nu)/(x[i-1]-x[i])
                        else: # (nu >= x[i]):
                            eta = (xiplus1 - nu)/(xiplus1-x[i])
                        aux1 = (1.0-1.0/2.0*delta_kron_(j,k))*eta
                        qjk = aggr_func(x[j], x[k], Aggr_extras)
                        t1 = aux1*qjk*N[j]*N[k]

                    term += t1 #it is set to zero at k iteration
            rhs_Ni[i] = term

        # Second Term in Eq. 29:
        for i in np.arange(0, Npts):
            term = 0.0
            for k in np.arange(1, Npts):
                t2 = aggr_func(x[i],x[k], Aggr_extras)*N[k]
                term += t2
            rhs_Ni[i] += -N[i] * term

        return

    if jit:
        rhs_pbe_numbers_aggr_base = numba.njit(rhs_pbe_numbers_aggr_base)

   # @numba.njit
    def rhs_pbe_numbers_inout(x, N, rhs_Ni, B0, NinOut):
        rhs_Ni[0] += B0
        rhs_Ni += NinOut
        return

    if jit:
        numba.njit(rhs_pbe_numbers_inout)

    #@numba.njit
    def rhs_pbe_numbers_aggr(x, N, rhs_Ni, B0, Aggr_extras):
        """
        Assumes that rhs_Ni is initialized at Zero!
        """
        if aggr_func(0,0, Aggr_extras) != -1:
            rhs_pbe_numbers_aggr_base(x, N, rhs_Ni, Aggr_extras)
        rhs_Ni[0] += B0
        return

    if jit:
        numba.njit(rhs_pbe_numbers_aggr)

   # @numba.njit
    def rhs_pbe_numbers_agg_inout(x, N, rhs_Ni, B0, NinOut, Aggr_extras):
        if aggr_func(0,0, Aggr_extras) != -1:
            rhs_pbe_numbers_aggr_base(x, N, rhs_Ni, Aggr_extras)
        rhs_Ni[0] += B0
        rhs_Ni += NinOut
        return

    if jit:
        numba.njit(rhs_pbe_numbers_agg_inout)

   #@numba.njit
    def rhs_pbe_numbers_nucl(x, N, rhs_Ni, B0):
        """
        Assumes that rhs_Ni is initialized at Zero!
        """
        rhs_Ni[0] += B0
        return

    if jit:
        numba.njit(rhs_pbe_numbers_nucl)

    if aggr_func is None and NinOut is False:
        ret_func = rhs_pbe_numbers_nucl
    elif aggr_func is None:
        ret_func = rhs_pbe_numbers_inout
    elif NinOut is False:
        ret_func = rhs_pbe_numbers_aggr
    else:
        ret_func = rhs_pbe_numbers_agg_inout

    # if jit:
    #     ret_func = numba.njit(ret_func)
    # Only the returned function is not compiled!

    return ret_func

## ---------------------------------
## GRID MOVEMENT EQUATIONS
## ---------------------------------
def create_pbe_grid_movement(as_array = False, jit = False):

    def pbe_grid_const_G_half(x, rhs_x, G):
        rhs_x[0] = G/2.0
        rhs_x[1:] = G
        return

    def pbe_grid_half(x, rhs_x, G):
        rhs_x[0] = G[0]/2.0
        rhs_x[1:] = G[1:]
        return

    if as_array:
        ret_func = pbe_grid_half
    else:
        ret_func = pbe_grid_const_G_half

    if jit:
        numba.njit(ret_func)

    return ret_func


## ---------------------------------
## DRIVERS FOR PBE SOLUTION
## ---------------------------------




## ---------------------------------
## UTILS:
## ---------------------------------
def delta_kron(i,j):
    """Just a simple auxiliary function for computing kronecker delta"""
    if i == j:
        return 1
    else:
        return 0
