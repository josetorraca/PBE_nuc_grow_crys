import numpy as np
import numba

@numba.njit()
def root_finding_newton(fun, J, x, eps, max_iter, args):
    """
    Solve nonlinear system fun(x)=0 by Newton's method.
    J is the Jacobian of fun(x). Both fun(x) and J must be functions of x.
    At input, x holds the start value. The iteration continues
    until ||F|| < eps.
    """
    F_value = fun(x, args)
    F_norm = np.linalg.norm(F_value, 2)  # l2 norm of vector
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter < max_iter:
        delta = np.linalg.solve(J(x, args), -F_value)
        x = x + delta
        F_value = fun(x, args)
        F_norm = np.linalg.norm(F_value, 2)
        iteration_counter += 1

    # Here, either a solution is found, or too many iterations
    if abs(F_norm) > eps:
        iteration_counter = -1
        raise ValueError('Maximum iteration reached in newton root finding!')
    return x, iteration_counter

@numba.njit()
def numeric_jacobian(fun, x, diff_eps, args):
    J = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        x1 = x.copy()
        x2 = x.copy()
        x1[i] += diff_eps
        x2[i] -= diff_eps
        f1 = fun(x1, args)
        f2 = fun(x2, args)
        J[:, i] = (f1 - f2) / (2 * diff_eps)

    return J

def create_jacobian(fun):

    @numba.njit()
    def numba_J(x, args):
        return numeric_jacobian(fun, x, 1e-8, args)
    return numba_J



# Testing:
@numba.njit()
def F(x, args):
    return np.array(
        [x[0]**2 - x[1] + x[0]*np.cos(args[0]*x[0]),
            x[0]*x[1] + np.exp(-x[1]) - x[0]**(-1)])

@numba.njit()
def J(x, args):
    return np.array(
        [[2*x[0] + np.cos(args[0]*x[0]) - args[0]*x[0]*np.sin(args[0]*x[0]), -1],
            [x[1] + x[0]**(-2), x[0] - np.exp(-x[1])]])

if __name__ == "__main__":

    expected = np.array([1.0, 0.0])
    tol = 1e-4
    x_guess = np.array([2.0, -1.0])
    args = (np.pi,)

    J_num = numeric_jacobian(F, x_guess, 1e-8, args)
    J_ext = J(x_guess, args)
    J_numba = create_jacobian(F)

    x, n = root_finding_newton(F, J_numba, x_guess, 0.0001, 100, args)
    print(n, x)
    error_norm = np.linalg.norm(expected - x, ord=2)
    assert error_norm < tol, 'norm of error =%g' % error_norm
    print('norm of error =%g' % error_norm)
