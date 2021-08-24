import numpy as np
from scikits.odes.odeint import odeint

#data of the oscillator
k = 4.0
m = 1.0
#initial position and speed data on t=0, x[0] = u, x[1] = \dot{u}, xp = \dot{x}
initx = [1, 0.1]

def rhseqn(t, x, xdot):
    """ we create rhs equations for the problem"""
    xdot[0] = x[1]
    xdot[1] = - k/m * x[0]


tout = np.linspace(0, 1, 11)
initial_values = np.array(initx)
output = odeint(rhseqn, tout, initial_values)
print(output.values.y)
