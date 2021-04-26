# Represents a time-stepping function for finite volume method
#
# Function signature
#
# def timestep_function(x, u, flux_method , c):
#     # do something
#     return u_next
#
# INPUT:
# x              centered locations of cells
# u              interested values at x positions
# flux_method    a flux method to compute flux with function signature:
#                f = flux_method(x, u, c):
#                See flux.py for more detail
# c              advection speed in x-direction
# dt             time step size
#
# OUTPUT:
# u_next         values at the next time step
#
# TODO:
# make the function able to handling non-periodic BCs

import numpy as np
from flux import *
from minfunc import *

# forward euler
def forward_euler(x, u, flux_method, c, dt):
    # here assume uniform grid
    dx = x[1] - x[0]
    
    # compute flux
    f = flux_method(x, u, c)
    u_next = u - (dt / dx) * (f - np.roll(f, 1))
    return u_next

# Suresh and Huynh (1997)
def RK3(x, u, flux_method, c, dt):
    # here assume uniform grid
    dx = x[1] - x[0]
    
    # CFL Number
    # cfl = c * dt / dx
    
    w0 = u
    f0 = flux_method(x, w0, c)
    w1 = w0 + (dt / dx) * (- f0 + np.roll(f0, 1))
    f1 = flux_method(x, w1, c)
    w2 = 3/4 * w0 + 1/4 * (w1 + (dt / dx) * (- f1 + np.roll(f1, 1)))
    f2 = flux_method(x, w2, c)
    w3 = 1/3 * w0 + 2/3 * (w2 + (dt / dx) * (- f2 + np.roll(f2, 1)))
    u_next = w3
    return u_next