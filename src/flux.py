# Represents a flux function for finite volume method
#
# Function signature
#
# def flux_function(x, u, c):
#     # do something
#     return f
#
# INPUT:
# x      centered locations of cells
# u      interested values at x positions
# c      advection speed in x-direction
#
# OUTPUT:
# f      flux at cell boundary on the right
#        f[j] is flux at x[j+1/2]
#
# TODO:
# make the function able to handling non-periodic BCs

import numpy as np
from minfunc import *

# 1st order upwind differencing
# equivalent to Godunov's Algorithm with piecewise constant
def first_order_upwind(x, u, c):
    f = c * u
    return f

# 2nd order centered differencing
def second_order_centered(x, u, c):
    f = c * (u + np.roll(u, -1)) / 2
    return f

# 2nd order upwind differencing
def second_order_upwind(x, u, c):
    f = c * (u + 0.5 * (u - np.roll(u, 1)))
    return f

# 3rd order upwind biased differencing
def third_order_upwind(x, u, c):
    u_left = np.roll(u, 1)
    u_right = np.roll(u, -1)
    f = c * (u + 1/4 * (u_right - u_left) + 1/12 * (u_right - 2 * u + u_left))
    return f

# 4th order centered differencing
def fourth_order_centered(x, u, c):
    f = c * (7/12 * (u + np.roll(u, -1)) - \
             1/12 * (np.roll(u, 1) + np.roll(u, -2)))
    return f

# intermediate complexity advection 5-point stencil with limiters
def ICA5(x, u, c):
    # Eq 2.12 Suresh and Huynh (1997)
    alpha = 4
    # limiters
    u1 = fourth_order_centered(x, u, c) / c
    uMPL = u + minmod(np.roll(u, -1) - u, alpha * (u - np.roll(u, 1)))
    uL = median(u1, u, uMPL)
    uMPR = np.roll(u, -1) + minmod(u - np.roll(u, -1), \
                                   alpha * (np.roll(u, -1) - np.roll(u, -2)))
    uR = median(u1, np.roll(u, -1), uMPR)
    
    # Rusanov's flux
    f = c/2 * (uL + uR) - np.abs(c)/2 * (uR - uL)
    
    return f