# Represents a interpolation function for finite volume method
#
# Function signature
#
# def interpolation_scheme(u):
#     # do something
#     return ubL
#
# INPUT:
# u      interested values at x positions    (Grid1DCartesian)
#        Assumes x is equally spaced.
#
# OUTPUT:
# ubL    values at the left of x_{j+1/2} boundary
#
# If you want to computevalues at the right of x_{j+1/2} boundary,
# supply the flipped version of u instead.
#
# ubL = interpolation_scheme(x, u)
# ubR = interpolation_scheme(x, flip(u))

import numpy as np
from minfunc import *

# 1st order upwind differencing
# equivalent to Godunov's Algorithm with piecewise constant
def first_order_upwind(u):
    ubL = u[-1,0]
    return ubL

# 2nd order centered differencing
def second_order_centered(u):
    ubL = (u[-1,0] + u[0,1]) / 2
    return ubL

# 2nd order upwind differencing
def second_order_upwind(u):
    ubL = u[-1,0] + 0.5 * (u[-1,0] - u[-2,-1])
    return ubL

# 3rd order upwind biased differencing
def third_order_upwind(u):
    u_left = u[-2,-1]
    u_right = u[0,1]
    ubL = u[-1,0] + 1/4 * (u_right - u_left) + 1/12 * (u_right - 2 * u[-1,0] + u_left)
    return ubL

# 4th order centered differencing
def fourth_order_centered(u):
    ubL = 7/12 * (u[-1,0] + u[0,1]) - 1/12 * (u[-2,-1] + u[1,2])
    return ubL

# 5th order upwind biased differencing
def fifth_order_upwind(u):
    ubL = 1./60 * (2 * u[-3,-2] - 13 * u[-2,-1] + 47 * u[-1,0] + 27 * u[0,1] - 3 * u[1,2])
    return ubL