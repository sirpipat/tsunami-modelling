# Represents a limiter for finite volume method
#
# Function signature
#
# def limiter(u, alpha, interp_func):
#     # do something
#     return ubL
#
# INPUT:
# u              interested values at x positions    (Grid1DCartesian)
#                Assumes x is equally spaced.
# alpha          the value alpha for MC limiter
# interp_func    function to interpolate u_{j+1/2}^L
#
# OUTPUT:
# ubL            limited values at the left of x_{j+1/2} boundary  (numpy.ndarray)
#
# If you want to computevalues at the right of x_{j+1/2} boundary,
# supply the flipped version of u instead.
#
# ubL = limiter(u, alpha, interp_func)
# ubR = np.flip(limiter(u.get_flip(), alpha, interp_func))

from interpolation_scheme import *
from Grid1DCartesian import *
from minfunc import *

# Second order accurate van Leer (1974) limiter
def interp_vanLeer3(u, alpha = 4, interp_func = third_order_upwind):
    ubL = interp_func(u)
    
    # Apply the Suresh-Huynh form of the van Leer
    # monotonicity-preserving limiters:
    u_MP = u[-1,0] + minmod(u[0,1] - u[-1,0], alpha * (u[-1,0] - u[-2,-1]))
    ubL = median(ubL, u[-1,0], u_MP)
    return ubL

# Fifth order accurate Suresh-Huynh (1997) limiter
def interp_SuHu5(u, alpha = 4, interp_func = fifth_order_upwind):
    ubL = interp_func(u)
    
    # calculate the local 2nd derivative at several points:
    d0 = u[0,1] - 2 * u[-1,0] + u[-2,-1]            # S&H Eq. 2.19
    d1 = u[1,2] - 2 * u[0,1] + u[-1,0]
    dm1 = u[-1,0] - 2 * u[-2,-1] + u[-3,-2]
    
    # apply limiters to the 2nd derivatives:
    d_M4   = minmod4(d0, d1, 4*d0-d1, 4*d1-d0)      # S&H Eq. 2.27
    d_M4m1 = minmod4(dm1, d0, 4*dm1-d0, 4*d0-dm1)
    
    # van Leer upper limit for monotonicity:
    u_UL = u[-1,0] + alpha * (u[-1,0] - u[-2,-1])   # S&H Eq. 2.8

    # allow interpolation beyond a smooth extrumum:
    u_MD = 0.5 * (u[-1,0] + u[0,1]) - 0.5 * d_M4    # S&H Eq. 2.28

    # extension of van Leer's upper limit near a smooth extremum:
    # "LC" = "Large Curvature"
    u_LC = u[-1,0] + 0.5 * (u[-1,0] - u[-2,-1]) + 4./3. * d_M4m1  # S&H Eq. 2.29
    
    # The following logic determines the interval I[u_min,u_max]
    # that is the intersection of the interval 
    # I[un[j],un[j+1],u_MD] and
    # I[un[j],u_UL,u_LC].
    #
    u_min = maxmax(minmin(u[-1,0], u[0,1], u_MD), minmin(u[-1,0], u_UL, u_LC))  # S&H Eq. 2.24a
    u_max = minmin(maxmax(u[-1,0], u[0,1], u_MD), maxmax(u[-1,0], u_UL, u_LC))  # S&H Eq. 2.24b

    # # If u_LC didn't modify limits, then we would have:
    # u_min = max(min(un[j],un[j+1],u_MD),min(un[j],u_UL))
    # u_max = min(max(un[j],un[j+1],u_MD),max(un[j],u_UL))
    # # This has very little effect.  So u_LC plays a crucial role

    ubL=median(ubL, u_min, u_max)    # S&H Eq. 2.26
    
    return ubL