import numpy as np

# Returns the lowest value among x, y, and z (optional) for scalar input or x_i, y_i, and z_i
# for any i for vector input. x, y, and z can be either scalars or vectors of the same
# length. If the input is a combination of vectors or vectors, all will be broadcasted
# to vectors.
def minmin(x, y, z = None):
    m = 1/2 * (x + y) - 1/2 * np.sign(x - y) * (x - y)
    if z is None:
        return m
    else:
        n = 1/2 * (m + z) - 1/2 * np.sign(m - z) * (m - z)
        return n
        
# Returns the value clostest to zero if x, y, and z (optional) have the same sign, zero otherwise.
# If x, y, or z are vectors, it will returns the value clostest to zero among x_i, y_i, and z_i,
# for any i.
def minmod(x, y, z = None):
    if z is None:
        return 0.5 * (np.sign(x) + np.sign(y)) * minmin(np.abs(x), np.abs(y))
    else:
        return 0.25 * (np.sign(x) + np.sign(y)) * np.abs(np.sign(y) + np.sign(z)) * \
               minmin(np.abs(x), np.abs(y), np.abs(z))

# Returns the median value among x, y, and z (optional) or x_i, y_i, and z_i for vector input.
def median(x, y, z):
    return x + minmod(y - x, z - x)

# Returns the highest value among x, y, and z (optional) for scalar input or x_i, y_i, and z_i
# for any i for vector input. x, y, and z can be either scalars or vectors of the same
# length. If the input is a combination of vectors or vectors, all will be broadcasted
# to vectors.
def maxmax(x, y, z = None):
    m = 1/2 * (x + y) + 1/2 * np.sign(x - y) * (x - y)
    if z is None:
        return m
    else:
        n = 1/2 * (m + z) + 1/2 * np.sign(m - z) * (m - z)
        return n