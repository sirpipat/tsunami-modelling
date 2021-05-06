"""@package Grid.py
Documentation for Grid Class (ABC)
"""

import numpy as np

class Grid(object):
    """
    Documentation for Grid class
    """
    
    """
    Representation
    
    _xmin
    _xmax
    _ndim       number of dimensions
    _ngrid      number of used grid cells
    _nghost     number of ghost cells appended to both ends
    _x          cell positions
    _u          values at cell positions
    """
    def __init__(self, xmin, xmax, ngrid, nghost):
        pass
        
    def set_grid(self, xmin, xmax, ngrid, nghost):
        pass
    
    def set_value(self, u, option = 'all'):
        pass
        
    def get_grid(self, option = 'all', shift = 0):
        pass
    
    def get_value(self, option = 'all', shift = 0):
        pass
    
    def get_ndims(self):
        return self._ndims
    
    def get_first_grid(self):
        return self._nghost
    
    def get_last_grid(self):
        return self._nghost + self._ngrid