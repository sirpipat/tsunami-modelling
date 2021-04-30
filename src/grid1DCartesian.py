"""@package Grid1DCartesian.py
Documentation for Grid1DCartesian Class
"""

import numpy as np
from grid1D import *

class Grid1DCartesian(Grid1D):
    """
    Documentation for Grid1DCartesian class
    """
    _type = "Cartesian"
    
    def __init__(self, xmin, xmax, ngrid, nghost):
        self._xmin = xmin * np.array([1])
        self._xmax = xmax * np.array([1])
        self._ngrid = ngrid  * np.array([1])
        self._nghost = nghost * np.array([1])
        
        dx = (xmax - xmin) / ngrid
        x = np.linspace(xmin - dx * nghost, \
                        xmax + dx * nghost, \
                        ngrid + 2 * nghost + 1)
        self._x = (x[0:-1] + x[1:]) / 2
        self._u = np.zeros(np.shape(self._x))
    
    def set_grid(self, xmin, xmax, ngrid, nghost):
        self._xmin = xmin * np.array([1])
        self._xmax = xmax * np.array([1])
        self._ngrid = ngrid  * np.array([1])
        self._nghost = nghost * np.array([1])
        
        dx = (xmax - xmin) / ngrid
        x = np.linspace(xmin - dx * nghost, \
                        xmax + dx * nghost, \
                        ngrid + 2 * nghost + 1)
        self._x = (x[0:-1] + x[1:]) / 2
        self._u = np.zeros(np.shape(self._x))
        
    def get_grid(self, option = 'all'):
        if option == 'grid':
            a = self.get_first_grid()
            b = self.get_last_grid()
            return np.copy(self._x[a[0]:b[0]])
        else:
            return np.copy(self._x)
        
    def get_value(self, option = 'all'):
        if option == 'grid':
            a = self.get_first_grid()
            b = self.get_last_grid()
            return np.copy(self._u[a[0]:b[0]])
        else:
            return np.copy(self._u)