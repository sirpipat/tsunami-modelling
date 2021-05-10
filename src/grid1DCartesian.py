"""@package Grid1DCartesian.py
Documentation for Grid1DCartesian Class
"""

from copy import deepcopy
import numpy as np
from Grid1D import *

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
        
    def set_value(self, u, option = 'all'):
        if option == 'grid':
            assert np.size(u) == self._ngrid
            a = self.get_first_grid()
            b = self.get_last_grid()
            self._u[a[0]:b[0]] = np.copy(u)
        else:
            assert np.size(u) == self._ngrid + 2 * self._nghost
            self._u = np.copy(u)
        
    def get_grid(self, option = 'all', shift = 0):
        if option == 'grid':
            a = self.get_first_grid()
            b = self.get_last_grid()
            return np.copy(self._x[(a[0]+shift):(b[0]+shift)])
        else:
            return np.copy(self._x)
        
    def get_value(self, option = 'all', shift = 0):
        if option == 'grid':
            a = self.get_first_grid()
            b = self.get_last_grid()
            return np.copy(self._u[(a[0]+shift):(b[0]+shift)])
        else:
            return np.copy(self._u)
        
###################################################
# Magic methods
###################################################

    def __abs__(self):
        r = deepcopy(self)
        r.set_value(np.abs(np.copy(self._u)), 'all')
        return r
    
    def __add__(self, other):
        return self.__radd__(other)
        
    def __radd__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(self_val + val, 'all')
        return r
    
    def __sub__(self, other):
        return self.__rsub__(other) * (-1)
    
    def __rsub__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(val - self_val, 'all')
        return r
    
    def __mul__(self, other):
        return self.__rmul__(other)
    
    def __rmul__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(self_val * val, 'all')
        return r
    
    def __truediv__(self, other):
        return self.__truediv__(other) ** (-1)
    
    def __rtruediv__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(val / self_val, 'all')
        return r
    
    def __pow__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(self_val ** val, 'all')
        return r
    
    # usage:
    # self[a]   returns self._u[nghost+a : nghost+ngrid+a]
    # self[a,b] returns self._u[nghost+a : nghost+ngrid+b]
    def __getitem__(self, items):
        if type(items) is int:
            return self.get_value('grid', items)
        else: 
            assert type(items) is tuple
            assert len(items) == 2
            a = self.get_first_grid()
            b = self.get_last_grid()
            return np.copy(self._u[(a[0]+items[0]):(b[0]+items[1])])
            
    def _check_other_input(self, other):
        if type(other) is Grid1DCartesian:
            assert other._xmin == self._xmin
            assert other._xmax == self._xmax
            assert other._ngrid == self._ngrid
            assert other._nghost == self._nghost
            val = other.get_value('all')
        elif type(other) is np.ndarray:
            assert np.size(other) == self._ngrid + 2 * self._nghost
            val = other
        else:
            val = other
        
        return val