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
        
        # define the grid
        dx = (xmax - xmin) / ngrid
        x = np.linspace(xmin - dx * nghost, \
                        xmax + dx * nghost, \
                        ngrid + 2 * nghost + 1)
        self._x = (x[0:-1] + x[1:]) / 2
        self._u = np.zeros(np.shape(self._x))
    
    """
    Sets the grid and resets the value to zero.
    Please set_value after that.
    """
    def set_grid(self, xmin, xmax, ngrid, nghost):
        self._xmin = xmin * np.array([1])
        self._xmax = xmax * np.array([1])
        self._ngrid = ngrid  * np.array([1])
        self._nghost = nghost * np.array([1])
        
        # define the grid
        dx = (xmax - xmin) / ngrid
        x = np.linspace(xmin - dx * nghost, \
                        xmax + dx * nghost, \
                        ngrid + 2 * nghost + 1)
        self._x = (x[0:-1] + x[1:]) / 2
        self._u = np.zeros(np.shape(self._x))
        
    """
    Sets values at the grid points.
    
    Options:
    - 'all'   set values at all points including ghost cells
    - 'grid'  set values only at cells inside interested domain and not ghost cells
    """
    def set_value(self, u, option = 'all'):
        if option == 'grid':
            if np.size(u) != self._ngrid:
                raise ValueError('Input does not have the same size as the domain\n')
            a = self.get_first_grid()
            b = self.get_last_grid()
            self._u[a[0]:b[0]] = deepcopy(u)
        else:
            if np.size(u) != self._ngrid + 2 * self._nghost:
                raise ValueError('Input does not have the same size as the grid\n')
            self._u = deepcopy(u)
    
    """
    Gets positions of the cell centers.
    
    Options:
    - 'all'   get positions of all points including ghost cells
    - 'grid'  get positions only of cells inside interested domain and not ghost cells
          shift indicate how many cell shift applied to the slicing
    """
    def get_grid(self, option = 'all', shift = 0):
        if option == 'grid':
            a = self.get_first_grid()
            b = self.get_last_grid()
            return deepcopy(self._x[(a[0]+shift):(b[0]+shift)])
        else:
            return deepcopy(self._x)
    
    """
    Gets values at the grid points.
    
    Options:
    - 'all'   get values at all points including ghost cells
    - 'grid'  get values only at cells inside interested domain and not ghost cells
          shift indicate how many cell shift applied to the slicing
    """
    def get_value(self, option = 'all', shift = 0):
        if option == 'grid':
            a = self.get_first_grid()
            b = self.get_last_grid()
            return deepcopy(self._u[(a[0]+shift):(b[0]+shift)])
        else:
            return deepcopy(self._u)
        
    """
    Returns a flipped version of the Grid
    """
    def get_flip(self):
        flipped = deepcopy(self)
        flipped.set_value(np.flip(self.get_value('all')), 'all')
        return flipped
        
###################################################
# Magic methods
###################################################
    """
    Returns a new Grid1DCartesian with the absolute values 
    """
    def __abs__(self):
        r = deepcopy(self)
        r.set_value(np.abs(deepcopy(self._u)), 'all')
        return r
    
    """
    Returns a new Grid1DCartesian with value added
    """
    def __add__(self, other):
        return self.__radd__(other)
    
    """
    Allows a Grid1DCartesian to be added from the left
    """
    def __radd__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(self_val + val, 'all')
        return r
    
    """
    Returns a new Grid1DCartesian with value subtracted
    """
    def __sub__(self, other):
        return self.__rsub__(other) * (-1)
    
    """
    Allows a Grid1DCartesian to be subtracted from the left
    """
    def __rsub__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(val - self_val, 'all')
        return r
    
    """
    Returns a new Grid1DCartesian with value multiplied
    """
    def __mul__(self, other):
        return self.__rmul__(other)
    
    """
    Allows a Grid1DCartesian to be multiplied from the left
    """
    def __rmul__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(self_val * val, 'all')
        return r
    
    """
    Returns a new Grid1DCartesian with value divided
    """
    def __truediv__(self, other):
        return self.__truediv__(other) ** (-1)
    
    """
    Allows a Grid1DCartesian to divide a value on the left
    """
    def __rtruediv__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(val / self_val, 'all')
        return r
    
    """
    Returns a new Grid1DCartesian with value powered by a float
    """
    def __pow__(self, other):
        # check input type
        val = self._check_other_input(other)
        
        # construct a result object
        r = deepcopy(self)
        self_val = self.get_value('all')
        r.set_value(self_val ** val, 'all')
        return r
    
    """
    Return a numpy array starting at a-th cell in the domain of interest
    and ending at b-th cell counting past the end of the domain of interest
    
    usage:
    self[a]   returns self._u[nghost+a : nghost+ngrid+a]
    self[a,b] returns self._u[nghost+a : nghost+ngrid+b]
    """
    def __getitem__(self, items):
        if type(items) is int:
            return self.get_value('grid', items)
        else: 
            assert type(items) is tuple
            assert len(items) == 2
            a = self.get_first_grid()
            b = self.get_last_grid()
            return deepcopy(self._u[(a[0]+items[0]):(b[0]+items[1])])
            
    """
    Checks if the input value is a Grid1DCartesian, a numpy array, or a float
    and handles accordingly.
    """
    def _check_other_input(self, other):
        if type(other) is Grid1DCartesian:
            if other._xmin != self._xmin or other._xmax != self._xmax or other._ngrid != self._ngrid or other._nghost != self._nghost:
                raise ValueError('Two grids do not have the same grid points.\n')
            val = other.get_value('all')
        elif type(other) is np.ndarray:
            if np.size(other) != self._ngrid + 2 * self._nghost:
                raise ValueError('Input does not have the same size as the grid\n')
            val = other
        else:
            val = other
        
        return val