from grid1DCartesian import Grid1DCartesian
import numpy as np

N = 2
NGHOST = 1
XMIN = -1
XMAX = 1
STEP_SIZE = (XMAX - XMIN) / N
PRECISION = 1e-5

def is_equal(a, b, p):
    return np.abs(a - b) < p    
    
def make_1D():   
    grid = Grid1DCartesian(XMIN, XMAX, N, NGHOST)
    return grid

def set_value(grid):
    x = grid.get_grid('grid')
    u = x
    
    # define values at the ghost cells as the value of the nearest grid cell
    ghost_indices = np.zeros(NGHOST, dtype = int)
    add_indices = np.concatenate((ghost_indices, ghost_indices + N))
    ghost_values = np.ones(NGHOST)
    add_values = np.concatenate((ghost_values * u[0], ghost_values * u[-1]))
    u_all = np.insert(u, add_indices, add_values)
    
    grid.set_value(u_all)
    return grid

def test_Grid1DCartesian():
    grid = make_1D()
    
    
    # test initialization
    assert is_equal(grid._xmin, XMIN, PRECISION)
    assert is_equal(grid._xmax, XMAX, PRECISION)
    assert grid._ngrid == N
    assert grid._nghost == NGHOST
    x = grid._x
    assert is_equal(x[0], XMIN + STEP_SIZE/2 - NGHOST * STEP_SIZE, PRECISION)
    assert is_equal(x[NGHOST], XMIN + STEP_SIZE/2, PRECISION)
    assert is_equal(x[NGHOST + N], XMAX + STEP_SIZE/2, PRECISION)
    u = grid._u
    assert is_equal(u[0], 0, PRECISION)
    assert is_equal(u[NGHOST], 0, PRECISION)
    assert is_equal(u[NGHOST + N], 0, PRECISION)
    
    # test set_value
    grid = set_value(grid)
    u = grid._u
    assert is_equal(u[0], XMIN + STEP_SIZE/2, PRECISION)
    assert is_equal(u[NGHOST], XMIN + STEP_SIZE/2, PRECISION)
    assert is_equal(u[NGHOST + N - 1], XMAX - STEP_SIZE/2, PRECISION)
    assert is_equal(u[NGHOST + N], XMAX - STEP_SIZE/2, PRECISION)
    assert is_equal(u[-1], XMAX - STEP_SIZE/2, PRECISION)
    
    # test get_grid
    xgrid = grid.get_grid('grid')
    assert np.size(xgrid) == N
    assert is_equal(xgrid[0], x[NGHOST], PRECISION)
    assert is_equal(xgrid[-1], x[NGHOST + N -1], PRECISION)
    
    xall = grid.get_grid('all')
    assert np.size(xall) == N + 2 * NGHOST
    assert is_equal(xall[0], x[0], PRECISION)
    assert is_equal(xall[NGHOST], x[NGHOST], PRECISION)
    assert is_equal(xall[NGHOST + N - 1], x[NGHOST + N -1], PRECISION)
    assert is_equal(xall[-1], x[-1], PRECISION)
    
    
    # test get_value
    ugrid = grid.get_value('grid')
    assert np.size(ugrid) == N
    assert is_equal(ugrid[0], u[NGHOST], PRECISION)
    assert is_equal(ugrid[-1], u[NGHOST + N -1], PRECISION)
    
    uall = grid.get_value('all')
    assert np.size(uall) == N + 2 * NGHOST
    assert is_equal(uall[0], u[0], PRECISION)
    assert is_equal(uall[NGHOST], u[NGHOST], PRECISION)
    assert is_equal(uall[NGHOST + N - 1], u[NGHOST + N -1], PRECISION)
    assert is_equal(uall[-1], u[-1], PRECISION)
    
    # test get_first_grid
    assert grid.get_first_grid() == NGHOST
    # test get_last_grid
    assert grid.get_last_grid() == NGHOST + N