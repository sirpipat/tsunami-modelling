import numpy as np
from ShallowWaterSim import *

'''
Workflow

def run_example():
    # define grid points
    xmin = 0
    xmax = 100
    N = 100
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = somefunction(x)

    # initial water surface
    s = someotherfunction(x)

    # initial speed
    u = anotherfunction(x)

    # boundary condition
    bc = 'non-reflecting'   # or 'periodic', 'reflecting' (TODO)

    # gravity acceleration 
    g = 9.8

    # initialize WaterWaveSim object
    ocean = WaterWaveSim(x, b, s, u, g, bc)

    # Alternatively, it can be initialized from an input file
    ocean = WaterWaveSim.fromfilename('directory/filename')

    # CFL number (0 < CFL < 1)
    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 100000

    output_directory = './output/'

    ocean.simulate(CFL, MAX_STEPS, times, output_directory)
'''

# Figure 13.1 from LeVeque (2002)
def run_LeVeque2002():
    # define grid points
    xmin = -6
    xmax = 6
    N = 240
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = -0 * (x ** 0)

    # initial water surface
    s = 0.5 * np.exp(- x**2 / 0.35**2) + 1

    # initial speed
    u = x * 0

    # boundary condition
    bc = 'periodic'

    # gravity acceleration 
    g = 1

    # initialize WaterWaveSim object
    tank = ShallowWaterSim(x, b, s, u, g, bc)
    
    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 100000

    output_directory = '../output/'

    times = np.array([0.5, 1, 2, 3])

    tank.simulate(CFL, MAX_STEPS, times, output_directory)
    
def run_simple():
    # define grid points
    xmin = -6
    xmax = 6
    N = 240
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = -0 * (x ** 0)

    # initial water surface
    s = 0.5 * np.exp(- x**2 / 0.35**2) + 1

    # initial speed
    u = x * 0 + 1

    # boundary condition
    bc = 'periodic'

    # gravity acceleration 
    g = 0

    # initialize WaterWaveSim object
    tank = ShallowWaterSim(x, b, s, u, g, bc)
    
    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 100000

    output_directory = '../output_simple/'

    times = np.arange(0.5,5.5,0.5)

    tank.simulate(CFL, MAX_STEPS, times, output_directory)
    
# PyClaw Example
def run_sill():
    # define grid points
    xmin = -1
    xmax = 1
    N = 500
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = -1 * (x ** 0)
    where = np.where(np.abs(x) <= 0.92)[0]
    b[where] = -1 + 0.8 * np.exp(-(x[where]/0.2)**2)

    # initial water surface
    s = 0 * x
    s = 0.1 * np.exp(- ((x+0.4)/0.2)**2)

    # initial speed
    u = x * 0

    # boundary condition
    bc = 'non-reflecting'

    # gravity acceleration 
    g = 9.8

    # initialize WaterWaveSim object
    tank = ShallowWaterSim(x, b, s, u, g, bc)
    
    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 100000

    output_directory = '../output_sill/'

    times = np.arange(0.1,1.1,0.1)

    tank.simulate(CFL, MAX_STEPS, times, output_directory)
    
# Hypothetical tsunami running up contiental slope toward the beach
def run_tsunami():
    # define grid points
    xmin = 0
    xmax = 200000      # lenght  = 200 km
    N = 10000          # spacing = 20 m
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = -4000 * (x ** 0)
    where = np.where(x <= 100000)[0]
    b[where] = -400
    where = np.where(x <= 80000)[0]
    b[where] = -200
    where = np.where(x <= 60000)[0]
    b[where] = -100
    where = np.where(x <= 40000)[0]
    b[where] = -50
    where = np.where(x <= 20000)[0]
    b[where] = -20
    where = np.where(x <= 10000)[0]
    b[where] = -1
    # smoothing
    for ii in range(10000):
        v = [1/4, 1/2, 1/4]
        b = np.convolve(b, v, mode = 'same')
        b[0] = -1
        b[-1] = -4000

    # initial water surface
    s = 0 * x
    where = np.where(np.abs(x-120000) <= 20000)[0]
    s[where] = 1 * np.sin(2*np.pi*(x[where]-120000)/80000) ** 2

    # initial speed
    u = x * 0

    # boundary condition
    bc = 'non-reflecting'

    # gravity acceleration 
    g = 9.8

    # initialize WaterWaveSim object
    ocean = ShallowWaterSim(x, b, s, u, g, bc)
    
    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 400000

    output_directory = '../output_tsunami/'

    times = np.arange(120,3961,120)

    ocean.simulate(CFL, MAX_STEPS, times, output_directory)
    
def run_Green_Law():
    # define grid points
    xmin = 0
    xmax = 200000      # length  = 200 km
    N = 10000          # spacing = 20 m
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = -4000 * (x ** 0)
    where = np.where(x <= 100000)[0]
    b[where] = - 1/25 * x[where]
    where = np.where(x <= 25)[0]
    b[where] = -1
    # smoothing
    for ii in range(10):
        v = [1/4, 1/2, 1/4]
        b = np.convolve(b, v, mode = 'same')
        b[0] = -1
        b[-1] = -4000

    # initial water surface
    s = 0 * x
    where = np.where(np.abs(x-150000) <= 20000)[0]
    s[where] = 1 * np.sin(2*np.pi*(x[where]-150000)/40000) ** 2

    # initial speed
    u = x * 0

    # boundary condition
    bc = 'non-reflecting'

    # gravity acceleration 
    g = 9.8

    # initialize WaterWaveSim object
    ocean4 = ShallowWaterSim(x, b, s, u, g, bc)

    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 420000

    output_directory = '../output_Green_Law/'

    times = np.arange(120,4201,120)

    ocean4.simulate(CFL, MAX_STEPS, times, output_directory)