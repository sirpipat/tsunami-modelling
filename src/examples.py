import numpy as np
from ShallowWaterSim import *

'''
This file contain workflow of using ShallowWaterSim objects as well as 
example code to produce the figures in the report/presentation. 
Read this before using the ShallowWaterSim object.

Workflow:

# import requried libraries
import numpy as np
from Grid1DCartesian import *
from interpolation_scheme import *
from limiters import *
from ShallowWaterSim import *

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
bc = 'non-reflecting'   # or 'periodic', 'reflecting'

# gravity acceleration 
g = 9.8

# initialize WaterWaveSim object
ocean = ShallowWaterSim(x, b, s, u, g, bc)

# Alternatively, it can be initialized from an input file
ocean = ShallowWaterSim.fromfilename('directory/filename')

# CFL number (0 < CFL < 1) , could be changed for other time-stepping scheme (TODO)
CFL = 0.1

# preventing the simulation to run eternally when dt is too small
MAX_STEPS = 100000

# time to take the snapshot
times = np.array([1, 2, 3])

# specify the output directory
output_directory = './output/'        # default value

# specify the problem type
problem = 'passive'                   # or 'shallowwater'

# specify the interpolation scheme
interp_scheme = fifth_order_upwind

# specify the limiter
limiter = interp_vanLeer3

# run the simulator
hGrid, uGrid = ocean.simulate(CFL, MAX_STEPS, times, problem, \
                              output_directory, interp_scheme, \
                              limiter)
'''
###################################################################################
# Example 1: convergence test on simple advection problem with SuHu5 limiter,
#            5th order upwind, CFL = 0.5
###################################################################################
def simple_convergence():
    # the number of grid cells
    # remove the last few elements if the runs take too long
    Ns = np.array([50,100,200,400,800,1600,3200,6400,12800, 25600])
    # RMS errors
    err_RMSs = np.zeros(np.shape(Ns))
    dxs = np.zeros(np.shape(Ns))
    for ii in range(np.size(Ns)):
        err, dx = residue_simple(Ns[ii])
        err_RMS = np.sqrt(np.sum(err ** 2) / np.size(err))
        err_RMSs[ii] = err_RMS
        dxs[ii] = dx
    
    # plot RMS error on log-log plot
    plt.loglog(dxs, err_RMSs, 'o-')
    plt.grid()
    plt.xlabel('dx', size = 12)
    plt.ylabel('RMS error', size = 12)
    plt.title('RMS error, CFL = 0.1')
    plt.savefig('../RMS_ERROR_5u_SuHu5_CFL_0p5.pdf', dpi = 300)
    
    # plot empirical rate of convergence (based on RMS error)
    p = np.log2((err_RMSs[:-2] - err_RMSs[1:-1]) / (err_RMSs[1:-1] - err_RMSs[2:]))
    plt.semilogx(dxs[:-2],p, '-o')
    plt.grid()
    plt.xlabel('dx', size = 12)
    plt.ylabel('empirical p', size = 12)
    plt.title('Empirical rate of convergence, CFL = 0.1')
    plt.savefig('../Empirical_p_5u_SuHu5_CFL_0p5.pdf', dpi = 300)
    
    # Example for reading files and plot the data (integration of water height)
    # the volume should be conserved!
    path = '../outputs_convergence/output_simple_5u_SuHu5_CFL_0p5/files/'
    files = np.sort(os.listdir(path=path))
    h = np.zeros(np.shape(files[1:]))
    I = np.zeros(np.shape(files[1:]))
    for ii, file in enumerate(files[1:]):
        dd = read_data(path + file)
        x = dd[3]
        h[ii] = x[1] - x[0]
        I[ii] = np.sum(dd[5]) * (x[1] - x[0])
    plt.semilogx(h, I, '-o')
    plt.grid()
    plt.xlabel('dx', size = 12)
    plt.ylabel('V', size = 12)
    plt.savefig('../Volume_5u_SuHu5_CFL_0p5.pdf', dpi = 300)

def residue_simple(N=100):
    # define grid points
    xmin = 0
    xmax = 1
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2
    dx = (xmax - xmin) / N

    # bathymetry
    b = -1 * (x ** 0)

    # initial water surface
    #xxx, xxx, s = init_smooth_cell(N, init_smooth)
    s = 1 * np.exp(- (x-0.5)**2 / (2 * 0.05**2)) + 0
    #where = np.where(np.abs(x - 0.7) <= 0.1)
    #s[where] = 1

    # initial speed
    u = x * 0 + 1

    # boundary condition
    bc = 'periodic'

    # gravity acceleration 
    g = 0

    # initialize WaterWaveSim object
    tank = ShallowWaterSim(x, b, s, u, g, bc)

    CFL = 0.5

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 150000

    output_directory = '../outputs_convergence/output_simple_5u_SuHu5_CFL_0.5/'

    times = np.array([1])

    problem = 'passive'
    interp_scheme = fifth_order_upwind
    limiter = interp_SuHu5

    hGrid, uGrid = tank.simulate(CFL, MAX_STEPS, times, problem, \
                                 output_directory, interp_scheme, \
                                 limiter)
    s_1cycle = hGrid.get_value('grid') + b
    err = s_1cycle - s
    return err, dx

###################################################################################
# Example 2: replication of Figure 13.1 from LeVeque (2002) using with SuHu5 
#            limiter, 5th order upwind on shallow water equations
###################################################################################
def run_LeVeque():
    # define grid points
    xmin = -5
    xmax = 5
    N = 1000
    x = np.linspace(xmin, xmax, N+1)
    x = (x[:-1] + x[1:]) / 2

    # bathymetry
    b = -0 * (x ** 0)

    # initial water surface
    s = 0.4 * np.exp(- x**2 / 0.35**2) + 1

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

    # specify the output directory
    output_directory = '../output_LeVeque2002_5u_SuHu5_1000/'

    times = np.array([0.5, 1, 2, 3])

    problem = 'shallowwater'
    interp_scheme = fifth_order_upwind
    limiter = interp_SuHu5

    hGrid, uGrid = tank.simulate(CFL, MAX_STEPS, times, problem, \
                                 output_directory, interp_scheme, \
                                 limiter)
    
    # reads the output files and makes a plot
    path = '../output_LeVeque2002_5u_SuHu5_1000/files/'
    files = np.sort(os.listdir(path=path))
    
    fig = plt.figure(figsize = [12,16])
    for ii, file in enumerate(files):
        dd = read_data(path + file)
        # plot h(x)
        ax1 = fig.add_subplot(5,2,2*ii+1)
        ax1.plot(dd[3],dd[5])
        ax1.grid()
        ax1.set_xlim((-5, 5))
        ax1.set_ylim((0.5, 1.5))
        ax1.set_title('h at t = %.1f' % dd[1], size = 14)

        # plot h(x)u(x)
        ax2 = fig.add_subplot(5,2,2*ii+2)
        ax2.plot(dd[3],dd[5]*dd[6])
        ax2.grid()
        ax2.set_xlim((-5, 5))
        ax2.set_ylim((-0.5, 0.5))
        ax2.set_title('hu at t = %.1f' % dd[1], size = 14)
    plt.savefig('../LeVeque2002_u5_SuHu5_1000.pdf', dpi = 150)
    

    
###################################################################################
# Example 3: shallow water wave over a hill ... compared with PyClaw example
#            http://www.clawpack.org/gallery/pyclaw/gallery/sill.html
#            You may try to install PyClaw to compare the result
###################################################################################
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

    output_directory = '../output_sill_5u_SuHu5/'

    times = np.arange(0.1,1.1,0.1)

    problem = 'shallowwater'
    interp_scheme = fifth_order_upwind
    limiter = interp_SuHu5

    hGrid, uGrid = tank.simulate(CFL, MAX_STEPS, times, problem, \
                                 output_directory, interp_scheme, \
                                 limiter)
    
###################################################################################
# Example 4: hypothetical tsunami propagation up the continental slope 
###################################################################################
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
    
    problem = 'shallowwater'
    interp_scheme = fifth_order_upwind
    limiter = interp_SuHu5

    ocean.simulate(CFL, MAX_STEPS, times, problem, \
                   output_directory, interp_scheme, \
                   limiter)

###################################################################################
# Example 5: test on Green's Law -- wave going on a sloped bottom
###################################################################################
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
    ocean = ShallowWaterSim(x, b, s, u, g, bc)

    CFL = 0.1

    # preventing the simulation to run eternally when dt is too small
    MAX_STEPS = 420000

    output_directory = '../output_Green_Law/'

    times = np.arange(120,4201,120)
    
    problem = 'shallowwater'
    interp_scheme = fifth_order_upwind
    limiter = interp_SuHu5

    ocean.simulate(CFL, MAX_STEPS, times, problem, \
                   output_directory, interp_scheme, \
                   limiter)