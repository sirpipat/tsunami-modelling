"""@package ShallowWaterSim.py
Documentation for ShallowWaterSim Class
"""

from copy import deepcopy
import os  # operating system / file utilities
import numpy as np
import matplotlib.pyplot as plt
from minfunc import *
from Grid1DCartesian import *

'''
Representation

_bGrid      bathymetry
_hGrid      water depth (from bottom to surface)
_uGrid      water flow speed
_g          gravitational acceleration
_bc         boundary conditions

'''

VERY_SMALL_NUMBER = 1e-7
class ShallowWaterSim(object):
    def __init__(self, x, b, s, u, g, bc):
        xmin = x[0] - (x[1] - x[0]) / 2
        xmax = x[-1] + (x[1] - x[0]) / 2
        Ngrid = np.size(x)
        if bc == 'periodic':
            Nghost = 0
        else:
            Nghost = 2
        
        # pad the values at the boundary for ghost cells
        b_padL = np.ones(Nghost) * b[0]
        b_padR = np.ones(Nghost) * b[-1]
        b = np.concatenate((b_padL, b, b_padR))
        
        s_padL = np.ones(Nghost) * s[0]
        s_padR = np.ones(Nghost) * s[-1]
        s = np.concatenate((s_padL, s, s_padR))
        
        u_padL = np.ones(Nghost) * u[0]
        u_padR = np.ones(Nghost) * u[-1]
        u = np.concatenate((u_padL, u, u_padR))
        
        self._bc = bc
        self._bGrid = Grid1DCartesian(xmin, xmax, Ngrid, Nghost)
        self._bGrid.set_value(b, 'all')
        
        h = maxmax(s - b, VERY_SMALL_NUMBER)
        
        self._hGrid = Grid1DCartesian(xmin, xmax, Ngrid, Nghost)
        self._hGrid.set_value(h, 'all')
        self._uGrid = Grid1DCartesian(xmin, xmax, Ngrid, Nghost)
        self._uGrid.set_value(u, 'all')
        self._g = g
        
    # "overloaded" constructor
    def fromfilename(cls, filename):
        data = np.genfromtxt(filename, skip_header = 2)
        x = data[:,0]
        b = data[:,1]
        s = data[:,2]
        u = data[:,3]
        
        fin = open(filename, 'r')
        g = float(fin.readline())
        bc = fin.readline()[0:-1]
        fin.close()
        
        return cls(x, b, s, u, g, bc)
    
    def simulate(self, cfl = 0.1, \
                 max_steps = 100000, \
                 times = np.array([60,120,180]), \
                 output_dir = 'outputs'):
        # create a subdirectory to hold outputs
        # (if it doesn't exist already)
        figures_dir = output_dir + "/figs"
        files_dir = output_dir + "/files"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(figures_dir, exist_ok=True)
        os.makedirs(files_dir, exist_ok=True)
        
        # set initial state and trackers
        steps = 0
        count = 0
        t = 0
        q = np.array([deepcopy(self._hGrid), deepcopy(self._uGrid)])
        x = self._bGrid.get_grid('all')
        dx = x[1] - x[0]
        
        # save initial states
        self._plot_result(q, self._bGrid, self._g, t, steps, figures_dir)
        self._save_result(q, self._bGrid, self._g, t, steps, files_dir)
        
        while t < times[-1]:
            # count the steps
            steps = steps + 1

            # determine appropriate timestep
            dt = cfl * dx / np.amax(np.abs(q[1][0,0]) + (self._g * q[0][0,0]) ** 0.5)
            # tell whether to take a snapshot
            if t + dt >= times[count]:
                take_snapshot = True
                count = count + 1
            else:
                take_snapshot = False
                
            # advect
            q, t = self._advect_1step(x, q, self._bGrid, self._g, t, dt, self._bc)
            
            # take snapshots
            if take_snapshot:
                self._plot_result(q, self._bGrid, self._g, t, steps, figures_dir)
                self._save_result(q, self._bGrid, self._g, t, steps, files_dir)

            # report every 10000 steps
            if steps % 10000 == 0:
                print('step %d completed: t = %.7f s, dt = %.7f s\n' % (steps, t, dt))

            if steps >= max_steps:
                print('The number of steps reach the maximum limit. Exit.\n')
                break
    
    # advect 1 step
    def _advect_1step(self, x, q, b, g, t0, dt, bc = 'periodic'):
        # here assumes uniform grid
        dx = x[1] - x[0]

        if bc == 'periodic':
            w0 = q
            f0 = self._flux(x, w0, b, g)
            df0 = - f0 + np.roll(f0, 1, axis = 1)
            df0Grid = deepcopy(q)
            df0Grid[0].set_value(df0[0,:], 'all')
            df0Grid[1].set_value(df0[1,:], 'all')
            w1 = w0 + (dt / dx) * df0Grid
            f1 = self._flux(x, w1, b, g)
            df1 = - f1 + np.roll(f1, 1, axis = 1)
            df1Grid = deepcopy(q)
            df1Grid[0].set_value(df1[0,:], 'all')
            df1Grid[1].set_value(df1[1,:], 'all')
            w2 = 3/4 * w0 + 1/4 * (w1 + (dt / dx) * df1Grid)
            f2 = self._flux(x, w2, b, g)
            df2 = - f2 + np.roll(f2, 1, axis = 1)
            df2Grid = deepcopy(q)
            df2Grid[0].set_value(df2[0,:], 'all')
            df2Grid[1].set_value(df2[1,:], 'all')
            w3 = 1/3 * w0 + 2/3 * (w2 + (dt / dx) * df2Grid)
            q_next = w3
        else:
            w0 = q
            f0 = self._flux(x, w0, b, g, bc)
            # flux differentiation
            df0 = - f0[:,1:] + f0[:,0:-1]
            # convert flux differentiation to Grid array of the same shape as q
            nghost = b.get_first_grid()[0]
            pad = np.zeros((2, nghost))
            df0 = np.concatenate((pad, df0, pad), axis = 1)
            df0Grid = deepcopy(q)
            df0Grid[0].set_value(df0[0,:], 'all')
            df0Grid[1].set_value(df0[1,:], 'all')
            # update value
            w1 = w0 + (dt / dx) * df0Grid

            f1 = self._flux(x, w1, b, g, bc)
            df1 = - f1[:,1:] + f1[:,0:-1]
            # convert flux differentiation to Grid array of the same shape as q
            df1 = np.concatenate((pad, df1, pad), axis = 1)
            df1Grid = deepcopy(q)
            df1Grid[0].set_value(df1[0,:], 'all')
            df1Grid[1].set_value(df1[1,:], 'all')
            # update value
            w2 = 3/4 * w0 + 1/4 * (w1 + (dt / dx) * df1Grid)

            f2 = self._flux(x, w2, b, g, bc)
            df2 = - f2[:,1:] + f2[:,0:-1]
            # convert flux differentiation to Grid array of the same shape as q
            df2 = np.concatenate((pad, df2, pad), axis = 1)
            df2Grid = deepcopy(q)
            df2Grid[0].set_value(df2[0,:], 'all')
            df2Grid[1].set_value(df2[1,:], 'all')
            # update value
            w3 = 1/3 * w0 + 2/3 * (w2 + (dt / dx) * df2Grid)
            q_next = w3

        t = t0 + dt

        return q_next, t
    
    def _flux(self, x, q, b, g, bc = 'periodic'):
        if bc == 'periodic':
            q0 = q[0].get_value('grid')
            q1 = q[1].get_value('grid')
            c_abs = np.abs(q1) + np.sqrt(g * q0)
            c_max = maxmax(c_abs, np.roll(c_abs, -1)) * 1.1
        else:
            c_abs = np.abs(q[1]) + (g * q[0]) ** 0.5
            c_max = maxmax(c_abs[0,1], c_abs[-1,0]) * 1.1

        if bc == 'periodic':
            bb = b.get_value('grid')
            hL, hR = self._find_uL_uR(x, q0)
            uL, uR = self._find_uL_uR(x, q1)
            bL, bR = self._find_uL_uR(x, bb)
        else:
            hL, hR = self._find_uL_uR(x, q[0], bc)
            uL, uR = self._find_uL_uR(x, q[1], bc)
            bL, bR = self._find_uL_uR(x, b, bc)

        f_hatL = self._flux_hat(hL, uL, bL, g)
        f_hatR = self._flux_hat(hR, uR, bR, g)

        # Rusanov's method
        f = 1/2 * (f_hatL + f_hatR) - 1/2 * np.abs(c_max) * np.array([hR - hL, uR - uL])
        return f

    # make the function name more descriptive and relevant
    def _find_uL_uR(self, x, u, bc = 'periodic'):
        alpha = 4
        if bc == 'periodic':
            u1 = (7/12 * (u + np.roll(u, -1))) - (1/12 * (np.roll(u, 1) + np.roll(u, -2)))

            uMPL = u + minmod(np.roll(u, -1) - u, alpha * (u - np.roll(u, 1)))
            uL = median(u1, u, uMPL)
            uMPR = np.roll(u, -1) + minmod(u - np.roll(u, -1), \
                                           alpha * (np.roll(u, -1) - np.roll(u, -2)))
            uR = median(u1, np.roll(u, -1), uMPR)
        else:
            u1 = (7/12 * (u[-1,0] + u[0,1])) - (1/12 * (u[-2,-1] + u[1,2]))

            uMPL = u[-1,0] + minmod(u[0,1] - u[-1,0], alpha * (u[-1,0] - u[-2,-1]))
            uL = median(u1, u[-1,0], uMPL)
            uMPR = u[0,1] + minmod(u[-1,0] - u[0,1], alpha * (u[0,1] - u[1,2]))
            uR = median(u1, u[0,1], uMPR)

        return uL, uR
    
    def _flux_hat(self, h, u, b, g):
        f = np.array([u*h, 1/2*(u**2)+g*(h+b)])
        return f
    
    def _plot_result(self, q, bGrid, g, t, steps, savedir):
        fig = plt.figure(figsize = [8,10])
        ax1 = fig.add_subplot(4,1,1)
        ax2 = fig.add_subplot(4,1,2)
        ax3 = fig.add_subplot(4,1,3)
        ax4 = fig.add_subplot(4,1,4)
        
        # plot bottom
        ax1.set_title('time = %.2f s, step = %d' % (t, steps))
        ax1.plot(bGrid.get_grid(), bGrid.get_value())
        ax1.set_xlim((bGrid._xmin, bGrid._xmax))
        ax1.grid()
        ax1.set_ylabel('bottom (m)')
        
        # plot water surface
        ax2.plot((q[0]+bGrid).get_grid(), (q[0]+bGrid).get_value())
        ax2.set_xlim((bGrid._xmin, bGrid._xmax))
        ax2.grid()
        ax2.set_ylabel('water surface (m)')
        
        # plot water flowing speed
        ax3.plot((q[1]).get_grid(), (q[1]).get_value())
        ax3.set_xlim((bGrid._xmin, bGrid._xmax))
        ax3.grid()
        ax3.set_ylabel('water speed (m/s)')

        # plot wave propagation speed
        ax4.plot(bGrid.get_grid(), np.sqrt(q[0].get_value() * g))
        ax4.set_xlim((bGrid._xmin, bGrid._xmax))
        ax4.grid()
        ax4.set_xlabel('position (m)')
        ax4.set_ylabel('wave speed (m/s)')
        
        # save figure
        savename = savedir + '/plot_step_%06d.pdf' % steps
        plt.savefig(savename, dpi=300)
        plt.close()
        
    def _save_result(self, q, bGrid, g, t, steps, savedir):
        headertxt = 'steps = %6d\nt     = %.7f s\ng     = %.7f m/s^2' % (steps, t, g)
        x = bGrid.get_grid('grid')
        b = bGrid.get_value('grid')
        s = (q[0]+bGrid).get_value('grid')
        u = q[1].get_value('grid')
        
        data = np.array([x, b, s, u])
        data = data.T
        
        fname = savedir + '/data_%06d.txt' % steps
        np.savetxt(fname, data, fmt = '%.7e', header = headertxt)