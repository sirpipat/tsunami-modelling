from Grid1DCartesian import *

# Applies boundary conditions for finite volume method
#
# INPUT
# u          a Grid1DCartesian grid
# bc         boundary condition
#            'periodic'
#            'reflecting_symmetric'
#            'reflecting_antisymmetric'
def apply_bc(u, bc):
    if bc == 'periodic':
        apply_periodic(u)
    elif bc == 'reflecting_symmetric':
        apply_reflecting_symmetric(u)
    elif bc == 'reflecting_antisymmetric':
        apply_reflecting_antisymmetric(u)
    else:
        raise Exception('%s is not set to an acceptable choice. Check spelling.' % bc)

def apply_periodic(u):
    Nghost = u._nghost[0]
    Ngrid = u._ngrid[0]
    for ii in range(Nghost):
        # ghost cells on the left
        u._u[ii] = u._u[Ngrid + ii]
        # ghost cells on the right
        u._u[Nghost + Ngrid + ii] = u._u[Nghost + ii]
        
def apply_reflecting_symmetric(u):
    Nghost = u._nghost[0]
    Ngrid = u._ngrid[0]
    for ii in range(Nghost):
        # ghost cells on the left
        u._u[ii] = u._u[Nghost]
        # ghost cells on the right
        u._u[Ngrid + ii] = u._u[Ngrid - 1]
        
def apply_reflecting_antisymmetric(u):
    Nghost = u._nghost[0]
    Ngrid = u._ngrid[0]
    for ii in range(Nghost):
        # ghost cells on the left
        u._u[ii] = - u._u[Nghost]
        # ghost cells on the right
        u._u[Ngrid + ii] = - u._u[Ngrid - 1]