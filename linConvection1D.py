import numpy as np
import matplotlib.pyplot as plt

def linearconvection(nx,nt):
    # nx = no of grid points, nt = no of time steps
    l = 2 # domain length
    cfl = 0.5 # CFL number
    c = 1 # wave speed
    dx = l/(nx-1) # delta x
    dt = cfl*dx/c # delta t

    u = np.ones(nx) # solution array
    u[int(.5/dx):int(1/dx+1)] = 2 # initial condition

    un = np.ones(nx) # temp array

    for n in range(nt): # solving in time
        un = u.copy()
        u[1:] = un[1:]-c*dt/dx*(un[1:]-un[:-1])

    plt.plot(np.linspace(0,l,nx),u) # plot after nt
    plt.show()

linearconvection(41,20)