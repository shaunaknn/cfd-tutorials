import numpy as np
import matplotlib.pyplot as plt

def nonlinearconvection(nx,nt):
    # nx = no of grid points, nt = no of time steps
    l = 2 # domain length
    dx = l/(nx-1) # delta x
    dt = nt/1000 # delta t

    u = np.ones(nx) # solution array
    u[int(.5/dx):int(1/dx+1)] = 2 # initial condition

    un = np.ones(nx) # temp array

    for n in range(nt): # solving in time
        un = u.copy()
        u[1:] = un[1:]-un[1:]*dt/dx*(un[1:]-un[:-1])

    plt.plot(np.linspace(0,l,nx),u) # plot after nt
    plt.show()

nonlinearconvection(41,20)