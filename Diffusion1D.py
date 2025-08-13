import numpy as np
import matplotlib.pyplot as plt

def diffusion(nx,nt):
    #nx = no of grid points, nt = no of time steps
    l = 2
    dx = l/(nx-1) #delta x
    nu = 0.3 #viscosity
    sigma = 0.2 #tuning parameter
    dt = sigma*dx**2/nu #delta t

    u = np.ones(nx) #solution array
    u[int(.5/dx):int(1/dx+1)] = 2 #initial condition

    un = np.ones(nx) #temp array

    for n in range(nt): #solving in time
        un = u.copy()
        u[1:-1] = un[1:-1]+nu*dt/dx**2*(un[2:]-2*un[1:-1]+un[:-2])

    plt.plot(np.linspace(0,l,nx),u) #plot after nt
    plt.show()

diffusion(41,20)
