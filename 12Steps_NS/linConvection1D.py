import numpy as np
import matplotlib.pyplot as plt

l = 2 # domain length
nx = 41 # no of grid points
nt = 20 # no of timesteps
cfl = 0.5 # CFL number
c = 1 # wave speed
dx = l/(nx-1) # delta x
dt = cfl*dx/(abs(c)+1e-6) # delta t

u = -1*np.ones(nx) # solution array
u[int(.5/dx):int(1/dx+1)] = 2 # initial condition

def linearconvection(u,nt,dx,dt):
    un = np.ones_like(u)
    for n in range(nt): # solving in time
        un = u.copy()
        u[1:] = un[1:]-c*dt/dx*(un[1:]-un[:-1])

    return u

plt.plot(np.linspace(0,l,nx),u,label='initial solution') # initial solution
plt.legend()
plt.show()

linearconvection(u,nt,dx,dt)
plt.plot(np.linspace(0,l,nx),u,label='standard upwind') # plot after nt
plt.legend()
plt.show()