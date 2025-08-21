import numpy as np
import matplotlib.pyplot as plt

nx = 121 # no of grid points
nt = 30 # no of time steps
l = 2 # domain length
dx = l/(nx-1) # delta x
#dt = nt/1000 # delta t

u = -1*np.ones(nx) # solution array
u[int(.5/dx):int(1/dx+1)] = -2 # initial condition

cfl = 0.9
c = np.max(np.abs(u))
dt = cfl*dx/(c+1e-6) # delta t, DrZGan approach

def nonlinearconvection(u,nx,nt,dx,dt):
    un = np.ones(nx) # temp soln array
    for n in range(nt): # solving in time
        un = u.copy()
        u[1:] = un[1:]-un[1:]*dt/dx*(un[1:]-un[:-1])

    return u

def nonlinearconvectionup(u,nx,nt,dx,dt): # flux splitting upwind scheme
    un = np.ones(nx) # temp soln array

    for n in range(nt): # solving in time
        un = u.copy()
        
        uplus = np.maximum(un/(np.abs(un)+1e-6),0.0)
        uminus = np.maximum(-un/(np.abs(un)+1e-6),0.0)

        # F_i+1/2 = u_i*a_i^+ + u_i+1*a_i+1^-, a is characteristic speed
        F = un[:-1]*uplus[:-1] + un[1:]*uminus[1:]

        u[1:-1] = un[1:-1]-un[1:-1]*dt/dx*(F[1:] - F[:-1])

    return u

plt.plot(np.linspace(0,l,nx),u,label = 'initial solution') # plot after nt
plt.legend()
plt.show()

nonlinearconvection(u,nx,nt,dx,dt)
plt.plot(np.linspace(0,l,nx),u,label = 'backward difference') # plot after nt
plt.legend()
plt.show()

u = -1*np.ones(nx) # solution array for 
u[int(.5/dx):int(1/dx+1)] = -2 # initial condition

nonlinearconvectionup(u,nx,nt,dx,dt)
plt.plot(np.linspace(0,l,nx),u,label = 'upwind') # plot after upwind scheme
plt.legend()
plt.show()