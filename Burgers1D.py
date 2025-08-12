import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from sympy import init_printing
from sympy.utilities.lambdify import lambdify

x,nu,t = sp.symbols('x nu t') #symbolic variables

phi = sp.exp(-(x-4*t)**2/(4*nu*(t+1)))+sp.exp(-(x-4*t-2*sp.pi)**2/(4*nu*(t+1)))
phiprime = phi.diff(x)
u = -2*nu*(phiprime/phi) + 4 #symbolic calculation of u using phi and phiprime

ufunc = lambdify((t,x,nu),u,modules='numpy') #turns symbolic equation to callable function

def burgers(nx,nt):
    #nx,nt = no of grid points, no of time steps
    dx = 2*np.pi/(nx-1)
    nu = 0.07
    dt = dx*nu

    x = np.linspace(0,2*np.pi,nx) #grid
    un = np.empty(nx) #temp solution
    t=0 #initial time

    #u = np.asarray([ufunc(t,x0,nu) for x0 in x]) #populate initial condition using new function
    u = ufunc(t,x,nu)

    for n in range(nt):
        un = u.copy()
        for i in range(1,nx-1):
            u[i] = un[i] - un[i]*dt/dx*(un[i]-un[i-1]) + nu*(dt/dx**2)*(un[i+1]-2*un[i]+un[i-1])
        u[0] = un[0] - un[0]*dt/dx*(un[0]-un[-2]) + nu*(dt/dx**2)*(un[1]-2*un[0]+un[-2])
        u[-1] = u[0]

    #u_analytical = np.asarray([ufunc(nt*dt,x1,nu) for x1 in x]) #analytical solution
    u_analytical = ufunc(nt*dt,x,nu)

    plt.figure(figsize=(11,7),dpi=100)
    plt.plot(x,u,marker='o',lw=2,label='computational')
    plt.plot(x,u_analytical,lw=2,label='analytical')
    plt.xlim([0,2*np.pi])
    plt.ylim([0,10])
    plt.legend()
    plt.show()

burgers(101,100)