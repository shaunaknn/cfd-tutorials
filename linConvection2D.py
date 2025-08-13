import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot_2dfield(X,Y,field,title):
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111,projection='3d')
    surf = ax.plot_surface(X,Y,field,cmap=cm.viridis)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    fig.colorbar(surf,shrink=0.5,aspect=8)
    plt.show()

def linearconvection2D(nx,ny,nt):
    # nx,ny = no of grid points, nt = no of time steps
    lx = 2 #domain dimensions
    ly = 2
    sigma = 0.2
    c = 1 #wave speed
    dx = lx/(nx-1) #delta x
    dy = ly/(ny-1) #delta y
    dt = sigma*dx # delta t

    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)
    X, Y = np.meshgrid(x,y)

    u = np.ones((ny,nx)) #solution array
    u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 #initial condition

    un = np.ones_like(u)

    for n in range(nt+1): #solving in time
        un = u.copy()
        u[1:,1:] = un[1:,1:]-c*dt/dx*(un[1:,1:]-un[1:,:-1])-c*dt/dy*(un[1:,1:]-un[:-1,1:])
        u[0,:], u[-1,:], u[:,0], u[:,-1] = 1, 1, 1, 1
    
    plot_2dfield(X,Y,u,'Final solution 2D linear convection')
    
linearconvection2D(81,81,100)