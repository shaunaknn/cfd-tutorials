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

def nonlinearconvection2D(nx,ny,nt):
    # nx,ny = no of grid points, nt = no of time steps
    lx = 2 #domain dimensions
    ly = 2
    sigma = 0.2
    dx = lx/(nx-1) #delta x
    dy = ly/(ny-1) #delta y
    dt = sigma*dx # delta t

    x = np.linspace(0,lx,nx)
    y = np.linspace(0,ly,ny)
    X, Y = np.meshgrid(x,y)

    u = np.ones((ny,nx)) #solution array u speed
    u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 #initial condition
    un = np.ones_like(u)

    v = np.ones((ny,nx)) #solution array v speed
    v[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2
    vn = np.ones_like(v) 

    for n in range(nt+1): #solving in time
        un = u.copy()
        vn = v.copy()
        u[1:,1:] = un[1:,1:] - un[1:,1:]*dt/dx*(un[1:,1:]-un[1:,:-1]) - vn[1:,1:]*dt/dy*(un[1:,1:]-un[:-1,1:])
        v[1:,1:] = vn[1:,1:] - un[1:,1:]*dt/dx*(vn[1:,1:]-vn[1:,:-1]) - vn[1:,1:]*dt/dy*(vn[1:,1:]-vn[:-1,1:])
        
        u[0,:], u[-1,:], u[:,0], u[:,-1] = 1, 1, 1, 1
        v[0,:], v[-1,:], v[:,0], v[:,-1] = 1, 1, 1, 1
    
    plot_2dfield(X,Y,u,'Final solution 2D non linear convection u')
    plot_2dfield(X,Y,v,'Final solution 2D non linear convection v')

linearconvection2D(81,81,100)
nonlinearconvection2D(101,101,80)