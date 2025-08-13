import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def linearadvection2D(nx,ny,nt):
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

    un = np.ones((ny,nx)) #temp array

    for n in range(nt): #solving in time
        un = u.copy()
        u[1:,1:] = un[1:,1:]-c*dt/dx*(un[1:,1:]-un[1:,:-1])-c*dt/dy*(un[1:,1:]-un[:-1,1:])
        u[0,:] = 1
        u[-1,:] = 1
        u[:,0] = 1
        u[:,-1] = 1
    
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X,Y,u,cmap=cm.viridis)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Final state')

    fig.colorbar(surf, shrink=0.5, aspect=5)  #Color bar
    plt.show()

def nonlinearadvection(nx,nt):
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

linearadvection2D(81,81,100)
#nonlinearadvection(41,20)