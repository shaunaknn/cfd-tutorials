import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot2d(x,y,field,title):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')

    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,field,rstride=1,cstride=1,cmap=cm.viridis,linewidth=0,antialiased=True)

    ax.set_zlim(1,2.5)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)

    fig.colorbar(surf,shrink=0.5,aspect=8)
    plt.show()

def diffusion2D(u,nt,dx,dy,dt,nu):
    un = np.empty_like(u)
    u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 #initial condition

    for n in range(nt+1): #solving in time
        un = u.copy()
        u[1:-1,1:-1] = un[1:-1,1:-1] + nu*dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2]) + nu*dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1])
        u[0,:], u[-1,:], u[:,0], u[:,-1] = 1, 1, 1, 1

    plot2d(x,y,u,f'Solution after {nt} timesteps')

lx, ly = 2, 2 #domain dimensions
nx, ny = 31, 31 #no of grid points
nt = 17 #number of timesteps
dx = lx/(nx-1) #delta x
dy = ly/(ny-1) #delta y

nu = 0.05 #viscosity
sigma = 0.25 #tuning parameter
dt = sigma*dx*dy/nu #delta t

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

u = np.ones((ny,nx)) #solution array

u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 #initial condition
plot2d(x,y,u,'Initial condition') #plot initial condition

diffusion2D(u,10,dx,dy,dt,nu)
diffusion2D(u,25,dx,dy,dt,nu)
diffusion2D(u,50,dx,dy,dt,nu)