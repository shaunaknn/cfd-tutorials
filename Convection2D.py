import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot2d(x,y,field,title):
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111,projection='3d')
    
    X,Y = np.meshgrid(x,y)
    surf = ax.plot_surface(X,Y,field,cmap=cm.viridis,rstride=2,cstride=2)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    fig.colorbar(surf,shrink=0.5,aspect=8)
    plt.show()

def nonlinearconvection2D(u,v,nt,dx,dy,dt):
    for n in range(nt+1): #solving in time
        un = u.copy()
        vn = v.copy()
        u[1:,1:] = un[1:,1:] - un[1:,1:]*dt/dx*(un[1:,1:]-un[1:,:-1]) - vn[1:,1:]*dt/dy*(un[1:,1:]-un[:-1,1:])
        v[1:,1:] = vn[1:,1:] - un[1:,1:]*dt/dx*(vn[1:,1:]-vn[1:,:-1]) - vn[1:,1:]*dt/dy*(vn[1:,1:]-vn[:-1,1:])
        
        u[0,:], u[-1,:], u[:,0], u[:,-1] = 1, 1, 1, 1 #boundary conditions
        v[0,:], v[-1,:], v[:,0], v[:,-1] = 1, 1, 1, 1
    
    return u, v

lx,ly = 2,2 #domain dimensions
nx,ny = 101,101 #no of grid points 
nt = 80 #no of time steps

sigma = 0.2 #stability parameter
dx = lx/(nx-1) #delta x
dy = ly/(ny-1) #delta y
dt = sigma*dx # delta t

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

u = np.ones((ny,nx)) #solution array u speed
v = np.ones((ny,nx)) #solution array v speed

u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 #initial condition
v[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 

plot2d(x,y,u,'initial u condition') #initial condition plots
plot2d(x,y,v,'initial v condition')

u, v = nonlinearconvection2D(u,v,nt,dx,dy,dt)

plot2d(x,y,u,'Final solution u')
plot2d(x,y,v,'Final solution v')