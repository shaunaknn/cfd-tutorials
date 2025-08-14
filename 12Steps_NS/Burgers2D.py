import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot2d(x,y,field,field2,title):
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111,projection='3d')
    
    X,Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,field,cmap=cm.viridis,rstride=1,cstride=1,linewidth=0,antialiased=True)
    ax.plot_surface(X,Y,field2,cmap=cm.viridis,rstride=1,cstride=1,linewidth=0,antialiased=True)

    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)

    plt.show()

def burgers2d(u,v,nt,dx,dy,dt,nu):
    for n in range(nt+1):
        un = u.copy()
        vn = v.copy()

        u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]*dt/dx*(un[1:-1,1:-1]-un[1:-1,:-2])\
              - vn[1:-1,1:-1]*dt/dy*(un[1:-1,1:-1]-un[:-2,1:-1])\
                + nu*dt/dx**2*(un[1:-1,2:]-2*un[1:-1,1:-1]+un[1:-1,:-2])\
                    + nu*dt/dy**2*(un[2:,1:-1]-2*un[1:-1,1:-1]+un[:-2,1:-1])
        
        v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1]*dt/dx*(vn[1:-1,1:-1]-vn[1:-1,:-2])\
            - vn[1:-1,1:-1]*dt/dy*(vn[1:-1,1:-1]-vn[:-2,1:-1])\
                + nu*dt/dx**2*(vn[1:-1,2:]-2*vn[1:-1,1:-1]+vn[1:-1,:-2])\
                    + nu*dt/dy**2*(vn[2:,1:-1]-2*vn[1:-1,1:-1]+vn[:-2,1:-1])

        u[0,:], u[-1,:], u[:,0], u[:,-1] = 1, 1, 1, 1 #boundary conditions
        v[0,:], v[-1,:], v[:,0], v[:,-1] = 1, 1, 1, 1

    return u, v

lx, ly = 2, 2 #domain dimensions
nx, ny = 41, 41 #no of grid points
nt = 120 #number of timesteps
dx = lx/(nx-1) #delta x
dy = ly/(ny-1) #delta y

nu = 0.01 #viscosity
sigma = 0.0009 #stability parameter
dt = sigma*dx*dy/nu #delta t

x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

u = np.ones((ny,nx)) #solution arrays
v = np.ones((ny,nx))

u[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2 #initial condition
v[int(.5/dy):int(1/dy+1),int(.5/dx):int(1/dx+1)] = 2

plot2d(x,y,u,v,'initial state')
u,v = burgers2d(u,v,nt,dx,dy,dt,nu)
plot2d(x,y,u,v,'final state')