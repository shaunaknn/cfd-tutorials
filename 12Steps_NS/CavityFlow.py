import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

lx, ly = 2, 2 #domain size
nx, ny = 41, 41 #no of domain points
nt = 700 #real timesteps count
nit = 50 #pseudo timestep count
dx = lx/(nx-1) #deltax
dy = ly/(ny-1) #deltay
x = np.linspace(0,lx,nx) #x-domain array
y = np.linspace(0,ly,ny) #y-domain array

rho = 1 #density
nu = 0.1 #kinematic viscosity
dt = 0.001 #timestep size

u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))

def plotcontour(x,y,field,title): #plots velocity vectors
    plt.figure(figsize=(11,7),dpi=100)
    X,Y = np.meshgrid(x,y)
    plt.contourf(X,Y,field,alpha=0.5,cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X,Y,field,cmap=cm.viridis)
    plt.quiver(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.show()

def streamlines(x,y,field,title): #plots velocity streamlines
    plt.figure(figsize=(11,7),dpi=100)
    X,Y = np.meshgrid(x,y)
    plt.contourf(X,Y,field,alpha=0.5,cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X,Y,field,cmap=cm.viridis)
    plt.streamplot(X,Y,u,v)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(x.min(),x.max())
    plt.ylim(y.min(),y.max())
    plt.title(title)
    plt.show()

def buildb(b,u,v,dx,dy,dt,rho): #separate source term of p poisson equation
    b[1:-1,1:-1] = rho*(1/dt*((u[1:-1,2:]-u[1:-1,:-2])/(2*dx) + (v[2:,1:-1]-v[:-2,1:-1])/(2*dy))\
                        - ((u[1:-1,2:]-u[1:-1,:-2])/(2*dx))**2\
                              - 2*((u[2:,1:-1]-u[:-2,1:-1])/(2*dy)*(v[1:-1,2:]- v[1:-1,:-2])/(2*dx))\
                              - ((v[2:,1:-1]-v[:-2,1:-1])/(2*dy))**2)
    
    return b

def ppoison(p,dx,dy,b):
    pn = np.empty_like(p)

    for q in range(nit):
        pn[:] = p
        p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,:-2])*dy**2 + (pn[2:,1:-1]+pn[:-2,1:-1])*dx**2)/(2*(dx**2 + dy**2))\
                        -dx**2 * dy**2/(2*(dx**2 + dy**2)) * b[1:-1,1:-1]
        
        p[-1,:] = 0
        p[0,:] = p[1,:]
        p[:,-1] = p[:,-2]
        p[:,0] = p[:,1]

    return p

def cavity(u,v,p,nt,dx,dy,dt,rho,nu):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny,nx))

    for n in range(nt):
        un[:] = u #similar to un = u.copy()
        vn[:] = v
        
        b = buildb(b,u,v,dx,dy,dt,rho)
        p = ppoison(p,dx,dy,b)

        u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]* dt/dx *(un[1:-1,1:-1]-un[1:-1,:-2])\
                        -vn[1:-1,1:-1]* dt/dy *(un[1:-1,1:-1]-un[:-2,1:-1])\
                        -dt/(2*rho*dx) * (p[1:-1,2:]-p[1:-1,:-2])\
                        + nu*(dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])\
                              + dt/dy**2 * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]))
        
        v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1]* dt/dx *(vn[1:-1,1:-1]-vn[1:-1,:-2])\
                        -vn[1:-1,1:-1]* dt/dy *(vn[1:-1,1:-1]-vn[:-2,1:-1])\
                        -dt/(2*rho*dy) * (p[2:,1:-1]-p[:-2,1:-1])\
                        + nu*(dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])\
                              + dt/dy**2 * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]))

        u[0,:] = 0 #bottom
        u[:,0] = 0 #left wall
        u[:,-1] = 0 #right wall
        u[-1,:] = 1 #lid
        v[0,:] = 0 #bottom
        v[-1,:] = 0 #lid
        v[:,0] = 0 #left wall
        v[:,-1] = 0 #right wall

    return u, v, p

u,v,p = cavity(u,v,p,nt,dx,dy,dt,rho,nu)
plotcontour(x,y,p,'Final time step velocity vectors')
streamlines(x,y,p,'Streamlines at final time step')
