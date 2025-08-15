import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

lx, ly = 2, 2 #domain size
nx, ny = 41, 41 #no of domain points
nt = 10 #real timesteps count
nit = 50 #pseudo timestep count
dx = lx/(nx-1) #deltax
dy = ly/(ny-1) #deltay
x = np.linspace(0,lx,nx) #x-domain array
y = np.linspace(0,ly,ny) #y-domain array

rho = 1 #density
nu = 0.1 #kinematic viscosity
dt = 0.01 #timestep size
F = 1 #source term

u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))

epsilon = 0.001
steps=0

def plotfield(x,y,u,v): #plots velocity vectors
    plt.figure(figsize=(11,7),dpi=100)
    X,Y = np.meshgrid(x,y)
    plt.quiver(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def buildb(u,v,dx,dy,dt,rho): #separate source term of p poisson equation
    b = np.zeros_like(u)
    b[1:-1,1:-1] = rho*(1/dt*((u[1:-1,2:]-u[1:-1,:-2])/(2*dx) + (v[2:,1:-1]-v[:-2,1:-1])/(2*dy))\
                        - ((u[1:-1,2:]-u[1:-1,:-2])/(2*dx))**2\
                              - 2*((u[2:,1:-1]-u[:-2,1:-1])/(2*dy) * (v[1:-1,2:]- v[1:-1,:-2])/(2*dx))\
                              - ((v[2:,1:-1]-v[:-2,1:-1])/(2*dy))**2)
    
    #periodic BC x = 2
    b[1:-1,-1] = rho*(1/dt*((u[1:-1,0]-u[1:-1,-2])/(2*dx) + (v[2:,-1]-v[:-2,-1])/(2*dy))\
                        - ((u[1:-1,0]-u[1:-1,-2])/(2*dx))**2\
                              - 2*((u[2:,-1]-u[:-2,-1])/(2*dy) * (v[1:-1,0]- v[1:-1,-2])/(2*dx))\
                              - ((v[2:,-1]-v[:-2,-1])/(2*dy))**2)
    
    #periodic BC x = 0
    b[1:-1,0] = rho*(1/dt*((u[1:-1,1]-u[1:-1,-1])/(2*dx) + (v[2:,0]-v[:-2,0])/(2*dy))\
                        - ((u[1:-1,1]-u[1:-1,-1])/(2*dx))**2\
                              - 2*((u[2:,0]-u[:-2,0])/(2*dy) * (v[1:-1,1]- v[1:-1,-1])/(2*dx))\
                              - ((v[2:,0]-v[:-2,0])/(2*dy))**2)

    return b

def ppoissonperiodic(p,dx,dy,b):
    pn = np.empty_like(p)

    for q in range(nit):
        pn[:] = p
        p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,:-2])*dy**2 + (pn[2:,1:-1]+pn[:-2,1:-1])*dx**2)/(2*(dx**2 + dy**2))\
                        -dx**2 * dy**2/(2*(dx**2 + dy**2)) * b[1:-1,1:-1]
        
        #periodic BC @ x=2
        p[1:-1,-1] = ((pn[1:-1,0]+pn[1:-1,-2])*dy**2 + (pn[2:,-1]+pn[:-2,-1])*dx**2)/(2*(dx**2 + dy**2))\
                        -dx**2 * dy**2/(2*(dx**2 + dy**2)) * b[1:-1,-1]
        
        #periodic BC @ x=0
        p[1:-1,-1] = ((pn[1:-1,1]+pn[1:-1,1])*dy**2 + (pn[2:,0]+pn[:-2,0])*dx**2)/(2*(dx**2 + dy**2))\
                        -dx**2 * dy**2/(2*(dx**2 + dy**2)) * b[1:-1,0]
        
        #top and bottom wall BCs
        p[-1,:] = p[-2,:] #dp/dy=0 @ y=2
        p[0,:] = p[1,:] #dp/dy=0 @ y=0

    return p

def channel(u,v,p,nt,dx,dy,dt,rho,nu,epsilon):
    un = np.empty_like(u)
    vn = np.empty_like(v)
    udiff, stepcount = 1, 0

    while udiff>epsilon:
        un[:] = u #similar to un = u.copy()
        vn[:] = v
        
        b = buildb(u,v,dx,dy,dt,rho)
        p = ppoissonperiodic(p,dx,dy,b)

        u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]* dt/dx *(un[1:-1,1:-1]-un[1:-1,:-2])\
                        -vn[1:-1,1:-1]* dt/dy *(un[1:-1,1:-1]-un[:-2,1:-1])\
                        -dt/(2*rho*dx) * (p[1:-1,2:]-p[1:-1,:-2])\
                        + nu*(dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])\
                              + dt/dy**2 * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]))\
                              +dt*F
        
        v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1]* dt/dx *(vn[1:-1,1:-1]-vn[1:-1,:-2])\
                        -vn[1:-1,1:-1]* dt/dy *(vn[1:-1,1:-1]-vn[:-2,1:-1])\
                        -dt/(2*rho*dy) * (p[2:,1:-1]-p[:-2,1:-1])\
                        + nu*(dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])\
                              + dt/dy**2 * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]))

        #periodic BCs
        u[1:-1,-1] = un[1:-1,-1] - un[1:-1,-1]* dt/dx *(un[1:-1,-1]-un[1:-1,-2])\
                        -vn[1:-1,-1]* dt/dy *(un[1:-1,-1]-un[:-2,-1])\
                        -dt/(2*rho*dx) * (p[1:-1,0]-p[1:-1,-2])\
                        + nu*(dt/dx**2 * (un[1:-1,0] - 2*un[1:-1,-1] + un[1:-1,-2])\
                              + dt/dy**2 * (un[2:,-1] - 2*un[1:-1,-1] + un[:-2,-1]))\
                              +dt*F
        
        u[1:-1,0] = un[1:-1,0] - un[1:-1,0]* dt/dx *(un[1:-1,0]-un[1:-1,-1])\
                        -vn[1:-1,0]* dt/dy *(un[1:-1,0]-un[:-2,0])\
                        -dt/(2*rho*dx) * (p[1:-1,1]-p[1:-1,-1])\
                        + nu*(dt/dx**2 * (un[1:-1,1] - 2*un[1:-1,0] + un[1:-1,-1])\
                              + dt/dy**2 * (un[2:,0] - 2*un[1:-1,0] + un[:-2,0]))\
                              +dt*F
        
        v[1:-1,1:-1] = vn[1:-1,-1] - un[1:-1,-1]* dt/dx *(vn[1:-1,-1]-vn[1:-1,-2])\
                        -vn[1:-1,-1]* dt/dy *(vn[1:-1,-1]-vn[:-2,-1])\
                        -dt/(2*rho*dy) * (p[2:,-1]-p[:-2,-1])\
                        + nu*(dt/dx**2 * (vn[1:-1,0] - 2*vn[1:-1,-1] + vn[1:-1,-2])\
                              + dt/dy**2 * (vn[2:,-1] - 2*vn[1:-1,-1] + vn[:-2,-1]))
        
        v[1:-1,1:-1] = vn[1:-1,0] - un[1:-1,0]* dt/dx *(vn[1:-1,0]-vn[1:-1,-1])\
                        -vn[1:-1,0]* dt/dy *(vn[1:-1,0]-vn[:-2,0])\
                        -dt/(2*rho*dy) * (p[2:,0]-p[:-2,0])\
                        + nu*(dt/dx**2 * (vn[1:-1,1] - 2*vn[1:-1,0] + vn[1:-1,-1])\
                              + dt/dy**2 * (vn[2:,0] - 2*vn[1:-1,0] + vn[:-2,0]))
        
        #wall BCs
        u[0,:] = 0 #bottom
        u[-1,:] = 0 #top
        v[0,:] = 0 #bottom
        v[-1,:] = 0 #top

        udiff = (np.sum(u)-np.sum(un))/np.sum(u)
        stepcount += 1

    return u, v, p, stepcount

u,v,p,steps = channel(u,v,p,nt,dx,dy,dt,rho,nu,epsilon)
plotfield(x,y,u,v)
print(steps)