import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

lx, ly = 1, 1 #domain size
nx, ny = 21, 21 #no of domain points
nt = 10000 #real timesteps count
nit = 50 #pseudo timestep count
dx = lx/(nx-1) #deltax
dy = ly/(ny-1) #deltay
x = np.linspace(0,lx,nx) #x-domain array
y = np.linspace(0,ly,ny) #y-domain array
c = 4 #lid velocity

rho = 1 #density
nu = 0.01 #kinematic viscosity
dt = 0.001 #timestep size
Re = c*max(lx,ly)/nu #Reynolds number
l1norm_target = 1e-4

u = np.zeros((ny,nx))
v = np.zeros((ny,nx))
p = np.zeros((ny,nx))

def plotfield(x,y,field,u,v,title,mode='quiver'): #plots velocity vectors
    plt.figure(figsize=(11,7),dpi=100)
    X,Y = np.meshgrid(x,y)
    plt.contourf(X,Y,field,alpha=0.5,cmap=cm.viridis)
    plt.colorbar()
    plt.contour(X,Y,field,cmap=cm.viridis)
    if mode == 'quiver':
        plt.quiver(X[::2,::2],Y[::2,::2],u[::2,::2],v[::2,::2])
    elif mode == 'stream':
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

def ppoisson(p,dx,dy,b): # no. of iterations specified version
    pn = np.empty_like(p)

    for q in range(nit):
        pn[:] = p
        p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,:-2])*dy**2 + (pn[2:,1:-1]+pn[:-2,1:-1])*dx**2)/(2*(dx**2 + dy**2))\
                        -dx**2 * dy**2/(2*(dx**2 + dy**2)) * b[1:-1,1:-1]
        
        p[-1,:] = 0 #p=0 at y=2
        p[0,:] = p[1,:] #dp/dy=0 at y=0
        p[:,-1] = p[:,-2] #dp/dx=0 at x=2
        p[:,0] = p[:,1] #dp/dx=0 at x=0

    return p

def ppoissonl1(p,dx,dy,b,l1norm_target): #l1 norm target version
    l1norm = 1
    itercount = 0
    eta = 1e-8
    pn = np.empty_like(p)

    while l1norm>l1norm_target:
        pn = p.copy()
        p[1:-1,1:-1] = ((pn[1:-1,2:]+pn[1:-1,:-2])*dy**2 + (pn[2:,1:-1]+pn[:-2,1:-1])*dx**2)/(2*(dx**2 + dy**2))\
                        -dx**2 * dy**2/(2*(dx**2 + dy**2)) * b[1:-1,1:-1]
        
        p[-1,:] = 0 #p=0 at y=2
        p[0,:] = p[1,:] #dp/dy=0 at y=0
        p[:,-1] = p[:,-2] #dp/dx=0 at x=2
        p[:,0] = p[:,1] #dp/dx=0 at x=0

        l1norm = (np.sum(np.abs(p - pn)))/np.sum(np.abs(pn)+eta)
        itercount += 1

    return p, itercount

def update_u(u,un,vn,p,dx,dy,dt,rho,nu): #update u velocity
    u[1:-1,1:-1] = un[1:-1,1:-1] - un[1:-1,1:-1]* dt/dx *(un[1:-1,1:-1]-un[1:-1,:-2])\
                        -vn[1:-1,1:-1]* dt/dy *(un[1:-1,1:-1]-un[:-2,1:-1])\
                        -dt/(2*rho*dx) * (p[1:-1,2:]-p[1:-1,:-2])\
                        + nu*(dt/dx**2 * (un[1:-1,2:] - 2*un[1:-1,1:-1] + un[1:-1,:-2])\
                              + dt/dy**2 * (un[2:,1:-1] - 2*un[1:-1,1:-1] + un[:-2,1:-1]))
    return u

def update_v(v,un,vn,p,dx,dy,dt,rho,nu): #update v velocity
    v[1:-1,1:-1] = vn[1:-1,1:-1] - un[1:-1,1:-1]* dt/dx *(vn[1:-1,1:-1]-vn[1:-1,:-2])\
                        -vn[1:-1,1:-1]* dt/dy *(vn[1:-1,1:-1]-vn[:-2,1:-1])\
                        -dt/(2*rho*dy) * (p[2:,1:-1]-p[:-2,1:-1])\
                        + nu*(dt/dx**2 * (vn[1:-1,2:] - 2*vn[1:-1,1:-1] + vn[1:-1,:-2])\
                              + dt/dy**2 * (vn[2:,1:-1] - 2*vn[1:-1,1:-1] + vn[:-2,1:-1]))
    return v

# upwind scheme implemented to handle large negative velocities
def compute_F(c):
    denom = np.abs(c) + 1e-6
    pos_part = np.maximum(c/denom,0)
    neg_part = np.maximum(-c/denom,0)
    return pos_part,neg_part

def update_uv_upwind(phi, un, vn, p, dx, dy, dt, rho, nu, is_u=True):
    """Updates a velocity component phi using upwind scheme for convective terms.
    is_u: Flag to switch between updating u and v.
    """
    
    # Compute positive/negative flux coefficients
    ew1, ew2 = compute_F(un)  # East-West coefficients based on un
    ns1, ns2 = compute_F(vn)  # North-South coefficients based on vn

    field = un.copy() if is_u else vn.copy()

    fe = field[1:-1, 1:-1]*ew1[1:-1,1:-1] + field[1:-1, 2:]*ew2[1:-1,1:-1]
    fw = field[1:-1, 0:-2]*ew1[1:-1,1:-1] + field[1:-1, 1:-1]*ew2[1:-1,1:-1]
        
    # North-South fluxes: transport un with vn-based coefficients  
    fn = field[1:-1, 1:-1]*ns1[1:-1,1:-1] + field[2:, 1:-1]*ns2[1:-1,1:-1]
    fs = field[0:-2, 1:-1]*ns1[1:-1,1:-1] + field[1:-1, 1:-1]*ns2[1:-1,1:-1]
    
    # Convective terms
    conv_x = un[1:-1,1:-1] * dt/dx * (fe - fw)
    conv_y = vn[1:-1,1:-1] * dt/dy * (fn - fs)

    # Pressure gradient
    if is_u:
        grad_p = dt/(2*rho*dx) * (p[1:-1,2:] - p[1:-1,:-2])
    else:
        grad_p = dt/(2*rho*dy) * (p[2:,1:-1] - p[:-2,1:-1])
        
    # Diffusion term
    diff = nu * (dt/dx**2 * (field[1:-1,2:] - 2*field[1:-1,1:-1] + field[1:-1,:-2])\
                 + dt/dy**2 * (field[2:,1:-1] - 2*field[1:-1,1:-1] + field[:-2,1:-1]))

    phi[1:-1,1:-1] = field[1:-1,1:-1] - conv_x - conv_y - grad_p + diff
    
    return phi

def applyBC(u,v,c): #apply boundary conditions
    u[0,:] = 0 #bottom wall
    u[:,0] = 0 #left wall
    u[:,-1] = 0 #right wall
    u[-1,:] = c #lid
    v[0,:] = 0 #bottom wall
    v[-1,:] = 0 #lid
    v[:,0] = 0 #left wall
    v[:,-1] = 0 #right wall
    return u,v

def cavity(u,v,p,nt,dx,dy,dt,rho,nu): #solve cavity flow
    un = np.empty_like(u)
    vn = np.empty_like(v)
    b = np.zeros((ny,nx))

    for n in range(nt):
        un[:] = u #similar to un = u.copy()
        vn[:] = v
        
        b = buildb(b,u,v,dx,dy,dt,rho) #build RHS of pressure-poisson eqn
        #p = ppoisson(p,dx,dy,b)
        p, _ = ppoissonl1(p,dx,dy,b,l1norm_target)

        #u = update_u(u,un,vn,p,dx,dy,dt,rho,nu)
        #v = update_v(v,un,vn,p,dx,dy,dt,rho,nu)

        u = update_uv_upwind(u, un, vn, p, dx, dy, dt, rho, nu, is_u=True)
        v = update_uv_upwind(v, un, vn, p, dx, dy, dt, rho, nu, is_u=False)

        u,v = applyBC(u,v,c)

    return u, v, p

u,v,p = cavity(u,v,p,nt,dx,dy,dt,rho,nu)
print(f"The reynolds number is {Re}")
plotfield(x,y,p,u,v,'Final time step velocity vectors','quiver')
plotfield(x,y,p,u,v,'Streamlines at final time step','stream')
