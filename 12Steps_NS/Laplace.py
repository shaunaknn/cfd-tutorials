import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot2d(x,y,field,title):
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111,projection='3d')
    
    X, Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,field,cmap=cm.viridis,rstride=1,cstride=1,linewidth=0,antialiased=False)

    ax.set_xlim(0,2)
    ax.set_ylim(0,1)
    ax.view_init(30,225)
    
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    
    plt.show()

def laplace2d(p,y,dx,dy,l1norm_target):
    l1norm = 1
    pn = np.empty_like(p)

    while l1norm>l1norm_target:
        pn = p.copy()
        p[1:-1,1:-1] = (dy**2*(pn[1:-1,2:]+pn[1:-1,:-2]) + dx**2*(pn[2:,1:-1]+pn[:-2,1:-1]))\
            /(2*(dx**2+dy**2))
        
        #boundary conditions
        p[:,0] = 0 #p=0 @x = 0
        p[:,-1] = y #p = y @x = 2
        p[0,:] = p[1,:] #dp/dy = 0 @y = 0
        p[-1,:] = p[-2,:] #dp/dy = 0 @y = 1

        l1norm = (np.sum(np.abs(p - pn)))/np.sum(np.abs(pn))

    return p

#Basic variable required
lx, ly = 2, 1 #domain lengths
nx, ny = 31, 31 #no of grid points
dx = lx/(nx-1) #grid size x
dy = ly/(ny-1) #grid size y

p = np.zeros((ny,nx))
x = np.linspace(0,lx,nx)
y = np.linspace(0,ly,ny)

#boundary conditions
p[:,0] = 0 #p=0 @x = 0
p[:,-1] = y #p = y @x = 2
p[0,:] = p[1,:] #dp/dy = 0 @y = 0
p[-1,:] = p[-2,:] #dp/dy = 0 @y = 1

plot2d(x,y,p,'initial value')
p = laplace2d(p,y,dx,dy,1e-4)
plot2d(x,y,p,'final value')