import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def plot2d(x,y,field,title):
    fig = plt.figure(figsize=(11,7),dpi=100)
    ax = fig.add_subplot(111,projection='3d')
    
    X, Y = np.meshgrid(x,y)
    ax.plot_surface(X,Y,field,cmap=cm.viridis,rstride=1,cstride=1,linewidth=0,antialiased=False)

    ax.view_init(30,225)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_title(title)
    
    plt.show()

#Basic variable required
xmin, xmax = 0, 2 #domain lengths
ymin, ymax = 0, 1
nx, ny = 50, 50 #no of grid points
dx = (xmax-xmin)/(nx-1) #grid size x
dy = (ymax-ymin)/(ny-1) #grid size y
nt = 100 #no of timesteps

x = np.linspace(xmin,xmax,nx)
y = np.linspace(ymin,ymax,ny)

p = np.zeros((ny,nx)) #solution array

b = np.zeros_like(p) #source term
b[int(ny/4),int(nx/4)], b[int(3*ny/4),int(3*nx/4)] = 100, -100

plot2d(x,y,b,'initial state of p') #initial solution

for i in range(nt):
    pn = p.copy()
    p[1:-1,1:-1] = (dy**2*(pn[1:-1,2:]+pn[1:-1,:-2]) + dx**2*(pn[2:,1:-1]+pn[:-2,1:-1])\
        - b[1:-1,1:-1]*dx**2*dy**2)/(2*(dx**2+dy**2))
        
    p[:,0], p[:,-1], p[0,:], p[-1,:] = 0, 0, 0, 0 #boundary conditions

plot2d(x,y,p,'final state of p') #final solution