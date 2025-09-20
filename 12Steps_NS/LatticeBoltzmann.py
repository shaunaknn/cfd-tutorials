import numpy as np
import matplotlib.pyplot as plt
import io, imageio.v2 as imageio
from tqdm import tqdm
import os
print("Saving GIF to:", os.getcwd())

# --- parameters
Nx = 400
Ny = 100
rho0 = 0.6
tau = 0.6
Nt = 4000

# D2Q9 velocities/weights
cxs = np.array([0,0,1,1,1,0,-1,-1,-1], dtype=np.int32)
cys = np.array([0,1,1,0,-1,-1,-1,0,1], dtype=np.int32)
weights = np.array([4/9,1/9,1/36,1/9,1/36,1/9,1/36,1/9,1/36], dtype=np.float32)
opposite = np.array([0,5,6,7,8,1,2,3,4], dtype=np.int32)

# initial distributions
F = np.ones((Ny, Nx, 9), dtype=np.float32)
np.random.seed(1)
F += 0.01 * np.random.rand(Ny, Nx, 9).astype(np.float32)

X, Y = np.meshgrid(np.arange(Nx), np.arange(Ny))
F[:, :, 3] += 2.0 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))

rho = F.sum(axis=2)
for i in range(9):
    F[:, :, i] *= (rho0 / rho)

cylinder = ((X - Nx/4)**2 + (Y - Ny/2)**2) < (Ny/4)**2

Feq = np.empty_like(F)

frames = []

# --- first pass: run a short sim to get global vorticity limits
test_steps = 200
ux = np.zeros((Ny, Nx))
uy = np.zeros((Ny, Nx))
vort_all = []
for it in range(test_steps):
    # streaming
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        F[:, :, i] = np.roll(np.roll(F[:, :, i], cy, axis=0), cx, axis=1)
    bndry = F[cylinder, :].copy()
    F[cylinder, :] = bndry[:, opposite]

    rho = F.sum(axis=2)
    rho_safe = rho + 1e-12
    ux = np.tensordot(F, cxs, axes=([2], [0])) / rho_safe
    uy = np.tensordot(F, cys, axes=([2], [0])) / rho_safe
    ux[cylinder] = 0; uy[cylinder] = 0
    cu = ux[..., None]*cxs + uy[..., None]*cys
    usq = ux**2 + uy**2
    Feq = (rho[..., None]*weights*(1+3*cu+4.5*cu**2-1.5*usq[...,None]))
    F += -(1.0/tau)*(F-Feq)

    if it % 20 == 0:
        dvdx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) * 0.5
        dudy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) * 0.5
        vorticity = dvdx - dudy
        vorticity[cylinder] = np.nan
        vort_all.append(vorticity)

vmin = min(np.nanmin(v) for v in vort_all)
vmax = max(np.nanmax(v) for v in vort_all)
# (optional: enforce symmetry around 0 for nicer rainbow colormap)
max_abs = max(abs(vmin), abs(vmax))
vmin, vmax = -max_abs, max_abs

print(f"Fixed color scale: vmin={vmin:.3f}, vmax={vmax:.3f}")

# --- main sim loop again for visualization
F = np.ones((Ny, Nx, 9), dtype=np.float32)  # reinit
F += 0.01 * np.random.rand(Ny, Nx, 9).astype(np.float32)
F[:, :, 3] += 2.0 * (1 + 0.2 * np.cos(2 * np.pi * X / Nx * 4))
rho = F.sum(axis=2)
for i in range(9):
    F[:, :, i] *= (rho0 / rho)

for it in tqdm(range(Nt)):
    for i, (cx, cy) in enumerate(zip(cxs, cys)):
        F[:, :, i] = np.roll(np.roll(F[:, :, i], cy, axis=0), cx, axis=1)
    bndry = F[cylinder, :].copy()
    F[cylinder, :] = bndry[:, opposite]

    rho = F.sum(axis=2)
    rho_safe = rho + 1e-12
    ux = np.tensordot(F, cxs, axes=([2], [0])) / rho_safe
    uy = np.tensordot(F, cys, axes=([2], [0])) / rho_safe
    ux[cylinder] = 0; uy[cylinder] = 0

    cu = ux[..., None]*cxs + uy[..., None]*cys
    usq = ux**2 + uy**2
    Feq = (rho[..., None]*weights*(1+3*cu+4.5*cu**2-1.5*usq[...,None]))
    F += -(1.0/tau)*(F-Feq)

    if (it % 50 == 0) or (it == Nt - 1):
        dvdx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) * 0.5
        dudy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) * 0.5
        vorticity = dvdx - dudy
        vorticity[cylinder] = np.nan

        fig, ax = plt.subplots(figsize=(4, 2), dpi=120)
        im = ax.imshow(vorticity, cmap='gist_rainbow',
                       origin='upper', vmin=vmin, vmax=vmax)  # << fix
        ax.imshow(~cylinder, cmap='gray', alpha=0.3, origin='upper')
        ax.set_axis_off()
        fig.colorbar(im, ax=ax, shrink=0.7)

        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))
        buf.close()

print("Number of frames:", len(frames))

#save gif if needed
imageio.mimsave("animation.gif", frames, duration=0.1, loop=0)
print("GIF saved as animation.gif")
