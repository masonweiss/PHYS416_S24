# Mason Weiss
# PHYS416 - Fick's Diffusion Model
# April 2024

# Cylindrical Configuration

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

# Model Parameters
NR = 61             # number of radial grid points, assuming symmetry in phi
NZ = 91             # number of height-wise grid points, assuming symmetry in phi
r_leakage = 0.5     # leakage percentage at outer barrier
z_leakage = 0.1     # leakage percentage at top & lower barrier
nstep = 10000       # number of timesteps
rfactor = 0.5       # factor by which to scale the critical radius
zfactor = 0.8       # factor by which to scale the critical height
animate_mov = True  # if ffmpeg not installed, animation will not work

initial_condition = 1
# 1 for gaussian bump
# 2 for delta function

# Physical Parameters
Sa = 0.1555     # absorption macroscopic cross-section (1/cm) of 4% enriched U-235
Sf = 0.5539     # fission macroscopic cross-section (1/cm) of 4% enriched U-235
St = 0.7235     # total macroscopic cross-section (1/cm) of 4% enriched U-235
init_E = 0.025  # neutron initial energy is approx 0.025 eV (room temp)
conv = 1.6e-19  # joules to ev
mass_n = 1.674e-27  # neutron mass(kg)
velo_n = np.sqrt(2*init_E*conv/mass_n)  # neutron velocity computed from energy
nu = 2.43       # average number of neutrons released per fission

D = velo_n/3/(St*100)         # diffusion coefficient
C = (nu*Sf - Sa)*velo_n*100   # creation coefficient

print(f"D = {D:0.4g}")
print(f"C = {C:0.4g}")

# System Analysis
Z_critical = np.pi * np.sqrt(3 * D / C)
R_critical = 2.404826 * np.sqrt(3 * D / 2 / C)  # from bessel function
V_critical = np.pi * (R_critical ** 2) * Z_critical

R = R_critical * rfactor
Z = Z_critical * zfactor

print(f' Critical radius is {R_critical:0.4g} meters')
print(f' Critical height is {Z_critical:0.4g} meters')
print(f' Critical volume is {V_critical:0.4g} cubic meters')

vol = Z * np.pi * (R ** 2)        # core volume

dr = R / (NR - 2.)                # radial grid spacing
dz = Z / (NZ - 2.)                # height grid spacing

tau = 0.5 / D / (1 / dr**2 + 1/dz**2)  # tau estimate

## initial conditions ##
rplot = (np.arange(0, NR) - 1 / 2) * dr
zplot = (np.arange(0, NZ) - 1 / 2) * dz
nn = np.zeros((NR, NZ))  # Initialize density to zero at all points

if initial_condition == 1:
    for r_idx in range(NR):
        for z_idx in range(NZ):
            # gaussian bump
            nn[r_idx][z_idx] = np.exp(-4/np.sqrt(dr*dz) * (rplot[r_idx] ** 2)) * np.exp(-4/np.sqrt(dr*dz) * ((zplot[z_idx]-zplot[int(NZ/2)])**2))
elif initial_condition == 2:
    nn[1][int(NZ/2)] = 1/dr
else:
    print("No Valid Initial Condition")
    assert False

## implement boundary conditions ##
# Homogeneous Inner Radius Neumann
nn[0, :] = nn[1, :]
# Outer Radius dn/dt = -n(1-r_leakage)/dr
nn[-1, :] = nn[-2, :] * (1 - r_leakage)
# Top dn/dt = -n(1-z_leakage)/dz
nn[1:(NR-1), 0] = nn[1:(NR-1), 1] * (1 - z_leakage)
# Bottom  dn/dt = =-n(1-z_leakage)/dz
nn[1:(NR-1), -1] = nn[1:(NR-1), -2] * (1 - z_leakage)

nn_new = np.copy(nn)

# plotting variables
speed = 10              # animation speed
pd = 20                 # number of iterations per plot for animation
num_plot = nstep/pd     # number of plots to generate

# function to compute average neutron density at a specific time
def avg_density(neu, rval, zval):
    avg_n = np.sum(np.matmul(rval[1:-1], neu[1:-1, 1:-1]) * dr * dz)
    avg_n = 2 * np.pi * avg_n / vol
    return avg_n

# initialize plotting arrays
nnplot = np.copy(nn)
nnmovplot = np.copy(nn)
tmovplot = np.array([0])
tplot = np.array([0])
nAve = np.array(avg_density(nn, rplot, zplot))

## FTCS Iteration ##
for istep in range(nstep + 1):
    # FTCS Scheme
    s1 = D * tau / (dr ** 2) * (nn[2:NR, 1:(NZ-1)] + nn[0:(NR-2), 1:(NZ-1)] - 2 * nn[1:(NR-1), 1:(NZ-1)])
    s2 = 0.5 * D * tau / dr * np.divide((nn[2:NR, 1:(NZ-1)] - nn[0:(NR-2), 1:(NZ-1)]), rplot[1:(NR - 1)][:, None])
    s3 = D * tau / (dz ** 2) * (nn[1:(NR-1), 2:NZ] + nn[1:(NR-1), 0:(NZ-2)] - 2 * nn[1:(NR-1), 1:(NZ-1)])
    s4 = (C * tau + 1) * nn[1:(NR - 1), 1:(NZ - 1)]
    nn_new[1:(NR-1), 1:(NZ-1)] = s1 + s2 + s3 + s4

    # Homogeneous Inner Radius Neumann Boundary Condition
    nn_new[0, :] = nn_new[1, :]

    # Outer Radius dn/dt = -n(1-r_leakage)/dr
    nn_new[-1, :] = nn_new[-2, :] * (1 - r_leakage)

    # Top dn/dt = -n(1-z_leakage)/dz
    nn_new[1:(NR-1), 0] = nn_new[1:(NR-1), 1] * (1 - z_leakage)

    # Bottom  dn/dt = =-n(1-z_leakage)/dz
    nn_new[1:(NR-1), -1] = nn_new[1:(NR-1), -2] * (1 - z_leakage)

    nn = np.copy(nn_new)

    # Record Density for Plotting
    if (istep % pd == 0):
        nnmovplot = np.block([[[nnmovplot]], [[nn]]])      # record for animation only num_plot times
        tmovplot = np.append(tmovplot, istep*tau)          # record time for animation

    nnplot = np.block([[[nnplot]], [[nn]]])                # record nn(i) for other plots each iteration
    tplot = np.append(tplot, istep * tau)                  # record time for plots
    nAve = np.append(nAve, avg_density(nn, rplot, zplot))  # record average density

# convert to microseconds before plotting
tplot = tplot * 1e6
tmovplot = tmovplot * 1e6

## PLOTTING ##
tt, rr_m = np.meshgrid(rplot, tplot)

# average neutron density vs radius vs time
fig = plt.figure(1)
plt.clf()
ax = plt.axes(projection='3d')
surf = ax.plot_surface(rr_m, tt, np.mean(nnplot, axis=2), rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
ax.set_xlabel('time t (ns)')
ax.set_ylabel('radius r (m)')
ax.set_zlabel('n(r, *, t)')
plt.suptitle(f'Neutron Diffusion for a Cylinder, r={round(R,5)}, z={round(Z,5)}')
plt.title('Mean over all heights at radius r')
plt.savefig(f'cylinder_neutron_radial_density{nstep}.png')
plt.show()

# animation of neutron density
zz, rr = np.meshgrid(zplot, rplot)
if(animate_mov):
    frate = 30
    fig = plt.figure(2)
    ax = plt.axes(projection='3d')
    plt.grid(True)
    surf_i = ax.plot_surface(rr, zz, nnmovplot[0, :, :], rstride=1, cstride=1, cmap=cm.jet, linewidth=0, vmin=0,
                             vmax=np.max(nnmovplot), antialiased=False)

    def update(n):
        ax.clear()
        surf = ax.plot_surface(rr, zz, nnmovplot[n, :, :], rstride=1, cstride=1, cmap=cm.jet, linewidth=0, vmin=0, vmax=np.max(nnmovplot), antialiased=False)
        ax.set_xlabel('radius r (m)')
        ax.set_ylabel('height z (m)')
        ax.set_zlabel('n(r, z, t)')
        ax.set_zlim([0, np.max(nnmovplot)])
        ax.set_ylim([0, Z])
        ax.set_xlim([0, R])
        surf.set_clim(vmin=0, vmax=np.max(nnmovplot))
        plt.suptitle(f'Neutron Diffusion for a Cylinder, r={round(R,5)}, z={round(Z,5)}')
        plt.title(f'time: {round(tmovplot[n], 10)} (ns)', fontsize=10)

    fig.colorbar(surf_i, shrink=0.5, aspect=5)
    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    anim = animation.FuncAnimation(fig, update, frames=int(num_plot), interval=speed*np.max(tplot)/num_plot)
    fig.suptitle('Neutron Diffusion', fontsize=14)
    FFwriter = animation.FFMpegWriter(fps=int(frate / 4))
    anim.save(f'cylinder_diffusion_{nstep}.mp4', writer=FFwriter)
    plt.close()

# average neutron density vs time
plt.figure(3)
plt.xlabel('time t (ns)')
plt.ylabel('mean neutron density')
plt.plot(tplot, nAve, 'r-', label='average neutron density')
plt.grid()
plt.legend()
plt.title('Neutron Diffusion')
plt.savefig(f'cylinder_avg_density_{nstep}.png')
plt.show()