# Mason Weiss
# PHYS416 - Fick's Diffusion Model
# April 2024

# Spherical Configuration

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.animation as animation

# Model Parameters
N = 61          # number of radial grid points, assuming symmetry in theta and phi
leakage = 0.5   # leakage percentage at barrier (as a decimal)
nstep = 10000   # number of timesteps
factor = 0.8    # factor by which to scale the dirichlet critical radius
animate_mov = True  # if ffmpeg not installed, animation will not work

initial_condition = 1
# 1 for gaussian bump
# 2 for delta function

# Physical Parameters
Sa = 0.1555     # absorption macroscopic cross-section (1/cm) of 4% enriched U-235
Sf = 0.5539     # fission macroscopic cross-section (1/cm) of 4% enriched U-235
St = 0.7235     # total macroscopic cross-section (1/cm) of 4% enriched U-235
init_E = 0.025  # neutron initial energy is 0.025 eV (room temp, thermal)
conv = 1.6e-19  # joules to ev
mass_n = 1.674e-27  # neutron mass(kg)
velo_n = np.sqrt(2*init_E*conv/mass_n)  # neutron velocity computed from energy
nu = 2.43       # average number of neutrons released per fission

D = velo_n/3/(St*100)         # diffusion coefficient
C = (nu*Sf - Sa)*velo_n*100   # creation coefficient

print(f"D = {D:0.4g}")
print(f"C = {C:0.4g}")

# System Analysis
R_critical = np.pi * np.sqrt(D / C)
V_critical = np.pi * 4/3 * (R_critical**3)
print(f'Critical radius is {R_critical:0.4g} meters')
print(f'Critical volume is {V_critical:0.4g} cubic meters')

# Set system size
R = R_critical * factor         # core radius
vol = np.pi * 4/3 * (R**3)      # core volume
h = R / (N - 2.)                # grid spacing
tau = 0.5 * h ** 2 / D          # maximum timestep for stability

## initial conditions
xplot = (np.arange(0, N)-1/2) * h
nn = np.zeros(N)  # Initialize density to zero at all points

if initial_condition == 1:
    for idx in range(N):
        # gaussian bump centered at origin
        nn[idx] = np.exp(-N/(2*h) * (xplot[idx] ** 2))
elif initial_condition == 2:
    nn[1] = 1/h
else:
    print("No Valid Initial Condition")
    assert False

## implement boundary conditions
# Homogeneous Left Neumann Boundary Condition
nn[0] = nn[1]
# Right Boundary Condition dn/dt = -n(1-leakage)/h
nn[-1] = nn[-2] * (1-leakage)

nn_new = np.copy(nn)

# plotting variables
speed = 10              # animation speed
pd = 20                 # number of plots for animation
num_plot = nstep/pd     # number of plots to generate

# function to compute average neutron density at a specific time
def avg_density(neu, xval):
    avg_n = np.sum(np.multiply(neu[1:-1], np.square(xval[1:-1]))) * h
    avg_n = 4 * np.pi * avg_n / vol
    return avg_n


# initialize plotting arrays
nnplot = np.copy(nn)
nnmovplot = np.copy(nn)
tmovplot = np.array([0])
tplot = np.array([0])
nAve = np.array(avg_density(nn, xplot))

## FTCS Iteration ##
for istep in range(nstep + 1):
    # FTCS Scheme
    nn_new[1:(N - 1)] = D*tau/(h**2) * (nn[2:N] + nn[0:(N - 2)] - 2 * nn[1:(N - 1)]) + (C*tau+1) * nn[1:(N - 1)] + D*tau/h * np.divide((nn[2:N] - nn[0:(N-2)]),xplot[1:(N-1)])

    # Homogeneous Left Neumann Boundary Condition
    nn_new[0] = nn_new[1]

    # Right Boundary Condition dn/dt = -n(1-leakage)/h
    nn_new[-1] = nn_new[-2] * (1-leakage)

    nn = np.copy(nn_new)

    # Record values for plotting
    if (istep % pd == 0):
        nnmovplot = np.vstack((nnmovplot, nn))      # record for animation only num_plot times
        tmovplot = np.append(tmovplot, istep*tau)   # record time for animation

    nnplot = np.vstack((nnplot, nn))                # record nn(i) for other plots each iteration
    tplot = np.append(tplot, istep * tau)           # record time for plots
    nAve = np.append(nAve, avg_density(nn, xplot))  # record average density

tplot = tplot * 1e6     # convert to µs before plotting
tmovplot = tmovplot * 1e6


## PLOTTING ##
# meshgrid plot of diffusion
tt, xx = np.meshgrid(xplot, tplot)
fig = plt.figure(1)
plt.clf()
ax = plt.axes(projection='3d')
ax.set_xlabel('time t (µs)')
ax.set_ylabel('radius r (m)')
ax.set_zlabel('n(r,t)')
surf = ax.plot_surface(xx, tt, nnplot, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
plt.title(f'Neutron Diffusion for a Sphere, r={round(R,5)}', fontsize=14)
plt.savefig(f'sphere_diffusion_{nstep}.png')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# animation of neutron density
if(animate_mov):
    frate = 30
    fig, ax = plt.subplots(1, 1)
    plt.xlabel('radius r (m)')
    plt.ylabel('n(r,t)')
    plt.grid(True)

    def update(n):
        ax.clear()
        ax.set_ylim([0, 1.1*np.max(nnmovplot)])
        ax.set_xlim([0, R])
        ax.plot(xplot, nnmovplot[0, :], '-', label=f'initial time')
        ax.plot(xplot, nnmovplot[n-1, :], '--', label=f'current time')
        plt.suptitle(f'Neutron Diffusion for a Sphere, r={round(R,5)}', fontsize=14)
        plt.title(f'time: {round(tmovplot[n], 10)} µs', fontsize=10)

    plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
    anim = animation.FuncAnimation(fig, update, frames=int(num_plot-2), interval=speed*np.max(tplot)/num_plot)
    FFwriter = animation.FFMpegWriter(fps=int(frate / 4))
    anim.save(f'sphere_diffusion_{nstep}.mp4', writer=FFwriter)
    plt.close()

# average neutron density vs time
plt.figure(3)
plt.xlabel('time t (µs)')
plt.ylabel('mean neutron density')
plt.plot(tplot, nAve, 'r-', label='average neutron density')
plt.grid()
plt.legend()
plt.title(f'Neutron Diffusion for a Sphere, r={round(R,5)}', fontsize=14)
plt.savefig(f'sphere_avg_density_{nstep}.png')
plt.show()

