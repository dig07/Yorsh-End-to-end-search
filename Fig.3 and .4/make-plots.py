import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

cpudat = np.load('timing-data/scaling_logl_costs_cpu_new-2.npy')
gpudat = np.load('timing-data/scaling_logl_costs_gpu_new-2.npy')

cpudat2d = np.load('timing-data/logl_costs_cpu_new_2-log.npy')
gpudat2d = np.load('timing-data/logl_costs_gpu_new_2-log.npy')

Mc_range = np.logspace(0, 2, 51)
f_GW_init_range = np.logspace(-3, -1, 51)
Mc_grid, f_GW_init_grid = np.meshgrid(Mc_range, f_GW_init_range, indexing='ij')

plt.figure(figsize=(4.,3.), dpi=150)
plt.pcolormesh(Mc_range, f_GW_init_range, gpudat2d.T * 1e6, rasterized=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Chirp mass, $\mathcal{M}_c$ [$M_\odot$]")
plt.ylabel(r"Initial GW frequency, $f_\mathrm{in}$ [Hz]")
cbar = plt.colorbar(label="GPU Likelihood cost [µs]")
plt.savefig("timings-2d.pdf", bbox_inches='tight')
plt.close()

Mcs = np.array([5, 20, 50])
finits = np.array([2e-3, 1e-2, 5e-2])
n_stat = np.logspace(0, np.log10(100_000), 51)

inds_x = [1, 2, 1, 2]
inds_y = [0, 0, 2, 2]

colors = ['C0', 'C1', 'C2', 'C3']
lses = ['-', '--', '-', '-']
fig, ax = plt.subplots(figsize=(5,4), dpi=150)
for (ix, iy, c, ls) in zip(inds_x, inds_y, colors, lses):
    ax.loglog(n_stat, gpudat[ix,iy], c=c, ls=ls, lw=1.2, label=fr'$\mathcal{{M}}_c={Mcs[ix]}, f_\mathrm{{low}}={finits[iy]}$')
    ax.axhline(cpudat[ix,iy], c=c, ls=ls, lw=0.6)
    print(ix, iy, cpudat[ix,iy])
ax.legend(frameon=False)
ax.set_ylabel('Cost per statistic evaluation (s)')
ax.set_xlabel('Batch size on GPU')
plt.savefig("timings-1d-scaling.pdf", bbox_inches='tight')
plt.close()