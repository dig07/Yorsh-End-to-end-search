import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Computer Modern"]

cpudat = np.load('timing-data/scaling_logl_costs_cpu_new-2.npy')
gpudat = np.load('timing-data/scaling_logl_costs_gpu_new-2.npy')
gpudat2d = np.load('timing-data/logl_costs_gpu_new_2-log.npy')

Mc_range = np.logspace(0, 2, 51)
f_GW_init_range = np.logspace(-3, -1, 51)
Mc_grid, f_GW_init_grid = np.meshgrid(Mc_range, f_GW_init_range, indexing='ij')

plt.figure(figsize=(3.5,2.7), dpi=150)
plt.pcolormesh(Mc_range, f_GW_init_range, gpudat2d.T * 1e6, rasterized=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"$\mathcal{M}_c$ [$\mathrm{M}_\odot$]")
plt.ylabel(r"$f_\mathrm{in}$ [Hz]")
cbar = plt.colorbar(label="Statistic GPU wall-time [µs]")
plt.savefig("timings-2d.pdf", bbox_inches='tight')
plt.close()

Mcs = np.array([5, 20, 50])
finits = np.array([2e-3, 1e-2, 5e-2])
n_stat = np.logspace(0, np.log10(100_000), 51)

inds_x = [0, 1, 1, 2]
inds_y = [0, 1, 2, 2]

colors = ['C0', 'C1', 'C2', 'C3']
fig, ax = plt.subplots(figsize=(4,3), dpi=150)
for (ix, iy, c) in zip(inds_x, inds_y, colors):
    label = f'{Mcs[ix]}, {int(finits[iy]*1000)}'
    ax.loglog(n_stat, gpudat[ix,iy], c=c, lw=1.2, label=label)
    ax.axhline(cpudat[ix,iy], c=c, ls='--', lw=1.2)
    print(ix, iy, cpudat[ix,iy])
ax.legend(frameon=False, alignment="left", title_fontsize=10, loc="lower left", title=r'$\mathcal{M}_c [\mathrm{M}_\odot], f_\mathrm{in}[\mathrm{mHz}]$')
ax.set_ylabel('Cost per statistic evaluation (s)')
ax.set_xlabel('Batch size of source parameters')
ax.set_ylim(4e-7, 1e-1)
ax.set_xlim(1, 1e5)
plt.savefig("timings-1d-scaling.pdf", bbox_inches='tight')
plt.close()