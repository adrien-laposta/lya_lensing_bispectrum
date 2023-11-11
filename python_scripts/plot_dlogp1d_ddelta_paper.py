from dict_tools import so_dict
import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from cosmo_tools import CosmoTools
import chiang17_interp_tools
from chiang17_interp_tools import load_chiang17_file


plt.style.use("custom.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dict_file", help = "Dict file")

args = parser.parse_args()
dict_file = args.dict_file

d = so_dict()
d.read_from_file(dict_file)

output_dir = d["output_dir"]

########################
### DEFINE COSMOLOGY ###
########################
print("Compute cosmology ...")
print("")
cosmo_params = d["cosmo_params"]
cosmology = CosmoTools(**cosmo_params)

###############################
### LOAD THE CHIANG+17 DATA ###
###############################
chiang17_p1d = load_chiang17_file(d["chiang17_p1d_file"])
chiang17_response = load_chiang17_file(d["chiang17_response_file"])
chiang17_extrapolation_pars = chiang17_interp_tools.load_logp1d_response_ratio_pars(d["chiang17_extrapolation_file"])

z_forest_list = np.array([2.00, 2.15, 2.29, 2.44, 2.59, 2.73, 2.88, 3.03, 3.18])

norm = matplotlib.colors.Normalize(vmin = z_forest_list.min(),
                                   vmax = z_forest_list.max())
cmap = matplotlib.cm.ScalarMappable(norm = norm, cmap = matplotlib.cm.jet)
cmap.set_array([])

plt.figure(figsize = (8, 6))
grid = plt.GridSpec(1, 20, hspace=0, wspace=0)
main = plt.subplot(grid[0, :-1])
cbar = plt.subplot(grid[0, -1], xticklabels = [], xticks = [])
colorbar = plt.colorbar(cmap, cax = cbar)
colorbar.set_label(r"$\bar{z}_\mathrm{forest}$")
main.set_xlabel(r"$k_\parallel$ [h/Mpc]")
main.set_ylabel(r"$\mathrm{dln}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/\mathrm{d}\bar{\delta}$ [Mpc/h]")

kth = np.linspace(0.175, 1.5, 500)
for z_forest in z_forest_list:

    if 2.2 <= z_forest <= 3.0:
        logp1d_response = chiang17_interp_tools.logp1d_response_interpolator(chiang17_p1d,
                                                                             chiang17_response)
    else:
        logp1d_response = chiang17_interp_tools.logp1d_response_extrapolator(chiang17_p1d, chiang17_response,
                                                                             chiang17_extrapolation_pars, d["zref"])

    d_logp1d_ddelta = logp1d_response(z_forest, kth)

    main.plot(kth, d_logp1d_ddelta, color = cmap.to_rgba(z_forest), ls = (0, (5, 5)),
              lw = 0.4)


for z_forest in [2.2, 2.4, 2.6, 2.8, 3.0]:

    logp1d_response = chiang17_interp_tools.logp1d_response_interpolator(chiang17_p1d, chiang17_response)
    main.plot(kth, logp1d_response(z_forest, kth), color = cmap.to_rgba(z_forest),
              lw = 1.6)

main.plot([], [], color = "gray", alpha = 0.5, ls = "solid", lw = 1, label = "Chiang et al. data")
main.plot([], [], color = "gray", alpha = 0.5, ls = (0, (5, 5)), lw = 0.8, label = "Interpolation/Extrapolation")

leg = main.legend(frameon=True, fontsize = 13)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")

# Finalize plots
main.set_xlim(kth.min(), kth.max())
main.set_ylim(0, 1.25)
plt.tight_layout()
plt.savefig(f"{output_dir}/dlogp1d_ddelta_paper.pdf")
