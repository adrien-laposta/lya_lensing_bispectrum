from lyman_p1d_tools import get_lyman_alpha_p1d, dP1D_ddelta_chiang17, dP1D_ddelta_aip15
from chiang17_interp_tools import load_chiang17_file
from cosmo_tools import CosmoTools
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from dict_tools import so_dict
import chiang17_interp_tools
import general_tools as gt
import forest_tools as ft
import bispec_tools as bt
import aip15_param_tools
import numpy as np
import matplotlib
import argparse
import pickle
import sys
import os

plt.style.use("custom.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-e", "--experiment", help = "Experiment to consider")
parser.add_argument("-d", "--dict_file", help = "Dict file")

args = parser.parse_args()
exp = args.experiment
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

plt.figure(figsize = (8, 6))

plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$\mathrm{d}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/\mathrm{d}\bar{\delta}$ [Mpc/h]")

kth = np.linspace(0.175, 1.5, 250)
z = 2.4

chiang17_extrapolation_pars = chiang17_interp_tools.load_logp1d_response_ratio_pars(d["chiang17_extrapolation_file"])
dp1d_ddelta_c17 = dP1D_ddelta_chiang17(kth, z, cosmology,
                                       chiang17_p1d, chiang17_response,
                                       chiang17_extrapolation_pars, zref = d["zref"])




aip15_pars = aip15_param_tools.aip15_par_extrapolation(z,
                                                           d["aip15_extrapolation_file_planck"])
aip15_pars_low_s8 = aip15_param_tools.aip15_par_extrapolation(z,
                                                                      d["aip15_extrapolation_file_low_s8"])
aip15_pars_mid_s8 = aip15_param_tools.aip15_par_extrapolation(z,
                                                                      d["aip15_extrapolation_file_mid_s8"])
aip15_pars_hi_s8 = aip15_param_tools.aip15_par_extrapolation(z,
                                                                      d["aip15_extrapolation_file_hi_s8"])
dp1d_ddelta_aip15 = dP1D_ddelta_aip15(kth, z, cosmology,
                                        aip15_pars_low_s8, aip15_pars_mid_s8,
                                        aip15_pars_hi_s8, **aip15_pars)
plt.plot(kth, dp1d_ddelta_c17, lw = 1.5, label = "Chiang et al. (2017) model")
plt.plot(kth, dp1d_ddelta_aip15, lw = 1., ls = (0, (5, 5)), label = "Finite differences with Arinyo-i-Prats et al. (2015)", color = "tab:red")

leg = plt.legend(frameon=True, fontsize = 13)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")

plt.title(r"$z_\mathrm{forest} = %.1f$" % z)

# Finalize plots
plt.xlim(kth.min(), kth.max())
#plt.ylim(0, 1.25)
plt.tight_layout()
plt.savefig(f"{output_dir}/dp1d_ddelta_comparison_paper.pdf")
