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

print(f"Running code for {exp}")

output_dir = d["output_dir"]
output_dir = f"{output_dir}/{exp}"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Outputs will be saved in {output_dir}/")

use_chiang17_extrapolation = d["extrapolate_chiang_response"]
suffix = ""
if not(use_chiang17_extrapolation):
    print("The response to the overdensity will be computed using finite differences")
    suffix = "_finite_diff"

########################
### DEFINE COSMOLOGY ###
########################
print("Compute cosmology ...")
print("")
cosmo_params = d["cosmo_params"]
cosmology = CosmoTools(**cosmo_params)

######################
### LENSING WIENER ###
######################
lmax_lensing = 4000
if exp != "Planck+BOSS":
    lensing_file = f"{d['data_dir']}/lensing_noise/kappa_noise_{d['clkk_files'][exp]}.dat"
else:
    lensing_file = f"{d['data_dir']}/lensing_noise/kappa_noise_planck.dat"
l, nlkk = np.loadtxt(lensing_file).T
nlkk = nlkk[(l >= 30) & (l <= lmax_lensing)]
l = l[(l >= 30) & (l <= lmax_lensing)]
clkk = cosmology.camb_results.get_lens_potential_cls(lmax = lmax_lensing)[:, 0] * (2*np.pi) / 4
clkk = clkk[30:]

wiener_filter = clkk / (clkk + nlkk)
Wl = np.array([l, wiener_filter]).T
clkk_w = wiener_filter ** 2 * clkk
nlkk_w = wiener_filter ** 2 * nlkk

# Plot Lensing signal
plt.figure(figsize = (8, 6))
plt.plot(l, clkk, color = "k")
plt.plot(l, clkk_w, label = r"Wiener filtered $C_\ell^{\kappa\kappa}$")
plt.plot(l, nlkk_w, label = r"Wiener filtered $N_\ell^{\kappa\kappa}$")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$C_\ell^{\kappa\kappa}$")
plt.yscale("log")
plt.xscale("log")
plt.ylim(1e-10, 1e-5)
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.tight_layout()
plt.savefig(f"{output_dir}/lensing_ps_wiener.png", dpi =300)

var_clkk = gt.get_real_space_corr(0., l, clkk_w)
var_nlkk = gt.get_real_space_corr(0., l, nlkk_w)
var_lensing = var_clkk + var_nlkk
print("===============")
print("=== LENSING ===")
print("===============")
print(f"Std dev of the signal : {np.sqrt(var_clkk)}")
print(f"Std dev of the noise : {np.sqrt(var_nlkk)}")
print(f"SNR : {np.sqrt(var_clkk/var_nlkk)}")
print(f"Total stddev : {np.sqrt(var_lensing)}")
print("")

###############################
### LOAD THE CHIANG+17 DATA ###
###############################
chiang17_p1d = load_chiang17_file(d["chiang17_p1d_file"])
chiang17_response = load_chiang17_file(d["chiang17_response_file"])

##################################################
### ONE-DIMENSIONAL LYMAN ALPHA POWER SPECTRUM ###
##################################################
##### Load infos from J. Guy's github
##### found at https://github.com/julienguy/simplelyaforecast

zmin, zmax = 2.2, 3.6 # QSO z

# These redshifts correspond to QSO z
zcenter = np.array([2.12, 2.28, 2.43, 2.59, 2.75, 2.91, 3.07, 3.23, 3.39, 3.55])
dndz = np.array([96, 81, 65, 52, 40, 30, 22, 16, 11, 7])
dz = zcenter[1] - zcenter[0]

dndz = dndz[(zcenter >= zmin) & (zcenter <= zmax)]
zcenter = zcenter[(zcenter >= zmin) & (zcenter <= zmax)]

binning_edges = np.concatenate((zcenter - dz / 2, [zcenter[-1] + dz / 2]))
bin_center = (binning_edges[:-1] + binning_edges[1:]) / 2

# Define the number density per square degree
nz = dndz * (binning_edges[1:] - binning_edges[:-1])
print(f"Number of L.O.S per sqdeg : {np.sum(nz)}")

Nz_act_desi = nz * d["areas"]["ACT+DESI-Y1"]
Nz_so_desi = nz * d["areas"]["SO+DESI-Y5"]
sv_area_planck_boss = 87085 / np.sum(nz)
Nz_planck_boss = nz * sv_area_planck_boss


# Plot number density
plt.figure(figsize = (8, 6))
plt.xlabel(r"$z$")
plt.ylabel(r"$N(z)$") #[$\mathrm{deg}^{-2}$]")
plt.bar(bin_center, Nz_so_desi,
        width = binning_edges[1:] - binning_edges[:-1],
        edgecolor = "k", linewidth = 1.4, facecolor = "darkblue",alpha=0.8,
        label = r"SO+DESI-Y5 [$N_\mathrm{tot} = %d$]" % np.sum(Nz_so_desi))
plt.bar(bin_center, Nz_planck_boss,
        width = binning_edges[1:] - binning_edges[:-1],
        edgecolor = "k", linewidth = 1.4, facecolor = "darkorange",alpha=0.8,
        label = "Planck+BOSS [$N_\mathrm{tot} = %d$]" % np.sum(Nz_planck_boss))
plt.bar(bin_center, Nz_act_desi,
        width = binning_edges[1:] - binning_edges[:-1],
        edgecolor = "k", linewidth = 1.4, facecolor = "forestgreen",alpha=0.8,
        label = "ACT+DESI-Y1 [$N_\mathrm{tot} = %d$]" % np.sum(Nz_act_desi))
leg = plt.legend(frameon = True, fontsize = 13)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")

plt.tight_layout()
plt.savefig(f"{output_dir}/number_density_paper.pdf")

# Read SNR file and get snr and z lists
snr_file = d["snr_file"]
snr_fits = fits.open(snr_file)[1].data
z_jguy = snr_fits["Z"][snr_fits["QSOTARGET"] == 1].clip(1e-5, 1e5)
snr_jguy = snr_fits["SNR"][snr_fits["QSOTARGET"] == 1].clip(1e-5, 1e5)

##### Binning properties
k_step = 0.1
k_h_over_Mpc = np.arange(chiang17_p1d["k"].min() - k_step / 2, 1.5+k_step, k_step)
k_center = (k_h_over_Mpc[1:] + k_h_over_Mpc[:-1]) / 2
mask = (k_center >= chiang17_p1d["k"].min()) & (k_center <= chiang17_p1d["k"].max())
k_center = k_center[mask]

k_th = np.linspace(k_center.min(), k_center.max(), 50)
k_plot_p1d = np.linspace(1e-3, 1.5, 200)

##### Total number of l-o-s
if exp != "Planck+BOSS":
    survey_area = d["areas"][exp]
else:
    survey_area = 87085 / np.sum(nz) # This number is extracted from Doux+15
ntot = nz * survey_area

##### P1D computation
print("=======================")
print("=== P1D computation ===")
print("=======================")

p1d_results = {}

norm = matplotlib.colors.Normalize(vmin = bin_center.min(),
                                   vmax = bin_center.max())
cmap = matplotlib.cm.ScalarMappable(norm = norm, cmap = matplotlib.cm.jet)
cmap.set_array([])

plt.figure(figsize = (8, 6))
plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$P^\mathrm{1D}_{\mathrm{Ly}\alpha}$ [Mpc/h]")

for i, zcenter in enumerate(bin_center):

    print(f"Computing the variance of the 1D power spectrum at z = {zcenter:.2f} ...")

    # Compute forest length
    rf_min_wl = 1040
    rf_max_wl = 1200
    redshift_center_forest, forest_length, z_length = ft.get_forest_length(zcenter, rf_min_wl, rf_max_wl, cosmology)

    # Plot
    plt.plot(k_plot_p1d, get_lyman_alpha_p1d(k_plot_p1d, redshift_center_forest, cosmology, use_npd13=True),
             color = cmap.to_rgba(zcenter),
             label = r"$\langle z_\mathrm{QSO} \rangle = %.2f$" % zcenter)
    # Compute forest power spectrum
    if use_chiang17_extrapolation:
        p1d = get_lyman_alpha_p1d(k_center, redshift_center_forest,
                                  cosmology, use_npd13 = True)
    else:
        aip15_pars = aip15_param_tools.aip15_par_extrapolation(redshift_center_forest,
                                                               d["aip15_extrapolation_file_planck"])
        p1d = get_lyman_alpha_p1d(k_center, redshift_center_forest,
                                  cosmology, use_npd13 = False, **aip15_pars)

    # Compute mean snr and noise power spectrum
    mask = (z_jguy >= binning_edges[i]) & (z_jguy <= binning_edges[i + 1])
    snr_cut = snr_jguy[mask]
    snr_mean = np.mean(snr_cut)

    p_noise = ft.p_noise_lya(snr_mean, zcenter, cosmology.H)

    # Apply spectrograph resolution
    # This value is an estimation from NPD+13 for BOSS
    # for DESI ?
    R_boss = 0.77
    delta_chi = 1.05 # 2pi/6h/Mpc
    w_spectro = ft.window_spectro(k_center, R_boss, delta_chi)
    p_noise = p_noise / w_spectro ** 2

    # Compute errors
    n_modes = k_step * forest_length / (2 * np.pi)
    p_tot = p1d + p_noise
    var_p_tot = 2 * p_tot ** 2 / n_modes
    cosmic_var = 2 * p1d ** 2 / n_modes

    p1d_results[zcenter] = {
        "k": k_center,
        "p1d": p1d,
        "p_noise": p_noise,
        "var": var_p_tot,
        "cosmic_var": cosmic_var,
        "length": forest_length,
        "z_length": z_length,
        "z_forest": redshift_center_forest
    }

plt.xlim(0, 1.5)
plt.ylim(0, 0.7)
plt.legend(ncol = 2)
plt.tight_layout()
plt.savefig(f"{output_dir}/lyman_p1d_paper.pdf")

pickle.dump(p1d_results, open(f"{output_dir}/p1d_forest{suffix}.pkl", "wb"))
print("Done !")
print("")

##############################
### BISPECTRUM COMPUTATION ###
##############################
print("==============================")
print("=== Bispectrum computation ===")
print("==============================")

bispec_results = {}
for i, zcenter in enumerate(bin_center):
    print(f"Computing the variance of the bispectrum at z = {zcenter:.2f} ...")
    p1d_result = p1d_results[zcenter]

    aip15_pars = aip15_param_tools.aip15_par_extrapolation(p1d_result["z_forest"],
                                                           d["aip15_extrapolation_file_planck"])

    # Compute the response to an overdensity
    if use_chiang17_extrapolation:
        chiang17_extrapolation_pars = chiang17_interp_tools.load_logp1d_response_ratio_pars(d["chiang17_extrapolation_file"])
        dp1d_ddelta = dP1D_ddelta_chiang17(k_center, p1d_result["z_forest"], cosmology,
                                           chiang17_p1d, chiang17_response,
                                           chiang17_extrapolation_pars, zref = d["zref"])

    else:
        aip15_pars_low_s8 = aip15_param_tools.aip15_par_extrapolation(p1d_result["z_forest"],
                                                                      d["aip15_extrapolation_file_low_s8"])
        aip15_pars_mid_s8 = aip15_param_tools.aip15_par_extrapolation(p1d_result["z_forest"],
                                                                      d["aip15_extrapolation_file_mid_s8"])
        aip15_pars_hi_s8 = aip15_param_tools.aip15_par_extrapolation(p1d_result["z_forest"],
                                                                      d["aip15_extrapolation_file_hi_s8"])
        dp1d_ddelta = dP1D_ddelta_aip15(k_center, p1d_result["z_forest"], cosmology,
                                        aip15_pars_low_s8, aip15_pars_mid_s8,
                                        aip15_pars_hi_s8, **aip15_pars)

    # Get the bispectrum
    cosmo_signal = bt.get_bispectrum(k_center, p1d_result["z_length"], Wl, p1d_result["z_forest"], cosmology, dp1d_ddelta)

    # DLA bias
    dla_bias = bt.get_dla_bispectrum(k_center, p1d_result["z_length"], Wl, p1d_result["z_forest"], cosmology, p1d_result["p1d"])

    # Continuum bias
    continuum_bias = bt.get_continuum_bispectrum(p1d_result["z_length"], Wl,
                                                 p1d_result["z_forest"], cosmology,
                                                 p1d_result["p1d"], **aip15_pars)

    # Total signal
    B_kappa_lya = (cosmo_signal + dla_bias + continuum_bias) * 1e5

    # Theory
    if use_chiang17_extrapolation:
        p1d_th = get_lyman_alpha_p1d(k_th, p1d_result["z_forest"],
                                     cosmology, use_npd13 = True)

        dp1d_ddelta_th = dP1D_ddelta_chiang17(k_th, p1d_result["z_forest"], cosmology,
                                             chiang17_p1d, chiang17_response,
                                             chiang17_extrapolation_pars, zref = d["zref"])
    else:
        p1d_th = get_lyman_alpha_p1d(k_th, p1d_result["z_forest"],
                                     cosmology, use_npd13 = False, **aip15_pars)
        dp1d_ddelta_th = dP1D_ddelta_aip15(k_th, p1d_result["z_forest"], cosmology,
                                           aip15_pars_low_s8, aip15_pars_mid_s8,
                                           aip15_pars_hi_s8, **aip15_pars)

    cosmo_signal_th = bt.get_bispectrum(k_th, p1d_result["z_length"], Wl, p1d_result["z_forest"], cosmology, dp1d_ddelta_th)
    dla_bias_th = bt.get_dla_bispectrum(k_th, p1d_result["z_length"], Wl, p1d_result["z_forest"], cosmology, p1d_th)
    continuum_bias_th = bt.get_continuum_bispectrum(p1d_result["z_length"], Wl,
                                                    p1d_result["z_forest"], cosmology,
                                                    p1d_th, **aip15_pars)
    B_kappa_lya_th = (cosmo_signal_th + dla_bias_th + continuum_bias_th) * 1e5

    # Variance of the bispectrum
    var_b = p1d_result["var"] * var_lensing / ntot[i]

    bispec_results[f"{zcenter:.2f}"] = {
        "k": k_center,
        "kth": k_th,
        "bispectrum": B_kappa_lya,
        "bispectrum_theory": B_kappa_lya_th,
        "bispectrum_cosmo": cosmo_signal * 1e5,
        "bispectrum_dla": dla_bias * 1e5,
        "bispectrum_cont": continuum_bias * 1e5,
        "var": var_b
    }

pickle.dump(bispec_results, open(f"{output_dir}/bispec_{exp}{suffix}.pkl", "wb"))
print("Done !")
print("")

for i, zcenter in enumerate(bin_center):

    bispec = bispec_results[f"{zcenter:.2f}"]
    forest = p1d_results[zcenter]

    kb, b = bispec["k"], bispec["bispectrum"]
    kp, p = forest["k"], forest["var"]

    print(b/1e5/np.sqrt(p*var_lensing))

