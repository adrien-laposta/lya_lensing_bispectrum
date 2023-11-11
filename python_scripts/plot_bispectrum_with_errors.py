import argparse
from dict_tools import so_dict
import pickle
import matplotlib.pyplot as plt
import numpy as np

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
output_dir = f"{output_dir}/{exp}"

suffix = ""
use_chiang17_extrapolation = d["extrapolate_chiang_response"]
if not(use_chiang17_extrapolation):
    suffix = "_finite_diff"

# Load result file
bispec_file_name = f"{output_dir}/bispec_{exp}{suffix}.pkl"
bispectrum_dict = pickle.load(open(bispec_file_name, "rb"))

# Define the different assumption for the correlation
# between the k-bins
correlation_assumption = {
    "No correlation": 0.,
    "Constant correlation [30%]": 0.3,
    "Constant correlation [50%]": 0.5,
    "Custom correlation [Doux et al. (2015)]": None
}

# Prepare the output file
ivw_results = {
    assumption: {
        "bispectrum": [],
        "covariance": []
    } for assumption in correlation_assumption
}

# Define the bins to combine together
# Note that we chose these combination
# because the bins at small scales have
# low SNR
bin_combination = {
    "B1": ["2.28"],
    "B2": ["2.43"],
    "B3": ["2.59"],
    "B4": ["2.75", "2.91"],
    "B5": ["3.07", "3.23", "3.39", "3.55"]
}

# Prepare the output file for the case
# where we merge some bins together
large_bins_results = {
    assumption: {
        bin_comb: {
            "bispectrum": [],
            "covariance": []
        } for bin_comb in bin_combination
    } for assumption in correlation_assumption
}

large_bins_nuisance = {
    bin_comb: {
        "cosmo": [],
        "dla": [],
        "cont": []
    } for bin_comb in bin_combination
}

for id_z, (zqso, results) in enumerate(bispectrum_dict.items()):

    print(f"Bin centered on {zqso}")

    k = results["k"]
    B_kappa_lya = results["bispectrum"]
    var_B_kappa_lya = results["var"] * 1e10
    std_B_kappa_lya = np.sqrt(var_B_kappa_lya)

    N = len(std_B_kappa_lya)
    for assumption, corr in correlation_assumption.items():

        if corr is None:

            cint, cmin, cmax = 0.2, 0.3, 0.7
            i, j = np.arange(N), np.arange(N)
            beta = (cint - cmin) / (N - 2)
            alpha = cmin - beta
            gamma = (cmax - alpha - beta * (N - 1)) / (N - 2)
            ii, jj = np.meshgrid(i, j)
            m = alpha + beta * ii + gamma * jj
            corrmat = np.triu(m) + np.triu(m).T

            corrmat -= np.diag(corrmat.diagonal())

            corrmat += np.diag(np.ones(N))

        else:
            corrmat = np.ones(shape = (N, N)) * corr
            corrmat += np.diag(np.ones(N)) * (1 - corr)

        cov = corrmat * np.outer(std_B_kappa_lya, std_B_kappa_lya)

        # Plot correlation matrix in each case
        if id_z == 0:
            plt.figure(figsize = (8, 6))
            plt.imshow(corrmat)
            plt.colorbar()
            plt.tight_layout()
            plt.savefig(f"{output_dir}/correlation_matrix_{assumption}.png", dpi = 300)

        snr = np.sqrt(B_kappa_lya @ np.linalg.inv(cov) @ B_kappa_lya)

        # Fill the results for IVW
        ivw_results[assumption]["covariance"].append(np.linalg.inv(cov))
        ivw_results[assumption]["bispectrum"].append(np.linalg.inv(cov) @ B_kappa_lya)

        for bin_comb, z_in_bin_list in bin_combination.items():
            if zqso in z_in_bin_list:
                large_bins_results[assumption][bin_comb]["covariance"].append(np.linalg.inv(cov))
                large_bins_results[assumption][bin_comb]["bispectrum"].append(np.linalg.inv(cov) @ B_kappa_lya)
                if corr is None:
                    large_bins_nuisance[bin_comb]["cosmo"].append(np.linalg.inv(cov) @ results["bispectrum_cosmo"])
                    large_bins_nuisance[bin_comb]["dla"].append(np.linalg.inv(cov) @ results["bispectrum_dla"])
                    large_bins_nuisance[bin_comb]["cont"].append(np.linalg.inv(cov) @ results["bispectrum_cont"])

        print(f"    SNR [{assumption}] = {snr:.2f}")

    print("")

print("Inverse variance weighting of the redshift bins")
for assumption in correlation_assumption:

    ivw_cov = np.linalg.inv(
        np.sum(
            ivw_results[assumption]["covariance"], axis = 0
        )
    )

    bispectrum = ivw_cov @ (
        np.sum(
            ivw_results[assumption]["bispectrum"], axis = 0
        )
    )

    snr = np.sqrt(bispectrum @ np.linalg.inv(ivw_cov) @ bispectrum)
    print(f"    SNR [{assumption}] = {snr:.2f}")
print("")

merged_bin_labels = {
    "B1": r"$2.20 \le z \le 2.35$",
    "B2": r"$2.35 \le z \le 2.51$",
    "B3": r"$2.51 \le z \le 2.67$",
    "B4": r"$2.67 \le z \le 2.99$",
    "B5": r"$2.99 \le z \le 3.63$"
}

print("=======================")
print("= Merged bins results =")
print("=======================")

plt.figure(figsize = (8.5, 20))
grid = plt.GridSpec(5, 1, hspace = 0, wspace = 0)

for id_bin, bin_comb in enumerate(bin_combination):
    print("")
    print(f"Results for bin {bin_comb} " + merged_bin_labels[bin_comb])
    print(f"--------------------------")
    if id_bin != 4:
        ax = plt.subplot(grid[id_bin], xticklabels = [])
    else:
        ax = plt.subplot(grid[id_bin])

    for assumption in correlation_assumption:

        large_bins_cov = np.linalg.inv(
            np.sum(large_bins_results[assumption][bin_comb]["covariance"], axis = 0)
        )

        bispectrum = large_bins_cov @ np.sum(large_bins_results[assumption][bin_comb]["bispectrum"], axis = 0)

        snr = np.sqrt(bispectrum @ np.linalg.inv(large_bins_cov) @ bispectrum)
        print(f"    SNR [{assumption}] = {snr:.2f}")

        if assumption == "Custom correlation [Doux et al. (2015)]":
            bispectrum_cosmo = large_bins_cov @ np.sum(large_bins_nuisance[bin_comb]["cosmo"], axis = 0)
            bispectrum_dla = large_bins_cov @ np.sum(large_bins_nuisance[bin_comb]["dla"], axis = 0)
            bispectrum_cont = large_bins_cov @ np.sum(large_bins_nuisance[bin_comb]["cont"], axis = 0)

            ax.axhline(0, color = "k", lw = 0.8)

            ax.plot(k, bispectrum_cosmo, label = "True signal" if id_bin == 4 else None, lw = 1.6)
            ax.plot(k, bispectrum_dla, label = "DLA contamination" if id_bin == 4 else None, lw = 0.4, ls = "--")
            ax.plot(k, bispectrum_cont, label = "Continuum bias" if id_bin == 4 else None, lw = 0.4, ls = "dotted")

            ax.errorbar(k, bispectrum, np.sqrt(large_bins_cov.diagonal()),
                        marker = "o", color = "tab:purple", markeredgecolor = "tab:purple",
                        markerfacecolor = "white", lw = 1.8, elinewidth = 2.,
                        capsize = 1.5, markersize = 8,
                        label = merged_bin_labels[bin_comb])

            ax.errorbar([], [], ls = "None", label = f"SNR = {snr:.2f}")

            ax.set_ylabel(r"$10^5B_{\kappa\mathrm{Ly}\alpha}$ [Mpc/h]")
            #ax.set_ylim(0, 5.5) # Planck+BOSS
            ax.set_ylim(0, 13.3) # SO+DESI-Y5
            #ax.set_ylim(0, 8.5) # ACT + DESI-Y1
            if id_bin == 4:
                ax.set_xlabel(r"$k_\parallel$ [h/Mpc]")
            ax.legend()

plt.tight_layout()
plt.savefig(f"{output_dir}/bispectrum_in_bins_paper.pdf")
