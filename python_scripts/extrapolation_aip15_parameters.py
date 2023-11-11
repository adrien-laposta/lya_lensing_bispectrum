### The goal of this script is to extrapolate the
### parameters from Arinyo-i-Prats 2015 that
### describe the 3D Lyman-alpha forest power
### spectrum in order to get an estimate of
### the continuum misestimation bias in our
### bispectrum.
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from dict_tools import so_dict
import pandas as pd
import numpy as np
import argparse
import pickle
import os

plt.style.use("custom.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dict_file", help = "Dict file")

args = parser.parse_args()
dict_file = args.dict_file

d = so_dict()
d.read_from_file(dict_file)

# Define output directory
output_dir = d["aip15_extrapolation_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define parameter labels
latex_labels = {"beta": r"$\beta$",
                "bTauDelta": r"$b_{\tau\delta}$",
                "bTauEta": r"$b_{\tau\eta}$",
                "q1": r"$q_1$",
                "q2": r"$q_2$",
                "kp": r"$k_p$",
                "kvav": r"$k_v^{a_v}$",
                "av": r"$a_v$",
                "bv": r"$b_v$"}

# Define a function to perform a 2 parameters linear fit
def linear_fit(x, y, yerr):
    """
    Perform a linear fit to the dataset (x, y)
    that have errorbars yerr and return the values
    of the parameters a and b.
    """
    model = lambda v,a,b: a * v + b
    def chi2(p):
        a,b = p
        return np.sum((y-model(x,a,b))**2 / yerr ** 2)

    results = minimize(chi2, [1,1])
    return results.x

for data_kind in ["planck", "low_s8", "mid_s8", "hi_s8"]:

    # Define data paths
    csv_file = d[f"aip15_table_{data_kind}"]
    csv_errors_file = d[f"aip15_table_{data_kind}"].replace(".csv", "_errors.csv")

    # Load the tables
    df_params = pd.read_csv(csv_file)
    df_param_errors = pd.read_csv(csv_errors_file)

    ### Plot the parameters and the associated linear models
    ### and save the results into a pickle file.
    # Define the labels
    labels = list(df_params.columns)[1:]

    # Define a redshift range to plot the model
    z_array = np.linspace(2, 3.5, 50)

    result_dict = {}

    fig, axes = plt.subplots(3, 3, figsize = (13, 13))

    for i, name in enumerate(labels):

        id_row = i // 3
        id_col = i % 3
        ax = axes[id_row, id_col]
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(latex_labels[name])

        x,y,yerr = df_params["z"], df_params[name], df_param_errors[name]
        # Perform the linear fit
        if (name == "q2") and (data_kind in ["low_s8", "mid_s8", "hi_s8"]):
            a,b = 0., 0.
        else:
            a,b = linear_fit(x,y,yerr)
        result_dict[name] = {"a": a, "b": b}

        ax.errorbar(x, y, yerr, marker = "o")
        lims = ax.get_ylim()

        ax.axhline(0, color = "k", ls = "--", zorder = 0)
        ax.plot(z_array, a*z_array+b, color = "darkorange", lw = 2.)

        ax.set_ylim(lims)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/aip15_pars_{data_kind}.png", dpi = 300)

    pickle.dump(result_dict, open(d[f"aip15_extrapolation_file_{data_kind}"], "wb"))
