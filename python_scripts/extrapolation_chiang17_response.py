### The goal of this script is to extrapolate
### the response of the 1D Lyman alpha forest
### power spectrum to a large scale overdensity
### outside of the redshift range used in Chiang17
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse
from dict_tools import so_dict
import os

plt.style.use("custom.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dict_file", help = "Dict file")

args = parser.parse_args()
dict_file = args.dict_file

d = so_dict()
d.read_from_file(dict_file)

# Define data paths
p1d_file = d["chiang17_p1d_file"]
p1d_response_file = d["chiang17_response_file"]

# Define output directory
output_dir = d["chiang17_extrapolation_dir"]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the tables
p1d_table = np.loadtxt(p1d_file).T
p1d_response_table = np.loadtxt(p1d_response_file).T

# Store the p1d and the response in a dict
p1d_dict = {}
p1d_response_dict = {}

k_para, p1ds = p1d_table[0], p1d_table[1:]
p1d_responses = p1d_response_table[1:]

z_str_list = ["2.2", "2.4", "2.6", "2.8", "3.0"]

for i, z_str in enumerate(z_str_list):
    p1d_dict[z_str] = p1ds[i]
    p1d_response_dict[z_str] = p1d_responses[i]

# Plot the P1D and the response
plt.figure(figsize = (8, 6))
plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$P_{\mathrm{Ly}\alpha}^{\mathrm{1D}}$ [Mpc/h]")

for z_str in z_str_list:
    plt.plot(k_para, p1d_dict[z_str], label = f"z = {z_str} [Chiang et al. 17]")
plt.xlim(k_para.min(), k_para.max())
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/chiang17_p1d.png", dpi = 300)

plt.figure(figsize = (8, 6))
plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$dP_{\mathrm{Ly}\alpha}^{\mathrm{1D}}/d\delta$ [Mpc/h]")

for z_str in z_str_list:
    plt.plot(k_para, p1d_response_dict[z_str], label = f"z = {z_str} [Chiang et al. 17]")
plt.xlim(k_para.min(), k_para.max())
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/chiang17_p1d_response.png", dpi = 300)

# Define and plot the response of the log P1D
logp1d_response_dict = {k: p1d_response_dict[k]/p1d_dict[k] for k in p1d_dict.keys()}
logp1d_response_dict["k_para"] = k_para
plt.figure(figsize = (8, 6))
plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$d\mathrm{ln}P_{\mathrm{Ly}\alpha}^{\mathrm{1D}}/d\delta$ [Mpc/h]")

for z_str in z_str_list:
    plt.plot(k_para, logp1d_response_dict[z_str], label = f"z = {z_str} [Chiang et al. 17]")
plt.xlim(k_para.min(), k_para.max())
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/chiang17_logp1d_response.png", dpi = 300)

# Plot the ratio of the logp1d response with the response at a reference redshift
for z_str_ref in z_str_list:
    plt.figure(figsize = (8, 6))
    plt.xlabel(r"$k_\parallel$ [h/Mpc]")
    plt.ylabel(r"$d\mathrm{ln}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/d\delta(z)/d\mathrm{ln}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/d\delta(z_\mathrm{ref})$")
    for z_str in z_str_list:
        logp1d_response_ratio = logp1d_response_dict[z_str] / logp1d_response_dict[z_str_ref]

        plt.plot(k_para, logp1d_response_ratio, label = f"z = {z_str}")

    plt.title(r"$z_\mathrm{ref} = %s$" % z_str_ref)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(k_para.min(), k_para.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/logp1d_response_ratio_with_ref_at_z{z_str_ref}.png", dpi = 300)

### The goal is now to perform a linear fit of these
### ratio (in logscale), and to extrapolate them

def model(k, a, b):
    return k ** a * np.exp(b)

def logscale_linear_fit(k, logp1d_response_ratio):

    def chi2(p):
        a, b = p
        theory = model(k, a, b)
        residual = logp1d_response_ratio - theory
        sum_squared_distance = np.sum(residual ** 2)
        return sum_squared_distance

    results = minimize(chi2, [0,0])

    return results.x

# Perform the individual fits
output_parameters = {}
for z_ref_str in z_str_list:
    output_parameters[z_ref_str] = {}
    for z_str in z_str_list:

        logp1d_response_ratio = logp1d_response_dict[z_str] / logp1d_response_dict[z_ref_str]
        a, b = logscale_linear_fit(k_para, logp1d_response_ratio)

        output_parameters[z_ref_str][z_str] = {"a": a, "b": b}

# Plot the results
plt.figure(figsize = (8, 6))
plt.xlabel(r"$z$")
plt.ylabel(r"$a$")
for z_ref_str, par_dict in output_parameters.items():
    x = list(par_dict.keys())
    y = [par_dict[z]["a"] for z in x]
    x = [float(el) for el in x]
    plt.plot(x, y, label = r"$z_\mathrm{ref} = %s$" % z_ref_str)
plt.xlim(min(x), max(x))
plt.legend()
plt.savefig(f"{output_dir}/individual_linear_fit_parameter_a.png", dpi = 300)

plt.figure(figsize = (8, 6))
plt.xlabel(r"$z$")
plt.ylabel(r"$b$")
for z_ref_str, par_dict in output_parameters.items():
    x = list(par_dict.keys())
    y = [par_dict[z]["b"] for z in x]
    x = [float(el) for el in x]
    plt.plot(x, y, label = r"$z_\mathrm{ref} = %s$" % z_ref_str)
plt.xlim(min(x), max(x))
plt.legend()
plt.savefig(f"{output_dir}/individual_linear_fit_parameter_b.png", dpi = 300)

### Now we want to test some parametric expression for these two parameters
### to check if we can model the ratio of the logp1d response easily.

# First for the a parameter
def a_model(z, z_ref, gamma, eta):
    X = (1 + z) / (1 + z_ref)
    # X - 1 ensures that for z = z_ref a = 0.
    return gamma * (X - 1) * X ** eta

def a_fit(z, z_ref, a_values):

    def chi2(p):
        gamma,eta = p
        a_theory = a_model(z, z_ref, gamma, eta)
        residual = a_values - a_theory
        sum_squared_distance = np.sum(residual ** 2)

        return sum_squared_distance

    results = minimize(chi2, [1,1])
    return results.x

plt.figure(figsize = (8, 6))
plt.xlabel(r"$(1+z)/(1+z_\mathrm{ref})$")
plt.ylabel(r"$a$")
for z_ref_str, par_dict in output_parameters.items():
    z_ref = float(z_ref_str)
    z_array = np.array([float(k) for k in par_dict])

    y = np.array([par_dict[k]["a"] for k in par_dict.keys()])
    X = (1 + z_array) / (1 + z_ref)

    l, = plt.plot(X, y, label = r"$z_\mathrm{ref} = %s$" % z_ref_str)

    gamma, eta = a_fit(z_array, z_ref, y)
    a_theory = a_model(z_array, z_ref, gamma, eta)

    plt.plot(X, a_theory, color = l.get_color(), ls = "--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/individual_linear_fit_parameter_a_parametrized.png", dpi = 300)

# Then for the b parameter
def b_model(z, z_ref, alpha):#, beta):
    # with a linear model in (1+z)/(1+z_ref)
    X = (1 + z) / (1 + z_ref)
    return alpha * X - alpha#+ beta

def b_fit(z, z_ref, b_values):

    def chi2(p):
        alpha = p #,beta = p
        b_theory = b_model(z, z_ref, alpha)#, beta)
        residual = b_values - b_theory
        sum_squared_distance = np.sum(residual ** 2)

        return sum_squared_distance

    results = minimize(chi2, [1])
    return results.x

plt.figure(figsize = (8, 6))
plt.xlabel(r"$(1+z)/(1+z_\mathrm{ref})$")
plt.ylabel(r"$b$")
for z_ref_str, par_dict in output_parameters.items():
    z_ref = float(z_ref_str)
    z_array = np.array([float(k) for k in par_dict])

    y = np.array([par_dict[k]["b"] for k in par_dict.keys()])
    X = (1 + z_array) / (1 + z_ref)

    l, = plt.plot(X, y, label = r"$z_\mathrm{ref} = %s$" % z_ref_str)

    alpha = b_fit(z_array, z_ref, y)#, beta = b_fit(z_array, z_ref, y)
    b_theory = b_model(z_array, z_ref, alpha)

    plt.plot(X, b_theory, color = l.get_color(), ls = "--")
plt.legend()
plt.tight_layout()
plt.savefig(f"{output_dir}/individual_linear_fit_parameter_b_parametrized.png", dpi = 300)

### Now we can perform a global fit for the different reference redshifts,
### and save the associated set of parameters.

def model(k, z, z_ref, alpha, gamma, eta):#, beta, gamma, eta):
    X = (1+z)/(1+z_ref)
    a_of_z = gamma * (X - 1) * X ** eta
    b_of_z = alpha * X - alpha
    return k**a_of_z * np.exp(b_of_z)

def fit_logp1d_response_ratio(k, logp1d_response_ratio_dict, z_ref):

    def chi2(p):
        #alpha,beta,gamma,eta = p
        alpha,gamma,eta = p

        sum_squared_distance = 0
        for z_str, logp1d_response_ratio in logp1d_response_ratio_dict.items():
            z = float(z_str)
            theory = model(k, z, z_ref, alpha, gamma, eta)
            residual = logp1d_response_ratio - theory
            sum_squared_distance += np.sum(residual**2)
        return sum_squared_distance

    results = minimize(chi2, [-4,8,4])

    return results.x

output_parameters = {}
for z_ref_str in z_str_list:
    z_ref = float(z_ref_str)

    logp1d_response_ratio_dict = {
        z_str: logp1d_response_dict[z_str] / logp1d_response_dict[z_ref_str]
        for z_str in z_str_list
    }

    alpha, gamma, eta = fit_logp1d_response_ratio(k_para, logp1d_response_ratio_dict, z_ref)

    output_parameters[z_ref_str] = {"alpha": alpha,
                                    "gamma": gamma, "eta": eta}

    # Plot logp1d response ratio
    plt.figure(figsize = (8, 6))
    plt.xlabel(r"$k_\parallel$ [h/Mpc]")
    plt.ylabel(r"$d\mathrm{ln}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/d\delta(z)/d\mathrm{ln}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/d\delta(z_\mathrm{ref})$")
    plt.title(r"$z_\mathrm{ref} = %s$" % z_ref_str)

    for z_str, logp1d_response_ratio in logp1d_response_ratio_dict.items():
        z = float(z_str)
        l, = plt.plot(k_para, logp1d_response_ratio, label = f"z = {z_str}")
        theory = model(k_para, z, z_ref, alpha, gamma, eta)
        plt.plot(k_para, theory, color = l.get_color(), ls = "--")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlim(k_para.min(), k_para.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fit_logp1d_response_ratio_with_ref_at_z{z_ref_str}.png", dpi = 300)

    # Plot logp1d response that have to be multiplied with the P1D that we are using
    plt.figure(figsize = (8,6))
    plt.xlabel(r"$k_\parallel$ [h/Mpc]")
    plt.ylabel(r"$d\mathrm{ln}P^\mathrm{1D}_{\mathrm{Ly}\alpha}/d\delta(z)$")
    plt.title(r"$z_\mathrm{ref} = %s$" % z_ref_str)

    for z_str, logp1d_response_ratio in logp1d_response_ratio_dict.items():
        z = float(z_str)
        l, = plt.plot(k_para, logp1d_response_ratio * logp1d_response_dict[z_ref_str], label = f"z = {z_str}")
        theory = model(k_para, z, z_ref, alpha, gamma, eta)
        plt.plot(k_para, theory * logp1d_response_dict[z_ref_str], color = l.get_color(), ls = "--")
    plt.xlim(k_para.min(), k_para.max())
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fit_logp1d_response_with_ref_at_z{z_ref_str}.png", dpi = 300)

# Save useful quantities
pickle.dump(logp1d_response_dict, open(f"{output_dir}/chiang17_logp1d_response.pkl", "wb"))
pickle.dump(output_parameters, open(d["chiang17_extrapolation_file"], "wb"))
