import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import pickle

def mean_transmission(z):
    """
    """
    return np.exp(-0.0023 * (1 + z) ** 3.65)

def interpolate_aip15_params(file_name):
    """
    """
    df = pd.read_csv(file_name)
    z_list = df["z"]
    interpolation_dict = {}
    for par_name in df:
        if par_name == "z": continue
        par_interp = interp1d(z_list, df[par_name])
        interpolation_dict[par_name] = par_interp
    return z_list, interpolation_dict

def aip15_pars(z, file_name):
    """
    """
    z_list, interpolation_dict = interpolate_aip15_params(file_name)
    assert z >= min(z_list), f"The value of z is below the interpolation range (z_min = {min(z_list)})"
    assert z <= max(z_list), f"The value of z is above the interpolation range (z_max = {max(z_list)})"

    par_dict = {}
    for par_name, par_interp in interpolation_dict.items():
        par_dict[par_name] = par_interp(z)

    par_dict["bF"] = par_dict["bTauDelta"] * np.log(mean_transmission(z))

    return par_dict

def aip15_par_extrapolation(z, aip15_extrapolation_file):
    """
    """
    fit_params = pickle.load(open(aip15_extrapolation_file, "rb"))
    par_dict = {}
    for par_name, ab_fit in fit_params.items():
        par_dict[par_name] = ab_fit["a"] * z + ab_fit["b"]
    par_dict["bF"] = par_dict["bTauDelta"] * np.log(mean_transmission(z))
    return par_dict
