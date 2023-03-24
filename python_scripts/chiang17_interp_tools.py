from scipy.interpolate import interp2d, interp1d
import numpy as np
import pickle

def load_chiang17_file(file_name):
    """
    """
    table = np.loadtxt(file_name)
    k, p1d_z = table[:, 0], table[:, 1:]
    zs = ["2.2", "2.4", "2.6", "2.8", "3.0"]
    output_dict = {z: p1d_z[:, i] for i,z in enumerate(zs)}
    output_dict["k"] = k
    return output_dict


def logp1d_response_interpolator(p1d_dict, p1d_response_dict):
    """
    """
    #return a function
    k_data = p1d_dict["k"]
    zs = ["2.2", "2.4", "2.6", "2.8", "3.0"]
    y = np.array([p1d_response_dict[z]/p1d_dict[z] for z in zs])

    def interpolator(z, k):

        assert z >= 2.2, f"The value of z is below the interpolation range (z_min = 2.2)"
        assert z <= 3.0, f"The value of z is above the interpolation range (z_max = 3.0)"

        assert k.min() >= min(k_data), f"The value of k is below the interpolation range (k_min = {min(k_data)})"
        assert k.max() <= max(k_data), f"The value of k is below the interpolation range (k_max = {max(k_data)})"
        result = interp2d(k_data, [2.2, 2.4, 2.6, 2.8, 3.0], y)
        return result(k, z)

    return interpolator

def load_logp1d_response_ratio_pars(pickle_file_name):
    """
    """
    pars = pickle.load(open(pickle_file_name, "rb"))
    return pars

def model_logp1d_response_ratio(z, k, zref, **kwargs):
    """
    """
    X = (1 + z) / (1 + float(zref))
    a_of_z = kwargs["gamma"] * (X - 1) * X ** kwargs["eta"]
    b_of_z = kwargs["alpha"] * X - kwargs["alpha"]
    return k ** a_of_z * np.exp(b_of_z)

def logp1d_response_extrapolator(p1d_dict, p1d_response_dict, pars, zref):
    """
    """
    params = pars[zref]
    logp1d_response_interp = logp1d_response_interpolator(p1d_dict, p1d_response_dict)

    def extrapolator(z, k):

        ratio = model_logp1d_response_ratio(z, k, zref, **params)
        return ratio * logp1d_response_interp(float(zref), k)

    return extrapolator
