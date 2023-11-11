from cosmo_tools import CosmoTools
from dict_tools import so_dict
import argparse
import matplotlib.pyplot as plt
import numpy as np
from general_tools import get_real_space_corr
import scipy.integrate

plt.style.use("custom.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dict_file", help = "Dict file")

args = parser.parse_args()
dict_file = args.dict_file

d = so_dict()
d.read_from_file(dict_file)

output_dir = d["output_dir"]

cosmo_params = d["cosmo_params"]
cosmology = CosmoTools(**cosmo_params)

lmax = 4000
l = np.arange(lmax + 1)
clkk = cosmology.camb_results.get_lens_potential_cls(lmax = lmax)[:, 0] * (2 * np.pi) / 4
l, clkk = l[30:], clkk[30:]

plt.figure(figsize = (8, 6))
plt.xlabel(r"$L$")
plt.ylabel(r"$C_L^{\kappa\kappa}$")

plt.plot(l, clkk, color = "k")

exp_labels = {
    "planck": "Planck 2018",
    "dr6_prelim": "ACT DR6 preliminary",
    "so": "SO baseline",
    "cmbs4": "CMB-S4"
}

wiener_results = {}
for exp, exp_label in exp_labels.items():

    lensing_file = f"{d['data_dir']}/lensing_noise/kappa_noise_{exp}.dat"
    l, nlkk = np.loadtxt(lensing_file).T
    nlkk = nlkk[(l >= 30) & (l <= lmax)]
    l = l[(l >= 30) & (l <= lmax)]

    wiener_filter = clkk / (clkk + nlkk)
    wiener_filtered_signal_and_noise = clkk ** 2 / (clkk + nlkk)

    wiener_results[exp] = {"wiener_filter": wiener_filter,
                           "wiener_filtered_signal_and_noise": wiener_filtered_signal_and_noise}

    plt.plot(l, nlkk, label = exp_label)

plt.yscale("log")
plt.xscale("log")
plt.xlim(30, lmax)
plt.ylim(1e-9, 1e-6)
leg = plt.legend(frameon=True, loc = "lower right")
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.tight_layout()
plt.savefig(f"{output_dir}/lensing_noise_summary_paper.pdf")


def get_95_contours(ell, w_ell, theta_rad):
    corr = get_real_space_corr(theta_rad, ell, w_ell)
    tot_area = scipy.integrate.trapz(corr, theta_rad)
    theta_tmp = np.deg2rad(0.01) # rad
    step = np.deg2rad(0.005)
    area = 0.
    while area / tot_area < 0.95:
        theta_tmp += step
        theta_vec = np.linspace(-theta_tmp, theta_tmp, 1000)
        corr = get_real_space_corr(theta_vec, ell, w_ell)
        area = scipy.integrate.trapz(corr, theta_vec)
    return np.rad2deg(theta_tmp)

def find_roots_of_array(array_x, array_y):

    right_roll = np.roll(array_y, 1)
    left_roll = np.roll(array_y, -1)

    right_sign = np.sign(right_roll * array_y)
    left_sign = np.sign(left_roll * array_y)

    right_roots = array_x[right_sign == -1.]
    left_roots = array_x[left_sign == -1.]

    return (right_roots + left_roots) / 2

theta_degree = np.linspace(-6, 6, 1500)
theta_rad = np.deg2rad(theta_degree)

def normalize_max(f, *args, **kwargs):
    y = f(*args, **kwargs)
    return y / y.max()

plt.figure(figsize = (8, 6))
plt.xlabel(r"$\theta$ [degrees]")
plt.ylabel(r"Angular correlation $\xi(\theta)$")

for exp, exp_label in exp_labels.items():

    wiener_filter = wiener_results[exp]["wiener_filter"]
    wiener_filtered_signal_and_noise = wiener_results[exp]["wiener_filtered_signal_and_noise"]

    wiener_filter_corr = normalize_max(get_real_space_corr, theta_rad, l, wiener_filter)
    wiener_filtered_corr = normalize_max(get_real_space_corr, theta_rad, l, wiener_filtered_signal_and_noise)

    line, = plt.plot(theta_degree, wiener_filter_corr, lw = 0.3, ls = (0, (5, 5)))#ls = "--")
    plt.plot(theta_degree, wiener_filtered_corr, color = line.get_color(), label = exp_label)

    wiener_filter_radius = get_95_contours(l, wiener_filter, theta_rad)
    wiener_filtered_radius = get_95_contours(l, wiener_filtered_signal_and_noise, theta_rad)
    print(f"[{exp}] Wiener filter 95% radius : {wiener_filter_radius:.02f} deg")
    print(f"[{exp}] Wiener filtered signal 95% radius : {wiener_filtered_radius:.02f} deg")

    roots_wiener_filter_rad = find_roots_of_array(theta_rad, wiener_filter_corr - 0.5)
    roots_wiener_filtered_rad = find_roots_of_array(theta_rad, wiener_filtered_corr - 0.5)
    print(f"[{exp}] Wiener filter 50% correlation limit : {np.rad2deg(roots_wiener_filter_rad)[1]} deg")
    print(f"[{exp}] Wiener filtered signal 50% correlation limit : {np.rad2deg(roots_wiener_filtered_rad)[1]} deg")


plt.plot([], [], color = "gray", alpha = 0.5, ls = "solid", lw = 1, label = "Wiener filtered signal correlation")
plt.plot([], [], color = "gray", alpha = 0.5, ls = (0, (5, 5)), lw = 0.8, label = "Wiener filter correlation")
plt.xlim(0., 2.5)
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.tight_layout()
plt.savefig(f"{output_dir}/correlation_lensing_paper.pdf")
