from dict_tools import so_dict
import argparse
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("custom.mplstyle")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dict_file", help = "Dict file")

args = parser.parse_args()
dict_file = args.dict_file

d = so_dict()
d.read_from_file(dict_file)

output_dir = d["output_dir"]

# These redshifts correspond to QSO z
zcenter = np.array([2.12, 2.28, 2.43, 2.59, 2.75, 2.91, 3.07, 3.23, 3.39, 3.55])
dndz = np.array([96, 81, 65, 52, 40, 30, 22, 16, 11, 7])
dz = zcenter[1] - zcenter[0]

#dndz = dndz[(zcenter >= zmin) & (zcenter <= zmax)]
#zcenter = zcenter[(zcenter >= zmin) & (zcenter <= zmax)]

binning_edges = np.concatenate((zcenter - dz / 2, [zcenter[-1] + dz / 2]))
bin_center = (binning_edges[:-1] + binning_edges[1:]) / 2

# Define the number density per square degree
nz = dndz * (binning_edges[1:] - binning_edges[:-1])

# Define the number density for various experiments
Nz_act_desi = nz * d["areas"]["ACT+DESI-Y1"]
Nz_so_desi = nz * d["areas"]["SO+DESI-Y5"]
sv_area_planck_boss = 87085 / np.sum(nz)
Nz_planck_boss = nz * sv_area_planck_boss

plt.figure(figsize = (8, 6))
plt.xlabel(r"$z$")
plt.ylabel(r"$n(z)$ [$\mathrm{deg}^{-2}$]")
plt.bar(bin_center, nz, width = binning_edges[1:] - binning_edges[:-1], edgecolor = "k", linewidth = 1.4)
plt.tight_layout()
plt.show()
