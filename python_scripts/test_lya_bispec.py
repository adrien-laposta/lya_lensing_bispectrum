from lya_bispec import LyaBispec
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use("~/Desktop/custom.mplstyle")

outputPlotDir = "../plots/lya_bispec"
if not os.path.exists(outputPlotDir):
    os.makedirs(outputPlotDir)

z = 2.4

bispec = LyaBispec(z)

# Test lensing efficiency kernel function Z
zList = np.linspace(0, 5, 500)
lensKernel = bispec.lensing_eff_kernel_z(zList)

plt.figure(figsize = (8, 6))
plt.plot(zList, lensKernel)
plt.xlabel(r"$z$")
plt.ylabel(r"$W_\kappa (z)$ [h/Mpc]")
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/lensing_eff_kernel_wrt_z.png", dpi = 300)
####

# Test lensing efficiency kernel function
zList = np.linspace(0, 1200, 5000)
chiList = bispec.forest.cambResults.comoving_radial_distance(zList)*bispec.h
lensKernel = bispec.lensing_eff_kernel(chiList)

plt.figure(figsize = (8, 6))
plt.plot(chiList, lensKernel)
plt.xlabel(r"$\chi$ [Mpc/h]")
plt.ylabel(r"$W_\kappa (\chi)$ [h/Mpc]")
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/lensing_eff_kernel_wrt_chi.png", dpi = 300)
####

# Test Wiener filtering
kPerpList = np.logspace(np.log10(0.0055), np.log10(0.62), 500)
wfPlanck = bispec.wiener_filter(kPerpList, "planck")
wfSO = bispec.wiener_filter(kPerpList, "so")
plt.figure(figsize = (8, 6))
plt.plot(kPerpList, wfPlanck, label = "Planck")
plt.plot(kPerpList, wfSO, label = "SO")
plt.xscale("log")
plt.xlabel(r"$k_\perp$ [h/Mpc]")
plt.ylabel(r"$\Lambda (k_\perp)$ (at $z = %.1f$)" % z)
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/wiener_filter_at_{z}.png", dpi = 300)
####

# Test Wiener filtering as a function of z
zmin, zmax = 0, 10
zList = np.linspace(zmin, zmax, 1000)
chi = bispec.get_chi_from_z(zList)
thetaPlanck = 0.87 * np.pi / 180
thetaSO = 0.45 * np.pi / 180
rCaracPlanck = chi * np.tan(thetaPlanck)
rCaracSO = chi * np.tan(thetaSO)
plt.figure(figsize = (8, 6))
plt.plot(zList, rCaracPlanck, label = "Planck")
plt.plot(zList, rCaracSO, label = "SO")
plt.xlabel(r"$z$")
plt.ylabel(r"$r_\perp^\mathrm{Wiener}$ [Mpc/h]")
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.xlim(zmin, zmax)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/r_wiener_wrt_z.png", dpi = 300)
####

# Test integrand fft
#kPerp = 0.05
#q = np.linspace(-10, 10, 2001)
#deltachi = 350
#int_fft = bispec.integrand_fft_p2d_delta_kappa(q, kPerp, deltachi)
#from test_fft import ifft
#chi,fchi = ifft(q, int_fft)
#plt.figure(figsize = (8, 6))
#plt.plot(q, int_fft)
#plt.xscale("log")
#plt.show()

#plt.figure()
#plt.plot(chi, fchi)
#plt.show()

# Test P2D
deltaZ = 0.2
chi = bispec.forest.cambResults.comoving_radial_distance(bispec.forest.z)*bispec.h
kperp = np.logspace(np.log10(5/chi), np.log10(2000/chi), 200)
p2d = np.array([bispec.PdeltaKappa(kP, deltaZ, "planck") for kP in kperp])
#p2d_scipy = np.array([bispec.PdeltaKappa_scipy(kP, deltaZ, "planck") for kP in kperp])
ell = kperp * chi

plt.figure(figsize = (8, 6))
plt.plot(ell, kperp**2 * p2d, label = "This work")
#plt.plot(ell, kperp**2 * p2d_scipy, label = "This work [scipy]")

test_chiang = np.loadtxt("test_chiang_p2d.dat")
plt.plot(test_chiang[:,0], test_chiang[:, 1], color = "r", ls = "--", label = "Chiang+17")
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell(\ell + 1)C_\ell^{\bar{\delta}\kappa}$ (at $z = %.1f$)" % z)
plt.xscale("log")
plt.yscale("log")
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/p2d_at_{z}.png", dpi = 300)

# Test sigma2 integrand
dsigma2_dlogk = np.array([bispec.dsigma2_dlogk(kP, deltaZ, "planck") for kP in kperp])
plt.figure(figsize = (8,6))
plt.plot(kperp, dsigma2_dlogk)
plt.xlabel(r"$k_\perp$ [h/Mpc]")
plt.ylabel(r"$d\sigma^2_{\bar{\delta}\kappa}/d\mathrm{ln}k$")
plt.xscale("log")
plt.savefig(f"{outputPlotDir}/dsigma2_dlogk_at_{z}.png", dpi = 300)

# Test bispectrum
plt.figure(figsize = (8, 6))
kpara = np.linspace(1e-2, 1.5, 40)
bispectrum = bispec.get_bispectrum(deltaZ, "planck", kpara)
plt.plot(kpara, 1e5 * bispectrum)
plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$10^5 B_{\kappa, \mathrm{Ly}\alpha}$ [Mpc/h]")
plt.xlim(0, 1.5)
plt.ylim(bottom = 0)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/bispectrum_at_{z}.png", dpi = 300)
