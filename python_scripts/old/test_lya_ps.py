from lya_ps import LyaForest
import numpy as np
import matplotlib.pyplot as plt
import os

plt.style.use("~/Desktop/custom.mplstyle")


outputPlotDir = "../plots/lya_p1d"
if not os.path.exists(outputPlotDir):
    os.makedirs(outputPlotDir)

z = 2.4
forest = LyaForest(z)

mus = [0, 0.13, 0.37, 0.63, 0.88]
k = np.logspace(-3, 2, 400)

# TEST D(k, mu)
Ds = np.array([forest.D(k, mu) for mu in mus])
plt.figure(figsize = (8, 6))
for i, mu in enumerate(mus):
    plt.plot(k, Ds[i], label = r"$\mu$ = %.2f" % mu)
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.xscale("log")
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$D(k, \mu)$ at $z = %.1f$" % z)
#plt.yscale("log")
plt.xlim(1e-1, 30)
plt.ylim(0, 2.5)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/D_at_z{z}.png", dpi = 300)

# TEST P3D
plt.figure(figsize = (8,6))
P3Ds = np.array([forest.get_P3D(k, mu) for mu in mus])
for i, mu in enumerate(mus):
    plt.plot(k, k**3 * P3Ds[i] / 2 / np.pi**2, label = r"$\mu$ = %.2f" % mu)
leg = plt.legend(frameon=True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.xscale("log")
plt.yscale("log")
plt.xlim(1e-2, 5e1)
plt.ylim(1e-6, 1e-1)
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$k^3 P_{3\mathrm{D}}(k, \mu)/2\pi^2$ at $z = %.1f$" % z)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/p3d_at_z{z}.png", dpi = 300)

# TEST PLya
kpara = np.logspace(-2, 2, 80)
plya = np.array([forest.get_P1D(kp) for kp in kpara])
plt.figure(figsize = (8,6))
plt.plot(kpara, plya)
plt.xlabel(r"$k_\parallel$ [h/Mpc]")
plt.ylabel(r"$P^{1\mathrm{D}}_{\mathrm{Ly}\alpha}$ [Mpc/h] (at $z = %.1f$)"% z)
plt.xscale("log")

np.savetxt(f"{outputPlotDir}/p1d_at_z{z}.dat", np.array([kpara, plya]).T)
plt.savefig(f"{outputPlotDir}/p1d_at_z{z}.png", dpi = 300)

# Test integrand
kparaList = [0.1, 0.3, 1.0, 3.2, 10]
kOut = []
integrandOut = []

for kPara in kparaList:
    mu = lambda k: kPara / k
    integrand = lambda k: (k>kPara) * k**2 * forest.get_P3D(k,
                                               mu(k)) / (2 * np.pi)
    #kPerpList = np.linspace(0, np.sqrt(1e2**2 - kPara**2), 5000)
    #kPerpList = np.logspace(-5, np.log(np.sqrt(1e2**2 - kPara**2))/np.log(10), 5000)
    kList = np.logspace(-2, 2, 5000)
    #integrandOut.append(integrand(kPerpList))
    #kOut.append(np.sqrt(kPerpList**2 + kPara**2))
    integrandOut.append(integrand(kList))
    kOut.append(kList)

plt.figure(figsize = (8,6))
for i, kPara in enumerate(kparaList):
    plt.plot(kOut[i], integrandOut[i], label = r"$k_\parallel = %.2f$" % kPara)
plt.legend()
plt.xscale("log")
plt.ylim(bottom=0)
plt.xlim(1e-2, 1e2)
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$dP^{1\mathrm{D}}(k_\parallel)/d\mathrm{ln}k$ [Mpc/h] (at $z = %.1f$)" % z)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/integrand_at_z{z}.png", dpi = 300)

# Test response to overdensity
dlogPlin_dDelta = forest.dlogPlin_dDelta(k)
dlogbF_dDelta = forest.dlogbF_dDelta()
dKaiser_mu1 = forest.dlog1pBetaMu_dDelta(1)
dlogD_dDelta_mu1 = forest.dlogD_dDelta(k, 1)
dlogD_dDelta_mu0 = forest.dlogD_dDelta(k, 0)

plt.figure(figsize = (8, 6))
plt.plot(k, dlogPlin_dDelta, label = r"$d\mathrm{ln}P_\mathrm{lin}/d\bar{\delta}$")
plt.plot(k, 2*dlogbF_dDelta*np.ones(len(k)), label = r"$d\mathrm{ln}b^2/d\bar{\delta}$")
plt.plot(k, 2*dKaiser_mu1*np.ones(len(k)), label = r"$d\mathrm{ln}(1+\beta\mu^2)^2/d\bar{\delta}$, $\mu = 1$")
plt.plot(k, dlogD_dDelta_mu1, label = r"$d\mathrm{ln}D/d\bar{\delta}$, $\mu = 1$")
plt.plot(k, dlogD_dDelta_mu0, label = r"$d\mathrm{ln}D/d\bar{\delta}$, $\mu = 0$")
leg = plt.legend(frameon = True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.xscale("log")
plt.xlim(1e-2, 1e2)
plt.ylim(-3, 12)
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$d\mathrm{ln}P^{3D}_{\mathrm{Ly}\alpha}/d\bar{\delta}$ (at $z = %.1f$)" % z)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/dlnp3d_ddelta_at_z{z}.png", dpi = 300)

# Total response of logP3D
dlogP3D_dDelta = [forest.dlogP3D_dDelta(k, mu) for mu in mus]
plt.figure(figsize = (8, 6))
for i, mu in enumerate(mus):
    plt.plot(k, dlogP3D_dDelta[i], label = r"$\mu = %.2f$" % mu)
# Remove that
kc, dp1dddc = np.loadtxt("data_chiang_fig2.dat").T
plt.plot(kc, dp1dddc, label = r"Chiang et al. $\mu = 0.875$")
#####
leg = plt.legend(frameon = True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.xscale("log")
plt.xlim(1e-2, 1e2)
plt.ylim(-3, 12)
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$d\mathrm{ln}P^{3D}_{\mathrm{Ly}\alpha}/d\bar{\delta}$ (at $z = %.1f$)" % z)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/dlnp3dtot_ddelta_at_z{z}.png", dpi = 300)

# Total response of P3D
dP3D_dDelta = [forest.dP3D_dDelta(k, mu) for mu in mus]
plt.figure(figsize = (8, 6))
for i, mu in enumerate(mus):
    plt.plot(k, dP3D_dDelta[i], label = r"$\mu = %.2f$" % mu)


leg = plt.legend(frameon = True)
leg.get_frame().set_linewidth(2)
leg.get_frame().set_edgecolor("k")
plt.xscale("log")
plt.xlim(1e-3, 1e2)
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$dP^{3D}_{\mathrm{Ly}\alpha}/d\bar{\delta}$ $[Mpc/h]^3$ (at $z = %.1f$)" % z)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/dp3d_ddelta_at_z{z}.png", dpi = 300)

# Total response of P1D
kpara = np.logspace(-2, 2, 200)
dplya_dDelta = np.array([forest.dP1D_dDelta(kp) for kp in kpara])
plt.figure(figsize = (8, 6))
plt.plot(kpara, dplya_dDelta)
np.savetxt(f"{outputPlotDir}/dp1d_dDelta_at_z{z}.dat", np.array([kpara, dplya_dDelta]).T)
plt.xscale("log")
plt.xlim(1e-2, 1e1)
plt.xlabel(r"$k$ [h/Mpc]")
plt.ylabel(r"$dP^{1D}_{\mathrm{Ly}\alpha}/d\bar{\delta}$ $[Mpc/h]$ (at $z = %.1f$)" % z)
plt.tight_layout()
plt.savefig(f"{outputPlotDir}/dp1d_ddelta_at_z{z}.png", dpi = 300)

# Test integrand of dP1D_dDelta  1.59415904e-02
kparas = np.logspace(np.log(1.59415904e-02*0.9)/np.log(10), np.log(1.59415904e-02*1.1)/np.log(10), 5)
integrand_dplya_ddelta = np.array([forest.integrand_dP1D_dDelta(kp) for kp in kparas])
integrand_blip = forest.integrand_dP1D_dDelta(1.59415904e-02)
plt.figure(figsize = (8, 6))
for i, integ in enumerate(integrand_dplya_ddelta):
    x = np.logspace(-3, 2, 1000)
    plt.plot(x, integ, label = r"$k_\parallel = %.5f$" % kparas[i])
plt.plot(x, integrand_blip, label = r"$k_\parallel^{blip}$", color = "k", ls = "--")
plt.xscale("log")
plt.xlabel(r"$k_\perp$ [h/Mpc]")
plt.ylabel(r"Integrand of $dP^{1D}_{\mathrm{Ly}\alpha}/d\bar{\delta}$ $[Mpc/h]$ (at $z = %.1f$)" % z)
plt.legend()
plt.savefig(f"{outputPlotDir}/integrand_dp1d_ddelta_at_z{z}.png", dpi = 300)

# Test integrand of dP1D_dDelta log
kparas = np.logspace(np.log(1.59415904e-02*0.9)/np.log(10), np.log(1.59415904e-02*1.1)/np.log(10), 5)
integrand_dplya_ddelta = np.array([forest.integrand_dP1D_dDelta_log(kp) for kp in kparas])
integrand_blip = forest.integrand_dP1D_dDelta_log(1.59415904e-02)
plt.figure(figsize = (8, 6))
for i, integ in enumerate(integrand_dplya_ddelta):
    x = np.logspace(-3, 2, 1000)
    plt.plot(x, integ, label = r"$k_\parallel = %.5f$" % kparas[i])
plt.plot(x, integrand_blip, label = r"$k_\parallel^{blip}$", color = "k", ls = "--", lw = 0.8)
plt.xscale("log")
plt.xlabel(r"$k_\perp$ [h/Mpc]")
plt.ylabel(r"Integrand of $dP^{1D}_{\mathrm{Ly}\alpha}/d\bar{\delta}$ $[Mpc/h]$ (at $z = %.1f$)" % z)
plt.legend()
plt.savefig(f"{outputPlotDir}/integrand_dp1d_ddelta_at_z{z}_log.png", dpi = 300)
