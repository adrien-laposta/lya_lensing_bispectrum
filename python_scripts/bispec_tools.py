from scipy.interpolate import interp1d
from general_tools import ifft
import scipy.integrate
import numpy as np
from lyman_p1d_tools import get_lyman_alpha_p3d
def wiener_filter(k_perp, Wl, z, cosmology):
    """
    """
    l, wiener = Wl.T
    chi = cosmology.get_chi_from_z(z)
    wiener_interp = interp1d(l, wiener)

    return wiener_interp(k_perp * chi)

def integrand_fft_p2d_delta_kappa(q, k_perp, deltachi, z, cosmology):
    """
    """
    k = np.sqrt(k_perp**2 + q ** 2)
    # np.sinc(x) = sin(pi * x) / (pi * x)
    return cosmology.matter_power(z, k) * np.sinc(q*deltachi/2/np.pi)

def PdeltaKappa(k_perp, deltaZ, Wl, z, cosmology):
    """
    """
    k = lambda q: np.sqrt(q**2+k_perp**2)
    z_down, z_up = z - deltaZ / 2, z + deltaZ / 2
    chi_up = cosmology.get_chi_from_z(z_up)
    chi_down = cosmology.get_chi_from_z(z_down)
    deltaChi = chi_up - chi_down

    chi_forest = cosmology.get_chi_from_z(z)
    q = np.linspace(-1, 1, 2001)
    Fq = integrand_fft_p2d_delta_kappa(q, k_perp, deltaChi, z, cosmology)
    chis,Fchi = ifft(q, Fq)
    Fchi = np.abs(Fchi)

    id = np.where((chis >= -deltaChi/2) & (chis <= deltaChi/2))
    chis, Fchi = chis[id], Fchi[id]

    lens_eff_kernel = cosmology.lensing_eff_kernel(chi_forest + chis, cosmology)
    PdeltaKappa = scipy.integrate.trapz(Fchi*lens_eff_kernel, chis)

    return wiener_filter(k_perp, Wl, z, cosmology) * PdeltaKappa

def dsigma2_dlogk(k_perp,deltaZ, Wl, z, cosmology):
    """
    """
    p2d = PdeltaKappa(k_perp, deltaZ, Wl, z, cosmology)
    return k_perp**2 / (2 * np.pi) * p2d

def sigma2deltaKappa(deltaZ, Wl, z, cosmology, kmin, kmax):
    """
    """
    integrand = lambda k_perp: k_perp / (2*np.pi) * PdeltaKappa(k_perp, deltaZ, Wl, z, cosmology)

    result = scipy.integrate.quad(integrand, kmin, kmax, epsabs=0.,epsrel=1e-2)[0]
    return result

def get_bispectrum(k_para, deltaZ, Wl, z, cosmology, dp1d_ddelta):
    """
    """
    chi = cosmology.get_chi_from_z(z)
    kmin = min(Wl[:, 0])/chi
    kmax = max(Wl[:, 0])/chi
    return dp1d_ddelta * sigma2deltaKappa(deltaZ, Wl, z, cosmology, kmin, kmax)

def integrand_fft_p2d_DeltaF_kappa(q, k_perp, deltachi, z, cosmology, **aip15_pars):
    """
    """
    k = np.sqrt(k_perp ** 2 + q ** 2)
    mu = np.abs(q / k)
    Plya = get_lyman_alpha_p3d(k, mu, z, cosmology, **aip15_pars)
    PdeltaF = np.sqrt(cosmology.matter_power(z, k) * Plya)
    # np.sinc(x) = sin(pi * x) / (pi * x)
    return PdeltaF * np.sinc(q*deltachi/2/np.pi)

def PDeltaFKappa(k_perp, deltaZ, Wl, z, cosmology, **aip15_pars):
    """
    """
    k = lambda q: np.sqrt(q**2+k_perp**2)
    z_down, z_up = z - deltaZ / 2, z + deltaZ / 2
    chi_up = cosmology.get_chi_from_z(z_up)
    chi_down = cosmology.get_chi_from_z(z_down)
    deltaChi = chi_up - chi_down

    chi_forest = cosmology.get_chi_from_z(z)

    q = np.linspace(-1, 1, 2001)
    Fq = integrand_fft_p2d_DeltaF_kappa(q, k_perp, deltaChi, z, cosmology, **aip15_pars)
    chis, Fchi = ifft(q, Fq)
    Fchi = np.abs(Fchi)

    id = np.where((chis >= -deltaChi/2) & (chis <= deltaChi/2))
    chis, Fchi = chis[id], Fchi[id]

    lens_eff_kernel = cosmology.lensing_eff_kernel(chi_forest + chis, cosmology)
    PdeltaKappa = scipy.integrate.trapz(Fchi*lens_eff_kernel, chis)

    return wiener_filter(k_perp, Wl, z, cosmology) * PdeltaKappa

def sigma2DeltaFKappa(deltaZ, Wl, z, cosmology, kmin, kmax, **aip15_pars):
    """
    """
    integrand = lambda k_perp: k_perp / (2*np.pi) * PDeltaFKappa(k_perp, deltaZ, Wl, z, cosmology, **aip15_pars)
    result = scipy.integrate.quad(integrand, kmin, kmax, epsabs=0., epsrel=1e-2)[0]
    return result

def get_continuum_bispectrum(deltaZ, Wl, z, cosmology, p1d_lya, **aip15_pars):
    """
    """
    chi = cosmology.get_chi_from_z(z)
    kmin = min(Wl[:, 0])/chi
    kmax = max(Wl[:, 0])/chi
    return 2 * p1d_lya * sigma2DeltaFKappa(deltaZ, Wl, z, cosmology, kmin, kmax, **aip15_pars)

def get_dla_par(z, p0, p1):
    """
    """
    return p0 * ((1+z)/(1+2.))**p1

def get_forest_weight(z):
    zTable = [2.0, 2.44, 3.01,3.49,4.43]
    wTable = [0.777, 0.696, 0.574, 0.457, 0.220]
    w = interp1d(zTable, wTable)
    return w(z)

def get_lls_weight(z):
    zTable = [2.0, 2.44, 3.01,3.49,4.43]
    wTable = [0.106, 0.149, 0.218, 0.270, 0.366]
    w = interp1d(zTable, wTable)
    return w(z)

def get_sub_weight(z):
    zTable = [2.0, 2.44, 3.01,3.49,4.43]
    wTable = [0.059, 0.081, 0.114, 0.143, 0.201]
    w = interp1d(zTable, wTable)
    return w(z)

def get_pdla_over_plya(z, a0, a1, b0, b1, c0, c1, k_para):

    a = get_dla_par(z, a0, a1)
    b = get_dla_par(z, b0, b1)
    c = get_dla_par(z, c0, c1)

    result = ((1+z)/(1+2.)) ** (-3.55)
    result /= (a * np.exp(b * k_para) - 1) ** 2
    result += c

    return result

def get_dla_bispectrum(k_para, deltaZ, Wl, z, cosmology, p1d_lya):
    """
    """
    bDLA = 2.17
    k_para_kms = k_para * cosmology.h / cosmology.H(z) * (1 + z)

    a0_LLS, a1_LLS = 2.2001, 0.0134
    b0_LLS, b1_LLS = 36.449, -0.0674
    c0_LLS, c1_LLS = 0.9849, -0.0631
    p1d_LLS_over_plya = get_pdla_over_plya(z,a0_LLS,a1_LLS, b0_LLS,b1_LLS, c0_LLS, c1_LLS, k_para_kms)

    a0_SUB, a1_SUB = 1.5083, 0.0994
    b0_SUB, b1_SUB = 81.388, -0.2287
    c0_SUB, c1_SUB = 0.8667, 0.0196
    p1d_SUB_over_plya = get_pdla_over_plya(z,a0_SUB,a1_SUB, b0_SUB,b1_SUB, c0_SUB, c1_SUB, k_para_kms)

    wLLS = get_lls_weight(z)
    wSUB = get_sub_weight(z)
    wF = get_forest_weight(z)
    wSum = wLLS + wSUB + wF
    wLLS, wSUB, wF = wLLS/wSum, wSUB/wSum, wF/wSum
    ptot_dla = wF + wSUB * p1d_SUB_over_plya + wLLS * p1d_LLS_over_plya - 1
    ptot_dla *= p1d_lya

    chi = cosmology.get_chi_from_z(z)
    kmin = min(Wl[:, 0])/chi
    kmax = max(Wl[:, 0])/chi

    return bDLA * ptot_dla * sigma2deltaKappa(deltaZ, Wl, z, cosmology, kmin, kmax)
