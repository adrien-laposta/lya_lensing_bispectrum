import numpy as np
import scipy.integrate
import chiang17_interp_tools

def D(k, mu, z, cosmology, **aip15_pars):#q1, q2, kvav, av, bv, kp):

    p3d_matter = cosmology.matter_power
    delta2 = p3d_matter(z, k) * k ** 3 / (2 * np.pi ** 2)
    Dt = aip15_pars["q1"] * delta2 + aip15_pars["q2"] * delta2 ** 2
    Dt *= 1 - (k ** aip15_pars["av"] / aip15_pars["kvav"]) * mu ** aip15_pars["bv"]
    Dt -= (k/aip15_pars["kp"]) ** 2
    return np.exp(Dt)

def get_lyman_alpha_p3d(k, mu, z, cosmology, **aip15_pars):#q1, q2, kvav, av, bv, kp, bF, beta):

    p3d_matter = cosmology.matter_power
    p3d_lyman_alpha = aip15_pars["bF"] ** 2 * (1 + aip15_pars["beta"] * mu ** 2) ** 2
    p3d_lyman_alpha *= p3d_matter(z, k) * D(k, mu, z, cosmology, **aip15_pars)
    return p3d_lyman_alpha

###################
# P1D computation #
###################
def get_lyman_alpha_p1d_aip15(k_para, z, cosmology, **aip15_pars):

    output_p1d = []
    for kp in k_para:
        mu = lambda k_perp: kp / np.sqrt(kp**2 + k_perp**2)
        integrand = lambda k_perp: k_perp * get_lyman_alpha_p3d(np.sqrt(kp**2 + k_perp**2),
                                                                mu(k_perp), z, cosmology,
                                                                **aip15_pars) / (2 * np.pi)
        integration_result = scipy.integrate.quad(integrand, 1e-3, 1e2, epsabs = 0., epsrel=1e-2)[0]
        output_p1d.append(integration_result)
    return np.array(output_p1d)

def get_lyman_alpha_p1d_npd13(k_para, z, cosmology):
    """
    Compute the Lyman-alpha
    1D power spectrum using
    NPD+13 template

    Parameters
    ----------
    kPara: 1D array
      scale in [h/Mpc]
    """
    z0, k0 = 3.0, 0.009 #s/km from NPD+13
    AF, nF, aF, BF, bF = 0.064, -2.55, -0.10, 3.55, -0.28

    k0 = k0 * cosmology.H(z) / (1 + z) # s/km to invMpc
    k0 = cosmology.invMpc_to_h_over_Mpc(k0)
    kP_over_pi = AF * ((1+z)/(1+z0))**BF * (k_para/k0)**(3+nF+aF*np.log(k_para/k0)+bF*np.log((1+z)/(1+z0)))

    return kP_over_pi / k_para * np.pi

def get_lyman_alpha_p1d(k_para, z, cosmology, use_npd13 = False, **aip15_pars):
    if use_npd13:
        return get_lyman_alpha_p1d_npd13(k_para, z, cosmology)
    else:
        return get_lyman_alpha_p1d_aip15(k_para, z, cosmology, **aip15_pars)

#######################
# Response of the P1D #
#######################
def dlog_plin_ddelta(k, z, cosmology):
    p3d_matter = cosmology.matter_power
    eps = 1e-4
    dplin_dlogk = (p3d_matter(z, k * (1 + eps)) - p3d_matter(z, k * (1 - eps))) / (2 * eps)

    dlog_plin_dlogk = dplin_dlogk / p3d_matter(z, k)

    output = 68/21
    output -= 1 + (1/3) * dlog_plin_dlogk
    return output

def dlog_bF_ddelta(pars_low, pars_mid, pars_high):
    bF_low = pars_low["bF"]
    bF_mid = pars_mid["bF"]
    bF_high = pars_high["bF"]

    dbF_dsigma8 = (bF_high - bF_low) / (0.88 - 0.64)
    dlog_bF_dlog_sigma8 = 0.76 / bF_mid * dbF_dsigma8
    dlog_bF_ddelta = 13/21 * dlog_bF_dlog_sigma8
    return dlog_bF_ddelta

def dlog_1pBetaMu_ddelta(pars_low, pars_mid, pars_high, mu):
    beta_low = pars_low["beta"]
    beta_mid = pars_mid["beta"]
    beta_high = pars_high["beta"]

    dbeta_dsigma8 = (beta_high - beta_low) / (0.88 - 0.64) * mu ** 2
    dlog_beta_dlogs8 = 0.76 / (1+beta_mid * mu**2) * dbeta_dsigma8
    dlog_beta_ddelta = 13/21 * dlog_beta_dlogs8

    return dlog_beta_ddelta

def dlog_D_ddelta(k, mu, z, cosmology, pars_low, pars_mid, pars_high):
    eps = 1e-4
    D_mid = lambda kc: D(kc, mu, z, cosmology, **pars_mid)

    dlogD_dlogk = (D_mid(k * (1+eps)) - D_mid(k* (1-eps))) / (2*eps) / D_mid(k)

    D_low = D(k, mu, z, cosmology, **pars_low)
    D_high = D(k, mu, z, cosmology, **pars_high)
    D_mid = D_mid(k)

    dD_dsigma8 = (D_high - D_low) / (0.88 - 0.64)
    dlogD_dlogs8 = 0.76 / D_mid * dD_dsigma8

    delta2 = cosmology.matter_power(z, k) * k ** 3 / (2 * np.pi ** 2)
    dlogD_dlogPlin = 2 * pars_mid["q1"] * delta2
    dlogD_dlogPlin += 4 * pars_mid["q2"] * delta2 ** 2
    dlogD_dlogPlin *= (1 - k**pars_mid["av"]/pars_mid["kvav"] * mu ** pars_mid["bv"])

    dlogD_dlogs8 += 2 * dlogD_dlogPlin

    result = 13/21 * dlogD_dlogs8 - 1/3 * dlogD_dlogk
    return result

def dlog_P3D_ddelta(k, mu, z, cosmology, pars_low, pars_mid, pars_high):

    result = dlog_plin_ddelta(k, z, cosmology)
    result += 2 * dlog_bF_ddelta(pars_low, pars_mid, pars_high)
    result += 2 * dlog_1pBetaMu_ddelta(pars_low, pars_mid, pars_high, mu)
    result += dlog_D_ddelta(k, mu, z, cosmology, pars_low, pars_mid, pars_high)

    return result

def dP3D_ddelta(k, mu, z, cosmology, pars_low, pars_mid, pars_high, **aip15_pars):
    return get_lyman_alpha_p3d(k,
                               mu, z,
                               cosmology,
                               **aip15_pars) * dlog_P3D_ddelta(k, mu, z, cosmology,
                                                               pars_low, pars_mid,
                                                               pars_high)

def dP1D_ddelta_aip15(k_para, z, cosmology, pars_low, pars_mid, pars_high, **aip15_pars):

    output = []
    for kp in k_para:
        mu = lambda k_perp: kp / np.sqrt(kp**2+k_perp**2)
        k = lambda k_perp: np.sqrt(kp**2+k_perp**2)
        integrand = lambda k_perp: (k_perp *
                                    get_lyman_alpha_p3d(k(k_perp), mu(k_perp), z, cosmology, **aip15_pars) *
                                    dlog_P3D_ddelta(k(k_perp), mu(k_perp), z, cosmology, pars_low, pars_mid, pars_high) / (2*np.pi))
        result = scipy.integrate.quad(integrand,1e-3,1e2,epsabs=0., epsrel=1.e-2)[0]
        output.append(result)
    return np.array(output)

def dP1D_ddelta_chiang17(k_para, z, cosmology, p1d_dict, p1d_response_dict, pars, zref):
    if 2.2 <= z <= 3.0:
        logp1d_response = chiang17_interp_tools.logp1d_response_interpolator(p1d_dict, p1d_response_dict)
    else:
        logp1d_response = chiang17_interp_tools.logp1d_response_extrapolator(p1d_dict, p1d_response_dict, pars, zref)

    d_logp1d_ddelta = logp1d_response(z, k_para)
    return d_logp1d_ddelta * get_lyman_alpha_p1d_npd13(k_para, z, cosmology)
