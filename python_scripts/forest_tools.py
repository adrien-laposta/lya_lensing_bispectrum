import numpy as np

def p_noise_lya(snr, z, h_of_z):
    lambda_lya = 1216.
    Hz = h_of_z(z)
    H0 = h_of_z(0.)
    cspeed = 3e5
    return 1./snr**2 * 1./lambda_lya * cspeed / (Hz / (H0/100))  # Mpc/h

def get_forest_length(z_qso, rf_min_wavelength, rf_max_wavelength, cosmology):
    lambda_lya = 1216.
    z_min = (1 + z_qso) * rf_min_wavelength / lambda_lya - 1
    z_max = (1 + z_qso) * rf_max_wavelength / lambda_lya - 1

    r_min = cosmology.get_chi_from_z(z_min) # Mpc / h
    r_max = cosmology.get_chi_from_z(z_max) # Mpc / h

    r_center = (r_min + r_max) / 2
    z_center = cosmology.get_z_from_chi(r_center)

    forest_length = r_max - r_min
    z_length = z_max - z_min

    return z_center, forest_length, z_length

def sinc(x):
    return np.where(x != 0, np.sin(x) / x, 1)

def window_spectro(k, R, deltaChi):
    return np.exp(- k**2 * R **2 / 2) * sinc(k*deltaChi/2)
