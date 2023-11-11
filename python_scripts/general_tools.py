import os
import numpy as np

def exists(path):
    return os.path.exists(path)

def create_directory(path):
    if not exists(path):
        os.makedirs(path)

def fft(t, f):

    f_shift = np.fft.ifftshift(f)
    fw = np.fft.fft(f_shift)
    fw = np.fft.fftshift(fw) * (t[1] - t[0]) #normalization

    n = t.size
    freq = np.fft.fftfreq(n, d = t[1] - t[0]) #equispaced t
    freq = np.fft.fftshift(freq)
    omega = freq * 2 * np.pi

    return omega, fw

def ifft(om, fw):

    f = om / (2*np.pi)
    df = f[1] - f[0] #equispaced f/om
    t = np.fft.fftshift(np.fft.fftfreq(f.size, d=df))
    dt = t[1] - t[0]

    fw_shift = np.fft.ifftshift(fw)
    ft = np.fft.ifft(fw_shift)

    ft = np.fft.fftshift(ft) / dt

    return t, ft

def get_real_space_corr(theta, l, Cl):
    cos_theta = np.cos(theta)
    fac = np.zeros(len(l) + 2)
    fac[2:] = (2 * l + 1) / (4 * np.pi) * Cl
    corr = np.polynomial.legendre.legval(cos_theta, fac)
    return corr
