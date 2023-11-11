import numpy as np
import matplotlib.pyplot as plt

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

if __name__ == "__main__":

    t = np.linspace(-50, 50, 101)
    sigma = 1.1
    f = np.exp(-0.5 * t ** 2 / sigma ** 2)

    om, fw = fft(t, f)

    t_recov, ft_recov = ifft(om, fw)

    plt.figure()
    plt.plot(t, f, label = "original")
    plt.plot(om, fw, label = "fft")
    plt.plot(om,  np.sqrt(2*np.pi*sigma**2) * np.exp(-om**2*sigma**2/2), label = "exact FT", ls = "--", color = "k")
    plt.plot(t_recov, ft_recov, label = "recovered original signal", ls = "--", color = "r")
    plt.legend()
    plt.show()
