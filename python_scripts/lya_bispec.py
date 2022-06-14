from lya_ps import LyaForest
import scipy.interpolate
import scipy.integrate
import numpy as np

class LyaBispec:


    def __init__(self, z):

        self.z = z
        self.forest = LyaForest(self.z)
        self.h = self.forest.cambResults.hubble_parameter(0) / 100


    def get_chi_from_z(self, z):
        """
        Get the comoving radial distance chi (Mpc / h)
        corresponding to the redshift z

        Parameters
        ----------
        z: 1D array
        """
        return self.forest.cambResults.comoving_radial_distance(z) * self.h

    def get_z_from_chi(self, chi):
        """
        Get the redshift associated with the
        comoving distance chi (Mpc/h)

        Parameters:
        chi: 1D array
        """
        return self.forest.cambResults.redshift_at_comoving_radial_distance(chi / self.h)

    def lensing_eff_kernel_z(self, z):
        """
        Get the lensing efficiency kernel such
        that \int dz W(z) delta(z, rperp) = kappa(rperp).
        W(z) is given in h/Mpc

        Parameters
        ----------
        z: 1D array
        """
        omegam = (self.forest.cambResults.get_Omega("cdm") +
                  self.forest.cambResults.get_Omega("baryon"))
        c = 3e5
        a = 1 / ( 1 + z )
        H = self.forest.cambResults.hubble_parameter(z) / self.h

        chi = self.get_chi_from_z(z)
        chistar = self.get_chi_from_z(1100)

        return 1.5 * omegam * 100 ** 2 / (a * H) * (chi / c) * (chistar - chi) / chistar


    def lensing_eff_kernel(self, chi):
        """
        Get the lensing efficiency kernel such
        that \int dchi W(chi) delta(chi, rperp) = kappa(rperp).
        W(chi) is given in h/Mpc

        Parameters
        ----------
        chi: 1D array
        """
        omegam = (self.forest.cambResults.get_Omega("cdm") +
                  self.forest.cambResults.get_Omega("baryon"))
        c = 3e5
        chistar = self.get_chi_from_z(1100)
        z = self.get_z_from_chi(chi)
        a = 1 / (1 + z)

        return 1.5 * omegam * (100 / c) ** 2 * chi * (chistar - chi) / chistar / a


    def wiener_filter(self, kPerp, exp):
        """
        Get the Wiener filter in the flat sky
        approximation (such that ell = kperp*chi)
        as a function of kPerp

        Parameters
        ----------
        kPerp: 1D array
          transverse scale in h/Mpc
        exp: string
          "planck" or "so"
        """
        file = f"/home/laposta-l/Documents/development/lya_lensing_bispectrum/data/wiener_filters/wiener_{exp}.dat"
        l, wiener = np.loadtxt(file).T
        chi = self.get_chi_from_z(self.z)
        self.kmax = np.max(l)/chi
        self.kmin = np.min(l)/chi
        print("Max k : ", np.max(l)/chi)
        print("Min k : ", np.min(l)/chi)
        wiener_l = scipy.interpolate.interp1d(l, wiener)

        return wiener_l(kPerp*chi)


    def integrand_fft_p2d_delta_kappa(self, q, kPerp, deltachi):
        """
        Get Plin * sinc(q*deltaChi/2) that enters in
        the computation of P2D

        Parameters
        ----------
        q: 1D array
          parallel scale in h/Mpc
        kPerp: 1D array
          transverse scale in h/Mpc
        deltachi: float
          width of the forest in Mpc/h
        """
        k = np.sqrt(q**2 + kPerp**2)
        # np.sinc(x) = sin(pi * x) / (pi * x)
        return self.forest.p3dMatter(self.z, k) * np.sinc(q*deltachi/2/np.pi)


    def PdeltaKappa(self, kPerp, deltaZ, exp):
        """
        Get PdeltaKappa given an experiment
        ("planck" or "so") as a function of kPerp.

        Parameters
        ----------
        kPerp: 1D array
          parallel scale in h/Mpc
        deltaZ: float
          width of the forest
        exp: string
          "planck" or "so"
        """
        k = lambda q: np.sqrt(q**2+kPerp**2)

        z_down, z_up = self.z - deltaZ / 2, self.z + deltaZ / 2

        chiUp = self.get_chi_from_z(z_up)
        chiDown = self.get_chi_from_z(z_down)
        deltaChi = chiUp - chiDown

        chiForest = self.get_chi_from_z(self.z)
        from test_fft import ifft

        q = np.linspace(-1e1, 1e1, 2001)
        Fq = self.integrand_fft_p2d_delta_kappa(q, kPerp, deltaChi)
        #chis, Fchi = fft(q, Fq)
        chis, Fchi = ifft(q, Fq)
        Fchi = np.abs(Fchi)

        id = np.where((chis >= -deltaChi/2) | (chis <= deltaChi/2))
        chis, Fchi = chis[id], Fchi[id]

        LensEffKernel = self.lensing_eff_kernel(chiForest + chis)

        PdeltaKappa = scipy.integrate.trapz(Fchi*LensEffKernel, chis)

        return self.wiener_filter(kPerp, exp) * PdeltaKappa


    def PdeltaKappa_scipy(self, kPerp, deltaZ, exp):

        k = lambda q: np.sqrt(q**2+kPerp**2)

        z_down, z_up = self.z - deltaZ, self.z + deltaZ

        chiUp = self.get_chi_from_z(z_up)
        chiDown = self.get_chi_from_z(z_down)
        deltaChi = chiUp - chiDown

        chiForest = self.get_chi_from_z(self.z)
        from test_fft import fft

        #q = np.linspace(-1e1, 1e1, 2001)
        #Fq = self.integrand_fft_p2d_delta_kappa(q, kPerp, deltaChi)
        intFChipp = lambda q, chipp: self.integrand_fft_p2d_delta_kappa(q, kPerp, deltaChi) * np.cos(q*chipp)
        Fchipp = lambda chipp: scipy.integrate.quad(intFChipp, -1e1, 1e1, epsabs=0., epsrel=1.e-2, args = (chipp,))[0]
        int_total = lambda chipp: Fchipp(chipp) * self.lensing_eff_kernel(chiForest + chipp)
        result = scipy.integrate.quad(int_total, -deltaChi/2, deltaChi/2, epsabs=0, epsrel=1e-2)[0]

        return self.wiener_filter(kPerp, exp) * result


    def dsigma2_dlogk(self, kPerp, deltaZ, exp):

        p2d = self.PdeltaKappa(kPerp, deltaZ, exp)
        return kPerp**2 / (2*np.pi) * p2d


    def sigma2deltaKappa(self, deltaZ, exp):

        integrand = lambda kperp: kperp / (2 * np.pi) * self.PdeltaKappa(kperp, deltaZ, exp)
        result = scipy.integrate.quad(integrand, self.kmin, self.kmax, epsabs = 0, epsrel = 1e-2)[0]

        return result


    def get_bispectrum(self, deltaZ, exp, kpara):

        dP1D_dDelta = np.array([self.forest.dP1D_dDelta(kp) for kp in kpara])
        return dP1D_dDelta * self.sigma2deltaKappa(deltaZ, exp)
