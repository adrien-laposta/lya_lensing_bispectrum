from lya_ps import LyaForest
from scipy.interpolate import interp1d
import scipy.integrate
import numpy as np
import pickle

class LyaBispec:

    def __init__(self, z, use_aip15_extrapolation = False):

        self.z = z
        self.forest = LyaForest(self.z, use_aip15_extrapolation)
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


    def wiener_filter(self, kPerp, Wl):
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
        #file = f"/home/laposta-l/Documents/development/lya_lensing_bispectrum/data/wiener_filters/wiener_{exp}.dat"
        #l, wiener = np.loadtxt(file).T
        l, wiener = Wl.T
        chi = self.get_chi_from_z(self.z)
        #self.kmax = np.max(l)/chi
        #self.kmin = np.min(l)/chi
        #print("Max k : ", np.max(l)/chi)
        #print("Min k : ", np.min(l)/chi)
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


    def PdeltaKappa(self, kPerp, deltaZ, Wl):
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

        q = np.linspace(-1, 1, 2001)
        Fq = self.integrand_fft_p2d_delta_kappa(q, kPerp, deltaChi)
        chis, Fchi = ifft(q, Fq)
        Fchi = np.abs(Fchi)

        id = np.where((chis >= -deltaChi/2) | (chis <= deltaChi/2))
        chis, Fchi = chis[id], Fchi[id]

        LensEffKernel = self.lensing_eff_kernel(chiForest + chis)

        PdeltaKappa = scipy.integrate.trapz(Fchi*LensEffKernel, chis)

        return self.wiener_filter(kPerp, Wl) * PdeltaKappa


    def dsigma2_dlogk(self, kPerp, deltaZ, Wl):

        p2d = self.PdeltaKappa(kPerp, deltaZ, Wl)
        return kPerp**2 / (2*np.pi) * p2d


    def sigma2deltaKappa(self, deltaZ, Wl):

        integrand = lambda kperp: kperp / (2 * np.pi) * self.PdeltaKappa(kperp, deltaZ, Wl)
        result = scipy.integrate.quad(integrand, self.kmin, self.kmax, epsabs = 0, epsrel = 1e-2)[0]

        return result


    def get_bispectrum(self, deltaZ, Wl, kpara, interpolated_func = None, use_chiang17_extrapolation = False):
        chi = self.get_chi_from_z(self.z)
        self.kmax = np.max(Wl[:,0])/chi
        self.kmin = np.min(Wl[:,0])/chi
        if interpolated_func is None:
            dP1D_dDelta = np.array([self.forest.dP1D_dDelta(kp, use_chiang17_extrapolation) for kp in kpara])
        else:
            dP1D_dDelta = interpolated_func(kpara, self.z)

        return dP1D_dDelta * self.sigma2deltaKappa(deltaZ, Wl)


    def integrand_fft_p2d_DeltaF_kappa(self, q, kPerp, deltachi):
        """
        Get PdeltaF * sinc(q*deltaChi/2) that enters in
        the computation of P2D_DeltaFkappa

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
        mu = np.abs(q / k)
        Plya = self.forest.get_P3D(k, mu)
        PdeltaF = np.sqrt(self.forest.p3dMatter(self.z, k) * Plya)
        # np.sinc(x) = sin(pi * x) / (pi * x)
        return PdeltaF * np.sinc(q*deltachi/2/np.pi)


    def PDeltaFKappa(self, kPerp, deltaZ, Wl):
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

        q = np.linspace(-1, 1, 2001)
        Fq = self.integrand_fft_p2d_DeltaF_kappa(q, kPerp, deltaChi)
        #chis, Fchi = fft(q, Fq)
        chis, Fchi = ifft(q, Fq)
        Fchi = np.abs(Fchi)

        id = np.where((chis >= -deltaChi/2) | (chis <= deltaChi/2))
        chis, Fchi = chis[id], Fchi[id]
        LensEffKernel = self.lensing_eff_kernel(chiForest + chis)

        PdeltaKappa = scipy.integrate.trapz(Fchi*LensEffKernel, chis)

        return self.wiener_filter(kPerp, Wl) * PdeltaKappa


    def sigma2DeltaFKappa(self, deltaZ, Wl):

        integrand = lambda kperp: kperp / (2 * np.pi) * self.PDeltaFKappa(kperp, deltaZ, Wl)
        result = scipy.integrate.quad(integrand, self.kmin, self.kmax, epsabs = 0, epsrel = 1e-2)[0]

        return result

    def get_continuum_bispectrum(self, deltaZ, Wl, kpara, use_npd13_template = False):
        p1d = np.array([self.forest.get_P1D(kp, use_npd13_template) for kp in kpara])
        return 2 * p1d * self.sigma2DeltaFKappa(deltaZ, Wl)

    def get_dla_a(self, a0, a1):
        return a0 * ((1+self.z)/(1 + 2)) ** a1

    def get_dla_b(self, b0, b1):
        return b0 * ((1+self.z)/(1 + 2)) ** b1

    def get_dla_c(self, c0, c1):
        return c0 * ((1+self.z)/(1 + 2)) ** c1

    def get_forest_weight(self):
        zTable = [2.0, 2.44, 3.01,3.49,4.43]
        wTable = [0.777, 0.696, 0.574, 0.457, 0.220]
        w = interp1d(zTable, wTable, kind = 'cubic')
        return w(self.z)

    def get_lls_weight(self):
        zTable = [2.0, 2.44, 3.01,3.49,4.43]
        wTable = [0.106, 0.149, 0.218, 0.270, 0.366]
        w = interp1d(zTable, wTable, kind = 'cubic')
        return w(self.z)

    def get_sub_weight(self):
        zTable = [2.0, 2.44, 3.01,3.49,4.43]
        wTable = [0.059, 0.081, 0.114, 0.143, 0.201]
        w = interp1d(zTable, wTable, kind = 'cubic')
        return w(self.z)

    def get_pdla_over_plya(self, a0, a1, b0, b1, c0, c1, kpara):

        a = self.get_dla_a(a0, a1)
        b = self.get_dla_b(b0, b1)
        c = self.get_dla_c(c0, c1)

        result = ((1+self.z)/(1+2)) ** (-3.55)
        result /= (a * np.exp(b * kpara) - 1) ** 2
        result += c

        return result

    def get_dla_bispectrum(self, kpara, deltaZ, Wl, use_npd13_template = False):
        bDLA = 2.17
        #p1d_LLS_over_plya = np.array([self.get_pdla_over_plya(2.2001, 0.0134, 36.449,
        #                                                      -0.0674,0.9849,-0.0631,kp/self.h  / self.forest.cambResults.hubble_parameter(self.z)) for kp in kpara])
        #p1d_SUB_over_plya = np.array([self.get_pdla_over_plya(1.5083, 0.0994, 81.388,
        #                                            -0.2287,0.8667,0.0196,kp/self.h / self.forest.cambResults.hubble_parameter(self.z)) for kp in kpara])
        p1d_LLS_over_plya = np.array([self.get_pdla_over_plya(2.2001, 0.0134, 36.449,
                                                              -0.0674,0.9849,-0.0631,kp*self.h  / self.forest.cambResults.hubble_parameter(self.z) * (1+self.z)) for kp in kpara])
        p1d_SUB_over_plya = np.array([self.get_pdla_over_plya(1.5083, 0.0994, 81.388,
                                                    -0.2287,0.8667,0.0196,kp*self.h / self.forest.cambResults.hubble_parameter(self.z) * (1+self.z)) for kp in kpara])

        p1d_forest = np.array([self.forest.get_P1D(kp, use_npd13_template) for kp in kpara])
        wLLS = self.get_lls_weight()
        wSUB = self.get_sub_weight()
        wF = self.get_forest_weight()
        wSum = wF + wSUB + wLLS
        wLLS, wSUB, wF = wLLS / wSum, wSUB / wSum, wF / wSum
        #wSUM = wSUB + wLLS
        #wLLS, wSUB = wLLS / wSUM, wSUB / wSUM
        ptot_dla = wF + wSUB * p1d_SUB_over_plya + wLLS * p1d_LLS_over_plya - 1
        #ptot_dla = wSUB * p1d_SUB_over_plya + wLLS * p1d_LLS_over_plya
        ptot_dla *= p1d_forest

        return bDLA * ptot_dla * self.sigma2deltaKappa(deltaZ, Wl)
