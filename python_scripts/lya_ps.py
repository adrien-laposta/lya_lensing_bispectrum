import camb
import numpy as np
import pandas as pd
import scipy.integrate
from camb import model
from scipy.interpolate import interp1d

class LyaForest:


    def __init__(self, z):
        self.z = z

        tablesDir = "../data/aip15_table"
        self.tableHs8 = pd.read_csv(f"{tablesDir}/aip_table_s088.csv")
        self.tableLs8 = pd.read_csv(f"{tablesDir}/aip_table_s064.csv")
        self.tableMs8 = pd.read_csv(f"{tablesDir}/aip_table_s076.csv")

        self.tablePlanck = pd.read_csv(f"{tablesDir}/aip_table_planck.csv")

        self.zTable = self.tablePlanck["z"]
        self.meanTransmission = np.exp(-0.0023 * (1 + self.z) ** 3.65)

        # Initialize P3D matter
        self.init_p3d_matter()
        # Initialize bias and fit parameters
        self.init_bias_and_fit_pars()


    def init_p3d_matter(self):
        """
        Initialize the matter power spectrum
        from CAMB (in Mpc/h units).
        """
        pars = camb.CAMBparams()
        pars.set_cosmology(cosmomc_theta = 0.0104092,
                           ombh2 = 0.02237,
                           omch2 = 0.1200,
                           tau = 0.0544)
        pars.InitPower.set_params(As=2.1e-9, ns=0.9649, r=0)
        pars.set_matter_power(redshifts = [self.z], kmax = 1e2)

        self.cambResults = camb.get_results(pars)

        # P(z, k)
        self.p3dMatter = camb.get_matter_power_interpolator(pars, zmin = 0, zmax = 5, kmax = 1e2,
                                                            nonlinear = False, extrap_kmax = True).P

    def init_bias_and_fit_pars(self):
        """
        Initialize the D(k, mu) parameters
        from a table.
        """
        for p in ["q1", "q2", "kp", "kvav", "av", "bv", "beta", "bF"]:
            setattr(self, p, getattr(self, f"f{p}")(self.tablePlanck))


    def fq1(self, table):
        """
        Compute the q1 parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the q1 values
          as a function of z
        """
        q1Table = table["q1"]
        q1Interp = interp1d(self.zTable, q1Table, kind = 'cubic')
        return q1Interp(self.z)


    def fq2(self, table):
        """
        Compute the q2 parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the q2 values
          as a function of z
        """
        q2Table = table["q2"]
        q2Interp = interp1d(self.zTable, q2Table, kind = 'cubic')
        return q2Interp(self.z)


    def fkp(self, table):
        """
        Compute the kp parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the kp values
          as a function of z
        """
        kpTable = table["kp"]
        kpInterp = interp1d(self.zTable, kpTable, kind = 'cubic')
        return kpInterp(self.z)


    def fkvav(self, table):
        """
        Compute the kv^(av) parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the kv^(av) values
          as a function of z
        """
        kvavTable = table["kvav"]
        kvavInterp = interp1d(self.zTable, kvavTable, kind = 'cubic')
        return kvavInterp(self.z)


    def fav(self, table):
        """
        Compute the av parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the av values
          as a function of z
        """
        avTable = table["av"]
        avInterp = interp1d(self.zTable, avTable, kind = 'cubic')
        return avInterp(self.z)


    def fbv(self, table):
        """
        Compute the bv parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the bv values
          as a function of z
        """
        bvTable = table["bv"]
        bvInterp = interp1d(self.zTable, bvTable, kind = 'cubic')
        return bvInterp(self.z)


    def fbeta(self, table):
        """
        Compute the beta parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the beta values
          as a function of z
        """
        betaTable = table["beta"]
        betaInterp = interp1d(self.zTable, betaTable, kind = 'cubic')
        return betaInterp(self.z)


    def fbTauDelta(self, table):
        """
        Compute the b_TauDelta parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the b_TauDelta values
          as a function of z
        """
        bTauDeltaTable = table["bTauDelta"]
        bTauDeltaInterp = interp1d(self.zTable, bTauDeltaTable, kind = 'cubic')
        return bTauDeltaInterp(self.z)


    def fbF(self, table):
        """
        Compute the bias parameter at
        redshift self.z given a table

        Parameters
        ----------
        table: dataframe
          containing the b_TauDelta values
          as a function of z
        """
        return self.fbTauDelta(table) * np.log(self.meanTransmission)

    def fD(self, k, mu, q1, q2, kvav, av, bv, kp):
        """
        Compute the non linear term that
        enters in the computation of Plya

        Parameters
        ----------
        k: 1D array
          scale in [h/Mpc]
        mu: float
        q1: float
        q2: float
        kvav: float
        av: float
        bv: float
        kp: float
        """
        delta2 = self.p3dMatter(self.z, k) * k**3 / (2 * np.pi**2)
        Dtemp = q1 * delta2 + q2 * delta2 ** 2
        Dtemp *= 1 - (k ** av / kvav) * mu ** bv
        Dtemp -= (k / kp) ** 2
        return np.exp(Dtemp)

    def D(self, k, mu):
        """
        Compute the non linear term that
        enters in the computation of Plya

        Parameters
        ----------
        k: 1D array
          scale in [h/Mpc]
        mu: float
        """
        return self.fD(k, mu, self.q1, self.q2, self.kvav,
                       self.av, self.bv, self.kp)


    def get_P3D(self, k, mu):
        """
        Compute the Lyman-alpha
        3D power spectrum

        Parameters
        ----------
        k: 1D array
          scale in [h/Mpc]
        mu: float
        """
        p3dLya = self.bF ** 2 * (1 + self.beta * mu**2)**2
        p3dLya *= self.p3dMatter(self.z, k) * self.D(k, mu)
        return p3dLya


    def get_P1D(self, kPara):
        """
        Integrate the Lya 3D power
        spectrum to get the P1D
        along the line of sight

        Parameters
        ----------
        kPara: float
          parallel scale in [h/Mpc]
        """
        mu = lambda kPerp: kPara / np.sqrt(kPara**2 + kPerp**2)
        integrand = lambda kPerp: kPerp * self.get_P3D(np.sqrt(kPara**2+kPerp**2),
                                                       mu(kPerp)) / (2 * np.pi)
        result = scipy.integrate.quad(integrand, 1e-3, 1e2, epsabs=0., epsrel=1.e-2)[0]
        return result


    # Response to matter overdensity
    def dlogPlin_dDelta(self, k):
        """
        Compute the response of the
        linear matter power spectrum
        to a large scale overdensity

        Parameters
        ----------
        k: 1D array
          scale in [h/Mpc]
        """
        epsilon = 1e-4
        dPlin_dlogk = (self.p3dMatter(self.z, k * (1 + epsilon)) -
                       self.p3dMatter(self.z, k * (1 - epsilon))) / (2*epsilon)
        dlogPlin_dlogk = dPlin_dlogk / self.p3dMatter(self.z, k)
        output = 68 / 21
        output -= 1 + 1/3 * dlogPlin_dlogk
        return output


    def dlogbF_dDelta(self):
        """
        Compute the response of the bias bF
        to a large scale overdensity
        """
        bHs8 = self.fbF(self.tableHs8)
        bLs8 = self.fbF(self.tableLs8)
        bMs8 = self.fbF(self.tableMs8)

        db_ds8 = (bHs8 - bLs8) / (0.88 - 0.64)
        dlogb_dlogs8 = 0.76 / bMs8 * db_ds8
        dlogb_dDelta = 13 / 21 * dlogb_dlogs8
        return dlogb_dDelta


    def dlog1pBetaMu_dDelta(self, mu):
        """
        Compute the response of the Kaiser term
        to a large scale overdensity

        Parameters
        ----------
        mu: float
        """
        betaHs8 = self.fbeta(self.tableHs8)
        betaLs8 = self.fbeta(self.tableLs8)
        betaMs8 = self.fbeta(self.tableMs8)

        dbeta_ds8 = (betaHs8 - betaLs8) / (0.88 - 0.64) * mu ** 2
        dlogbeta_dlogs8 = 0.76 / (1 + betaMs8 * mu ** 2) * dbeta_ds8
        dlogbeta_dDelta = 13 / 21 * dlogbeta_dlogs8

        return dlogbeta_dDelta

    def dlogD_dDelta(self, k, mu):
        """
        Compute the response of the
        non-linear term to a large scale
        overdensity

        Parameters
        ---------
        k: 1D array
          scale in h/Mpc
        mu: float
        """
        epsilon = 1e-4
        fDMs8 = lambda kc : self.fD(kc, mu, self.fq1(self.tableMs8),
                           self.fq2(self.tableMs8),
                           self.fkvav(self.tableMs8),
                           self.fav(self.tableMs8),
                           self.fbv(self.tableMs8),
                           self.fkp(self.tableMs8))
        dlogD_dlogk =  (fDMs8(k * (1 + epsilon)) -
                        fDMs8(k * (1 - epsilon))) / (2 * epsilon) / fDMs8(k)
        #dlogD_dlogk =  (self.D(k * (1 + epsilon), mu) -
        #                self.D(k * (1 - epsilon), mu)) / (2 * epsilon) / self.D(k, mu)

        DHs8 = self.fD(k, mu, self.fq1(self.tableHs8),
                       self.fq2(self.tableHs8),
                       self.fkvav(self.tableHs8),
                       self.fav(self.tableHs8),
                       self.fbv(self.tableHs8),
                       self.fkp(self.tableHs8))
        DLs8 = self.fD(k, mu, self.fq1(self.tableLs8),
                       self.fq2(self.tableLs8),
                       self.fkvav(self.tableLs8),
                       self.fav(self.tableLs8),
                       self.fbv(self.tableLs8),
                       self.fkp(self.tableLs8))
        DMs8 = self.fD(k, mu, self.fq1(self.tableMs8),
                       self.fq2(self.tableMs8),
                       self.fkvav(self.tableMs8),
                       self.fav(self.tableMs8),
                       self.fbv(self.tableMs8),
                       self.fkp(self.tableMs8))

        dD_ds8 = (DHs8 - DLs8) / (0.88 - 0.64)
        dlogD_dlogs8 = 0.76 / DMs8 * dD_ds8

        delta2 = self.p3dMatter(self.z, k) * k**3 / (2 * np.pi**2)
        dlogD_dlogPlin = 2 * self.fq1(self.tableMs8) * delta2
        dlogD_dlogPlin += 4 * self.fq2(self.tableMs8) * delta2 ** 2
        dlogD_dlogPlin *= (1 - k**self.fav(self.tableMs8)/self.fkvav(self.tableMs8) * mu ** self.fbv(self.tableMs8))

        dlogD_dlogs8 += 2 * dlogD_dlogPlin

        result = 13 / 21 * dlogD_dlogs8 - 1/3 * dlogD_dlogk
        return result

    def dlogP3D_dDelta(self, k, mu):
        """
        Compute the response of the
        3D Lya power spectrum to a
        large scale overdensity

        Parameters
        ---------
        k: 1D array
          scale in h/Mpc
        mu: float
        """
        result = self.dlogPlin_dDelta(k)
        result += 2 * self.dlogbF_dDelta()
        result += 2 * self.dlog1pBetaMu_dDelta(mu)
        result += self.dlogD_dDelta(k, mu)

        return result


    def dP3D_dDelta(self, k, mu):

        return self.get_P3D(k, mu) * self.dlogP3D_dDelta(k, mu)


    def dP1D_dDelta(self, kPara):
        """
        Compute the response of the
        1D Lya power spectrum to a
        large scale overdensity

        Parameters
        ---------
        kPara: float
          parallel scale in h/Mpc
        """
        mu = lambda kPerp: kPara / np.sqrt(kPara**2 + kPerp**2)
        k = lambda kPerp: np.sqrt(kPara**2+kPerp**2)
        integrand = lambda kPerp: (kPerp *
                                   self.get_P3D(k(kPerp),mu(kPerp)) *
                                   self.dlogP3D_dDelta(k(kPerp), mu(kPerp)) / (2 * np.pi))
        result = scipy.integrate.quad(integrand, 1e-3, 1e2, epsabs=0., epsrel=1.e-2)[0]
        return result

    def integrand_dP1D_dDelta(self, kPara):

        mu = lambda kPerp: kPara / np.sqrt(kPara**2 + kPerp**2)
        k = lambda kPerp: np.sqrt(kPara**2+kPerp**2)
        integrand = lambda kPerp: (kPerp *
                                   self.get_P3D(k(kPerp),mu(kPerp)) *
                                   self.dlogP3D_dDelta(k(kPerp), mu(kPerp)) / (2 * np.pi))
        kPerp = np.logspace(-3, 2, 1000)

        return integrand(kPerp)

    def integrand_dP1D_dDelta_log(self, kPara):
        mu = lambda kPerp: kPara / np.sqrt(kPara**2 + kPerp**2)
        k = lambda kPerp: np.sqrt(kPara**2+kPerp**2)
        integrand = lambda kPerp: (kPerp**2 *
                                   self.get_P3D(k(kPerp),mu(kPerp)) *
                                   self.dlogP3D_dDelta(k(kPerp), mu(kPerp)) / (2 * np.pi))
        kPerp = np.logspace(-3, 2, 1000)

        return integrand(kPerp)
