import camb

class CosmoTools:

    def __init__(self, **kwargs):


        pars = camb.CAMBparams()
        pars.set_cosmology(cosmomc_theta = kwargs["cosmomc_theta"],
                           ombh2 = kwargs["ombh2"],
                           omch2 = kwargs["omch2"],
                           tau = kwargs["tau"])
        pars.InitPower.set_params(As = kwargs["As"], ns = kwargs["ns"])
        pars.set_for_lmax(5000, lens_potential_accuracy=1.)

        cambResults = camb.get_results(pars)
        p3d = camb.get_matter_power_interpolator(pars, zmin = 0.,
                                                 zmax = 4., kmax = 1e2,
                                                 nonlinear=False,extrap_kmax=True).P

        self.camb_results = cambResults
        self.matter_power = p3d


    def H(self, z):
        return self.camb_results.hubble_parameter(z)

    @property
    def h(self):
        return self.H(0) / 100.

    # Units definition
    def Mpc_to_Mpc_over_h(self, x_Mpc):
        x_Mpc_over_h = x_Mpc * self.h
        return x_Mpc_over_h

    def Mpc_over_h_to_Mpc(self, x_Mpc_over_h):
        x_Mpc = 1 / self.h * x_Mpc_over_h
        return x_Mpc

    def invMpc_to_h_over_Mpc(self, k_invMpc):
        k_h_over_Mpc = 1 / self.h * k_invMpc
        return k_h_over_Mpc

    def h_over_Mpc_to_invMpc(self, k_h_over_Mpc):
        k_invMpc = self.h * k_h_over_Mpc
        return k_invMpc

    def get_chi_from_z(self, z):
        """
        Get the comoving radial distance chi (Mpc / h)
        corresponding to the redshift z

        Parameters
        ----------
        z: 1D array
        """
        chi_Mpc = self.camb_results.comoving_radial_distance(z)
        return self.Mpc_to_Mpc_over_h(chi_Mpc)

    def get_z_from_chi(self, chi):
        """
        Get the redshift associated with the
        comoving distance chi (Mpc/h)

        Parameters:
        chi: 1D array
        """
        chi_Mpc = self.Mpc_over_h_to_Mpc(chi)
        return self.camb_results.redshift_at_comoving_radial_distance(chi_Mpc)

    def lensing_eff_kernel_z(self, z, cosmology):
        """
        Get the lensing efficiency kernel such
        that \int dz W(z) delta(z, rperp) = kappa(rperp).
        W(z) is given in h/Mpc

        Parameters
        ----------
        z: 1D array
        """
        omegam = (cosmology.camb_results.get_Omega("cdm") +
                  cosmology.camb_results.get_Omega("baryon"))
        c = 3e5
        a = 1 / ( 1 + z )
        Hz = cosmology.H(z) / cosmology.h

        chi = cosmology.get_chi_from_z(z)
        chistar = cosmology.get_chi_from_z(1100)

        return 1.5 * omegam * 100 ** 2 / (a * H) * (chi / c) * (chistar - chi) / chistar


    def lensing_eff_kernel(self, chi, cosmology):
        """
        Get the lensing efficiency kernel such
        that \int dchi W(chi) delta(chi, rperp) = kappa(rperp).
        W(chi) is given in h/Mpc

        Parameters
        ----------
        chi: 1D array
        """
        omegam = (cosmology.camb_results.get_Omega("cdm") +
                  cosmology.camb_results.get_Omega("baryon"))
        c = 3e5
        chistar = cosmology.get_chi_from_z(1100)
        z = cosmology.get_z_from_chi(chi)
        a = 1 / (1 + z)

        return 1.5 * omegam * (100 / c) ** 2 * chi * (chistar - chi) / chistar / a
