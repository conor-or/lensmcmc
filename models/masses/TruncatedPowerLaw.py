from models.masses.MassModel import *


class TruncatedPowerLaw(MassModel):

    def __init__(self, params):
        """
        Truncated power-law model from Paper III. Mass-density
        is truncated at an elliptical radius truncation_radius.
        Equations referenced are in Mass profiles III.
        """
        self.b = params['lensing_strength']
        self.r = params['truncation_radius']
        self.t = params['slope']

        if 'ellipticity_x' in params.keys():
            self.q = 1.0 - np.hypot(params['ellipticity_x'],
                                    params['ellipticity_y'])
            self.a = np.arctan2(params['ellipticity_y'],
                                params['ellipticity_x'])

        else:
            self.q = params['axis_ratio']
            self.a = params['position_angle']

        # Parameters defined in the paper
        self.nu = self.r / self.b

        # Normalisation (eq. 6)
        if self.nu <= 1:
            self.kT = (2 - self.t) / (2 * self.nu ** 2)
        else:
            self.kT = (2 - self.t) / (2 * self.nu ** self.t)

    def convergence(self, grid):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1)
        """

        # Rotate coordinates
        x = grid[0] * np.cos(self.a) + grid[1] * np.sin(self.a)
        y = grid[1] * np.cos(self.a) - grid[0] * np.sin(self.a)

        # Elliptical radius
        R = np.hypot(x * self.q, y)

        # Inside truncation radius
        kappa_inner = self.kT * (self.r / R) ** self.t

        # Outside truncation radius
        kappa_outer = np.zeros_like(R)

        return (kappa_inner * (R <= self.r) +
                kappa_outer * (R > self.r))

    def deflection_angle(self, grid, max_terms=10):
        """
        Returns the complex deflection angle from eq. 14
        """
        # Rotate coordinates
        z = (grid[0] + 1j * grid[1]) * np.exp(- 1j * self.a)

        # Elliptical radius
        R = np.hypot(z.real * self.q, z.imag)

        # Factors common to inner and outer parts
        factors = (self.b ** 2) / (self.q * z)

        # Hypergeometric functions
        # (in order of appearance in eq. 14
        F1 = hyp2f1_series(self.t, self.q, R, z, max_terms=max_terms)
        F2 = hyp2f1_series(self.t, self.q, self.r, z, max_terms=max_terms)

        # theta < break radius (first line of eq. 14)
        inner_part = factors * F1 * (self.r / R) ** (self.t - 2)

        # theta > break radius (second line of eq. 14)
        outer_part = factors * F2

        # Combine and take the conjugate
        alpha = (inner_part * (R <= self.r) +
                 outer_part * (R > self.r)).conjugate()

        # Rotate the components
        return alpha * np.exp(1j * self.a)

    def shear(self, grid, max_terms=10):
        """
        Returns the complex shear
        """

        ### TODO

        return np.zeros_like(grid[0])
