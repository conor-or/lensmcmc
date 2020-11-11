from models.masses.MassModel import *
from models.masses.SinglePowerLaw import SinglePowerLaw


class BrokenPowerLaw(MassModel):

    def __init__(self, params):
        """
        Elliptical, homoeoidal mass model with an inner_slope
        and outer_slope, continuous in density across break_radius.
        Position angle is defined to be zero on x-axis and
        +ve angle rotates the lens anticlockwise

        The grid variable is a tuple of (theta_1, theta_2), where
        each theta_1, theta_2 is itself a 2D array of the x and y
        coordinates respectively.~
        """
        self.b = params['lensing_strength']
        self.r = params['break_radius']
        self.t1 = params['inner_slope']
        self.t2 = params['outer_slope']

        if 'ellipticity_x' in params.keys():
            self.q = 1.0 - np.hypot(params['ellipticity_x'],
                                    params['ellipticity_y'])
            self.a = np.arctan2(params['ellipticity_y'],
                                params['ellipticity_x'])

        else:
            self.q = params['axis_ratio']
            self.a = params['position_angle']

        # Parameters defined in the notes
        self.nu = self.r / self.b
        self.dt = (2 - self.t1) / (2 - self.t2)

        # Normalisation (eq. 5)
        if self.nu < 1:
            self.kB = (2 - self.t1) / (
                    (2 * self.nu ** 2) * (1 + self.dt * (self.nu ** (self.t2 - 2) - 1)))
        else:
            self.kB = (2 - self.t1) / (2 * self.nu ** 2)

    def convergence(self, grid):
        """
        Returns the dimensionless density kappa=Sigma/Sigma_c (eq. 1)
        """
        root_q = np.sqrt(self.q)

        # Rotate coordinates
        x = grid[0] * np.cos(self.a) + grid[1] * np.sin(self.a)
        y = grid[1] * np.cos(self.a) - grid[0] * np.sin(self.a)

        # Elliptical radius
        R = np.hypot(x * self.q, y)

        # Inside break radius
        kappa_inner = self.kB * (self.r / R) ** self.t1

        # Outside break radius
        kappa_outer = self.kB * (self.r / R) ** self.t2

        return (kappa_inner * (R <= self.r) +
                kappa_outer * (R > self.r))

    def deflection_angle(self, grid, max_terms=10):
        """
        Returns the complex deflection angle from eq. 18 and 19
        """
        # Rotate coordinates
        z = (grid[0] + 1j * grid[1]) * np.exp(- 1j * self.a)

        # Elliptical radius
        R = np.hypot(z.real * self.q, z.imag)

        # Factors common to eq. 18 and 19
        factors = 2 * self.kB * (self.r ** 2) / (self.q * z * (2 - self.t1))

        # Hypergeometric functions
        # (in order of appearance in eq. 18 and 19)
        # These can also be computed with scipy.special.hyp2f1(), it's
        # much slower can be a useful test
        F1 = hyp2f1_series(self.t1, self.q, R, z, max_terms=max_terms)
        F2 = hyp2f1_series(self.t1, self.q, self.r, z, max_terms=max_terms)
        F3 = hyp2f1_series(self.t2, self.q, R, z, max_terms=max_terms)
        F4 = hyp2f1_series(self.t2, self.q, self.r, z, max_terms=max_terms)

        # theta < break radius (eq. 18)
        inner_part = factors * F1 * (self.r / R) ** (self.t1 - 2)

        # theta > break radius (eq. 19)
        outer_part = factors * (F2 + self.dt * (((self.r / R) ** (self.t2 - 2)) * F3 - F4))

        # Combine and take the conjugate
        alpha = (inner_part * (R <= self.r) +
                 outer_part * (R > self.r)).conjugate()

        # Rotate the components
        return alpha * np.exp(1j * self.a)

    def shear(self, grid, max_terms=10):
        """
        Returns the complex shear
        """

        # Rotate coordinates
        x, y = self.rotation(grid)
        z = x + 1j * y

        # Get q dashed
        q_ = (1 - self.q ** 2) / (self.q ** 2)

        # Get models for SPLs with inner and outer slopes
        spl_1 = SinglePowerLaw({
            'lensing_strength': self.b, 'slope': self.t1,
            'axis_ratio': self.q, 'position_angle': self.a
        })
        spl_2 = SinglePowerLaw({
            'lensing_strength': self.b, 'slope': self.t2,
            'axis_ratio': self.q, 'position_angle': self.a
        })

        # Inner shear
        inner_shear = 2 * self.kB * ((self.nu ** self.t1) / (2 - self.t1)) * spl_1.shear(grid)

        # Outer shear
        pre_factor = 2 * self.kB / (2 - self.t1)
        norm = ((self.r / z) ** 2) / self.q
        t1 = (self.t1 - self.t2) / np.sqrt(1 - q_ * (self.r / z) ** 2)
        t2 = (1 - self.t1) * hyp2f1_series(self.t1, self.q, self.r, z)
        t3 = (1 - self.t2) * hyp2f1_series(self.t2, self.q, self.r, z)
        t4 = (self.nu ** self.t2) * self.dt * spl_2.shear(grid)
        outer_shear = pre_factor * (norm * (t1 + t2 - t3) + t4)

        # Combine
        mask = self.elliptical_radius(grid) < self.r
        return (inner_shear * mask + outer_shear * (1 - mask)).conjugate()
