from models.masses.MassModel import *


class SinglePowerLaw(MassModel):
    """
    A traditional single power-law model with
    a 2D logarithmic slope t, lensing strength b,
    axis ratio q and position angle a
    """

    def __init__(self, params):

        self.b = params['lensing_strength']
        self.t = params['slope']

        # Account for ellipticity/position angle transform
        if 'ellipticity_x' in params.keys():
            self.q = 1.0 - np.hypot(params['ellipticity_x'],
                                    params['ellipticity_y'])
            self.a = np.arctan2(params['ellipticity_y'],
                                params['ellipticity_x'])

        else:
            self.q = params['axis_ratio']
            self.a = params['position_angle']

    def convergence(self, grid):
        """
        Dimensionless density profile
        """
        x, y = self.rotation(grid)
        er = self.elliptical_radius((x, y))

        return ((2 - self.t) / 2) * (self.b / er) ** self.t

    def deflection_angle(self, grid):
        """
        Complex deflection angle
        """
        # Rotate coordinates
        x, y = self.rotation(grid)
        z = x + 1j * y

        # Elliptical radius
        er = self.elliptical_radius((x, y))

        f1 = (self.b ** 2) / (self.q * z)
        f2 = (self.b / er) ** (self.t - 2)
        f3 = hyp2f1_series(self.t, self.q, er, z)

        # Rotate back into original frame
        return (f1 * f2 * f3).conjugate() * np.exp(- 1j * self.a)

    def shear(self, grid):
        """
        Complex shear
        """
        # Rotate coordinates
        x, y = self.rotation(grid)
        z = x + 1j * y

        t1 = (1 - self.t) * self.deflection_angle(grid).conjugate() / z
        t2 = self.convergence(grid) * z.conjugate() / z

        return (t1 - t2).conjugate()

    def potential(self, grid):
        """
        Lensing potential
        """
        # Get deflection angle
        alpha = self.deflection_angle(grid)

        return (grid[0] * alpha.real + grid[1] * alpha.imag) / (2 - self.t)

