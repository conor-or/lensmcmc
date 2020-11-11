from models.sources.SourceModel import *


class Sersic(SourceModel):

    def __init__(self, params):

        self.x = params['x_position']
        self.y = params['y_position']
        self.r = params['radius']
        self.n = params['sersic_index']
        self.I0 = params['brightness']

        # Account for ellipticity/position angle transformation
        if 'ellipticity_x' in params.keys():
            self.q = 1.0 - np.hypot(params['ellipticity_x'],
                                    params['ellipticity_y'])
            self.a = np.arctan2(params['ellipticity_y'],
                                params['ellipticity_x'])

        else:
            self.q = params['axis_ratio']
            self.a = params['position_angle']

    def sersic_b(self):
        """
        Normalisation constant
        """
        return (2 * self.n) - (1.0 / 3.0) + (4.0 / (405.0 * self.n)) + \
               (46.0 / (25515.0 * self.n ** 2)) + (131.0 / (148174.0 * self.n ** 3))

    def brightness_profile(self, beta):
        """
        Creates a Sersic source centered on (srcx, srcy)
        with a radius srcr, peak brightness srcb and Sersuc index m.
        """

        # Shift to source centre
        z = beta - self.x - 1j * self.y

        # Rotate
        z *= np.exp(- 1j * self.a)

        # Transform to elliptical coords.
        rootq = np.sqrt(self.q)
        r = np.hypot(rootq * z.real, z.imag / rootq)

        # Call the sersic profile
        return self.I0 * np.exp(- self.sersic_b() * (r / self.r) ** (1.0 / self.n))
