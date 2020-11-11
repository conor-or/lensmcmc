from models.sources.SourceModel import *
from scipy.interpolate import RectBivariateSpline


class InterpolatedImage(SourceModel):

    def __init__(self, params, x, y, img):
        """
        Source model with brightness function interpolated
        from img, with pixel coordinates x and y.
        
        Source params gives centre of source (x and y position)
        and a scale factor r which scales the image in the source
        plane.
        """

        # Parameters
        self.x = params['x_position']
        self.y = params['y_position']
        self.r = params['radius']  # Arbitrary scale factor for source
        self.source_function = RectBivariateSpline(x * self.r, y * self.r, img)

    def brightness_profile(self, beta):
        # Shift to source centre
        z = beta - self.x - 1j * self.y

        # Call interpolation
        return self.source_function(z.real.flatten(), z.imag.flatten(), grid=False).reshape(z.shape)
