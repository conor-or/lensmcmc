import numpy as np
import matplotlib.pyplot as plt
from tools import observation_yaml_loader
from tools.lensing import pixels, snr_set
from tools.plotting import dark_mode, rcdefaults


class Observation:

    def __init__(self, file):
        """
        Creates and stores one mock observation
        with settings specified in file
        """

        # Load observation parameters, mass model and source model
        self.p, MassModel, SourceModel = observation_yaml_loader(file)

        # Save the above
        self.massModel = MassModel(self.p['lens'])
        self.sourceModel = SourceModel(self.p['src'])

        # Initialise pixel grid
        self.pix = pixels(self.p['obs']['pixel_width'],
                          self.p['obs']['field_of_view'])

        # Compute noiseless image plane
        self.img_noiseless = self.massModel.image_plane_sub_pix(self.pix, self.sourceModel)

        # Get the noise level and mask
        self.noise_level, self.mask = snr_set(self.img_noiseless,
                                              self.p['obs']['sig_noise_ratio'],
                                              self.p['obs']['mask_level'],
                                              self.p['obs']['conv_kernel_size'])

        # Create the noise field
        self.noise_field = np.random.normal(0.0, self.noise_level, self.img_noiseless.shape)

        # Add the noise to the image
        self.image_plane = self.img_noiseless + self.noise_field

    def lens_param_string(self):

        param_string = ''
        for k in self.p['lens'].keys():
            if type(self.p['lens'][k]) == str:
                param_string += '{:18s} {:s}\n'.format(k, self.p['lens'][k])
            else:
                param_string += '{:18s} {:.2f}\n'.format(k, self.p['lens'][k])

        return param_string[:-1]

    def source_param_string(self):

        param_string = ''
        for k in self.p['src'].keys():
            if type(self.p['src'][k]) == str:
                param_string += '{:18s} {:s}\n'.format(k, self.p['src'][k])
            else:
                param_string += '{:18s} {:.2f}\n'.format(k, self.p['src'][k])

        return param_string[:-1]

    def plot(self):

        dark_mode()
        fig, ax = plt.subplots()

        ext = self.p['obs']['field_of_view'] / 2.0

        # Image plane
        ax.imshow(self.image_plane, interpolation='none', origin='lower',
                  extent=[-ext, ext, -ext, ext], cmap='magma', zorder=-100)

        # Source
        ax.plot([self.sourceModel.x], [self.sourceModel.y], 'wx')

        # Calculate critical curve and caustic
        crit, caus = self.massModel.critical_curve_caustic()

        # Critical curve
        for c in [crit, caus]:
            ax.plot(c[:, 0], c[:, 1], 'w', alpha=0.2)

        # Break radius
        if 'break_radius' in self.p['lens'].keys():
            ax.contour(self.pix[0], self.pix[1],
                       self.massModel.elliptical_radius(self.pix),
                       levels=[self.p['lens']['break_radius']],
                       colors='w', linestyles='--', alpha=0.2)

        # Source Parameters
        ax.text(-2.9, -2.9, self.source_param_string(), family='monospace',
                ha='left', va='bottom', size=10, alpha=0.5, color='w')

        # Lens Parameters
        ax.text(-2.9, 2.9, self.lens_param_string(), family='monospace',
                ha='left', va='top', size=10, alpha=0.5, color='w')

        # Grid lines
        ax.axhline(0.0, linewidth=1.0, alpha=0.1, color='w')
        ax.axvline(0.0, linewidth=1.0, alpha=0.1, color='w')

        # format_axes(ax)
        ax.set(xticklabels=[], yticklabels=[])
        fig.set_size_inches(6, 6)
        fig.tight_layout()
        rcdefaults()

        return fig
