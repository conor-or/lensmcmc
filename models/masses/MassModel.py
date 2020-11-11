import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from tools.lensing import pixels


def hyp2f1_series(t, q, r, z, max_terms=10):
    """
    Computes the Hypergeometric function numerically
    according to the recipe in Paper III.
    """

    # U from
    q_ = (1 - q ** 2) / (q ** 2)
    u = 0.5 * (1 - np.sqrt(1 - q_ * (r / z) ** 2))

    # First coefficient
    a_n = 1.0

    # Storage for sum
    F = np.zeros_like(z, dtype='complex64')

    for n in range(max_terms):
        F += a_n * (u ** n)
        a_n *= ((2 * n) + 4 - (2 * t)) / ((2 * n) + 4 - t)

    return F


class MassModel:
    """
    Generic class for handling lens mass models. Specific
    mass models must define:
    * Convergence
    * Shear
    * Deflection angle
    """

    def magnification(self, grid):
        """
        Scalar magnification from convergence and shear
        """
        k = self.convergence(grid)
        s = self.shear(grid)

        return 1.0 / (((1 - k) ** 2) - np.abs(s) ** 2)

    def rotation(self, grid):
        """
        Rotates the passed coordinates anticlockwise by an angle a
        """
        x = grid[0] * np.cos(self.a) - grid[1] * np.sin(self.a)
        y = grid[0] * np.sin(self.a) + grid[1] * np.cos(self.a)
        return x, y

    def elliptical_radius(self, grid):
        """
        Elliptical radius by R_e^2 = (qx)^2 + y^2
        """
        return np.hypot(self.q * grid[0], grid[1])

    def ray_trace(self, grid, source_model, sub=1):
        """
        Finds the image plane surface brightness with
        sub-pixelisation. Use sub=1 for no sub-pixelisation
        """

        if sub == 1:

            alpha = self.deflection_angle(grid)
            theta = grid[0] + 1j * grid[1]
            beta = theta - alpha

            return source_model.brightness_profile(beta)

        # Storage for grids and constants
        x1, x2 = [], []
        pwd = np.abs(grid[0][0, 0] - grid[0][0, 1])
        fov = grid[0].max() - grid[0].min() + pwd
        N = int(fov / pwd)

        # Loop over sub-pixelisation size and create
        # up to p.sub sub-pixelised grids
        for s in range(1, sub + 1):
            # Define 1D grid
            x = np.linspace(- (fov / 2.0) + (pwd / 2.0),
                            (fov / 2.0) - (pwd / 2.0),
                            N * s)

            # Define 2D grid and reshape
            xx, yy = map(lambda x: x.reshape(N, s, N, s),
                         np.meshgrid(x, x))

            # Add to list
            x1.append(xx)
            x2.append(yy)

        # Initial calculation with no sub-pixelisation
        img_ = np.sqrt(self.ray_trace((x1[0], x2[0]), source_model, sub=1))
        img = img_.mean(axis=3).mean(axis=1)

        # Find the level of detail (0-sub) necessary for that pixel
        msk = np.clip(np.array(np.ceil(sub * img / np.max(img)), dtype='int') - 1, 0, sub - 1)

        # Loop over the pixels and extract the necessary sub-grid for that pixel
        x1_sub, x2_sub, p_ix, t = [], [], [], 0
        for i in range(N):
            for j in range(N):
                # Pick out sub pixel coordinates for this pixel
                u = x1[msk[i, j]][i, :, j, :].flatten()
                v = x2[msk[i, j]][i, :, j, :].flatten()

                # Add coordinates and parent pixel indices to list
                x1_sub.append(u)
                x2_sub.append(v)

                p_ix.append([x for x in range(t, t + len(u))])
                t += len(u)

        # Collapse above into flat array
        x1_sub = np.array([x for s in x1_sub for x in s])
        x2_sub = np.array([x for s in x2_sub for x in s])

        # Calculate brightness in sub-pixels
        img_sub = self.ray_trace((x1_sub, x2_sub), source_model, sub=1)

        # Take the mean of the sub pixels for each pixel
        img = np.array([img_sub[p].mean() for p in p_ix])

        # Transform 1D -> 2D
        return img.reshape(N, N)

    def lens_equation(self, x, source_model):
        """
        The lens equation for this model, images are at
        zeroes
        """
        alpha = self.deflection_angle((x[0], x[1]))
        a1, a2 = alpha.real, alpha.imag
        return np.hypot(source_model.x - x[0] + a1, source_model.y - x[1] + a2)

    def image_finder(self, source_model, labels=False,
                     initial=np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])):
        """
        Finds the image positions given a source model
        """

        positions_ = []

        # Loop over images
        for i in range(len(initial)):

            # Find roots of lens equation
            roots = minimize(self.lens_equation, x0=initial[i], tol=1.0E-12,
                             args=(source_model), method='SLSQP')
            positions_.append(roots.x)

        positions_ = np.array(positions_)
        obj = np.array([self.lens_equation(p, source_model) for p in positions_])
        img_mask = obj < 0.0001
        positions = positions_[img_mask]

        if labels:

            # Label images according to angle from x-axis
            angle = np.arctan2(positions[:, 1], positions[:, 0])
            return positions, np.argsort(np.argsort(angle + ((angle < 0) * 2 * np.pi))) + 1

        else:
            return positions

    def critical_curve_caustic(self, resolution=0.01, field_of_view=6.0):

        # Get high resolution pixel grid
        high_res_pix = pixels(resolution, field_of_view)

        # Get critical curves as the contour where det(M) -> infinity
        fig, ax = plt.subplots()
        contour = ax.contour(high_res_pix[0], high_res_pix[1],
                              1.0 / self.magnification(high_res_pix),
                              levels=[0.0], colors='none')
        plt.close(fig)

        # Discard the smaller (inner) curve
        outer_curve_index = np.argmax([len(c) for c in contour.allsegs[0]])
        critical_curve = contour.allsegs[0][outer_curve_index]

        # Lens the critical curve to get the caustic
        alpha = self.deflection_angle((critical_curve[:, 0], critical_curve[:, 1]))
        caustic = critical_curve[:, 0] - alpha.real, critical_curve[:, 1] - alpha.imag

        # Put both in same shaped array
        return critical_curve, np.array(caustic).T
