from tools import generic_yaml_loader
from importlib import import_module
import numpy as np
import os
import time


class EmceeLikelihood:

    def __init__(self, obs, fit_params,
                 priors_file=None, widths_file=None):

        # Set default prior and width files
        self.default_priors = os.environ['LENSMCMC'] + '/settings/defaults/priors.yaml'
        self.default_widths = os.environ['LENSMCMC'] + '/settings/defaults/widths.yaml'

        # Copy data to object
        self.obs, self.p = obs, fit_params

        # Get true parameter values
        self.mu = []
        for model in self.p['model']['active_parameters'].keys():
            for key in self.p['model']['active_parameters'][model]:
                self.mu.append(self.p['model']['params'][model][key])
        self.mu = np.array(self.mu)

        # Get priors
        if priors_file:
            self.priors = self.priors_loader(priors_file)
        else:
            self.priors = self.priors_loader(self.default_priors)

        # Get starting widths
        if widths_file:
            self.widths = self.widths_loader(widths_file)
        else:
            self.widths = self.widths_loader(self.default_widths)

    def prior_check(self, theta):
        """
        Checks points in the parameter space aganst the prior probability
        """
        return all([self.priors[i][0] <= theta[i] <= self.priors[i][1] for i in range(len(theta))])

    def priors_loader(self, priors_file):
        """
        Load the priors from priors_file for the active_parameters
        """
        priors_ = generic_yaml_loader(priors_file)
        priors = []
        for key in self.p['model']['active_parameters'].keys():
            priors += [priors_[key][k] for k in
                       self.p['model']['active_parameters'][key]]
        return priors

    def widths_loader(self, widths_file):
        """
        Load the starting Gaussian widths for the active parameters
        """
        widths_ = generic_yaml_loader(widths_file)
        widths = []
        for key in self.p['model']['active_parameters'].keys():
            widths += [widths_[key][k] for k in
                       self.p['model']['active_parameters'][key]]
        return widths

    def theta_dict(self, theta):
        """
        Converts emcee's theta into a dictionary of model parameters
        for use with Linsen models
        """

        # Copy specified model parameters
        model_params = self.p['model']['params'].copy()

        # Replace active parameters with theta
        i = 0
        for key in self.p['model']['active_parameters'].keys():
            for parameter in self.p['model']['active_parameters'][key]:
                model_params[key][parameter] = theta[i]
                i += 1

        return model_params


class EmceeTest(EmceeLikelihood):
    """
    Test likelihood for the emcee sampler
    """

    def __init__(self, obs, fit_params, priors_file, widths_file):

        # Copy data and initialise super class
        super().__init__(obs, fit_params, priors_file, widths_file)

        # Covariance matrix
        self.cov = np.diag(self.widths)

    def __call__(self, theta, *args, **kwargs):
        """
        For a parameter vector theta, return a Gaussian
        log-likelihood centered on the true value with stdvs
        set by the starting widths.
        """

        if self.prior_check(theta):

            X = np.matrix(theta)
            M = np.matrix(self.mu)
            C = np.matrix(self.cov)
            log_likelihood = -0.5 * (X - M) * C.I * (X - M).T
            time.sleep(np.random.uniform(0.000001, 0.02))

        else:

            log_likelihood = - np.inf

        return log_likelihood


class EmceeExtendedSource(EmceeLikelihood):

    def __init__(self, obs, fit_params, priors_file=None, widths_file=None):

        # Copy data and initialise super class
        super().__init__(obs, fit_params, priors_file, widths_file)

        # Load mass model
        lens_module = import_module('models.masses.' + self.p['model']['lens_model'])
        self.MassModel = getattr(lens_module, self.p['model']['lens_model'])

        # Load source model
        src_module = import_module('models.sources.' + self.p['model']['source_model'])
        self.SourceModel = getattr(src_module, self.p['model']['source_model'])

    def __call__(self, theta, *args, **kwargs):

        if self.prior_check(theta):

            # Collate fixed and fitted parameters
            model_params = self.theta_dict(theta)

            # Make source model
            src = self.SourceModel(model_params['src'])

            # Make lens model
            lens = self.MassModel(model_params['lens'])

            # Get model image plane
            model_image_plane = lens.image_plane_sub_pix(
                self.obs.pix, src, sub=self.obs.p['obs']['sub_pixel_lvl']
            )

            # Calculate chisq.
            logprob = - 0.5 * np.sum(
                (self.obs.image_plane - model_image_plane) ** 2.0 /
                (self.obs.noise_level ** 2.0)
            )

            # Check the probability is valid and return
            return logprob if not np.isnan(logprob) else - np.inf

        else:
            return - np.inf
