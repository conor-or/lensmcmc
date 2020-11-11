import numpy as np
from datetime import datetime as dt
from tools import fitting_yaml_loader
from tools import likelihoods as likelihoods
from tools import samplers as samplers
from emcee import EnsembleSampler
from math import floor


class ModelFit:
    
    def __init__(self, obs, file, priors_file=None, widths_file=None,
                 pool=None, it=0):
        """
        Fits the model specified in file to the observation
        in obs
        """

        # Save observation
        self.obs = obs

        # Load the fitting settings from file
        self.p = fitting_yaml_loader(file, obs=self.obs)

        # Get likelihood
        Likelihood = getattr(likelihoods, self.p['mcmc']['likelihood'])

        # Initialise likelihood
        self.likelihood = Likelihood(self.obs, self.p,
                                     priors_file, widths_file)

        # Get sampler
        Sampler = getattr(samplers, self.p['mcmc']['sampler'])

        # Initialise sampler
        self.sampler = Sampler(self.p, self.likelihood, pool=pool, it=it)
