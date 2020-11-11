import os
import sys
import shutil
import numpy as np
from tools import generic_yaml_loader, check_master, mkdir_check
from tools.observation import Observation
from tools.modelfit import ModelFit
from math import ceil


class ObservationFit:
    """
    Makes a mock observation according to the observation
    settings file and fits a model according to model
    setttings file
    """

    def __init__(self, save_dir,
                 observation_settings_file=None,
                 model_settings_file=None, priors_file=None,
                 widths_file=None, pool=None):

        # Copy save directory
        self.save_dir = save_dir
        it = int(self.save_dir.split('/')[-1])

        # Initialise on master process
        if check_master(pool):
            self.initialise_file_structure(observation_settings_file,
                                           model_settings_file, priors_file,
                                           widths_file)

            # Make mock observation
            self.observation = Observation(self.obs_file)

            # Save the observation to file
            self.save_observation()

        # Initialise fitting procedure
        self.modelfit = ModelFit(self.observation, self.mod_file,
                                 priors_file, widths_file, pool=pool, it=it)

    def initialise_file_structure(self, obs, mod, priors, widths):

        # If no settings are given, use the ones in the save_dir
        # else make the new directory and copy the given files

        # Observation
        try:
            obs_path = self.save_dir + '/settings/observation_settings.yaml'
            if (not obs) and os.path.exists(obs_path):
                self.obs_file = obs_path
            else:
                mkdir_check(self.save_dir)
                mkdir_check(self.save_dir + '/settings')
                shutil.copy(obs, obs_path)
                self.obs_file = obs_path

            # Model
            mod_path = self.save_dir + '/settings/model_settings.yaml'
            if (not mod) and os.path.exists(mod_path):
                self.mod_file = mod_path
            else:
                mkdir_check(self.save_dir)
                mkdir_check(self.save_dir + '/settings')
                shutil.copy(mod, mod_path)
                self.mod_file = mod_path
        except TypeError:
            print("Error: Settings files not found")
            sys.exit(0)

        # If not using default priors and widths copy those too
        if priors:
            shutil.copy(priors, self.save_dir + '/settings/priors.yaml')
        if widths:
            shutil.copy(widths, self.save_dir + '/settings/widths.yaml')

    def save_observation(self):

        # Make directory
        self.obs_dir = self.save_dir + '/observation'
        self.plt_dir = self.save_dir + '/plots'
        for path in [self.obs_dir, self.plt_dir]:
            if not os.path.exists(path):
                os.mkdir(path)

        # Save image plane plot
        fig = self.observation.plot()
        fig.savefig(self.plt_dir + '/image_plane.png', facecolor=fig.get_facecolor())

        # Save data
        np.save(self.obs_dir + '/image.npy', self.observation.image_plane)
        np.save(self.obs_dir + '/noise.npy', self.observation.noise_field)
        np.save(self.obs_dir + '/noise_level.npy',
                np.array(self.observation.noise_level))
        np.save(self.obs_dir + '/image_noiseless.npy',
                self.observation.img_noiseless)

    def fit_model(self, pool=None):

        if not check_master(pool):
            pool.wait()
            sys.exit(0)

        # Perform burn-in
        self.modelfit.sampler.run_burn_in(pool=pool)

        # Perform sampling
        self.modelfit.sampler.run_sampling(pool=pool)

        # Prune final chain
        self.modelfit.sampler.format_chain()

        # Save fitting results
        if check_master(pool):
            self.save_results()

    def save_results(self):

        # Make directory
        result_dir = self.save_dir + '/results'
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)

        # Save MCMC chain
        np.save(result_dir + '/chain.npy', self.modelfit.sampler.final_chain)

        # Save log probability at each sample
        np.save(result_dir + '/lnpost.npy', self.modelfit.sampler.logprob)

        # Save corner plot
        fig = self.corner_plot()
        fig.savefig(self.plt_dir + '/corner.png', facecolor=fig.get_facecolor())

        # Save chain plot
        fig = self.chain_plot()
        fig.savefig(self.plt_dir + '/chains.png', facecolor=fig.get_facecolor())

    def corner_plot(self):

        from matplotlib import rcdefaults
        from tools.plotting import dark_mode
        from corner import corner
        dark_mode()

        # Load latex labels from file
        latex_labels = generic_yaml_loader(os.environ['LINSEN'] +
                                           '/settings/defaults/latex_labels.yaml')

        labels = []
        for key in self.modelfit.p['model']['active_parameters']:
            for param in self.modelfit.p['model']['active_parameters'][key]:
                labels.append(latex_labels[key][param])

        # Flatten chain
        chain_shape = self.modelfit.sampler.final_chain.shape
        samples = self.modelfit.sampler.final_chain.reshape(chain_shape[0] * chain_shape[1],
                                                            chain_shape[2])

        fig = corner(samples, labels=labels, truths=self.modelfit.likelihood.mu,
                     color='coral', truth_color='darkturquoise', bins=30,
                     fill_contours=False, show_titles=True, plot_datapoints=False,
                     levels=(1.0 - np.exp(-0.5 * np.array([1, 2, 3]))), quantiles=[0.16, 0.84],
                     hist_kwargs={'lw': 2.0, 'histtype': 'stepfilled', 'alpha': 0.8},
                     plot_density=False, no_fill_contours=True)

        fig.patch.set_facecolor([0.2] * 3)
        fig.set_size_inches(12, 12)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.0, wspace=0.0)

        fig.text(0.66, 0.88, self.modelfit.sampler.stats(),
                 family='monospace', size=12, alpha=0.8)

        rcdefaults()

        return fig

    def chain_plot(self):

        from matplotlib import rcdefaults
        from tools.plotting import dark_mode, rcParams
        import matplotlib.pyplot as plt
        dark_mode()
        rcParams['font.family'] = 'monospace'

        # Load labels for active parameters
        labels = []
        for key in self.modelfit.p['model']['active_parameters']:
            for param in self.modelfit.p['model']['active_parameters'][key]:
                labels.append(key + ':' + param)

        # Initialise fig with rows of 3
        fig, ax = plt.subplots(ceil(len(labels) / 3), 3, sharex=True)

        # Loop over parameters
        for j, a in enumerate(ax.flatten()[:len(labels)]):
            # Plot burn-in walkers
            for i in range(self.modelfit.sampler.burn_chain.shape[0]):
                a.plot(range(0, self.modelfit.sampler.burn_chain.shape[1]),
                       self.modelfit.sampler.burn_chain[i, :, j], 'darkturquoise', alpha=0.1)

            # Plot sampling walkers
            for i in range(self.modelfit.sampler.final_chain.shape[0]):
                a.plot(range(self.modelfit.sampler.burn_chain.shape[1] - 1,
                             self.modelfit.sampler.burn_chain.shape[1] +
                             self.modelfit.sampler.final_chain.shape[1] - 1),
                       self.modelfit.sampler.final_chain[i, :, j], 'coral', alpha=0.1)

            # Set labels etc
            a.set_ylabel(labels[j])
            a.set_xlabel('Step')
            a.xaxis.grid(True, alpha=0.6)
            a.axhline(self.modelfit.likelihood.mu[j], color='w', linestyle='dotted', alpha=0.8)

        # Delete unused axes
        if len(labels) % 3 != 0:
            for a in ax.flatten()[len(labels):]:
                fig.delaxes(a)

        fig.set_size_inches(12, 3 * ceil(len(labels) / 3))
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.0)
        rcdefaults()

        return fig
