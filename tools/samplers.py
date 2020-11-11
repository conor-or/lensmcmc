import numpy as np
import os
from tools import check_master
from emcee import EnsembleSampler
from datetime import datetime as dt
from scipy.stats import mode
from sklearn.cluster import DBSCAN
from math import floor
from jinja2 import Template


class EmceeEnsembleSampler:

    def __init__(self, params, likelihood, pool=None, it=0):

        # Storage
        if pool:
            self.n_cores = pool.size + 1
        else:
            self.n_cores = 1
        self.likelihood = likelihood
        self.p = params
        self.pre_burn_position = None
        self.post_burn_position = None

        # File for writing progress
        self.prog_fname = (
                os.environ['LINSEN_OUTPUT'] + ('/.progress/000_000_%03d.txt' % it)
        )

    def run_sampling(self, pool=None):

        # Take the time at the start of sampling
        self.sample_start_time = dt.now()

        # Respawn the walkers from the final burn-in position
        self.redistribute_walkers()

        # Initialise new sampler for final chain
        self.final_sampler = EnsembleSampler(
            self.p['mcmc']['walkers_initial'] * self.p['mcmc']['walkers_factor'],
            len(self.likelihood.mu), self.likelihood, pool=pool
        )

        # Run the sampler and write progress to file
        for i, a in enumerate(
                self.final_sampler.sample(self.post_burn_position,
                                          iterations=(self.p['mcmc']['final_iterations'] + 10))
        ):
            if check_master(pool):
                with open(self.prog_fname, 'w') as f:
                    f.write(self.write_progress(i, self.p['mcmc']['final_iterations'] + 10,
                                                self.sample_start_time, 'S'))

        # Record the finish time
        self.sample_finish_time = dt.now()

        # Prune the chain to remove dead walkers and drop second burn-in
        self.format_chain()

    def run_burn_in(self, pool=None):

        # Initialise sampler for burn-in
        self.burn_in_sampler = EnsembleSampler(
            self.p['mcmc']['walkers_initial'],
            len(self.likelihood.mu), self.likelihood, pool=pool
        )

        # Record start time
        self.burn_start_time = dt.now()

        # Initialise walkers
        self.walker_init()

        # Run the sampler and write progress to file
        for i, a in enumerate(
                self.burn_in_sampler.sample(self.pre_burn_position,
                                            iterations=self.p['mcmc']['burn_in_iterations'])
        ):

            if check_master(pool):
                with open(self.prog_fname, 'w') as f:
                    f.write(self.write_progress(i, self.p['mcmc']['burn_in_iterations'],
                                                self.burn_start_time, 'B'))

        # Save the chain
        self.burn_chain = self.burn_in_sampler.chain

    def walker_init(self):
        """
        Initialises the positions of the walkers as either
        - Gaussian ball around the max. likelihood (if centre=True)
        - Uniformly distributed across the prior volume (if centre=False)
        """

        # Get features of the likelihood
        means = self.likelihood.mu
        stdev = self.likelihood.widths
        prior = self.likelihood.priors
        n_w = self.p['mcmc']['walkers_initial']

        if self.p['mcmc']['start_walkers'] == 'Centre':
            p0 = [np.clip(np.random.normal(means[i], stdev[i], n_w).astype('float32'), prior[i][0], prior[i][1])
                  for i in range(len(self.likelihood.mu))]
        else:
            p0 = [np.random.uniform(prior[i][0], prior[i][1], n_w).astype('float32')
                  for i in range(len(self.likelihood.mu))]

        self.pre_burn_position = np.transpose(np.array(p0))

    def redistribute_walkers(self):
        """
        Given a chain of walkers, creates {factor} copies of the walkers
        distributed in Gaussian kernels around each point. The original points
        are retained to further speed up burn-in
        """

        # Get the final position
        p0 = self.burn_chain[:, -1, :]

        if self.p['mcmc']['walkers_factor'] > 1:

            # Get variance in each dimension at final position
            v = np.array(0.1 * p0.std(axis=0))

            # Get factor - 1 coordinates around each point
            p1 = []
            for p in p0:
                p1.append(np.array([np.random.normal(p[i], 0.1 * v[i], self.p['mcmc']['walkers_factor'] - 1)
                                    for i in range(len(v))]).transpose())
            p1 = np.array(p1)

            # Add the 1 set of real samples to the new set of factor - 1 fake samples
            self.post_burn_position = np.hstack((p0.reshape(len(p0), 1, self.burn_chain.shape[-1]), p1)).reshape(
                len(self.burn_chain) * self.p['mcmc']['walkers_factor'],
                self.burn_chain.shape[-1]
            ).astype('float32')

        else:

            self.post_burn_position = p0

    def format_chain(self, eps=0.05):

        if self.p['options']['prune']:

            prune_params = []
            for i, model in enumerate(self.p['options']['prune_params']):
                for param in self.p['options']['prune_params'][model]:
                    if i == 0:
                        lens_offset = 0
                    else:
                        lens_offset = len(self.p['model']['active_parameters']['lens'])
                    prune_params.append(self.p['model']['active_parameters'][model].index(param) + lens_offset)

            # Find the clusters via DBSCAN
            dbscan = DBSCAN(eps=eps)
            clusters = dbscan.fit_predict(self.final_sampler.chain[:, :, prune_params].mean(axis=1))

            # Get the walkers belonging to the most populous cluster
            self.prune_list = np.where(clusters == mode(clusters)[0][0])[0]

            # Discard pruned walkers
            self.final_chain = self.final_sampler.chain[self.prune_list, 10:, :]

            # Discard pruned likelihoods
            self.logprob = self.final_sampler.lnprobability[self.prune_list, 10:]

        else:

            self.final_chain = self.final_sampler.chain[:, 10:, :]
            self.logprob = self.final_sampler.lnprobability[:, 10:]

    def stats(self):
        text = """Sampler                {{ sm }}
No. Cores              {{ nc }}
Start Time             {{ st }}
End Time               {{ et }}
Time Taken             {{ tt }}
Acceptance Fraction    {{ af }}
Prune Fraction         {{ pf }}"""

        if self.p['options']['prune']:
            pf = round(1.0 - (float(len(self.prune_list)) /
                              float(self.post_burn_position.shape[0])), 4)
        else:
            pf = 0.0
        af = round(np.mean(self.final_sampler.acceptance_fraction), 4)
        t = Template(text)
        td = str(self.sample_finish_time - self.burn_start_time).split('.')[0]

        return t.render(sm='Emcee', af=af, pf=pf,
                        nc=self.n_cores,
                        st=self.burn_start_time.strftime("%D %H:%M"),
                        et=self.sample_finish_time.strftime("%D %H:%M"),
                        tt=td)

    @staticmethod
    def write_progress(j, n, t0, s, w=20):

        # Fraction of run complete
        frac = float(j + 1) / float(n)

        # Avg. time per iteration so far and time left in seconds
        time_per = float((dt.now() - t0).seconds) / float(j + 1)
        time_lft = float(time_per * (n - (j + 1)))

        # Finish time based on previous
        pred_time_h = floor(time_lft / 3600.0)
        pred_time_m = floor((time_lft % 3600) / 60.0)

        # Write progress string
        prog = (('%5d/%5d |' % (j + 1, n) + '#' * int(frac * w) +
                 ' ' * (w - int(frac * w)) + ('| %3d%% %s' % (int(100 * frac), s)) +
                 (' | %02dh%02dm' % (pred_time_h, pred_time_m))))

        return prog
