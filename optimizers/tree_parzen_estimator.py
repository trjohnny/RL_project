import scipy.stats as st
from scipy.stats import kstest
import numpy as np
from sklearn.neighbors import KernelDensity

from optimizers.optimizer import Optimizer


class TPE(Optimizer):
    def __init__(self, agent, hyperparameters_grid, n_seed=4, n_total=20, gamma=.2):
        super().__init__(agent)
        self.n_samples_dist = 100
        self.space = hyperparameters_grid
        self.n_seed = n_seed
        self.n_total = n_total
        self.gamma = gamma

    def optimize(self, episodes):
        return self.fmin(episodes, self.n_seed, self.n_total, self.gamma)

    @staticmethod
    def __best_fit_distribution(data):
        """
        Determine best fit distribution for the given data. Uses Kolmogorov-Smirnov test.
        """
        # Normalize data
        data = (data - np.mean(data)) / np.std(data)

        dist_names = ['norm', 'lognorm', 'expon', 'gamma', 'weibull_min', 'weibull_max']
        best_dist_name = ''
        best_dist_params = ()
        min_ks_stat = float('inf')

        for dist_name in dist_names:
            dist = getattr(st, dist_name)
            try:
                # Skip distributions that can't be fit
                if not hasattr(dist, 'fit'):
                    continue
                # Parameters estimation
                dist_params = dist.fit(data)

                # Kolmogorov-Smirnov test
                ks_stat, ks_pval = kstest(data, dist_name, dist_params)
                # Save if best
                if ks_stat < min_ks_stat:
                    min_ks_stat = ks_stat
                    best_dist_name = dist_name
                    best_dist_params = dist_params
            except:
                pass

        # Create distribution with best parameters
        best_dist = getattr(st, best_dist_name)
        distribution = best_dist(*best_dist_params)

        return distribution.cdf

    @staticmethod
    def choose_next_hps(l_kde, g_kde, n_samples):
        """
        Consumes KDE's for l(x) and g(x), samples n_samples from
        l(x) and evaluates each sample with respect to g(x)/l(x).
        The sample which maximizes this quantity is returned as the
        next set of hyperparameters to test.
        """
        samples = l_kde.sample(n_samples)

        l_score = l_kde.score_samples(samples)
        g_score = g_kde.score_samples(samples)

        EI = g_score / l_score

        hps = samples[np.argmax(EI)]

        return hps, np.max(EI)

    def sample_priors(self, n_samples, episodes):
        """
        Consumes search space defined by priors and returns
        n_samples.
        """

        # Sample from each prior
        seed = [self.space[hp].sample(n_samples) for hp in self.space]

        # Convert the list of numpy arrays into a list of tuples
        seed = list(map(tuple, zip(*seed)))

        # Calculate objective for each pair in the seed
        seed_obj = [self.objective(*hp, episodes=episodes) for hp in seed]

        # Combine the seed and seed_obj into a list of tuples
        trials = [(seed[i] + (seed_obj[i],)) for i in range(len(seed))]

        return trials

    @staticmethod
    def segment_distributions(trials, gamma):
        """
        Splits samples into l(x) and g(x) distributions based on our
        quantile cutoff gamma (using rmse as criteria).

        Returns a kerned density estimator (KDE) for l(x) and g(x),
        respectively.
        """
        cut = np.quantile(trials[:, -1], gamma)

        l_x = trials[trials[:, -1] < cut][:, :-1]

        mask = np.isin(trials[:, :-1], l_x)
        g_x = trials[~np.any(mask, axis=1), :-1]

        l_kde = KernelDensity(kernel='gaussian', bandwidth=5.0)
        g_kde = KernelDensity(kernel='gaussian', bandwidth=5.0)

        l_kde.fit(l_x)
        g_kde.fit(g_x)

        return l_kde, g_kde

    def fmin(self, episodes, n_seed, n_total, gamma):
        """
        Consumes a hyperparameter search space, number of iterations for seeding
        and total number of iterations and performs Bayesian Optimization. TPE
        can be sensitive to choice of quantile cutoff, which we control with gamma.
        """

        print(f"Starting first {n_seed} trials...")

        # Seed priors
        trials = self.sample_priors(n_seed, episodes)
        EIs = []

        # Not really sure if that ever will help
        # cdf_function = self.__best_fit_distribution(y)
        # cdf = cdf_function(y_star)

        print(f"Starting next {n_total - n_seed} trials...")

        for i in range(n_seed, n_total):
            print(f"--- Trial #{i + 1} ---")
            # Segment trials into l and g distributions
            l_kde, g_kde = self.segment_distributions(trials, gamma)

            # Determine next pair of hyperparameters to test
            hps, EI = self.choose_next_hps(l_kde, g_kde, self.n_samples_dist)

            # Evaluate with fn and add to trials
            result = np.concatenate([hps, self.objective(*hps, episodes=episodes)])

            trials = np.append(trials, np.array([result]), axis=0)

            EIs.append(EI)

        return trials, EIs