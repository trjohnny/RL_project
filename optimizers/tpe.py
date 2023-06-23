import scipy.stats as st
from scipy.stats import kstest
import numpy as np
from sklearn.neighbors import KernelDensity


class TPE:
    def __init__(self, Agent, hyperparameters_grid):
        self.Agent = Agent
        self.space = hyperparameters_grid
        self.n_samples_dist = 100

    def objective(self, *hyperparameters):
        # training the Agent

        agent = self.Agent(*hyperparameters)
        rewards1 = agent.train(agent, episodes=400)

        agent = self.Agent(*hyperparameters)
        rewards2 = agent.train(agent, episodes=400)

        agent = self.Agent(*hyperparameters)
        rewards3 = agent.train(agent, episodes=400)

        # FIN = agent.finished
        TOT_REW = (sum(rewards1) + sum(rewards2) + sum(rewards3)) / (3 * len(rewards1))
        # DONE_RATIO = agent.dones / len(rewards1)

        return -TOT_REW

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

    def sample_priors(self, n_samples):
        """
        Consumes search space defined by priors and returns
        n_samples.
        """

        # Sample from each prior
        seed = [self.space[hp].sample(n_samples) for hp in self.space]

        # Convert the list of numpy arrays into a list of tuples
        seed = list(map(tuple, zip(*seed)))

        # Calculate objective for each pair in the seed
        seed_obj = [self.objective(*hp) for hp in seed]

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

    def fmin(self, n_seed=4, n_total=50, gamma=.2):
        return self.__fmin__(n_seed, n_total, gamma)

    def __fmin__(self, n_seed, n_total, gamma):
        """
        Consumes a hyperparameter search space, number of iterations for seeding
        and total number of iterations and performs Bayesian Optimization. TPE
        can be sensitive to choice of quantile cutoff, which we control with gamma.
        """

        print(f"Starting first {n_seed} trials...")

        # Seed priors
        trials = self.sample_priors(n_seed)
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
            result = np.concatenate([hps, self.objective(*hps)])

            trials = np.append(trials, np.array([result]), axis=0)

            EIs.append(EI)

        return trials, EIs


class Choice:
    """
    Class encapsulates behavior for a uniform distribution alongside a set of objects.
    """

    def __init__(self, objects):
        self.objects = objects

    def sample(self, n_samples):
        return np.random.choice(self.objects, n_samples)


class Uniform:
    """
    Class encapsulates behavior for a uniform distribution.
    """

    def __init__(self, min_, max_):
        self.min = min_
        self.max = max_

    def sample(self, n_samples):
        return np.random.uniform(self.min, self.max, n_samples)


class Normal:
    """
    Class encapsulates behavior for a normal distribution.
    """

    def __init__(self, mean_, std_):
        self.mean = mean_
        self.std = std_

    def sample(self, n_samples):
        return np.random.normal(self.mean, self.std, n_samples)


class LogNormal:
    """
    Class encapsulates behavior for a lognormal distribution.
    """

    def __init__(self, mean_, std_):
        self.mean = mean_
        self.std = std_

    def sample(self, n_samples):
        return np.random.lognormal(self.mean, self.std, n_samples)


class QUniform:
    """
    Class encapsulates behavior for a q_uniform distribution.
    """

    def __init__(self, min_, max_, step):
        self.min = min_
        self.max = max_
        self.step = step

    def sample(self, n_samples):
        return np.around(np.random.uniform(self.min, self.max, n_samples) / self.step) * self.step


class QNormal:
    """
    Class encapsulates behavior for a q_normal distribution.
    """

    def __init__(self, logmean, logstd, step):
        self.logmean = logmean
        self.logstd = logstd
        self.step = step

    def sample(self, n_samples):
        return np.around(np.random.normal(self.logmean, self.logstd, n_samples) / self.step) * self.step


class QLogNormal:
    """
    Class encapsulates behavior for a q_lognormal distribution.
    """

    def __init__(self, logmean, logstd, step):
        self.logmean = logmean
        self.logstd = logstd
        self.step = step

    def sample(self, n_samples):
        return np.around(np.random.lognormal(self.logmean, self.logstd, n_samples) / self.step) * self.step
