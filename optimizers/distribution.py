import numpy as np


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
