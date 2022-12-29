import numpy as np
from scipy.special import erfinv
import scipy.stats


class distribution:
    def __init__(self, type):
        self.type = type


class univariatDistribution(distribution):
    def __init__(self, type):
        self.n_dimensions = 1
        super().__init__(type)


class normDistribution(univariatDistribution):
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = scale
        super().__init__("norm")

    def rvs(self, n):
        return scipy.stats.norm.rvs(self.loc, self.scale, n)

    def pdf(self, x):
        return scipy.stats.norm.pdf(x, self.loc, self.scale)

    def cdf(self, x):
        return scipy.stats.norm.cdf(x, self.loc, self.scale)

    def ppf(self, x):
        return scipy.stats.norm.ppf(x, self.loc, self.scale)

    def cdf_dx(self, x):
        sigma = self.scale
        mu = self.loc
        return np.exp(-(x-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma)

    def ppf_dx(self, x):
        sigma = self.scale
        mu = self.loc
        return np.sqrt(2*np.pi) * sigma * np.exp(erfinv((-1+2*x))**2)

    def cdf_dx_dx(self, x):
        sigma = self.scale
        mu = self.loc
        return -np.exp(-(x-mu)**2/(2*sigma**2))*(x-mu)/(np.sqrt(2*np.pi)*sigma**3)

    def ppf_dx_dx(self, x):
        sigma = self.scale
        mu = self.loc
        return 2*np.sqrt(2)*np.pi * sigma * np.exp(2*erfinv((-1+2*x))**2)*erfinv(2*x-1)


class lognormDistribution(univariatDistribution):
    def __init__(self, s=1, loc=0., scale=1.):
        self.s = s
        self.loc = loc
        self.scale = scale
        super().__init__("lognorm")

    def rvs(self, n):
        return scipy.stats.lognorm.rvs(self.s, self.loc, self.scale, n)

    def pdf(self, x):
        return scipy.stats.lognorm.pdf(x, self.s, self.loc, self.scale)

    def cdf(self, x):
        return scipy.stats.lognorm.cdf(x, self.s, self.loc, self.scale)

    def ppf(self, x):
        return scipy.stats.lognorm.ppf(x, self.s, self.loc, self.scale)

    def cdf_dx(self, x):
        sigma = self.s
        mu = self.loc
        return np.exp(-(np.log(x)-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma*x)

    def ppf_dx(self, x):
        sigma = self.s
        mu = self.loc
        return np.sqrt(2*np.pi)*sigma*np.exp(np.sqrt(2)*sigma*erfinv(2*x-1)+(erfinv(2*x-1))**2+mu)

    def cdf_dx_dx(self, x):
        sigma = self.s
        mu = self.loc
        return -np.exp(-(np.log(x)-mu)**2/(2*sigma**2))/(np.sqrt(2*np.pi)*sigma*x**2) - 2 * ((np.log(x)-mu)*np.exp(-(np.log(x)-mu)**2/(2*sigma**2)))/(2*np.sqrt(2*np.pi)*sigma**3*x**2)

    def ppf_dx_dx(self, x):
        sigma = self.s
        mu = self.loc
        return np.sqrt(2)*np.pi*sigma*(2*erfinv(2*x-1)+np.sqrt(2)*sigma) * np.exp(np.sqrt(2)*sigma*erfinv(2*x-1)+2*(erfinv(2*x-1))**2+mu)


class uniformDistribution(univariatDistribution):
    def __init__(self, loc=0., scale=1.):
        self.loc = loc
        self.scale = scale
        super().__init__("uniform")

    def rvs(self, n):
        return scipy.stats.uniform.rvs(self.loc, self.scale, n)

    def pdf(self, x):
        return scipy.stats.uniform.pdf(x, self.loc, self.scale)

    def cdf(self, x):
        return scipy.stats.uniform.cdf(x, self.loc, self.scale)

    def ppf(self, x):
        return scipy.stats.uniform.ppf(x, self.loc, self.scale)

    def cdf_dx(self, x):
        return 1./self.scale*np.ones(x.shape)

    def ppf_dx(self, x):
        return self.scale*np.ones(x.shape)

    def cdf_dx_dx(self, x):
        return np.zeros(x.shape)

    def ppf_dx_dx(self, x):
        return np.zeros(x.shape)

class multivariatDistribution(distribution):
    def __init__(self, n_dimensions):
        self.n_dimensions = n_dimensions
        super().__init__("multivariatDistribution")

    def setUnivariatDistributions(self, distributions):
        self.distributions = []
        for i in range(0, self.n_dimensions):
            self.distributions.append(distributions[i])

    def rvs(self, n):
        s = np.zeros((n, self.n_dimensions))
        for i in range(0, self.n_dimensions):
            s[:, i] = self.distributions[i].rvs(n)
        return s

    def pdf(self, x):
        pdf = np.ones((len(x[:, 0]), ))
        for i in range(0, self.n_dimensions):
            pdf *= self.distributions[i].pdf(x[:, i])
        return pdf

    def cdf(self, x):
        q = np.zeros(x.shape)
        for i in range(0, self.n_dimensions):
            q[:, i] = self.distributions[i].cdf(x[:, i])
        return q

    def ppf(self, x):
        s = np.zeros(x.shape)
        for i in range(0, self.n_dimensions):
            s[:, i] = self.distributions[i].ppf(x[:, i])
        return s


def mapping_v(x, distribution1, distribution2):
    if distribution1.n_dimensions != distribution2.n_dimensions:
        raise NameError("mapping: The two distributions are not compatible")
    return distribution2.ppf(distribution1.cdf(x))


def mapping_dx(x, distribution1, distribution2):
    if distribution1.n_dimensions != distribution2.n_dimensions:
        raise NameError("mapping_dx: The two distributions are not compatible")
    return distribution2.ppf_dx(distribution1.cdf(x)) * distribution1.cdf_dx(x)


def mapping_dx_dx(x, distribution1, distribution2):
    if distribution1.n_dimensions != distribution2.n_dimensions:
        raise NameError(
            "mapping_dx_dx: The two distributions are not compatible")
    return distribution2.ppf_dx_dx(distribution1.cdf(x)) * distribution1.cdf_dx(x)**2 + distribution2.ppf_dx(distribution1.cdf(x)) * distribution1.cdf_dx_dx(x)


class mapping:
    def __init__(self, standard, other):
        self.standard = standard
        self.other = other

    def toNormal(self, x):
        return mapping_v(x, self.other, self.standard)

    def fromNormal(self, x):
        return mapping_v(x, self.standard, self.other)
