from bilby.core.prior import Prior, ConditionalPowerLaw, Uniform, conditional_prior_factory
import numpy as np

class OrderedAmplitude(ConditionalPowerLaw):
    def __init__(self, order, base_name, latex_label, minimum, maximum):
        super(OrderedAmplitude, self).__init__(
            minimum=minimum,
            maximum=maximum,
            alpha=-1 + order,
            name="_".join([base_name, str(order)]),
            latex_label=latex_label,
            condition_func=self._condition,
        )
        self.base_name = base_name
        self.order = order
        self._previous_name = "_".join([base_name, str(order - 1)])
        self._required_variables = [self._previous_name]
        self.__class__.__name__ = "OrderedAmplitude"
        self.__class__.__qualname__ = "OrderedAmplitude"

    def _condition(self, reference_params, **kwargs):
        return dict(
            minimum=reference_params["minimum"], maximum=kwargs[self._previous_name]
        )

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


class Discrete(Prior):
    """
    A unit spaced discrete prior class.

    Parameters
    ----------
    minimum: int
        The minimum allowed value (inclusive)
    maximum: int
        The maximum allowed value (inclusive)
    """

    def __init__(
        self, minimum, maximum, name=None, latex_label=None, boundary=None
    ):
        super(Discrete, self).__init__(
            name=name, latex_label=latex_label, boundary=boundary
        )
        if minimum >= maximum:
            raise ValueError(
                "Maximum must be greater than minimum for discrete prior"
            )
        self.minimum = minimum
        self.maximum = maximum

    @property
    def n_bins(self):
        return (self.maximum - self.minimum + 1)

    def prob(self, val):
        prob = 1 / self.n_bins
        return prob

    def rescale(self, val):
        val = np.atleast_1d(val)
        val *= self.n_bins
        val += self.minimum
        if isinstance(val, (float, int)) or len(val) == 1:
            val = int(val)
        else:
            val = val.astype(int)
        return val

class SpikeSlabPowerLawPrior(Prior):
    def __init__(self, alpha, minimum, maximum, peak, slab_probability, name=None, latex_label=None,
                 unit=None, boundary=None):
        """Slab-spike powerlaw prior.

        Parameters
        ----------
        alpha: float
            Power law exponent parameter
        minimum: float
            See superclass
        maximum: float
            See superclass
        peak: float
            Sets the peak of the spike
        slab_probability: float
            Sets the probability of the sample being drawn from the slab.
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass
        boundary: str
            See superclass
        """
        super(SpikeSlabPowerLawPrior, self).__init__(name=name, latex_label=latex_label,
                                       unit=unit,
                                       minimum=minimum, maximum=maximum,
                                       boundary=boundary)
        self.peak = peak
        self.slab_probability = slab_probability
        self.alpha = alpha

    def powerlaw_rescale(self, val):
        if self.alpha == -1:
            return self.minimum * np.exp(val * np.log(self.maximum / self.minimum))
        else:
            return (self.minimum ** (1 + self.alpha) + val *
                    (self.maximum ** (1 + self.alpha) - self.minimum ** (1 + self.alpha))) ** (1. / (1 + self.alpha))

    def powerlaw_prob(self, val):
        if self.alpha == -1:
            return np.nan_to_num(1 / val / np.log(self.maximum / self.minimum)) * self.is_in_prior_range(val)
        else:
            return np.nan_to_num(val ** self.alpha * (1 + self.alpha) /
                                 (self.maximum ** (1 + self.alpha) -
                                  self.minimum ** (1 + self.alpha))) * self.is_in_prior_range(val)

    def powerlaw_cdf(self, val):
        if self.alpha == -1:
            _cdf = (np.log(val / self.minimum) /
                    np.log(self.maximum / self.minimum))
        else:
            _cdf = (val ** (self.alpha + 1) - self.minimum ** (self.alpha + 1)) / \
                (self.maximum ** (self.alpha + 1) - self.minimum ** (self.alpha + 1))
        _cdf = np.minimum(_cdf, 1)
        _cdf = np.maximum(_cdf, 0)
        return _cdf

    def rescale(self, val):
        L_min = self.slab_probability * self.powerlaw_cdf(self.peak)
        L_max = self.slab_probability * (1 - self.powerlaw_cdf(self.peak))

        if not isinstance(val, float):
            arr = np.zeros(len(val))

            before_peak = (val <= L_min)
            after_peak = (val >= 1-L_max)
            at_peak = ~(before_peak | after_peak)

            arr[before_peak] += self.powerlaw_rescale(val[before_peak]/self.slab_probability)
            arr[after_peak] += self.powerlaw_rescale((val[after_peak] - (1-self.slab_probability))/self.slab_probability)
            arr[at_peak] += self.peak * val[at_peak] ** 0
            return arr

        else:
            if val < L_min:
                return self.powerlaw_rescale(val/self.slab_probability)
            elif val > 1-L_max:
                return self.powerlaw_rescale((val - (1-self.slab_probability))/self.slab_probability)
            else:
                return self.peak * val ** 0


    def prob(self, val):
        if not isinstance(val, float):
            arr = self.powerlaw_prob(val) * self.slab_probability
            at_peak = (val==self.peak)
            arr[at_peak] = np.inf * (1-self.slab_probability)

            return arr

        else:
            if val == self.peak:
                return np.inf * (1-self.slab_probability)
            else:
                return self.powerlaw_prob(val) * self.slab_probability


    def cdf(self, val):
        if not isinstance(val, float):
            arr = self.powerlaw_cdf(val) * self.slab_probability
            after_peak = (val>self.peak)
            arr[after_peak] = self.powerlaw_cdf(val[after_peak]) * self.slab_probability + (1-self.slab_probability)

            return arr

        else:
            if val <= self.peak:
                return self.powerlaw_cdf(val) * self.slab_probability
            else:
                return self.powerlaw_cdf(val) * self.slab_probability + (1-self.slab_probability)


ConditionalSpikeSlabPowerLawPrior = conditional_prior_factory(SpikeSlabPowerLawPrior)

class OrderedSpikeSlabAmplitude(ConditionalSpikeSlabPowerLawPrior):
    def __init__(self, order, base_name, latex_label, minimum, maximum, peak, slab_probability):
        super(OrderedSpikeSlabAmplitude, self).__init__(
            minimum=minimum,
            maximum=maximum,
            peak=peak,
            slab_probability=slab_probability,
            alpha=-1 + order,
            name="_".join([base_name, str(order)]),
            latex_label=latex_label,
            condition_func=self._condition,
        )
        self.base_name = base_name
        self.order = order
        self._previous_name = "_".join([base_name, str(order - 1)])
        self._required_variables = [self._previous_name]
        self.__class__.__name__ = "OrderedSpikeSlabAmplitude"
        self.__class__.__qualname__ = "OrderedSpikeSlabAmplitude"

    def _condition(self, reference_params, **kwargs):
        return dict(
            minimum=reference_params["minimum"], maximum=kwargs[self._previous_name]
        )

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


class Discrete(Prior):
    """
    A unit spaced discrete prior class.
    
    Parameters
    ----------
    minimum: int
        The minimum allowed value (inclusive)
    maximum: int
        The maximum allowed value (inclusive)
    """

    def __init__(
        self, minimum, maximum, name=None, latex_label=None, boundary=None
    ):
        super(Discrete, self).__init__(
            name=name, latex_label=latex_label, boundary=boundary
        )
        if minimum >= maximum:
            raise ValueError(
                "Maximum must be greater than minimum for discrete prior"
            )
        self.minimum = minimum
        self.maximum = maximum

    @property
    def n_bins(self):
        return (self.maximum - self.minimum + 1)

    def prob(self, val):
        prob = 1 / self.n_bins
        return prob

    def rescale(self, val):
        val = np.atleast_1d(val)
        val *= self.n_bins
        val += self.minimum
        if isinstance(val, (float, int)) or len(val) == 1:
            val = int(val)
        else:
            val = val.astype(int)
        return val
