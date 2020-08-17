from bilby.core.prior import Prior, ConditionalPowerLaw, Uniform
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

    def _condition(self, reference_parms, **kwargs):
        return dict(
            minimum=reference_parms["minimum"], maximum=kwargs[self._previous_name]
        )

    def __repr__(self):
        return Prior.__repr__(self)

    def get_instantiation_dict(self):
        instantiation_dict = Prior.get_instantiation_dict(self)
        for key, value in self.reference_params.items():
            if key in instantiation_dict:
                instantiation_dict[key] = value
        return instantiation_dict


class SpikeSlabPrior(Prior):
    def __init__(self, peak, slab_probability, name=None, latex_label=None,
                 unit=None, boundary=None, slab_prior=Uniform, **slab_prior_kwargs):
        """Slab-spike prior. Requires location of the peak and the prior probability of the slab.

        Parameters
        ----------
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
        slab_prior_kwargs:
            Remaining kwargs are the parameters for the slab prior
        """
        super(SpikeSlabPrior, self).__init__(name=name, latex_label=latex_label,
                                       unit=unit,
                                       boundary=boundary)
        self.peak = peak
        self.slab_probability = slab_probability
        self.slab_prior = slab_prior(**slab_prior_kwargs)

    def rescale(self, val):
        L_min = self.slab_probability * self.slab_prior.cdf(self.peak)
        L_max = self.slab_probability * (1 - self.slab_prior.cdf(self.peak))

        if not isinstance(val, float):
            arr = np.zeros(len(val))

            before_peak = (val < L_min)
            after_peak = (val > 1-L_max)
            at_peak = (val > L_min)*(val < 1-L_max)

            arr[before_peak] += self.slab_prior.rescale(val[before_peak]/self.slab_probability)
            arr[after_peak] += self.slab_prior.rescale((val[after_peak] - (1-self.slab_probability))/self.slab_probability)
            arr[at_peak] += self.peak * val[at_peak] ** 0
            return arr

        else:
            if val < L_min:
                return self.slab_prior.rescale(val/self.slab_probability)
            elif val > 1-L_max:
                return self.slab_prior.rescale((val - (1-self.slab_probability))/self.slab_probability)
            else:
                return self.peak * val ** 0


    def prob(self, val):
        if not isinstance(val, float):
            arr = self.slab_prior.prob(val) * self.slab_probability
            at_peak = (val==self.peak)
            arr[at_peak] = np.inf * (1-self.slab_probability)

            return arr

        else:
            if val == self.peak:
                return np.inf * (1-self.slab_probability)
            else:
                return self.slab_prior.prob(val) * self.slab_probability


    def cdf(self, val):
        if not isinstance(val, float):
            arr = self.slab_prior.cdf(val) * self.slab_probability
            after_peak = (val>self.peak)
            arr[after_peak] = self.slab_prior.cdf(val[after_peak]) * self.slab_probability + (1-self.slab_probability)

            return arr

        else:
            if val <= self.peak:
                return self.slab_prior.cdf(val) * self.slab_probability
            else:
                return self.slab_prior.cdf(val) * self.slab_probability + (1-self.slab_probability)
