from bilby.core.prior import Prior, ConditionalPowerLaw


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
