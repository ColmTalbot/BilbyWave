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
