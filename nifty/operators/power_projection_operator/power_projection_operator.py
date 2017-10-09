from ... import Field,\
                FieldArray
from ..linear_operator import LinearOperator


class PowerProjection(LinearOperator):
    def __init__(self, domain, target, spaces=0, default_spaces=None):
        self._domain = self._parse_domain(domain)
        self._target = self._parse_domain(target)
        self.pindex = self.target[spaces].pindex
        super(PowerProjection, self).__init__(default_spaces)

    def _times(self,x,spaces):
        projected_x = self.pindex.bincount(weights=x.weight(1).val.real)
        y = Field(self.target, val=projected_x).weight(-1)
        return y

    def _adjoint_times(self,x,spaces):
        if spaces is None:
            spaces = 0
        y = Field(self.domain, val=1.)
        axes = x.domain_axes
        spec = x.val.get_full_data()

        spec = self._spec_to_rescaler(spec, spaces, axes)
        y.val.apply_scalar_function(lambda x: x * spec.real,
                                            inplace=True)

        return y

    def _spec_to_rescaler(self, spec, power_space_index,axes):

        # weight the random fields with the power spectrum
        # therefore get the pindex from the power space
        # take the local data from pindex. This data must be compatible to the
        # local data of the field given the slice of the PowerSpace
        # local_distribution_strategy = \
        #     result.val.get_axes_local_distribution_strategy(
        #         result.domain_axes[power_space_index])
        #
        # if self.pindex.distribution_strategy is not local_distribution_strategy:
        #     self.logger.warn(
        #         "The distribution_stragey of pindex does not fit the "
        #         "slice_local distribution strategy of the synthesized field.")

        # Now use numpy advanced indexing in order to put the entries of the
        # power spectrum into the appropriate places of the pindex array.
        # Do this for every 'pindex-slice' in parallel using the 'slice(None)'s
        local_pindex = self.pindex.get_local_data(copy=False)

        local_blow_up = [slice(None)]*len(spec.shape)
        local_blow_up[axes[power_space_index][0]] = local_pindex
        # here, the power_spectrum is distributed into the new shape
        local_rescaler = spec[local_blow_up]
        return local_rescaler
    @property
    def domain(self):
        return self._domain

    @property
    def target(self):
        return self._target

    @property
    def unitary(self):
        return False