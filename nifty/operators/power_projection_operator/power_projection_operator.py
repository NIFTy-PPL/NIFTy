from ... import Field,\
                FieldArray
from ..linear_operator import LinearOperator
from ...spaces.power_space import PowerSpace

class PowerProjection(LinearOperator):
    def __init__(self, domain, target, default_spaces=None):
        self._domain = self._parse_domain(domain)
        self._target = self._parse_domain(target)
        if len(self._domain)!=1 or len(self._target)!=1:
            raise ValueError("Operator only works over one space")
        if not self._domain[0].harmonic:
            raise ValueError("domain must be a harmonic space")
        if not isinstance(self._target[0], PowerSpace):
            raise ValueError("target must be a PowerSpace")
        self.pindex = self.target[0].pindex
        super(PowerProjection, self).__init__(default_spaces)

    def _times(self,x,spaces):
        if spaces is None:
            spaces = 0
        projected_x = self.pindex.bincount(
            weights=x.weight(1,spaces=spaces).val.real,
            axis=x.domain_axes[spaces])
        tgt_domain = list(x.domain)
        tgt_domain[spaces] = self._target[0]
        y = Field(tgt_domain, val=projected_x).weight(-1, spaces=spaces)
        return y

    def _adjoint_times(self,x,spaces):
        if spaces is None:
            spaces = 0
        tgt_domain = list(x.domain)
        tgt_domain[spaces] = self._domain[0]
        y = Field(tgt_domain, val=1.)
        axes = x.domain_axes[spaces]
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
        local_blow_up[axes[power_space_index]] = local_pindex
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
