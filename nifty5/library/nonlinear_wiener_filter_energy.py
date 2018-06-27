# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# Copyright(C) 2013-2018 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik
# and financially supported by the Studienstiftung des deutschen Volkes.

from ..models.constant import Constant
from .unit_log_gauss import UnitLogGauss
from .hamiltonian import Hamiltonian


def NonlinearWienerFilterEnergy(measured_data, data_model, sqrtN, iteration_controller):
        d = measured_data.lock()
        residual = Constant(data_model.position, d) - data_model
        lh = UnitLogGauss(sqrtN.inverse(residual))
        return Hamiltonian(lh, iteration_controller)

