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
# Copyright(C) 2013-2019 Max-Planck-Society
#
# NIFTy is being developed at the Max-Planck-Institut fuer Astrophysik.

import os

import numpy as np

from . import dobj
from .domains.gl_space import GLSpace
from .domains.hp_space import HPSpace
from .domains.power_space import PowerSpace
from .domains.rg_space import RGSpace
from .domains.log_rg_space import LogRGSpace
from .domain_tuple import DomainTuple
from .field import Field

# relevant properties:
# - x/y size
# - x/y/z log
# - x/y/z min/max
# - colorbar/colormap
# - axis on/off
# - title
# - axis labels
# - labels


def _mollweide_helper(xsize):
    xsize = int(xsize)
    ysize = xsize//2
    res = np.full(shape=(ysize, xsize), fill_value=np.nan, dtype=np.float64)
    xc, yc = (xsize-1)*0.5, (ysize-1)*0.5
    u, v = np.meshgrid(np.arange(xsize), np.arange(ysize))
    u, v = 2*(u-xc)/(xc/1.02), (v-yc)/(yc/1.02)

    mask = np.where((u*u*0.25 + v*v) <= 1.)
    t1 = v[mask]
    theta = 0.5*np.pi-(
        np.arcsin(2/np.pi*(np.arcsin(t1) + t1*np.sqrt((1.-t1)*(1+t1)))))
    phi = -0.5*np.pi*u[mask]/np.maximum(np.sqrt((1-t1)*(1+t1)), 1e-6)
    phi = np.where(phi < 0, phi+2*np.pi, phi)
    return res, mask, theta, phi



def _rgb_data(spectral_cube):
    def _eye_sensitivity(energy_bins, spacing=None):
        from scipy.ndimage import zoom
        rgb_high=[[9.85234460e-03, 6.13406271e-04, 0.00000000e+00],
                [1.04118690e-02, 6.49048870e-04, 0.00000000e+00],
               [1.09713934e-02, 6.84691469e-04, 0.00000000e+00],
               [1.15309178e-02, 7.20334068e-04, 0.00000000e+00],
               [1.20904422e-02, 7.55976668e-04, 0.00000000e+00],
               [1.29039276e-02, 8.08680908e-04, 0.00000000e+00],
               [1.37595706e-02, 8.64217381e-04, 0.00000000e+00],
               [1.46152136e-02, 9.19753853e-04, 0.00000000e+00],
               [1.54708566e-02, 9.75290326e-04, 0.00000000e+00],
               [1.63264996e-02, 1.03082680e-03, 0.00000000e+00],
               [1.71821426e-02, 1.08636327e-03, 0.00000000e+00],
               [1.81667722e-02, 1.15137696e-03, 0.00000000e+00],
               [1.93698228e-02, 1.23243904e-03, 0.00000000e+00],
               [2.05728734e-02, 1.31350111e-03, 0.00000000e+00],
               [2.17759241e-02, 1.39456318e-03, 0.00000000e+00],
               [2.29789747e-02, 1.47562526e-03, 0.00000000e+00],
               [2.41820254e-02, 1.55668733e-03, 0.00000000e+00],
               [2.53874406e-02, 1.63793709e-03, 0.00000000e+00],
               [2.70586835e-02, 1.75616085e-03, 0.00000000e+00],
               [2.87299264e-02, 1.87438460e-03, 0.00000000e+00],
               [3.04011692e-02, 1.99260836e-03, 0.00000000e+00],
               [3.20724121e-02, 2.11083211e-03, 0.00000000e+00],
               [3.37436550e-02, 2.22905587e-03, 0.00000000e+00],
               [3.54148979e-02, 2.34727962e-03, 0.00000000e+00],
               [3.75526858e-02, 2.50677171e-03, 0.00000000e+00],
               [3.98413542e-02, 2.67960994e-03, 0.00000000e+00],
               [4.21300226e-02, 2.85244818e-03, 0.00000000e+00],
               [4.44186909e-02, 3.02528642e-03, 0.00000000e+00],
               [4.67073593e-02, 3.19812466e-03, 0.00000000e+00],
               [4.89960277e-02, 3.37096290e-03, 0.00000000e+00],
               [5.17840545e-02, 3.59459245e-03, 0.00000000e+00],
               [5.48783544e-02, 3.84937399e-03, 0.00000000e+00],
               [5.79726543e-02, 4.10415554e-03, 0.00000000e+00],
               [6.10669543e-02, 4.35893709e-03, 0.00000000e+00],
               [6.41612542e-02, 4.61371864e-03, 0.00000000e+00],
               [6.72555541e-02, 4.86850018e-03, 0.00000000e+00],
               [7.09715657e-02, 5.20598062e-03, 0.00000000e+00],
               [7.51114716e-02, 5.59984666e-03, 0.00000000e+00],
               [7.92513775e-02, 5.99371269e-03, 0.00000000e+00],
               [8.33912835e-02, 6.38757873e-03, 0.00000000e+00],
               [8.75311894e-02, 6.78144477e-03, 0.00000000e+00],
               [9.16710953e-02, 7.17531081e-03, 0.00000000e+00],
               [9.67048710e-02, 7.69960083e-03, 0.00000000e+00],
               [1.02165218e-01, 8.28613156e-03, 0.00000000e+00],
               [1.07625564e-01, 8.87266229e-03, 0.00000000e+00],
               [1.13085911e-01, 9.45919302e-03, 0.00000000e+00],
               [1.18546257e-01, 1.00457237e-02, 0.00000000e+00],
               [1.24006604e-01, 1.06322545e-02, 0.00000000e+00],
               [1.30860134e-01, 1.14285362e-02, 0.00000000e+00],
               [1.37932889e-01, 1.22578234e-02, 0.00000000e+00],
               [1.45005644e-01, 1.30871106e-02, 0.00000000e+00],
               [1.52078399e-01, 1.39163978e-02, 0.00000000e+00],
               [1.59151155e-01, 1.47456850e-02, 0.00000000e+00],
               [1.66454559e-01, 1.56214122e-02, 0.00000000e+00],
               [1.75033740e-01, 1.67540109e-02, 0.00000000e+00],
               [1.83612920e-01, 1.78866096e-02, 0.00000000e+00],
               [1.92192101e-01, 1.90192083e-02, 0.00000000e+00],
               [2.00771282e-01, 2.01518070e-02, 0.00000000e+00],
               [2.09350463e-01, 2.12844057e-02, 0.00000000e+00],
               [2.18438958e-01, 2.27487266e-02, 0.00000000e+00],
               [2.27958699e-01, 2.44939217e-02, 0.00000000e+00],
               [2.37478439e-01, 2.62391168e-02, 0.00000000e+00],
               [2.46998180e-01, 2.79843119e-02, 0.00000000e+00],
               [2.56517920e-01, 2.97295070e-02, 0.00000000e+00],
               [2.66089164e-01, 3.14923971e-02, 0.00000000e+00],
               [2.77541802e-01, 3.39016757e-02, 0.00000000e+00],
               [2.88994440e-01, 3.63109543e-02, 0.00000000e+00],
               [3.00447079e-01, 3.87202330e-02, 0.00000000e+00],
               [3.11899717e-01, 4.11295116e-02, 0.00000000e+00],
               [3.23352355e-01, 4.35387902e-02, 0.00000000e+00],
               [3.36149249e-01, 4.64923760e-02, 0.00000000e+00],
               [3.49819910e-01, 4.97997613e-02, 0.00000000e+00],
               [3.63490571e-01, 5.31071466e-02, 0.00000000e+00],
               [3.77161232e-01, 5.64145319e-02, 0.00000000e+00],
               [3.90831894e-01, 5.97219172e-02, 0.00000000e+00],
               [4.04905270e-01, 6.33547279e-02, 0.00000000e+00],
               [4.20027945e-01, 6.78354525e-02, 0.00000000e+00],
               [4.35150619e-01, 7.23161770e-02, 0.00000000e+00],
               [4.50273293e-01, 7.67969016e-02, 0.00000000e+00],
               [4.65395968e-01, 8.12776261e-02, 0.00000000e+00],
               [4.80491754e-01, 8.58174413e-02, 0.00000000e+00],
               [4.94910505e-01, 9.18451595e-02, 0.00000000e+00],
               [5.09329255e-01, 9.78728776e-02, 0.00000000e+00],
               [5.23748005e-01, 1.03900596e-01, 0.00000000e+00],
               [5.38166755e-01, 1.09928314e-01, 0.00000000e+00],
               [5.52585506e-01, 1.15956032e-01, 0.00000000e+00],
               [5.67615464e-01, 1.23568964e-01, 9.47898839e-07],
               [5.82723782e-01, 1.31385129e-01, 2.01732317e-06],
               [5.97832099e-01, 1.39201294e-01, 3.08674750e-06],
               [6.12940417e-01, 1.47017459e-01, 4.15617183e-06],
               [6.28048735e-01, 1.54833624e-01, 5.22559616e-06],
               [6.43098813e-01, 1.64307246e-01, 5.84877257e-06],
               [6.58136078e-01, 1.74145478e-01, 6.37378263e-06],
               [6.73173344e-01, 1.83983711e-01, 6.89879268e-06],
               [6.88210609e-01, 1.93821943e-01, 7.42380273e-06],
               [7.03247875e-01, 2.03660175e-01, 7.94881278e-06],
               [7.17546471e-01, 2.15436756e-01, 8.72244917e-06],
               [7.31700230e-01, 2.27593405e-01, 9.54483582e-06],
               [7.45853990e-01, 2.39750055e-01, 1.03672225e-05],
               [7.60007749e-01, 2.51906704e-01, 1.11896091e-05],
               [7.74161508e-01, 2.64063354e-01, 1.20119958e-05],
               [7.86497696e-01, 2.78333181e-01, 1.32818969e-05],
               [7.98704366e-01, 2.92753591e-01, 1.45836873e-05],
               [8.10911036e-01, 3.07174002e-01, 1.58854778e-05],
               [8.23117706e-01, 3.21594412e-01, 1.71872683e-05],
               [8.35186518e-01, 3.36208819e-01, 1.85747438e-05],
               [8.46139602e-01, 3.52393309e-01, 2.06556971e-05],
               [8.57092687e-01, 3.68577799e-01, 2.27366504e-05],
               [8.68045771e-01, 3.84762289e-01, 2.48176037e-05],
               [8.78998855e-01, 4.00946779e-01, 2.68985570e-05],
               [8.89215651e-01, 4.17579320e-01, 2.94434547e-05],
               [8.98143940e-01, 4.34995949e-01, 3.28002551e-05],
               [9.07072229e-01, 4.52412578e-01, 3.61570554e-05],
               [9.16000518e-01, 4.69829208e-01, 3.95138558e-05],
               [9.24928807e-01, 4.87245837e-01, 4.28706562e-05],
               [9.31880728e-01, 5.04651340e-01, 4.76834190e-05],
               [9.37955635e-01, 5.22051906e-01, 5.31422653e-05],
               [9.44030542e-01, 5.39452472e-01, 5.86011115e-05],
               [9.50105449e-01, 5.56853038e-01, 6.40599578e-05],
               [9.55892875e-01, 5.74294183e-01, 6.98501401e-05],
               [9.58948160e-01, 5.92120984e-01, 7.87892608e-05],
               [9.62003444e-01, 6.09947784e-01, 8.77283815e-05],
               [9.65058728e-01, 6.27774585e-01, 9.66675021e-05],
               [9.68114013e-01, 6.45601386e-01, 1.05606623e-04],
               [9.72354205e-01, 6.64398627e-01, 1.17838668e-04],
               [9.77489728e-01, 6.83929146e-01, 1.32558884e-04],
               [9.82625251e-01, 7.03459664e-01, 1.47279100e-04],
               [9.87760774e-01, 7.22990182e-01, 1.61999316e-04],
               [9.92509943e-01, 7.42196432e-01, 1.77817359e-04],
               [9.94261297e-01, 7.58886608e-01, 2.02153721e-04],
               [9.96012651e-01, 7.75576783e-01, 2.26490084e-04],
               [9.97764004e-01, 7.92266959e-01, 2.50826447e-04],
               [9.99515358e-01, 8.08957134e-01, 2.75162810e-04],
               [9.99065878e-01, 8.23559025e-01, 3.11123629e-04],
               [9.97791085e-01, 8.37377810e-01, 3.51443619e-04],
               [9.96516292e-01, 8.51196595e-01, 3.91763610e-04],
               [9.95241498e-01, 8.65015380e-01, 4.32083600e-04],
               [9.93237351e-01, 8.77470091e-01, 4.83190079e-04],
               [9.90172100e-01, 8.87940271e-01, 5.49989336e-04],
               [9.87106849e-01, 8.98410452e-01, 6.16788594e-04],
               [9.84041597e-01, 9.08880632e-01, 6.83587852e-04],
               [9.80871738e-01, 9.19166847e-01, 7.57052109e-04],
               [9.77122292e-01, 9.28433790e-01, 8.67444063e-04],
               [9.73372847e-01, 9.37700733e-01, 9.77836018e-04],
               [9.69623402e-01, 9.46967676e-01, 1.08822797e-03],
               [9.65873957e-01, 9.56234619e-01, 1.19861993e-03],
               [9.59779661e-01, 9.61388103e-01, 1.37907137e-03],
               [9.53593770e-01, 9.66380906e-01, 1.56225950e-03],
               [9.47407879e-01, 9.71373709e-01, 1.74544764e-03],
               [9.41221987e-01, 9.76366512e-01, 1.92863578e-03],
               [9.35862130e-01, 9.81296800e-01, 2.20723373e-03],
               [9.30666137e-01, 9.86214687e-01, 2.50475852e-03],
               [9.25470143e-01, 9.91132574e-01, 2.80228330e-03],
               [9.20274150e-01, 9.96050461e-01, 3.09980808e-03],
               [9.11776049e-01, 9.96814428e-01, 3.53678623e-03],
               [9.02278626e-01, 9.96321289e-01, 4.01596737e-03],
               [8.92781202e-01, 9.95828150e-01, 4.49514851e-03],
               [8.83283779e-01, 9.95335011e-01, 4.97432965e-03],
               [8.70357183e-01, 9.90138809e-01, 5.66803016e-03],
               [8.56351813e-01, 9.83463085e-01, 6.42921569e-03],
               [8.42346443e-01, 9.76787360e-01, 7.19040121e-03],
               [8.28341073e-01, 9.70111636e-01, 7.95158673e-03],
               [8.14214130e-01, 9.61934330e-01, 9.03521277e-03],
               [8.00059079e-01, 9.53409844e-01, 1.01933900e-02],
               [7.85904028e-01, 9.44885358e-01, 1.13515672e-02],
               [7.71748976e-01, 9.36360872e-01, 1.25097445e-02],
               [7.54803293e-01, 9.23647801e-01, 1.42317318e-02],
               [7.37618069e-01, 9.10575194e-01, 1.60021149e-02],
               [7.20432846e-01, 8.97502586e-01, 1.77724979e-02],
               [7.02994012e-01, 8.83987616e-01, 1.96135712e-02],
               [6.82848697e-01, 8.65751846e-01, 2.22090338e-02],
               [6.62703383e-01, 8.47516077e-01, 2.48044965e-02],
               [6.42558069e-01, 8.29280307e-01, 2.73999591e-02],
               [6.21021574e-01, 8.08629266e-01, 3.03177497e-02],
               [5.96317925e-01, 7.82479626e-01, 3.39693509e-02],
               [5.71614275e-01, 7.56329987e-01, 3.76209521e-02],
               [5.46910626e-01, 7.30180347e-01, 4.12725533e-02],
               [5.21885772e-01, 7.02842258e-01, 4.56468260e-02],
               [4.96626730e-01, 6.74637674e-01, 5.05479963e-02],
               [4.71367687e-01, 6.46433091e-01, 5.54491667e-02],
               [4.46108644e-01, 6.18228507e-01, 6.03503370e-02],
               [4.22732390e-01, 5.90585713e-01, 6.78261191e-02],
               [3.99556150e-01, 5.63002600e-01, 7.55754081e-02],
               [3.76379909e-01, 5.35419486e-01, 8.33246972e-02],
               [3.54150919e-01, 5.08679668e-01, 9.15660981e-02],
               [3.34345779e-01, 4.84097693e-01, 1.01066727e-01],
               [3.14540639e-01, 4.59515718e-01, 1.10567355e-01],
               [2.94735499e-01, 4.34933743e-01, 1.20067983e-01],
               [2.77694238e-01, 4.13550010e-01, 1.30415519e-01],
               [2.61791044e-01, 3.93483202e-01, 1.41111781e-01],
               [2.45887851e-01, 3.73416393e-01, 1.51808043e-01],
               [2.30724201e-01, 3.54217289e-01, 1.63258477e-01],
               [2.18819558e-01, 3.38841970e-01, 1.78032377e-01],
               [2.06914916e-01, 3.23466651e-01, 1.92806277e-01],
               [1.95010274e-01, 3.08091332e-01, 2.07580177e-01],
               [1.85839746e-01, 2.96264763e-01, 2.28113511e-01],
               [1.77786849e-01, 2.85888824e-01, 2.51001139e-01],
               [1.69733951e-01, 2.75512886e-01, 2.73888768e-01],
               [1.61966598e-01, 2.65368567e-01, 2.98662491e-01],
               [1.54926257e-01, 2.55813966e-01, 3.28238311e-01],
               [1.47885916e-01, 2.46259366e-01, 3.57814131e-01],
               [1.40845574e-01, 2.36704765e-01, 3.87389952e-01],
               [1.34384065e-01, 2.27560262e-01, 4.24364167e-01],
               [1.27986519e-01, 2.18461077e-01, 4.62155940e-01],
               [1.21588973e-01, 2.09361892e-01, 4.99947714e-01],
               [1.15488374e-01, 2.00339219e-01, 5.38691241e-01],
               [1.09616987e-01, 1.91375605e-01, 5.78169419e-01],
               [1.03745600e-01, 1.82411992e-01, 6.17647598e-01],
               [9.78980944e-02, 1.73260478e-01, 6.54076290e-01],
               [9.21142738e-02, 1.63607897e-01, 6.82373018e-01],
               [8.63304532e-02, 1.53955317e-01, 7.10669746e-01],
               [8.05663926e-02, 1.44323302e-01, 7.38641004e-01],
               [7.55830929e-02, 1.35503863e-01, 7.53752198e-01],
               [7.05997933e-02, 1.26684424e-01, 7.68863392e-01],
               [6.56164936e-02, 1.17864985e-01, 7.83974586e-01],
               [6.22882135e-02, 1.11589324e-01, 8.05751119e-01],
               [5.93247626e-02, 1.05874409e-01, 8.28996946e-01],
               [5.63613116e-02, 1.00159494e-01, 8.52242773e-01],
               [5.41909435e-02, 9.58527781e-02, 8.80230772e-01],
               [5.24365127e-02, 9.22846018e-02, 9.10705835e-01],
               [5.06820820e-02, 8.87164255e-02, 9.41180899e-01],
               [4.90109904e-02, 8.51181293e-02, 9.61647451e-01],
               [4.74127283e-02, 8.14935115e-02, 9.73367629e-01],
               [4.58144663e-02, 7.78688937e-02, 9.85087806e-01],
               [4.42430778e-02, 7.42331402e-02, 9.91441525e-01],
               [4.27043214e-02, 7.05838651e-02, 9.91278831e-01],
               [4.11655651e-02, 6.69345899e-02, 9.91116137e-01],
               [3.94185919e-02, 6.30133732e-02, 9.79110351e-01],
               [3.73708609e-02, 5.86993522e-02, 9.49997875e-01],
               [3.53231300e-02, 5.43853312e-02, 9.20885399e-01],
               [3.32966294e-02, 5.01302108e-02, 8.89766895e-01],
               [3.13012767e-02, 4.59615062e-02, 8.55705260e-01],
               [2.93059241e-02, 4.17928016e-02, 8.21643626e-01],
               [2.73987252e-02, 3.79727389e-02, 7.83328843e-01],
               [2.56039108e-02, 3.45971493e-02, 7.39591852e-01],
               [2.38090964e-02, 3.12215596e-02, 6.95854862e-01],
               [2.21391780e-02, 2.81738767e-02, 6.51344383e-01],
               [2.05887070e-02, 2.54397958e-02, 6.06094160e-01],
               [1.90382361e-02, 2.27057149e-02, 5.60843937e-01],
               [1.73497807e-02, 2.01171617e-02, 5.08432174e-01],
               [1.55765090e-02, 1.76180613e-02, 4.51618362e-01],
               [1.38032374e-02, 1.51189609e-02, 3.94804551e-01],
               [1.21218105e-02, 1.29757867e-02, 3.41210630e-01],
               [1.04688473e-02, 1.09429181e-02, 2.88614589e-01],
               [8.81588406e-03, 8.91004948e-03, 2.36018548e-01],
               [7.40566041e-03, 7.40847074e-03, 1.95596905e-01],
               [6.01017453e-03, 5.93914888e-03, 1.55914421e-01],
               [4.69529289e-03, 4.56170905e-03, 1.18703848e-01],
               [3.81412524e-03, 3.67866644e-03, 9.47940869e-02],
               [2.93295759e-03, 2.79562384e-03, 7.08843254e-02],
               [2.20646541e-03, 2.07474126e-03, 5.17641069e-02],
               [1.70753686e-03, 1.59243385e-03, 3.96904214e-02],
               [1.20860831e-03, 1.11012644e-03, 2.76167359e-02],
               [8.89261197e-04, 8.07902163e-04, 2.01786925e-02],
               [6.52132098e-04, 5.88125582e-04, 1.48629913e-02],
               [4.15003000e-04, 3.68349000e-04, 9.54729000e-03]]
        rgb_high = np.array(rgb_high)
        # if spacing != None:
        #     spacing = np.arange(0, 1, 1 / energy_bins)

        rgb = zoom(rgb_high.T,(1,energy_bins/len(rgb_high.T[0])))


        return np.clip(rgb,1e-15, rgb.max())
    rgb = _eye_sensitivity(spectral_cube.shape[-1])
    rgb_data = np.tensordot(spectral_cube, rgb, axes=[-1, -1])
    rgb_data = np.log(rgb_data)
    rgb_data -= rgb_data.min()
    rgb_data /= rgb_data.max()
    return rgb_data

def _find_closest(A, target):
    # A must be sorted
    idx = np.clip(A.searchsorted(target), 1, len(A)-1)
    idx -= target - A[idx-1] < A[idx] - target
    return idx


def _makeplot(name):
    import matplotlib.pyplot as plt
    if dobj.rank != 0:
        plt.close()
        return
    if name is None:
        plt.show()
        plt.close()
        return
    extension = os.path.splitext(name)[1]
    if extension in (".pdf", ".png", ".svg"):
        plt.savefig(name)
        plt.close()
    else:
        raise ValueError("file format not understood")


def _limit_xy(**kwargs):
    import matplotlib.pyplot as plt
    x1, x2, y1, y2 = plt.axis()
    x1 = kwargs.pop("xmin", x1)
    x2 = kwargs.pop("xmax", x2)
    y1 = kwargs.pop("ymin", y1)
    y2 = kwargs.pop("ymax", y2)
    plt.axis((x1, x2, y1, y2))


def _register_cmaps():
    try:
        if _register_cmaps._cmaps_registered:
            return
    except AttributeError:
        _register_cmaps._cmaps_registered = True

    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.pyplot as plt
    planckcmap = {'red':   ((0., 0., 0.), (.4, 0., 0.), (.5, 1., 1.),
                            (.7, 1., 1.), (.8, .83, .83), (.9, .67, .67),
                            (1., .5, .5)),
                  'green': ((0., 0., 0.), (.2, 0., 0.), (.3, .3, .3),
                            (.4, .7, .7), (.5, 1., 1.), (.6, .7, .7),
                            (.7, .3, .3), (.8, 0., 0.), (1., 0., 0.)),
                  'blue':  ((0., .5, .5), (.1, .67, .67), (.2, .83, .83),
                            (.3, 1., 1.), (.5, 1., 1.), (.6, 0., 0.),
                            (1., 0., 0.))}
    he_cmap = {'red':   ((0., 0., 0.), (.167, 0., 0.), (.333, .5, .5),
                         (.5, 1., 1.), (1., 1., 1.)),
               'green': ((0., 0., 0.), (.5, 0., 0.), (.667, .5, .5),
                         (.833, 1., 1.), (1., 1., 1.)),
               'blue':  ((0., 0., 0.), (.167, 1., 1.), (.333, .5, .5),
                         (.5, 0., 0.), (1., 1., 1.))}
    fd_cmap = {'red':   ((0., .35, .35), (.1, .4, .4), (.2, .25, .25),
                         (.41, .47, .47), (.5, .8, .8), (.56, .96, .96),
                         (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                         (.9, .5, .5), (1., .4, .4)),
               'green': ((0., 0., 0.), (.2, 0., 0.), (.362, .88, .88),
                         (.5, 1., 1.), (.638, .88, .88), (.8, .25, .25),
                         (.9, .3, .3), (1., .2, .2)),
               'blue':  ((0., .35, .35), (.1, .4, .4), (.2, .8, .8),
                         (.26, .8, .8), (.41, 1., 1.), (.44, .96, .96),
                         (.5, .8, .8), (.59, .47, .47), (.8, 0., 0.),
                         (1., 0., 0.))}
    fdu_cmap = {'red':   ((0., 1., 1.), (0.1, .8, .8), (.2, .65, .65),
                          (.41, .6, .6), (.5, .7, .7), (.56, .96, .96),
                          (.59, 1., 1.), (.74, .8, .8), (.8, .8, .8),
                          (.9, .5, .5), (1., .4, .4)),
                'green': ((0., .9, .9), (.362, .95, .95), (.5, 1., 1.),
                          (.638, .88, .88), (.8, .25, .25), (.9, .3, .3),
                          (1., .2, .2)),
                'blue':  ((0., 1., 1.), (.1, .8, .8), (.2, 1., 1.),
                          (.41, 1., 1.), (.44, .96, .96), (.5, .7, .7),
                          (.59, .42, .42), (.8, 0., 0.), (1., 0., 0.))}
    pm_cmap = {'red':   ((0., 1., 1.), (.1, .96, .96), (.2, .84, .84),
                         (.3, .64, .64), (.4, .36, .36), (.5, 0., 0.),
                         (1., 0., 0.)),
               'green': ((0., .5, .5), (.1, .32, .32), (.2, .18, .18),
                         (.3, .8, .8),  (.4, .2, .2), (.5, 0., 0.),
                         (.6, .2, .2), (.7, .8, .8), (.8, .18, .18),
                         (.9, .32, .32), (1., .5, .5)),
               'blue':  ((0., 0., 0.), (.5, 0., 0.), (.6, .36, .36),
                         (.7, .64, .64), (.8, .84, .84), (.9, .96, .96),
                         (1., 1., 1.))}

    plt.register_cmap(cmap=LinearSegmentedColormap("Planck-like", planckcmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("High Energy", he_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Map", fd_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Faraday Uncertainty",
                                                   fdu_cmap))
    plt.register_cmap(cmap=LinearSegmentedColormap("Plus Minus", pm_cmap))


def _plot1D(f, ax, **kwargs):
    import matplotlib.pyplot as plt

    for i, fld in enumerate(f):
        if not isinstance(fld, Field):
            raise TypeError("incorrect data type")
        if i == 0:
            dom = fld.domain
            if (len(dom) != 1):
                raise ValueError("input field must have exactly one domain")
        else:
            if fld.domain != dom:
                raise ValueError("domain mismatch")
    dom = dom[0]

    label = kwargs.pop("label", None)
    if not isinstance(label, list):
        label = [label] * len(f)

    linewidth = kwargs.pop("linewidth", 1.)
    if not isinstance(linewidth, list):
        linewidth = [linewidth] * len(f)

    alpha = kwargs.pop("alpha", None)
    if not isinstance(alpha, list):
        alpha = [alpha] * len(f)

    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))

    if isinstance(dom, RGSpace):
        plt.yscale(kwargs.pop("yscale", "linear"))
        npoints = dom.shape[0]
        dist = dom.distances[0]
        xcoord = np.arange(npoints, dtype=np.float64)*dist
        for i, fld in enumerate(f):
            ycoord = fld.to_global_data()
            plt.plot(xcoord, ycoord, label=label[i],
                     linewidth=linewidth[i], alpha=alpha[i])
        _limit_xy(**kwargs)
        if label != ([None]*len(f)):
            plt.legend()
        return
    elif isinstance(dom, LogRGSpace):
        plt.yscale(kwargs.pop("yscale", "log"))
        npoints = dom.shape[0]
        xcoord = dom.t_0 + np.arange(npoints-1)*dom.bindistances[0]
        for i, fld in enumerate(f):
            ycoord = fld.to_global_data()[1:]
            plt.plot(xcoord, ycoord, label=label[i],
                     linewidth=linewidth[i], alpha=alpha[i])
        _limit_xy(**kwargs)
        if label != ([None]*len(f)):
            plt.legend()
        return
    elif isinstance(dom, PowerSpace):
        plt.xscale(kwargs.pop("xscale", "log"))
        plt.yscale(kwargs.pop("yscale", "log"))
        xcoord = dom.k_lengths
        for i, fld in enumerate(f):
            ycoord = fld.to_global_data()
            plt.plot(xcoord, ycoord, label=label[i],
                     linewidth=linewidth[i], alpha=alpha[i])
        _limit_xy(**kwargs)
        if label != ([None]*len(f)):
            plt.legend()
        return
    raise ValueError("Field type not(yet) supported")


def _plot2D(f, ax, **kwargs):
    import matplotlib.pyplot as plt

    dom = f.domain

    if len(dom) > 2:
        raise ValueError("DomainTuple can have at most two entries.")

    # check for multifrequency plotting
    have_rgb = False
    if len(dom) == 2:
        if (not isinstance(dom[1], RGSpace)) or len(dom[1].shape) != 1:
            raise TypeError("need 1D RGSpace as second domain")
        rgb = _rgb_data(f.to_global_data())
        have_rgb = True

    label = kwargs.pop("label", None)

    foo = kwargs.pop("norm", None)
    norm = {} if foo is None else {'norm': foo}

    ax.set_title(kwargs.pop("title", ""))
    ax.set_xlabel(kwargs.pop("xlabel", ""))
    ax.set_ylabel(kwargs.pop("ylabel", ""))
    dom = dom[0]
    if not have_rgb:
        cmap = kwargs.pop("colormap", plt.rcParams['image.cmap'])

    if isinstance(dom, RGSpace):
        nx, ny = dom.shape
        dx, dy = dom.distances
        if have_rgb:
            im = ax.imshow(
                rgb, extent=[0, nx*dx, 0, ny*dy], origin="lower", **norm)
        else:
            im = ax.imshow(
                f.to_global_data().T, extent=[0, nx*dx, 0, ny*dy],
                vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                cmap=cmap, origin="lower", **norm)
            plt.colorbar(im)
        _limit_xy(**kwargs)
        return
    elif isinstance(dom, (HPSpace, GLSpace)):
        import pyHealpix
        xsize = 800
        res, mask, theta, phi = _mollweide_helper(xsize)
        if have_rgb:
            res = np.full(shape=res.shape+(3,), fill_value=1., dtype=np.float64)

        if isinstance(dom, HPSpace):
            ptg = np.empty((phi.size, 2), dtype=np.float64)
            ptg[:, 0] = theta
            ptg[:, 1] = phi
            base = pyHealpix.Healpix_Base(int(np.sqrt(dom.size//12)), "RING")
            if have_rgb:
                res[mask] = rgb[base.ang2pix(ptg)]
            else:
                res[mask] = f.to_global_data()[base.ang2pix(ptg)]
        else:
            ra = np.linspace(0, 2*np.pi, dom.nlon+1)
            dec = pyHealpix.GL_thetas(dom.nlat)
            ilat = _find_closest(dec, theta)
            ilon = _find_closest(ra, phi)
            ilon = np.where(ilon == dom.nlon, 0, ilon)
            if have_rgb:
                res[mask] = rgb[ilat*dom[0].nlon + ilon]
            else:
                res[mask] = f.to_global_data()[ilat*dom.nlon + ilon]
        plt.axis('off')
        if have_rgb:
            plt.imshow(res, origin="lower")
        else:
            plt.imshow(res, vmin=kwargs.get("zmin"), vmax=kwargs.get("zmax"),
                       cmap=cmap, origin="lower")
            plt.colorbar(orientation="horizontal")
        return
    raise ValueError("Field type not(yet) supported")


def _plot(f, ax, **kwargs):
    _register_cmaps()
    if isinstance(f, Field):
        f = [f]
    f = list(f)
    if len(f) == 0:
        raise ValueError("need something to plot")
    if not isinstance(f[0], Field):
            raise TypeError("incorrect data type")
    dom1 = f[0].domain
    if (len(dom1)==1 and
        (isinstance(dom1[0],PowerSpace) or
            (isinstance(dom1[0], (RGSpace, LogRGSpace)) and
             len(dom1[0].shape) == 1))):
        _plot1D(f, ax, **kwargs)
        return
    else:
        if len(f) != 1:
            raise ValueError("need exactly one Field for 2D plot")
        _plot2D(f[0], ax, **kwargs)
        return
    raise ValueError("Field type not(yet) supported")


class Plot(object):
    def __init__(self):
        self._plots = []
        self._kwargs = []

    def add(self, f, **kwargs):
        """Add a figure to the current list of plots.

        Notes
        -----
        After doing one or more calls `plot()`, one also needs to call
        `plot_finish()` to output the result.

        Parameters
        ----------
        f: Field or list of Field
            If `f` is a single Field, it must be defined on a single `RGSpace`,
            `PowerSpace`, `HPSpace`, `GLSpace`.
            If it is a list, all list members must be Fields defined over the
            same one-dimensional `RGSpace` or `PowerSpace`.
        title: string
            title of the plot.
        xlabel: string
            Label for the x axis.
        ylabel: string
            Label for the y axis.
        [xyz]min, [xyz]max: float
            Limits for the values to plot.
        colormap: string
            Color map to use for the plot (if it is a 2D plot).
        linewidth: float or list of floats
            Line width.
        label: string of list of strings
            Annotation string.
        alpha: float or list of floats
            transparency value
        """
        self._plots.append(f)
        self._kwargs.append(kwargs)

    def output(self, **kwargs):
        """Plot the accumulated list of figures.

        Parameters
        ----------
        title: string
            Title of the full plot.
        nx, ny: int
            Number of subplots to use in x- and y-direction.
            Default: square root of the numer of plots, rounded up.
        xsize, ysize: float
            Dimensions of the full plot in inches. Default: 6.
        name: string
            If left empty, the plot will be shown on the screen,
            otherwise it will be written to a file with the given name.
            Supported extensions: .png and .pdf. Default: None.
        """
        import matplotlib.pyplot as plt
        nplot = len(self._plots)
        fig = plt.figure()
        if "title" in kwargs:
            plt.suptitle(kwargs.pop("title"))
        nx = kwargs.pop("nx", int(np.ceil(np.sqrt(nplot))))
        ny = kwargs.pop("ny", int(np.ceil(np.sqrt(nplot))))
        if nx*ny < nplot:
            raise ValueError(
                'Figure dimensions not sufficient for number of plots. '
                'Available plot slots: {}, number of plots: {}'
                .format(nx*ny, nplot))
        xsize = kwargs.pop("xsize", 6)
        ysize = kwargs.pop("ysize", 6)
        fig.set_size_inches(xsize, ysize)
        for i in range(nplot):
            ax = fig.add_subplot(ny, nx, i+1)
            _plot(self._plots[i], ax, **self._kwargs[i])
        fig.tight_layout()
        _makeplot(kwargs.pop("name", None))
