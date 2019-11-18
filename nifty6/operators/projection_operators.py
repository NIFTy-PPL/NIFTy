from .linear_operator import LinearOperator
from ..domain_tuple import DomainTuple
from ..domains.rg_space import RGSpace
from ..domains.hp_space import HPSpace
from ..domains.gl_space import GLSpace
from ..field import Field
from ..utilities import infer_space

import numpy as np
import pyHealpix as hp
from scipy.sparse import csr_matrix


class ProjectionOperator(LinearOperator):
    def __init__(self, domain, pointing, radius, target=None, shape=None, space=None):
        
        # make sure we work on the correct space
        self._domain = DomainTuple.make(domain)
        self._space  = infer_space(self._domain, space)
        
        # ensure correct types and values for angular arguments
        self._pointing = tuple([float(p) for p in pointing])
        if (len(self._pointing) != 2 or (
            not ((0. <= self._pointing[0] <= np.pi)
                 and (0. <= self._pointing[1] <= 2*np.pi)
                )
        )):
            raise ValueError('pointing: must be tuple of (co-latitude, longitude) in radians')
        
        # ensure disc is smaller than sphere
        self._radius = float(radius)
        if self._radius >= np.pi:
            raise ValueError('radius: must be strictly less than pi')
        
        # prepare sparse projection matrix generation
        if isinstance(self._domain[self._space], HPSpace) or isinstance(self._domain[self._space], GLSpace): # forward projection
            self._base   = self._get_base(self._domain[self._space])
            if target == None:
                # if not specified, shape is as close as possible to domain's angular resolution
                if shape == None:
                    if isinstance(self._domain[self._space], HPSpace):
                        self._shape = tuple([int(8. * self._domain[self._space].nside * self._radius / np.pi)]*2)
                    else: # domain is GLSpace
                        self._shape = tuple([
                            int(2. * self._domain[self._space].nlat * self._radius / np.pi), # theta
                            int(self._domain[self._space].nlon * self._radius / np.pi),      # phi
                        ])
                else:
                    self._shape = tuple([int(s) for s in shape])
                    if len(self._shape) != 2:
                        raise ValueError('shape: tuple of length two required')
                        
                self._distances = self._make_distances()
                domains = list(self._domain._dom)
                domains[self._space] = RGSpace(self._shape, self._distances)
                self._target    = DomainTuple.make(domains)
                
            else:
                self._target = DomainTuple.make(target)
                if isinstance(self._target[self._space], RGSpace):
                    self._target    = DomainTuple.make(target)
                    self._shape     = target.shape
                    self._distances = self._make_distances()
                
                else:
                    raise ValueError('target[space]: must be RGSpace if domain is HPSpace or GLSpace')
                
            # generate sparse forward projection matrix
            self._mat = self._make_for_mat()
            
        elif isinstance(self._domain[self._space], RGSpace): # backward projection
            self._target = DomainTuple.make(target)
            if isinstance(self._target[self._space], HPSpace) or isinstance(self._target[self._space], GLSpace):
                self._shape     = self._domain[self._space].shape
                self._base      = self._get_base(self._target[self._space])
                self._distances = self._make_distances()
                self._mat       = self._make_bac_mat()
            else:
                raise ValueError('target: must be HPSpace or GLSpace if domain is RGSpace')
                
        else:
            raise ValueError('domain[space]: must be HPSpace, GLSpace or RGSpace')
            
        # since this operator is generally not invertible we only support two modes
        self._capability = self.TIMES | self.ADJOINT_TIMES
    
    
    def __repr__(self):
        """Returns a representation string for type casting to string."""
        
        return (
            self.__class__.__name__ +
            "(domain={}, pointing={}, radius={}, target={}, shape={})"
            .format(
                self._domain,
                self._pointing,
                self._radius,
                self._target,
                self._shape
            )
        )
    
    
    def apply(self, x, mode):
        """Applies the projection to a given field :attr:`x`."""
        
        # make sure the operator is applied correctly
        self._check_input(x, mode)
        # extract value array from field
        values = x.to_global_data()
        
        if mode == self.TIMES:
            # number of axes for requested space
            axes = self._domain.axes[self._space]
            if len(axes) > 1: # RGSpace must be reshaped for matrix multiplication
                shape = list(self._domain.shape)
                shape[self._space] = -1
                del shape[self._space + 1]
                shape = tuple(shape)
            else:
                shape = self._domain.shape
                
            result = self._mat.dot(values.reshape(shape).transpose())
            result = result.transpose().reshape(self._target.shape)
            
            return Field.from_global_data(self._target, result)
        
        else:
            # number of axes for requested space
            axes = self._target.axes[self._space]
            if len(axes) > 1: # RGSpace must be reshaped for matrix multiplication
                shape = list(self._target.shape)
                shape[self._space] = -1
                del shape[self._space + 1]
                shape = tuple(shape)
            else:
                shape = self._target.shape
                
            result = self._mat.transpose().dot(values.reshape(shape).transpose())
            result = result.transpose().reshape(self._domain.shape)
            
            return Field.from_global_data(self._domain, result)
    
    
    def _make_distances(self):
        """Returns the internal units of the projection plane."""
        
        # calculate distance of projection plane from origin
        d = np.cos(self._radius)
        # calculate geometric length of projection plane
        l = 2 * np.sqrt(1 - d * d)
        # divide by shape in each direction
        distances = tuple([l / s for s in self._shape])
        
        return distances
    
    
    def _make_for_mat(self):
        """The matrix defined in this function describes a surjective mapping from the
        projected disc on the sphere into the projection plane. It should therefore be
        used for the forward projection."""
        
        raise NotImplementedError
    
    
    def _make_bac_mat(self):
        """The matrix defined in this function describes a surjective mapping from the
        projection plane into the projected disc on the sphere. It should therefore be
        used for the back projection."""
        
        raise NotImplementedError
    
    
    def _get_base(self, domain):
        """Returns an implementation of the coordinate arithmetic for :attr:`domain`.
        This method is the reason why projection operators can comfortably be written
        for both GLSpace and HPSpace."""
        
        if isinstance(domain, HPSpace):
            # Healpix_Base implments HEALPix geometry
            return hp.Healpix_Base(
                nside  = domain.nside,
                scheme = 'RING'
            )
        else: # domain is GLSpace
            #return GLPix_Base(
            #    nlat = domain.nlat,
            #    nlon = domain.nlon
            #)
            raise NotImplementedError
    
    
    def _get_local_basis(self):
        """Returns a tuple of the three local base vectors spanning the tangential space:
        * n   : normal
        * e_t : theta unit vector
        * e_p : phi unit vector
        in that order.
        """
        
        t = self._pointing[0] # theta
        p = self._pointing[1] # phi
        n = np.array([ # normal
            np.sin(t) * np.cos(p),
            np.sin(t) * np.sin(p),
            np.cos(t)
        ])
        e_t = np.array([ # unit vector in theta direction
             np.cos(t) * np.cos(p),
             np.cos(t) * np.sin(p),
            -np.sin(t)
        ])
        e_p = np.array([ # unit vector in phi direction
            -np.sin(p),
             np.cos(p),
             0
        ])
        
        return n, e_t, e_p
    
    
    def _get_disc_pixels(self):
        """Returns a tuple with the all pixel indices within the projected disc."""
        # query pixel ranges for disc
        disc = self._base.query_disc(
            ptg    = self._pointing,
            radius = self._radius
        )
        # generate tuple of pixels
        pixels = tuple([])
        for sector in disc:
            pixels += tuple(range(sector[0],sector[1]))

        return pixels
    
    
    def _get_disc_vectors(self):
        """Returns an array of unit vectors pointing to all pixels in the projected disc."""
        # query pixel ranges for disc
        pixels = self._get_disc_pixels()
        # generate unit pointings to points d in disc
        return self._base.pix2vec(pixels)
    
    
    def _ang2vec(self, ang):
        x = np.sin(ang[:,0]) * np.cos(ang[:,1])
        y = np.sin(ang[:,0]) * np.sin(ang[:,1])
        z = np.cos(ang[:,0])
        vec = np.hstack((
            x.reshape(-1,1),
            y.reshape(-1,1),
            z.reshape(-1,1)
        ))
        return vec


class StereographicProjectionOperator(ProjectionOperator):
    """Projects pixels whose centers fall within disc of `radius` around `pointing`
    stereographically into an RGSpace and vice versa.
    
    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        domain[space] needs to be either :class:`HPSpace` or :class:`GLSpace`.
    pointing : tuple of float
        Pointing around which the projected disc is spanned. Must be a single
        (co-latitude, longitude) tuple.
    radius : float
        Radius of the disc around `pointing`. Must be a single float and less then pi/2.
    shape : int
        Shape of resulting RGSpace. Must be a tuple int. Default is to be as close
        as possible to domains angular resolution.
    space : int
        Index of space in `domain` on which the operator shall act.
        Default is 0.
    """
    
    def __init__(self, *args, **kwargs):
        super(StereographicProjectionOperator, self).__init__(*args, **kwargs)
    
    
    def _make_for_mat(self):
        # normal vector and local basis of projection plane
        n, e_t, e_p = self._get_local_basis()
        # distance of projection plane from origin
        d = np.cos(self._radius)
        # projection base vector
        b = d * n
        # plane pixel indices
        ppi = np.arange(self._shape[0] * self._shape[1])
        # plane pixel to internal coordinates
        v_t = np.floor(ppi % self._shape[0])
        v_p = np.floor(ppi / self._shape[0])
        # subtract projection center
        v_t -= self._shape[0] / 2.
        v_p -= self._shape[1] / 2.
        # convert to canonical units
        v_t *= self._distances[0]
        v_p *= self._distances[1]
        # vectors within projection plane
        v = np.outer(v_t, e_t) + np.outer(v_p, e_p)
        # pointings from projection anticenter to plane pixels
        p = b + n + v
        # normalize anticenter to plane pointings
        p /= np.linalg.norm(p, axis=1).reshape(-1,1)
        # length of p from projection anticenter to sphere
        l = 2 * np.einsum('ij, ij->i', p, n[np.newaxis,:]).reshape(-1,1)
        # pointing from origin to sphere pixel
        u = -n + l * p
        # convert pointings to angles
        spi = self._base.vec2pix(u)
        # nonzero entries of projection mapping matrix
        mapping = (np.ones(spi.shape, dtype=np.int64), (spi, ppi))
        # shape to map from spherical pixels to projection pixels
        shape = tuple([self._base.npix(), self._shape[0]**2])
        # represent as compressed sparse row matrix
        mat = csr_matrix(mapping, shape)
        
        return mat.transpose()
    
    
    def _make_bac_mat(self):
        # normal vector and local basis of projection plane
        n, e_t, e_p = self._get_local_basis()
        # distance of projection plane from origin
        d = np.cos(self._radius)
        # projection base vector
        b = d * n
        # unit pointings to points d in disc
        u = self._get_disc_vectors()
        # vectors from projection anticenter to points d in disc
        a = u + n
        # lenghts of intersection vectors with plane
        l = (1. + d) * np.reciprocal(a.dot(n))
        # intersection pointings from projection anticenter
        p = l.reshape(a.shape[0],1) * a
        # vectors from projection center to intersections
        v = p - n - b
        # project v into local basis
        v_t = v.dot(e_t)
        v_p = v.dot(e_p)
        # convert into the projection plane's units
        v_t /= self._distances[0]
        v_p /= self._distances[1]
        # push vector bases to projection center
        v_t += self._shape[0] / 2
        v_p += self._shape[1] / 2
        # plane pixel index
        ppi = (self._shape[0] * v_p.astype('int') + v_t.astype('int'))
        # disc pixel index
        dpi = self._get_disc_pixels()
        # nonzero entries of projection mapping matrix
        mapping = (np.ones(ppi.shape, dtype=np.int64), (ppi, dpi))
        # shape to map from spherical pixels to projection pixels
        shape = tuple([self._shape[0]**2,self._base.npix()])
        # represent as compressed sparse row matrix
        mat = csr_matrix(mapping, shape)
        
        return mat.transpose()



class GnomonicProjectionOperator(ProjectionOperator):
    """Projects pixels whose center fall within disc of `radius` around `pointing`
    gnomonically into an RGSpace and vice versa.
    
    Parameters
    ----------
    domain : Domain, DomainTuple or tuple of Domain
        domain[space] needs to be either :class:`HPSpace` or :class:`GLSpace`.
    pointing : tuple of float
        Pointing around which the projected disc is spanned. Must be a single
        (co-latitude, longitude) tuple.
    radius : float
        Radius of the disc around `pointing`. Must be a single float and less then pi/2.
    shape : int
        Shape of resulting RGSpace. Must be a tuple int. Default is to be as close
        as possible to domains angular resolution.
    space : int
        Index of space in `domain` on which the operator shall act.
        Default is 0.
    """
    
    
    def __init__(self, *args, **kwargs):
        super(GnomonicProjectionOperator, self).__init__(*args, **kwargs)
    
        
    def _make_for_mat(self):
        # normal vector and local basis of projection plane
        n, e_t, e_p = self._get_local_basis()
        # distance of projection plane from origin
        d = np.cos(self._radius)
        # projection base vector
        b = d * n
        # plane pixel indices
        ppi = np.arange(self._shape[0] * self._shape[1])
        # plane pixel to internal coordinates
        v_t = np.floor(ppi % self._shape[0])
        v_p = np.floor(ppi / self._shape[0])
        # subtract projection center
        v_t -= self._shape[0] / 2.
        v_p -= self._shape[1] / 2.
        # convert to canonical units
        v_t *= self._distances[0]
        v_p *= self._distances[1]
        # vectors within projection plane
        v = np.outer(v_t, e_t) + np.outer(v_p, e_p)
        # pointings from origin to plane pixels
        p = b + v
        # convert pointings to angles
        spi = self._base.vec2pix(p)
        # nonzero entries of projection mapping matrix
        mapping = (np.ones(spi.shape, dtype=np.int64), (spi, ppi))
        # shape to map from spherical pixels to projection pixels
        shape = tuple([self._base.npix(),self._shape[0]**2])
        # represent as compressed sparse row matrix
        mat = csr_matrix(mapping, shape)
        
        return mat.transpose()
    
    
    def _make_bac_mat(self):
        # normal vector and local basis of projection plane
        n, e_t, e_p = self._get_local_basis()
        # distance of projection plane from origin
        d = np.cos(self._radius)
        # projection base vector
        b = d * n
        # unit pointings to points d in disc
        u = self._get_disc_vectors()
        # lenghts of intersection vectors with plane
        l = d * np.reciprocal(u.dot(n))
        # intersection pointings
        p = l.reshape(u.shape[0],1) * u
        # vectors from projection center to intersections
        v = p - b
        # project v into local basis
        v_t = v.dot(e_t)
        v_p = v.dot(e_p)
        # convert into the projection plane's units
        v_t /= self._distances[0]
        v_p /= self._distances[1]
        # push vector bases to projection center
        v_t += self._shape[0] / 2
        v_p += self._shape[1] / 2
        # plane pixel index
        ppi = (self._shape[0] * v_p.astype('int') + v_t.astype('int'))
        # disc pixel index
        dpi = self._get_disc_pixels()
        # nonzero entries of projection mapping matrix
        mapping = (np.ones(ppi.shape, dtype=np.int64), (ppi, dpi))
        # shape to map from spherical pixels to projection pixels
        shape = tuple([self._shape[0]**2,self._base.npix()])
        # represent as compressed sparse row matrix
        mat = csr_matrix(mapping, shape)
        
        return mat.transpose()
