# -*- coding: utf-8 -*-

import numpy as np
from nifty.nifty_mpi_data import distributed_data_object
from nifty.nifty_about import about

# Try to import pyfftw. If this fails fall back to gfft. 
# If this fails fall back to local gfft_rg

try:
    import pyfftw
    fft_machine='pyfftw'
except(ImportError):
    try:
        import gfft
        fft_machine='gfft'
        about.infos.cprint('INFO: Using gfft')
    except(ImportError):
        import gfft_rg as gfft
        fft_machine='gfft_fallback'
        about.infos.cprint('INFO: Using builtin "plain" gfft version 0.1.0')


def fft_factory():
    """
        A factory for fast-fourier-transformation objects.

        Parameters
        ----------
        None

        Returns
        -----
        fft: Returns a fft_object depending on the available packages.
        Hierarchy: pyfftw -> gfft -> built in gfft.
        
    """       
    if fft_machine == 'pyfftw':
        return fft_fftw()
        
    elif fft_machine == 'gfft' or 'gfft_fallback':
        return fft_gfft()



class fft(object):
    """
        A generic fft object without any implementation.

        Parameters
        ----------
        None
    """
    def transform(self, val, domain, codomain, **kwargs):
        """
            A generic ff-transform function. 
            
            Parameters
            ----------
            field_val : distributed_data_object
                The value-array of the field which is supposed to 
                be transformed.
            
            domain : nifty.rg.nifty_rg.rg_space
                The domain of the space which should be transformed.
            
            codomain : nifty.rg.nifty_rg.rg_space
                The taget into which the field should be transformed.
        """
        return None    
    
if fft_machine == 'pyfftw':
    ## The instances of plan_and_info store the fftw plan and all 
    ## other information needed in order to perform a mpi-fftw transformation            
    class _fftw_plan_and_info(fft):
        def __init__(self,domain,codomain,fft_fftw_context,**kwargs):
            self.compute_plan_and_info(domain, codomain, fft_fftw_context, 
                                       **kwargs)
            
        def set_plan(self, x):
            self.plan=x                    
        def get_plan(self):
            return self.plan
        
        def set_domain_centering_mask(self,x):
            self.domain_centering_mask=x
        def get_domain_centering_mask(self):
            return self.domain_centering_mask

        def set_codomain_centering_mask(self,x):
            self.codomain_centering_mask=x
        def get_codomain_centering_mask(self):
            return self.codomain_centering_mask
        
        def compute_plan_and_info(self, domain, codomain, fft_fftw_context, 
                                  **kwargs):
            
            self.input_dtype = 'complex128'
            self.output_dtype = 'complex128'
            
            self.global_input_shape = domain.shape()   
            self.global_output_shape = codomain.shape()
            self.fftw_local_size = pyfftw.local_size(self.global_input_shape)

            self.in_zero_centered_dimensions = domain.paradict['zerocenter']
            self.out_zero_centered_dimensions = codomain.paradict['zerocenter']

            self.local_node_dimensions = np.append((self.fftw_local_size[1],), 
                                                   self.global_input_shape[1:])
            self.offsetQ = self.fftw_local_size[2]%2
            
            if codomain.fourier == True:
                self.direction = 'FFTW_FORWARD'
            else:
                self.direction = 'FFTW_BACKWARD'
                
            ##compute the centering masks                    
            self.set_domain_centering_mask(
                fft_fftw_context.get_centering_mask(
                    self.in_zero_centered_dimensions,
                    self.local_node_dimensions,
                    self.offsetQ))
                                        
            self.set_codomain_centering_mask(
                fft_fftw_context.get_centering_mask(
                    self.out_zero_centered_dimensions,
                    self.local_node_dimensions,
                    self.offsetQ))
            
            self.set_plan(
                pyfftw.create_mpi_plan(
                    input_shape=self.global_input_shape, 
                    input_dtype=self.input_dtype, 
                    output_dtype=self.output_dtype, 
                    direction=self.direction,
                    flags=["FFTW_ESTIMATE"],
                    **kwargs)
                )
                
    class fft_fftw(fft):  
        """
            The pyfftw pendant of a fft object.

            Parameters
            ----------
            None
            
        """
        def __init__(self):
            ## The plan_dict stores the plan_and_info objects which correspond
            ## to a certain set of (field_val, domain, codomain) sets.
            self.plan_dict = {}
            
            ## initialize the dictionary which stores the values from 
            ## get_centering_mask
            self.centering_mask_dict = {}      
         
        def get_centering_mask(self, to_center_input, dimensions_input, 
                               offset_input=0):
            """
                Computes the mask, used to (de-)zerocenter domain and target 
                fields.
                
                Parameters
                ----------
                to_center_input : tuple, list, numpy.ndarray
                    A tuple of booleans which dimensions should be 
                    zero-centered.
                
                dimensions_input : tuple, list, numpy.ndarray
                    A tuple containing the masks desired shape.
                
                offset_input : int, boolean
                    Specifies whether the zero-th dimension starts with an odd
                    or and even index, i.e. if it is shifted.
                    
                Returns
                -------
                result : np.ndarray
                    A 1/-1-alternating mask. 
            """
            ## cast input
            to_center = np.array(to_center_input)
            dimensions = np.array(dimensions_input)  
            
            if np.all(dimensions == np.array(1)) or \
                np.all(dimensions == np.array([1])):
                return dimensions
            ## The dimensions of size 1 must be sorted out for computing the
            ## centering_mask. The depth of the array will be restored in the 
            ## end.
            size_one_dimensions = []            
            temp_dimensions = [] 
            temp_to_center = []
            for i in range(len(dimensions)):
                if dimensions[i]==1:
                    size_one_dimensions += [True]
                else:
                    size_one_dimensions += [False]                    
                    temp_dimensions += [dimensions[i]]
                    temp_to_center += [to_center[i]]
            dimensions = np.array(temp_dimensions)
            to_center = np.array(temp_to_center)
            ## cast the offset_input into the shape of to_center
            offset = np.zeros(to_center.shape,dtype=int)
            offset[0] = int(offset_input)
            ## check for dimension match
            if to_center.size != dimensions.size:
                raise TypeError(\
                        'The length of the supplied lists does not match.')

            ## build up the value memory
            ## compute an identifier for the parameter set
            temp_id = tuple((tuple(to_center),tuple(dimensions),tuple(offset)))
            if not temp_id in self.centering_mask_dict:
                ## use np.tile in order to stack the core alternation scheme 
                ## until the desired format is constructed. 
                core = np.fromfunction(
                    lambda *args : (-1)**\
                        (np.tensordot(to_center,args + \
                        offset.reshape(offset.shape + \
                            (1,)*(np.array(args).ndim - 1)),1)),\
                    (2,)*to_center.size)
                    
                centering_mask = np.tile(core,dimensions//2)           
                ## for the dimensions of odd size corresponding slices must be added
                for i in range(centering_mask.ndim):
                    ## check if the size of the certain dimension is odd or even
                    if (dimensions%2)[i]==0:
                        continue
                    ## prepare the slice object
                    temp_slice = (slice(None),)*i + (slice(-2,-1,1),) +\
                                 (slice(None),)*(centering_mask.ndim -1 - i)
                    ## append the slice to the centering_mask                    
                    centering_mask = np.append(centering_mask, 
                                            centering_mask[temp_slice],axis=i)                
                ## Add depth to the centering_mask where the length of a 
                ## dimension was one
                temp_slice = ()
                for i in range(len(size_one_dimensions)):
                    if size_one_dimensions[i] == True:
                        temp_slice += (None,)
                    else:
                        temp_slice += (slice(None),)
                centering_mask = centering_mask[temp_slice]
                self.centering_mask_dict[temp_id] = centering_mask        
            return self.centering_mask_dict[temp_id]
                

        def _get_plan_and_info(self,domain,codomain,**kwargs):
            ## generate a id-tuple which identifies the domain-codomain setting                
            temp_id = (domain.__identifier__(), codomain.__identifier__())
            ## generate the plan_and_info object if not already there                
            if not temp_id in self.plan_dict:
                self.plan_dict[temp_id]=_fftw_plan_and_info(domain, codomain,
                                                            self, **kwargs)
            return self.plan_dict[temp_id]
        
        def transform(self, val, domain, codomain, **kwargs):
            """
                The pyfftw transform function. 
                
                Parameters
                ----------
                val : distributed_data_object or numpy.ndarray
                    The value-array of the field which is supposed to 
                    be transformed.
                
                domain : nifty.rg.nifty_rg.rg_space
                    The domain of the space which should be transformed.
                
                codomain : nifty.rg.nifty_rg.rg_space
                    The taget into which the field should be transformed.
    
                **kwargs : *optional*
                    Further kwargs are passed to the create_mpi_plan routine.
                
                Returns
                -------
                result : np.ndarray
                    Fourier-transformed pendant of the input field.
            """
            current_plan_and_info=self._get_plan_and_info(domain, codomain, 
                                                          **kwargs)
            ## Prepare the environment variables
            local_size = current_plan_and_info.fftw_local_size
            local_start = local_size[2]
            local_end = local_start + local_size[1]
            
            ## Prepare the input data
            ## Case 1: val is a distributed_data_object
            if isinstance(val, distributed_data_object):
                return_val = val.copy_empty(global_shape =\
                    tuple(current_plan_and_info.global_output_shape),
                    dtype = np.complex128)
                ## If the distribution strategy of the d2o is fftw, extract 
                ## the data directly
                if val.distribution_strategy == 'fftw':
                    local_val = val.get_local_data()
                else:
                    local_val = val.get_data(slice(local_start, local_end))
            ## Case 2: val is a numpy array carrying the full data
            else:
                local_val = val[slice(local_start, local_end)]

            local_val *= current_plan_and_info.get_codomain_centering_mask()
            
            ## Define a abbreviation for the fftw plan                
            p = current_plan_and_info.get_plan()
            ## load the field into the plan
            if p.has_input:
                p.input_array[:] = local_val
            ## execute the plan
            p()
            result = p.output_array * current_plan_and_info.\
                                get_domain_centering_mask()

            ## renorm the result according to the convention of gfft            
            if current_plan_and_info.direction == 'FFTW_FORWARD':
                result = result/float(result.size)
            else:
                result *= float(result.size)
                    
            ## build the return object according to the input val
            try:
                if return_val.distribution_strategy == 'fftw':
                    return_val.set_local_data(data = result)
                else:
                    return_val.set_data(data = result, 
                                        key = slice(local_start, local_end)) 
                
                ## If the values living in domain are purely real, the 
                ## result of the fft is hermitian
                if domain.paradict['complexity'] == 0:
                    return_val.hermitian = True
                    
            ## In case the input val was not a distributed data obect, the try
            ## will produce a NameError
            except(NameError): 
                return_val = distributed_data_object(
                    global_shape =\
                        tuple(current_plan_and_info.global_output_shape), 
                    dtype = np.complex128,
                    distribution_strategy='fftw')
                return_val.set_local_data(data = result)
                return_val = return_val.get_full_data()
                        
            return return_val
            

    
elif fft_machine == 'gfft' or 'gfft_fallback':
    class fft_gfft(fft):                    
        """
            The gfft pendant of a fft object.

            Parameters
            ----------
            None
            
        """        
        def transform(self, val, domain, codomain, **kwargs):
            """
                The gfft transform function. 
                
                Parameters
                ----------
                val : numpy.ndarray or distributed_data_object
                    The value-array of the field which is supposed to 
                    be transformed.
                
                domain : nifty.rg.nifty_rg.rg_space
                    The domain of the space which should be transformed.
                
                codomain : nifty.rg.nifty_rg.rg_space
                    The taget into which the field should be transformed.
    
                **kwargs : *optional*
                    Further kwargs are not processed.
                
                Returns
                -------
                result : np.ndarray
                    Fourier-transformed pendant of the input field.
            """
            naxes = (np.size(domain.para)-1)//2
            if(codomain.fourier):
                ftmachine = "fft"
            else:
                ftmachine = "ifft"       
            ## if the input is a distributed_data_object, extract the data
            if isinstance(val, distributed_data_object):
                d2oQ = True
                val = val.get_full_data()
            ## transform and return
            if(domain.datatype==np.float64):
                temp = gfft.gfft(val.astype(np.complex128), 
                                in_ax=[], 
                                out_ax=[], 
                                ftmachine=ftmachine, 
                                in_zero_center=domain.para[-naxes:].\
                                    astype(np.bool).tolist(), 
                                out_zero_center=codomain.para[-naxes:].\
                                    astype(np.bool).tolist(), 
                                enforce_hermitian_symmetry = \
                                    bool(codomain.para[naxes]==1),
                                W=-1,
                                alpha=-1,
                                verbose=False)
            else:
                temp = gfft.gfft(val,
                                in_ax=[],
                                out_ax=[],
                                ftmachine=ftmachine,
                                in_zero_center=domain.para[-naxes:].\
                                    astype(np.bool).tolist(),
                                out_zero_center=codomain.para[-naxes:].\
                                    astype(np.bool).tolist(),
                                enforce_hermitian_symmetry = \
                                    bool(codomain.para[naxes]==1),
                                W=-1,
                                alpha=-1,
                                verbose=False)
            if d2oQ == True:
                val.set_full_data(temp)
            else:
                val = temp
                
            return val
                