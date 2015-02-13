# -*- coding: utf-8 -*-

import numpy as np

# Try to import pyfftw. If this fails fall back to gfft. If this fails fall back to local gfft_rg

try:
    import pyfftw
    fft_machine='pyfftw'
except(ImportError):
    try:
        import gfft
        fft_machine='gfft'
        #about.infos.cprint('INFO: Using gfft')
    except(ImportError):
        import gfft_rg as gfft
        fft_machine='gfft_fallback'
        #about.infos.cprint('INFO: Using builtin "plain" gfft version 0.1.0')
    

'''
The fft_factory checks which fft module is available and returns the appropriate fft object.
The fft objects must get 3 parameters:
    1. field_val:
        The value-array of the field which is supposed to be transformed
    2. rg_space:
        The field's underlying rg_space
    3. codaim
        The rg_space into which the field is transformed
'''

def fft_factory():       
    if fft_machine == 'pyfftw':
        return fft_fftw()
        
    elif fft_machine == 'gfft' or 'gfft_fallback':
        return fft_gfft()



class fft(object):
    def transform(self,field_val,domain,codomain,**kwargs):
        return None    
    
if fft_machine == 'pyfftw':
    ## The instances of plan_and_info store the fftw plan and all 
    ## other information needed in order to perform a mpi-fftw transformation            
    class _fftw_plan_and_info(object):
        def __init__(self,domain,codomain,fft_fftw_context):
            self.compute_plan_and_info(domain,codomain,fft_fftw_context)
            
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
        
        def compute_plan_and_info(self, domain, codomain,fft_fftw_context):
            
            self.input_dtype = 'complex128'
            self.output_dtype = 'complex128'
            
            self.global_input_shape = domain.dim(split=True)   
            self.global_output_shape = codomain.dim(split=True)
            self.fftw_local_size = pyfftw.local_size(self.global_input_shape)

            self.in_zero_centered_dimensions = domain.zerocenter()[::-1]
            self.out_zero_centered_dimensions = codomain.zerocenter()[::-1]

            self.local_node_dimensions = np.append((self.fftw_local_size[1],),self.global_input_shape[1:])
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
                    flags=["FFTW_ESTIMATE"])
                )
                
    class fft_fftw(fft):        
        ## initialize the dictionary which stores the values from get_centering_mask
        centering_mask_dict = {}               
        def get_centering_mask(self, to_center_input, dimensions_input, offset_input=0):
            ## cast input
            to_center = np.array(to_center_input)
            dimensions = np.array(dimensions_input)  
            ## cast the offset_input into the shape of to_center
            offset = np.zeros(to_center.shape,dtype=int)
            offset[0] = int(offset_input)
            ## check for dimension match
            if to_center.size != dimensions.size:
                raise TypeError('The length of the supplied lists does not match')
            ## check that every dimension is larger than 1
            if np.any(dimensions == 1):
                return TypeError('Every dimensions must have an extent greater than 1')
            ## build up the value memory
            ## compute an identifier for the parameter set
            temp_id = tuple((tuple(to_center),tuple(dimensions),tuple(offset)))
            if not temp_id in self.centering_mask_dict:
                ## use np.tile in order to stack the core alternation scheme 
                ## until the desired format is constructed. 
                core = np.fromfunction(lambda *args : (-1)**(np.tensordot(to_center,args+offset.reshape(offset.shape+(1,)*(np.array(args).ndim-1)),1)), (2,)*to_center.size)
                centering_mask = np.tile(core,dimensions//2)           
                ## for the dimensions of odd size corresponding slices must be added
                for i in range(centering_mask.ndim):
                    ## check if the size of the certain dimension is odd or even
                    if (dimensions%2)[i]==0:
                        continue
                    ## prepare the slice object
                    temp_slice=(slice(None),)*i + (slice(-2,-1,1),) + (slice(None),)*(centering_mask.ndim -1 - i)
                    ## append the slice to the centering_mask                    
                    centering_mask = np.append(centering_mask,centering_mask[temp_slice],axis=i)                
                self.centering_mask_dict[temp_id] = centering_mask
            return self.centering_mask_dict[temp_id]
                
        ## The plan_dict stores the plan_and_info objects which correspond
        ## to a certain set of (field_val, domain, codomain) sets.
        plan_dict = {}
        def get_plan_and_info(self,domain,codomain):
            ## generate a id-tuple which identifies the domain-codomain setting                
            temp_id = (domain.__identifier__(), codomain.__identifier__())
            ## generate the plan_and_info object if not already there                
            if not temp_id in self.plan_dict:
                self.plan_dict[temp_id]=_fftw_plan_and_info(domain,codomain,self)
            return self.plan_dict[temp_id]
        
        def transform(self,field_val,domain,codomain):
            current_plan_and_info=self.get_plan_and_info(domain,codomain)
            ## Prepare the input data
            field_val*=current_plan_and_info.get_codomain_centering_mask()
            ## Define a abbreviation for the fftw plan                
            p = current_plan_and_info.get_plan()
            ## load the field into the plan
            if p.has_input:
                p.input_array[:] = field_val
            ## execute the plan
            p()
            return p.output_array*current_plan_and_info.get_domain_centering_mask()
        
    
elif fft_machine == 'gfft' or 'gfft_fallback':
    class fft_gfft(fft):                    
        def transform(self,field_val,domain,codomain):
            naxes = (np.size(domain.para)-1)//2
            if(codomain.fourier):
                ftmachine = "fft"
            else:
                ftmachine = "ifft"       
            ## transform and return
            if(domain.datatype==np.float64):
                return gfft.gfft(field_val.astype(np.complex128),in_ax=[],out_ax=[],ftmachine=ftmachine,in_zero_center=domain.para[-naxes:].astype(np.bool).tolist(),out_zero_center=codomain.para[-naxes:].astype(np.bool).tolist(),enforce_hermitian_symmetry=bool(codomain.para[naxes]==1),W=-1,alpha=-1,verbose=False)
            else:
                return gfft.gfft(field_val,in_ax=[],out_ax=[],ftmachine=ftmachine,in_zero_center=domain.para[-naxes:].astype(np.bool).tolist(),out_zero_center=codomain.para[-naxes:].astype(np.bool).tolist(),enforce_hermitian_symmetry=bool(codomain.para[naxes]==1),W=-1,alpha=-1,verbose=False)
    