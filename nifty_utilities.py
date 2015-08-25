# -*- coding: utf-8 -*-

import numpy as np

def hermitianize(x):
    ## make the point inversions
    flipped_x = _hermitianize_inverter(x)
    flipped_x = flipped_x.conjugate()
    ## check if x was already hermitian        
    if (x == flipped_x).all():
        return x
    ## average x and flipped_x. 
    ## Correct the variance by multiplying sqrt(0.5)
    x = (x + flipped_x) * np.sqrt(0.5)
    ## The fixed points of the point inversion must not be avaraged.
    ## Hence one must multiply them again with sqrt(0.5)
    ## -> Get the middle index of the array
    mid_index = np.array(x.shape, dtype=np.int)//2
    dimensions = mid_index.size
    ## Use ndindex to iterate over all combinations of zeros and the
    ## mid_index in order to correct all fixed points.
    for i in np.ndindex((2,)*dimensions):
        temp_index = tuple(i*mid_index)
        x[temp_index] *= np.sqrt(0.5)
    try:
        x.hermitian = True
    except(AttributeError):
        pass
        
    return x



def _hermitianize_inverter(x):
    ## calculate the number of dimensions the input array has
    dimensions = len(x.shape)
    ## prepare the slicing object which will be used for mirroring
    slice_primitive = [slice(None),]*dimensions
    ## copy the input data
    y = x.copy()
    ## flip in every direction
    for i in xrange(dimensions):
        slice_picker = slice_primitive[:]
        slice_picker[i] = slice(1, None,None)
        slice_inverter = slice_primitive[:]
        slice_inverter[i] = slice(None, 0, -1)

        try:
            y.inject(to_slices=slice_picker, data=y, 
                     from_slices=slice_inverter)
        except(AttributeError):
            y[slice_picker] = y[slice_inverter]
    return y
    
    
def direct_dot(x, y):
    ## the input could be fields. Try to extract the data
    try:
        x = x.get_val()
    except(AttributeError):
        pass
    try:
        y = y.get_val()
    except(AttributeError):
        pass
    ## try to make a direct vdot
    try:
        return x.vdot(y)
    except(AttributeError):
        pass
    
    try:
        return y.vdot(x)
    except(AttributeError):
        pass        

    ## fallback to numpy 
    return np.vdot(x, y)    
    
    
def convert_nested_list_to_object_array(x):
    ## if x is a nested_list full of ndarrays all having the same size,
    ## np.shape returns the shape of the ndarrays, too, i.e. too many 
    ## dimensions
    possible_shape = np.shape(x)
    ## Check if possible_shape goes too deep.
    dimension_counter = 0    
    current_extract = x
    for i in xrange(len(possible_shape)):
        if isinstance(current_extract, list) == False and\
            isinstance(current_extract, tuple) == False:
            break
        current_extract = current_extract[0]
        dimension_counter += 1
    real_shape = possible_shape[:dimension_counter]
    ## if the numpy array was not encapsulated at all, return x directly    
    if real_shape == ():
        return x
    ## Prepare the carrier-object    
    carrier = np.empty(real_shape, dtype = np.object)
    for i in xrange(np.prod(real_shape)):
        ii = np.unravel_index(i, real_shape)
        try:        
            carrier[ii] = x[ii]
        except(TypeError):
            extracted = x
            for j in xrange(len(ii)):                    
                extracted = extracted[ii[j]]
            carrier[ii] = extracted     
    return carrier


def field_map(ishape, function, *args):
    if ishape == ():
        return function(*args)
    else:
        if args == ():
            result = np.empty(ishape, dtype=np.object)
            for i in xrange(np.prod(ishape)):
                ii = np.unravel_index(i, ishape)
                result[ii] = function()
            return result
        else:
            ## define a helper function in order to clip the get-indices
            ## to be suitable for the foreign arrays in args.
            ## This allows you to do operations, like adding to fields 
            ## with ishape (3,4,3) and (3,4,1)
            def get_clipped(w, ind):
                w_shape = np.array(np.shape(w))
                get_tuple = tuple(np.clip(ind, 0, w_shape-1))
                return w[get_tuple]
            result = np.empty_like(args[0])
            for i in xrange(np.prod(result.shape)):
                ii = np.unravel_index(i, result.shape)
                result[ii] = function(*map(
                                        lambda z: get_clipped(z, ii), args)
                                      )
                #result[ii] = function(*map(lambda z: z[ii], args))
            return result    