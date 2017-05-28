import numpy as np

# classic features

def calc_peak(wv, peak_index=None):
    return np.amax(wv)

def calc_energy(wv, peak_index=None):
    return reduce(lambda x, acc: (x * x) + acc, wv, 0)
 
def calc_valley(wv, peak_index=None):
    if peak_index is None:
        peak_index = np.argmax(wv)
    return np.amin(wv[peak_index:])

def calc_trough(wv, peak_index=None):
    if peak_index is None:
        peak_index = np.argmax(wv)
    return np.amin(wv[:peak_index])
 

 # additional features

 def calc_fwhm(wv, dt_ms, peak_index=None):
    if peak_index is None:
        peak_index = np.argmax(wv)

    hm = wv[peak_index] / 2.  # half-max
    def get_hm_index(side):                                             
        if side == 'l':                                                 
            indices = np.arange(peak_index - 1, -1, -1)                 
        elif side == 'r':                                               
            indices = np.arange(peak_index, len(wv), 1)                 
        else:                                                           
            raise ValueError('get_hm_index parameter side must = "l" or "r"')

        for i in indices:                                               
            if i < len(wv) - 1:                                         
                ub = abs(wv[i + 1] - wv[i])  # upper bound              
            else:                                                       
                raise IndexError('FWHM calculation cannot find index in ' + side)

            if i > 0:                                                   
                lb = abs(wv[i] - wv[i - 1])   # lower bound             
            else:                                                       
                raise IndexError('FWHM calculation cannot find index in ' + side)

            if wv[i] - lb <= hm <= wv[i] + ub:                          
                return i                                                
        raise IndexError('FWHM could not find index')                   

        lhm_index = get_hm_index('l')                                       
        rhm_index = get_hm_index('r')                                       
        return (rhm_index - lhm_index) * dt_ms


def calc_p2vt(wv, dt_ms, peak_index=None):
    if peak_index is None:
        peak_index = np.argmax(wv)

    valley_index = np.argmin(wv[peak_index:]) + peak_index
    return (valley_index - peak_index) * dt_ms

