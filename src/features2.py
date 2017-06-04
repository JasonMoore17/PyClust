import numpy as np

PEAK_INDEX = 17

def double_resolution(wv, dt_ms):                                               
    wv_inbetween = []                                                           
    for i in range(wv.size - 1):                                                
        wv_inbetween.append((wv[i] + wv[i + 1]) / 2.)                           
    wv_inbetween.append(wv[-1] + (wv[-1] - wv_inbetween[-1]))                   
    wv_inbetween = np.array(wv_inbetween)                                       
    wv_new = np.array([[wv], [wv_inbetween]]).transpose().flatten()             
    return wv_new, dt_ms / 2.


# classic features

def calc_peak(wv, peak_index=None):
    return np.amax(wv)

def calc_energy(wv, peak_index=None):
    #return reduce(lambda x, acc: (x * x) + acc, wv, 0)
    return np.sum(wv * wv)
 
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


# naive, because noisy cases could result in unpredictable values
def calc_fwhm_naive(wv, dt_ms):
    wv, dt_ms = double_resolution(wv, dt_ms)
    wv, dt_ms = double_resolution(wv, dt_ms)
    argmax = np.argmax(wv)
    amax = wv[argmax]
    vdist = np.vectorize(lambda x: abs(amax / 2. - x))
    argLhm = np.argmin(vdist(wv[:argmax + 1]))
    argRhm = np.argmin(vdist(wv[argmax:])) + argmax
    return (argRhm - argLhm) * dt_ms


def calc_p2vt_naive(wv, dt_ms):
    wv, dt_ms = double_resolution(wv, dt_ms)
    wv, dt_ms = double_resolution(wv, dt_ms)
    peakIndex = np.argmax(wv)
    valleyIndex = np.argmin(wv[peakIndex:]) + peakIndex
    return (valleyIndex - peakIndex) * dt_ms		

	
def calculate_features(wv, peak_index=None, special=False, dt_ms=None):
    feats = []
    feats.append(calc_peak(wv, peak_index))
    feats.append(calc_energy(wv, peak_index))
    feats.append(calc_valley(wv, peak_index))
    feats.append(calc_trough(wv, peak_index))

    if special:
        if dt_ms is None:
            raise ValueError('calculating special features requires non-None dt_ms')
        feats.append(calc_fwhm(wv, peak_index))
        feats.append(calc_p2vt(wv, peak_index))

    return feats


def calc_feats_naive(wv, dt_ms):
    feats = []
    feats.append(calc_peak(wv, PEAK_INDEX))
    feats.append(calc_energy(wv, PEAK_INDEX))
    feats.append(calc_valley(wv, PEAK_INDEX))
    feats.append(calc_trough(wv, PEAK_INDEX))
    feats.append(calc_fwhm_naive(wv, dt_ms))
    feats.append(calc_p2vt_naive(wv, dt_ms))
    return feats


