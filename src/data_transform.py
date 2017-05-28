import numpy as np
import matplotlib.pyplot as plt
import os

import classifier

ROOT = classifier.ROOT

##################################################################
# Feature Calculation
##################################################################
def double_resolution(wv, dt_ms):                                               
    wv_inbetween = []                                                           
    for i in range(wv.size - 1):                                                
        wv_inbetween.append((wv[i] + wv[i + 1]) / 2.)                           
    wv_inbetween.append(wv[-1] + (wv[-1] - wv_inbetween[-1]))                   
    wv_inbetween = np.array(wv_inbetween)                                       
    wv_new = np.array([[wv], [wv_inbetween]]).transpose().flatten()             
    return wv_new, dt_ms / 2.

def calc_fwhm(spikes):
    def calculate_spike(wv):                                                
        dt_ms = 1.
        wv, dt_ms = double_resolution(wv, dt_ms)                   
        wv, dt_ms = double_resolution(wv, dt_ms)                   
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
        print('fhwm: ' + str((rhm_index - lhm_index) * dt_ms))
        return (rhm_index - lhm_index) * dt_ms                              

    return np.apply_along_axis(calculate_spike, 1, spikes) 


def calc_p2vt(spikes):                                              
    def calculate_spike(wv):                                                
        dt_ms = 1.
        #wv, dt_ms = double_resolution(wv, dt_ms)                   
        #wv, dt_ms = double_resolution(wv, dt_ms)                   
        print('dt_ms: ' + str(dt_ms))
        peak_index = np.argmax(wv)
        valley_index = np.argmin(wv[peak_index:]) + peak_index              
        print('peak_index: ' + str(peak_index))
        print('valley_index: ' + str(valley_index))
        print('valley_index - peak_index: ' + str(valley_index - peak_index))
        plt.figure()
        plt.plot(range(len(wv)), wv, 'o')
        plt.show()
        return (valley_index - peak_index) * dt_ms
    return np.apply_along_axis(calculate_spike, 1, spikes)


##################################################################
# Data Transformations
##################################################################

# X = NxD matrix of N samples with D attributes

def baseline_to_zero(X):
    def per_row(row):
        argmax = np.argmax(row)
        argmin = np.argmin(row[:argmax])
        offset = np.mean(row[:argmin])
        return np.array(map(lambda x: x - offset, row))

    return np.array(map(lambda row: per_row(row), X))


def normalize(X):
    def per_row(row):
        div = np.amax(row)
        return np.array(map(lambda x: x / div, row))

    return np.array(map(lambda row: per_row(row), X))


##################################################################


# convert files from srcroot using transformation func and save to destroot
def io_transform(func, srcroot, dstroot):
    print('Executing io_transform: loading from ' + srcroot + ' to ' + dstroot)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, srcroot)):
        csvs = filter(lambda f: f.endswith('.csv'), filenames)              
        for fname in csvs:                                                  
            print('loading from file: ' + os.path.join(dirpath, fname))
            X, y = classifier.load_data(dirpath, fname)
            X_new = func(X)

            dt_ms = classifier.load_dt_ms(dirpath, fname)
            print('loaded dt_ms: ' + str(dt_ms))

            subj_sess = dirpath[(len(ROOT) + 2 + len(srcroot)) :]
            src_fpath = os.path.join(dirpath, fname)

            print('subj_sess: ' + subj_sess)
            print('dstroot: ' + dstroot)

            dst_dirpath = os.path.join(ROOT, dstroot, subj_sess)
            print('saving to directory: ' + dst_dirpath)
            if not os.path.exists(dst_dirpath):
                os.makedirs(dst_dirpath)

            rows = np.array(map(lambda X_row, y_label: np.roll(np.append(X_row, y_label), 1), X_new, y))
            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, np.array([dt_ms]), fmt='%g', delimiter=',', header='dt_ms')
            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header='label,attributes')
            
            
if __name__ == '__main__':
    X, y = classifier.load_data('raw/means')
    io_transform(lambda x: x, 'raw/means', 'identity')
