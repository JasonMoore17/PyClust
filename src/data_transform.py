import numpy as np
import os

# for debugging
import matplotlib.pyplot as plt

import classifier
import features2

ROOT = classifier.ROOT
PEAK_INDEX = 17

##################################################################
# Data Transformations
##################################################################

# X = NxD matrix of N samples with D attributes

def zero_baseline(X, dt_ms=0.025):
    def per_row(row):
        argmin = np.argmin(row[:PEAK_INDEX])
        offset = np.mean(row[:argmin + 1])
        return np.array(map(lambda x: x - offset, row))

    return np.array(map(lambda row: per_row(row), X))


def normalize(X, dt_ms=0.025):
    def per_row(row):
        div = np.amax(row)
        return np.array(map(lambda x: x / div, row))

    return np.array(map(lambda row: per_row(row), X))


def bsln_norm(X, dt_ms=0.025):
    X_bsln = zero_baseline(X)
    return normalize(X_bsln)


def calc_features(X, dt_ms=0.025):
    def per_row(row):
        #peak_index = np.argmax(row)
        feats = features2.calculate_features(row, peak_index=PEAK_INDEX)
        return np.array(feats)

    return np.array(map(lambda row: per_row(row), X))


def calc_feats_naive(X, dt_ms=0.025): 
    def per_row(row):
        feats = features2.calc_feats_naive(row, dt_ms)
        return np.array(feats)
    
    return np.array(map(lambda row: per_row(row), X))


##################################################################


# convert files from srcroot using transformation func and save to destroot
def data_transform(func, srcroot, dstroot, header='label,attributes'):
    print('Executing io_transform: loading from ' + srcroot + ' to ' + dstroot)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, srcroot)):
        csvs = filter(lambda f: f.endswith('.csv'), filenames)              
        for fname in csvs:                                                  
            print('loading from file: ' + os.path.join(dirpath, fname))
            X, y = classifier.load_data(dirpath, fname)
            dt_ms = classifier.load_dt_ms(dirpath, fname)
            X_new = func(X, dt_ms=dt_ms)

            subj_sess = dirpath[(len(ROOT) + 2 + len(srcroot)) :]
            src_fpath = os.path.join(dirpath, fname)

            dst_dirpath = os.path.join(ROOT, dstroot, subj_sess)
            print('saving to directory: ' + dst_dirpath)
            if not os.path.exists(dst_dirpath):
                os.makedirs(dst_dirpath)

            print(y)
            rows = np.array(map(lambda X_i, y_i: np.append(y_i, X_i), X_new, y))

            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, np.array([dt_ms]), fmt='%g', delimiter=',', header='dt_ms')
            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header=header)

            
if __name__ == '__main__':
    #def identity(X, dt_ms):
    #    return X
    #data_transform(identity, 'raw/means', 'identity/means')
    data_transform(calc_feats_naive, 'raw/means', 'feats_naive/means', 
            header='label,peak,energy,valley,trough,fwhm,p2vt')
    data_transform(calc_feats_naive, 'raw/members', 'feats_naive/members',
            header='label,peak,energy,valley,trough,fwhm,p2vt')

    #data_transform(calc_features, 'raw/means', 'feats/means', 
    #        header='label,peak,energy,valley,trough')
    #data_transform(bsln_norm, 'raw/means', 'bsln_norm/means')
    #data_transform(calc_features, 'blsn_norm/means', 'bsln_norm_feats/means', 
    #        header='label,peak,energy,valley,trough')

    #data_transform(calc_features, 'raw/members', 'feats/members', 
    #        header='label,peak,energy,valley,trough')
    #data_transform(bsln_norm, 'raw/members', 'bsln_norm/members')
    #data_transform(bsln_norm, 'raw/means', 'bsln_norm/means')
    #data_transform(calc_features, 'bsln_norm/means', 'bsln_norm_feats/means', 
    #        header='label,peak,energy,valley,trough')
    #data_transform(calc_features, 'bsln_norm/members', 'bsln_norm_feats/members', 
    #        header='label,peak,energy,valley,trough')
    
