import numpy as np
import os

# for debugging
import matplotlib.pyplot as plt

import classifier
import features2

ROOT = classifier.ROOT

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


def calc_features(X):
    def per_row(row):
        peak_index = np.argmax(row)
        feats = features2.calculate_features(row, peak_index)
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
            X_new = func(X)

            dt_ms = classifier.load_dt_ms(dirpath, fname)

            subj_sess = dirpath[(len(ROOT) + 2 + len(srcroot)) :]
            src_fpath = os.path.join(dirpath, fname)

            dst_dirpath = os.path.join(ROOT, dstroot, subj_sess)
            print('saving to directory: ' + dst_dirpath)
            if not os.path.exists(dst_dirpath):
                os.makedirs(dst_dirpath)

            rows = np.array(map(lambda X_i, y_i: np.append(y_i, X_i), X_new, y))

            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, np.array([dt_ms]), fmt='%g', delimiter=',', header='dt_ms')
            with open(os.path.join(dst_dirpath, fname), 'a') as f:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header=header)
            
            
if __name__ == '__main__':
    srcroot = 'raw/means'
    dstroot = 'feats'
    data_transform(calc_features, srcroot, dstroot, header='label,peak,energy,valley,trough')
