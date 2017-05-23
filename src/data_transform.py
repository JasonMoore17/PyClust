import numpy as np
import matplotlib.pyplot as plt
import os

import classifier

ROOT = classifier.ROOT

##################################################################
# Data Transformations
##################################################################

# X = NxD matrix of N samples with D attributes

def baseline_to_zero(X):
    def per_row(row):
        argmax = np.argmax(row)
        offset = np.amin(row[0:argmax])
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
            X, y = classifier.load_data(os.path.join(dirpath, fname))
            X_new = func(X)

            src_fpath = os.path.join(dirpath, fname)
            subj_sess = os.path.split(src_fpath[len(ROOT) + 1 + len(srcroot):])[0]

            print('saving to file: ' + os.path.join(ROOT, dstroot, subj_sess))
            dst_dirpath = os.path.join(ROOT, dstroot, subj_sess)
            if not os.path.exists(dst_dirpath):
                os.makedirs(dst_dirpath)

            rows = np.array(map(lambda X_row, y_label: np.roll(np.append(X_row, y_label), 1), X, y))
            with open(os.path.join(dst_dirpath, fname, 'w')) as f:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header='label,attributes')
            
            
if __name__ == '__main__':
    X, y = classifier.load_data('raw/means')
    io_transform(lambda x: x, 'raw/means', 'identity')
    #nplots = 3
    #for i in range(nplots):
    #    plt.figure()
    #    plt.title('Not baseline zeroed')
    #    plt.plot(range(X.shape[1]), X[i])
    #    plt.figure()
    #    plt.title('Baseline zeroed')
    #    plt.plot(range(X.shape[1]), X_bsln0[i])
    #    plt.show()
