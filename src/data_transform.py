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
def io_transform(func, srcroot, destroot):
    print('ROOT: ' + ROOT)
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, srcroot)):
        csvs = filter(lambda f: f.endswith('.csv'), filenames)              
        for fname in csvs:                                                  
            fpath = os.path.join(dirpath, fname)
            print('fpath: ' + fpath)
            X, y = classifier.load_data(srcroot)
            X_new = func(X)

            subj_sess = os.path.split(fpath[len(ROOT) + 1 + len(srcroot):])[0]
            print('subj_sess: ' + subj_sess)
            
            
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
