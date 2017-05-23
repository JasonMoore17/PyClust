""" 
    This module handles training data obtained from PyClust.
    labeled data is stored in a directory tree structure rooted at 
    PyClust/data/clf_data.
"""

import os
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
import features

PEAK_INDEX = 17
CLASS_P = 1
CLASS_I = 2
CLASS_J = 3
ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'clf_data')

############################################################################
# Extra Features
############################################################################

# if the original waveform wv has 60 timestamps, the output is
# a waveform with 120 timestamps
def double_resolution(wv, dt_ms):                                               
    wv_inbetween = []                                                           
    for i in range(wv.size - 1):                                                
        wv_inbetween.append((wv[i] + wv[i + 1]) / 2.)                           

    wv_inbetween.append(wv[-1] + (wv[-1] - wv_inbetween[-1]))                   
    wv_inbetween = np.array(wv_inbetween)                                       
    wv_new = np.array([[wv], [wv_inbetween]]).transpose().flatten()             
    return wv_new, dt_ms / 2.


# compute full width at half maximum (FWHM)
def get_fwhm(wv, dt_ms):                                                
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

        print('indices for side: ' + side)
        print(indices)

        for i in indices:                                               
            if i < len(wv) - 1:                                         
                ub = abs(wv[i + 1] - wv[i])  # upper bound              
            else:                                                       
                raise IndexError('index ' + str(i) + ' could not find upper bound for side ' + side)

            if i > 0:                                                   
                lb = abs(wv[i] - wv[i - 1])   # lower bound             
            else:                                                       
                raise IndexError('index ' + str(i) + ' could not find lower bound for side ' + side)

            if wv[i] - lb <= hm <= wv[i] + ub:                          
                return i                                                

        raise IndexError('FWHM calculation cannot find index in ' + side)

    lhm_index = get_hm_index('l')                                       
    rhm_index = get_hm_index('r')                                       
    return (rhm_index - lhm_index) * dt_ms


############################################################################

# predict the label to save
def get_label(cluster, dt_ms):
    # if the original waveform wv has 60 timestamps, the output is
    # a waveform with 120 timestamps
    def double_resolution(wv, dt_ms):
        wv_inbetween = []
        for i in range(wv.size - 1):
            wv_inbetween.append((wv[i] + wv[i + 1]) / 2.)
        wv_inbetween.append((wv_inbetween[wv_inbetween.size - 1] 
                - wv_inbetween[wv_inbetween.size - 2]) 
                + wv_inbetween[wv_inbetween.size - 1])
        wv_inbetween = np.array(wv_inbetween)
        wv_new = np.array([[wv], [wv_inbetween]]).transpose().flatten()
        return (wv_new, dt_ms / 2.)

    # compute full width at half maximum (FWHM)
    def get_fwhm(wv, dt_ms):
        argmax = np.argmax(wv)

        # align waveform to 0 for baseline left of amplitude
        min = np.min(wv[:argmax])
        voffset = np.vectorize(lambda x: x - min)
        wv = voffset(wv)
        max = np.amax(wv)

        vdist = np.vectorize(lambda x: abs(max / 2. - x))
        argLhm = np.argmin(vdist(wv[:argmax]))
        argRhm = np.argmin(vdist(wv[argmax:])) + argmax
        return (argRhm - argLhm) * dt_ms

    # compute time from peak to valley
    def get_p2vt(wv, dt_ms):
        peakIndex = np.argmax(wv)
        valleyIndex = np.argmin(wv[peakIndex:]) + peakIndex
        return (valleyIndex - peakIndex) * dt_ms

    chans = cluster.wv_mean.shape[1]
    counts = np.zeros(3)

    for chan in range(chans):
        wv = cluster.wv_mean[:, chan]
        wv2xres, dt_ms2xres = double_resolution(wv, dt_ms)
        fwhm = get_fwhm(wv2xres, dt_ms2xres)
        p2vt = get_p2vt(wv2xres, dt_ms2xres)
        if cluster.stats['csi'] == np.NAN:
            return 0  # unlabeled
        elif cluster.stats['csi'] > 10:  # Pyramidal
            if 1.6 * fwhm + p2vt > 0.95:
                counts[1] += 1  # Pyramidal
            else:
                counts[0] += 1  # unlabeled
        else:
            if 1.6 * fwhm + p2vt < 0.95:
                counts[2] += 1  # Interneuron
            else:
                counts[0] += 1  # unlabeled

    if counts[1] > counts[2]:
        return 1
    elif counts[2] > counts[1]:
        return 2
    else:
        return 0


# Load all clustr mean data from PyClust/data/clf_data
# returns (X, y) where X is n x d matrix of attributes and y is n x 1 vector of labels
def load_data(target_path='', target_file=''):
    data = None

    # look through entire directory tree rooted at 'clf_data/<target_path>'
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, target_path)):
        if target_file == '':
            csvs = filter(lambda f: f.endswith('.csv'), filenames)
            for fname in csvs:
                if data is None:
                    data = np.loadtxt(os.path.join(dirpath, fname), delimiter=',', skiprows=0)
                else:
                    data = np.append(data, np.loadtxt(os.path.join(dirpath, fname), delimiter=','),
                                     axis=0)

        else:
            for fname in filter(lambda f: f == target_file, filenames):
                data = np.loadtxt(os.path.join(dirpath, fname), delimiter=',', skiprows=0)

    if data is None:
        return None
    else:
        X = np.delete(data, 0, axis=1)
        y = data[:,0].astype(int)

    return X, y


# compute cross-validation error of classifier
# X, y = data attributes and labels
# nfolds = number of stratified folds
# clf = classifier object
def get_cv_error(X, y, clf, n_splits=5):
    # stratified k-folds
    skf = StratifiedKFold(n_splits=n_splits)
    total_error = 0.0
    n_trials = 0
    for train_indices, test_indices in skf.split(X, y):
        n_trials += 1
        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # count number of mispredictions from current fold
        n_mispreds = 0.0
        for i in range(y_test.size):
            if not y_test[i] == y_pred[i]:
                # add penalty for misprediction
                n_mispreds += 1.0

        # add to total error
        total_error += n_mispreds / float(y_test.size)
        
    return total_error / float(n_trials)


# Compute optimal SVM hyperparameter C using cross-validation
# Returns: opt_c = optimal C, min_error = min error for opt_c
def get_opt_c(X, y, kernel='linear'):
    c_range = [10.0 ** i for i in range(-5, 5)]

    try:
        min_error = float('inf')
    except:
        min_error = 1e30000

    for c in c_range:
        clf = SVC(C=c, kernel=kernel)
        error = get_cv_error(X, y, clf)
        #print('c: ', c)
        #print('error: ', error)

        if error < min_error:
            min_error = error
            opt_c = c

    return opt_c, min_error


# Compute the error of training classifier 'clf' on training data
# 'X_train', 'y_train'. Test on samples 'X_test', 'y_test'.
def get_error(X_test, y_test, clf):
    y_pred = clf.predict(X_test)

    # count number of mispredictions
    n_mispreds = 0.0
    for i in range(y_test.size):
        if not y_pred[i] == y_test[i]:
            n_mispreds += 1.0
    
    return n_mispreds / float(y_test.size)


def normalize(X, mode='total', max_peak=None):
    if mode == 'total':
        # each spike adjusts by the max peak of all spikes
        if max_peak is None:
            max_peak = np.amax(map(lambda row: np.amax(row), X))
        return np.array(map(lambda row: map(lambda x: x / max_peak, row), X)), max_peak

    elif mode == 'each':
        # each spike adjusts by the max peak of that spike; i.e. each amplitude
        # is scaled to 1
        def div_row_by(row, factor):
            if factor == 0.:
                return row
            else:
                return map(lambda x: x / factor, row)

        #return np.array(map(lambda row: map(lambda x: x / np.amax(row), row), X))
        return np.array(map(lambda row: div_row_by(row, np.amax(row)), X))

    else:
        raise ValueError('invalid argument for parameter "mode"')


def save_to_file(X, y, fname, fprefix=''):
    rows = np.array(map(lambda X_row, y_label: np.roll(np.append(X_row, y_label), 1), X, y))

    fpath = os.path.join(ROOT, fprefix)
    if not os.path.exists(fpath):
        os.makedirs(fpath)

    with open(os.path.join(fpath, fname), 'w') as f:
        np.savetxt(f, rows, fmt='%g', delimiter=',', header='label,waveform')
    return True


if __name__ == '__main__':

    X, y = load_data('raw/means')
    #print(np.arange(0, 60, 0.5))
    #print(double_resolution(X[i], 1.))
    for i in range(3):
        max = np.amax(X[i])
        plt.figure()
        plt.title('mean')
        plt.plot(np.arange(0, 60, 0.5), double_resolution(X[i], 1.)[0])
        plt.plot(np.arange(0, 60, 0.5), [max / 2. for i in range(120)], 'r')
        plt.show()
        fwhm = get_fwhm(X[i], 1.)
        print('fwhm: ' + str(fwhm))
       
    #for i in range(3):
    #    #fwhm = get_fwhm(X[i], 1.)
    #    #print('fwhm' + str(fwhm))
    #    plt.figure()
    #    plt.title('raw')
    #    #plt.plot(range(X_train.shape[1]), X_train[i])
    #    plt.plot(range(X.shape[1]), X[i])
    #    plt.figure()
    #    plt.title('raw - double resolution')
    #    row, _ = double_resolution(X[i], 1.)
    #    plt.plot(range(row.size), row)
    #    plt.show()


    ## switch to show plots
    #verbose = False

    #print_count = True

    #for mode in ['raw', 'normalized-by-max', 'normalized-by-each']:
    #    # load, fit, and predict
    #    print'mode: ' + mode
    #    print('loading data')
    #    if mode == 'raw':
    #        X_train, y_train = load_data('raw/means')
    #        X_test, y_test = load_data('raw/members')
    #    elif mode == 'normalized-by-max':
    #        X_train, y_train = load_data('normalized/means', 'normalized_means_by_total.csv')
    #        X_test, y_test = load_data('normalized/members', 'normalized_members_by_total.csv')
    #    elif mode == 'normalized-by-each':
    #        X_train, y_train = load_data('normalized/means', 'normalized_means_by_each.csv')
    #        X_test, y_test = load_data('normalized/members', 'normalized_members_by_each.csv')
    #    else:
    #        raise ValueError

    #    print('X_test.shape: ' + str(X_test.shape))
    #    print('y_test.shape: ' + str(y_test.shape))
    #    X_test = X_test[500:2500]
    #    y_test = y_test[500:2500]

    #    print('X_test.shape: ' + str(X_test.shape))
    #    print('y_test.shape: ' + str(y_test.shape))

    #    if verbose:
    #        for i in range(3):
    #            plt.figure()
    #            plt.title('raw')
    #            #plt.plot(range(X_train.shape[1]), X_train[i])
    #            plt.plot(range(X_test.shape[1]), X_test[i])
    #            plt.show()


    #    ########################################################################
    #    # SVC fitting
    #    ########################################################################
    #    for kern in ['linear', 'rbf']:
    #        print('kernel: ' + kern)
    #        print('optimizing classifier fit')
    #        opt_c, min_error = get_opt_c(X_train, y_train, kern)
    #        print('min CV error with training data: ', min_error)
    #        clf = SVC(C=opt_c, kernel=kern)
    #        clf.fit(X_train, y_train)

    #        # total error for random sample members
    #        error = get_error(X_test, y_test, clf)
    #        print('error with normalized (total): ', error)

    #        # total error for random exclusive P, I sample members
    #        get_p_indices = np.vectorize(lambda x: x == 1)
    #        get_i_indices = np.vectorize(lambda x: x == 2)
    #        i_indices = get_p_indices(y_test)
    #        p_indices = get_i_indices(y_test)
    #        if print_count:
    #            i_count = len(filter(lambda x: x == True, i_indices))
    #            p_count = len(filter(lambda x: x == True, p_indices))
    #            print('I count: ' + str(i_count))
    #            print('P count: ' + str(p_count))
    #        X_test_p = X_test[p_indices]
    #        X_test_i = X_test[i_indices]
    #        y_test_p = y_test[p_indices]
    #        y_test_i = y_test[i_indices]

    #        error_p = get_error(X_test_p, y_test_p, clf)
    #        error_i = get_error(X_test_i, y_test_i, clf)

    #        print('p error: ', error_p)
    #        print('i error: ', error_i)
        

    ########################################################################
    # PCA stuff
    ########################################################################

    #print('opt_c: ', opt_c)
    #print('min_error: ', min_error)

    #clf = SVC(kernel='rbf')
    #clf = SVC(kernel='linear')
    #error_rate = get_cv_error(X, y, clf)
    #print('error_rate: ', error_rate)
    #get_opt_c(X, y, clf)

    #n_components = 10
    #PCA = decomposition.KernelPCA(n_components=n_components, kernel='rbf')
    #Z = PCA.fit_transform(X)

    #get_p_indices = np.vectorize(lambda x: x == 1)
    #get_i_indices = np.vectorize(lambda x: x == 2)
    #Zp = Z[get_p_indices(y)]
    #Zi = Z[get_i_indices(y)]

    #for i in np.arange(0, n_components - 1, 2):
    #    plt.figure()
    #    plt.xlabel('Principle Component ' + str(i))
    #    plt.ylabel('Principle Component ' + str(i + 1))
    #    plt.plot(Zp[:, i], Zp[:, i + 1], 'ro', Zi[:, i], Zi[:, i + 1], 'bo')
    #    plt.show()
    #print(Z)
