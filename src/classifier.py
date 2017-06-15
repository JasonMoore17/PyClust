import os
import numpy as np
from sklearn import decomposition
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold

import features
import features2

class LoadingError(Exception):
    pass

PEAK_INDEX = 17
CLASS_P = 1
CLASS_I = 2
CLASS_J = 3
ROOT = os.path.join(os.path.dirname(__file__), '..', 'data', 'clf_data')


# Load all cluster mean data from PyClust/data/clf_data
# returns (X, y) where X is n x d matrix of attributes and y is n x 1 vector of labels
def load_data(target_path='', target_file='', verbose=False):
    data = None

    if not target_file == '':
        if target_path == '':
            raise ValueError('target_path must be nonempty if target_file is nonempty')

    # look through entire directory tree rooted at 'clf_data/<target_path>'
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, target_path)):
        if target_file == '':
            csvs = filter(lambda f: f.endswith('.csv'), filenames)
            for fname in csvs:
                src = os.path.join(dirpath, fname)

                if verbose:
                    print('loading from ' + src)

                if data is None:
                    data = np.loadtxt(src, delimiter=',', skiprows=2)
                    if data.shape == (0,):
                        data = None
                else:
                    append_data = np.loadtxt(src, delimiter=',', skiprows=2)
                    if append_data.ndim <= 1:
                        append_data = np.array([append_data])

                    data = np.append(data, append_data, axis=0)
        else:
            if target_file in filenames:
                data = np.loadtxt(os.path.join(dirpath, target_file), delimiter=',', skiprows=2)
                break

    if data is None:
        raise LoadingError('Unable to load data with parameters ' + str((target_path, target_file)))
    else:
        X = np.delete(data, 0, axis=1)
        y = data[:,0].astype(int)

    return X, y


def load_dt_ms(path, fname):
    with open(os.path.join(ROOT, path, fname)) as f:
        for i, row in enumerate(f):
            if i == 1:
                return float(row)


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


def get_test_error(X_test, y_test, clf):
    err_all = get_error(X_test, y_test, clf)
    p_indices = np.array(map(lambda y_i: y_i == CLASS_P, y_test))
    i_indices = np.array(map(lambda y_i: y_i == CLASS_I, y_test))
    err_p = get_error(X_test[p_indices], y_test[p_indices], clf)
    err_i = get_error(X_test[i_indices], y_test[i_indices], clf)
    return err_all, err_p, err_i


##############################################################################
# Top Modules
##############################################################################

def get_opt_clf(kernel='linear'):
    attrname = 'bsln_norm'
    trainroot = os.path.join(attrname, 'means')
    X_train, y_train = load_data(trainroot)
    min_cv_error = float('inf')

    if kernel == 'linear':
        param_range = [1]  # place holder
    elif kernel == 'rbf' or kernel == 'sigmoid':
        param_range = [1]  # place holder
        #param_range = [10.0 ** i for i in range(-2, 8)]
    #    param_range = [10.0]
    elif kernel == 'poly':
        param_range = np.arange(2, 10, 1)

    for c in [10.0 ** i for i in range(-5, 5)]:
        for param in param_range:
            if kernel == 'poly':
                clf = SVC(kernel='linear', C=c, degree=param)
            else:
                clf = SVC(kernel='linear', C=c)
            cv_error = get_cv_error(X_train, y_train, clf)
            if cv_error < min_cv_error:
                min_cv_error = cv_error
                opt_c = c
                #if not kernel == 'linear':
                #    opt_param = param

    #print('optimal parameters for ' + kernel + ' classifier: ')
    if kernel == 'linear':
        clf = SVC(kernel='linear', C=opt_c)
        print('C=' + str(opt_c))
    elif kernel == 'rbf' or kernel == 'sigmoid':
        clf = SVC(kernel=kernel, C=opt_c)
        print('C=' + str(opt_c))
    elif kernel == 'poly':
        clf = SVC(kernel='poly', C=opt_c, degree=param)
        print('C=' + str(opt_c))
        print('degree=' + str(param))
    else:
        raise ValueError('invalid kernel')

    clf.fit(X_train, y_train)
    return clf
    

# since there are C channels for each spike, we take a vote to get its class
def get_indices(clf, spikes):
    def normalize_spike(wv):
        amax = np.amax(wv)
        return np.array(map(lambda x: x / amax, wv))

    N = spikes.shape[0]
    C = spikes.shape[2]
    p_counts = np.zeros((N, C))
    i_counts = np.zeros((N, C))
    for c in range(C):
        X = spikes[:, :, c]
        X = np.array(map(lambda wv: normalize_spike(wv), X))
        y_pred = clf.predict(X)
        for n in range(N):
            if y_pred[n] == CLASS_P:
                p_counts[n, c] += 1
            elif y_pred[n] == CLASS_I:
                i_counts[n, c] += 1
            else:
                raise IndexError
    p_votes_by_chan = np.sum(p_counts, axis=1)
    i_votes_by_chan = np.sum(i_counts, axis=1)

    thresh = C / 2
    p_inds = np.array(map(lambda x: x > thresh, p_votes_by_chan))
    i_inds = np.array(map(lambda x: x > thresh, i_votes_by_chan))
    j_inds = np.array(map(lambda p, i: not p and not i, p_inds, i_inds))

    return p_inds, i_inds, j_inds


# testing
if __name__ == '__main__':
    trainroot = os.path.join('bsln_norm', 'means')
    X_test, y_test = load_data(trainroot)


    #for kernel in ['linear', 'rbf', 'poly', 'sigmoid']:
    #    if kernel == 'poly':
    #        PCA = decomposition.KernelPCA(n_components=8, kernel=kernel, degree=9)
    #    else:
    #        PCA = decomposition.KernelPCA(n_components=8, kernel=kernel)
    #    Z = PCA.fit_transform(X_test)

    #    get_p_indices = np.vectorize(lambda x: x == 1)
    #    get_i_indices = np.vectorize(lambda x: x == 2)
    #    Zp = Z[get_p_indices(y_test)]
    #    Zi = Z[get_i_indices(y_test)]

    #    for i in np.arange(0, 8, 2):
    #        plt.figure()
    #        plt.title('kernel: ' + kernel)
    #        plt.xlabel('Principle Component ' + str(i))
    #        plt.ylabel('Principle Component ' + str(i + 1))
    #        plt.plot(Zp[:, i], Zp[:, i + 1], 'ro', Zi[:, i], Zi[:, i + 1], 'bo')
    #        plt.show()


    #p_inds = np.array(map(lambda x: x == CLASS_P, y_test))
    #i_inds = np.array(map(lambda x: x == CLASS_I, y_test))
    #print('P count: ' + str(sum(p_inds)))
    #print('I count: ' + str(sum(i_inds)))

    testroot = os.path.join('test', 'bsln_norm', 'members')
    X_test, y_test = load_data(testroot)
    print('X_test: ' + str(X_test.shape))
    kernels = ['linear', 'rbf', 'poly', 'sigmoid']
    errs_all = []
    errs_p = []
    errs_i = []
    for kernel in kernels:
        clf = get_opt_clf(kernel)
        print('kernel: ' + kernel)
        err_all, err_p, err_i  = get_test_error(X_test, y_test, clf)
        print('test error: ' + str(err_all))
        print('P error: ' + str(err_p))
        print('I error: ' + str(err_i))
        print('')
        errs_all.append(err_all)
        errs_p.append(err_p)
        errs_i.append(err_i)

    plt.bar(np.arange(len(kernels)), errs_all, align='center', alpha=0.5)
    plt.title('Overall misclassification rate')
    plt.xticks(np.arange(len(kernels)), kernels)
    plt.show()

    plt.bar(np.arange(len(kernels)), errs_p, align='center', alpha=0.5)
    plt.title('P misclassification rate')
    plt.xticks(np.arange(len(kernels)), kernels)
    plt.show()

    plt.bar(np.arange(len(kernels)), errs_i, align='center', alpha=0.5)
    plt.title('I misclassification rate')
    plt.xticks(np.arange(len(kernels)), kernels)
    plt.show()

