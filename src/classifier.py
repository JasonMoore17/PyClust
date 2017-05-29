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
def load_data(target_path='', target_file=''):
    data = None

    if not target_file == '':
        if target_path == '':
            raise ValueError('target_path must be nonempty if target_file is nonempty')

    # look through entire directory tree rooted at 'clf_data/<target_path>'
    for dirpath, dirnames, filenames in os.walk(os.path.join(ROOT, target_path)):
        if target_file == '':
            csvs = filter(lambda f: f.endswith('.csv'), filenames)
            for fname in csvs:
                if data is None:
                    data = np.loadtxt(os.path.join(dirpath, fname), delimiter=',', skiprows=2)
                else:
                    data = \
                    np.append(data, np.loadtxt(os.path.join(dirpath, fname), 
                            delimiter=',', skiprows=2), axis=0)
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
       

