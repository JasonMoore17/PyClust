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


# predict the label to save
def get_label(cluster, dt_ms):
    # if the original waveform wv has 60 timestamps, the output is
    # a waveform with 120 timestamps
    def double_resolution(wv, dt_ms):
        wv_inbetween = []
        for i in range(wv.size - 1):
            wv_inbetween.append((wv[i] + wv[i + 1]) / 2.)
        wv_inbetween.append(wv[wv.size - 1])
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

# classifier data
class Dataset:
    P = 1
    I = 2
    J = 3
    def __init__(self, subject=None, session=None, fname=None):
        # 'PyClust/data/clf_data'
        self.root = os.path.join(os.path.dirname(__file__), '..', 'data', 
                'clf_data')

        # used for directory structure of data files under classifier/data
        self.subject = subject
        self.session = session
        self.fname = fname

        # keeps track of which file and cluster has been added
        self.saved = set()

    # returns the directory path for the new data
    def __get_file_path(self):
        if self.subject == None or self.session == None:
            return None
        return os.path.join(self.root, self.subject, self.session)

    # Creates path for new file if it does not exist
    def make_path(self):
        path = self.__get_file_path()
        if not os.path.exists(path):
            os.makedirs(path)

    def is_saved(self, subject, session, fname, clust_num):
        return (subject, session, fname, clust_num) in self.saved

    # saves labeled cluster members to file ; returns success or failure
    def cluster_to_file(self, clust, clust_num, label, fname=None):

        pathname = self.__get_file_path()
        if not pathname:
            return False
        if label < 1 or label > 3:
            return False
        if not os.path.exists(pathname):
            os.makedirs(pathname)

        wv_mean = clust.wv_mean
        rows = []
        for chan in range(wv_mean.shape[1]):
            row = wv_mean[:, chan]
            #plt.plot(range(row.size), row)
            #plt.show()
            listrow = row.tolist()
            listrow.append(float(label))
            row = np.array(listrow)
            row = np.roll(row, 1)  # make label show first
            rows.append(row)
        rows = np.array(rows)


        if fname == None:
            fpathname = os.path.join(pathname, self.fname + '.csv')

        with open(fpathname, 'a') as f:
            if os.path.exists(fpathname):
                np.savetxt(f, rows, fmt='%g', delimiter=',')
            else:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header='label,waveform')

        self.saved.add((self.subject, self.session, self.fname, clust_num))
        return True

# Load all data from PyClust/data/clf_data
# returns (X, y) where X is n x d matrix of attributes and y is n x 1 vector of labels
def load_data():
    root = os.path.join(os.path.dirname(__file__), '..', 'data', 'clf_data')
    data = None
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('.csv'):
            #if fname in ['TT5.csv', 'TT7.csv', 'TT10.csv', 'TT13.csv', 'TT17.csv']:
            #if fname in ['TT7.csv', 'TT10.csv']:
                if data is None:
                    data = np.loadtxt(os.path.join(dirpath, fname), delimiter=',', skiprows=0)
                else:
                    data = np.append(data, np.loadtxt(os.path.join(dirpath, fname), delimiter=','),
                                     axis=0)

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
def cv_error(X, y, clf, n_splits=5):
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

if __name__ == '__main__':
    X, y = load_data()
    print('X.shape:', X.shape)
    print('y.shape:', y.shape)
    #clf = SVC(kernel='rbf')
    clf = SVC(kernel='linear')
    error_rate = cv_error(X, y, clf)
    print('error_rate: ', error_rate)

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
