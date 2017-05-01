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
class DataSaver:
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
        self.saved_mean = set()
        self.saved_members = set()

    # returns the directory path for the new data
    def __get_file_path(self):
        if self.subject == None or self.session == None:
            return None
        return os.path.join(self.root, self.subject, self.session)

    # Creates path for new file if it does not exist
    def __make_path(self):
        path = self.__get_file_path()
        if not os.path.exists(path):
            os.makedirs(path)

    def is_saved(self, subject, session, fname, clust_num, mode='mean'):
        if mode is 'mean':
            return (subject, session, fname, clust_num) in self.saved_mean
        elif mode is 'members':
            return (subject, session, fname, clust_num) in self.saved_members
        else:
            return False

    # Saves labeled mean cluster waveforms of each channel to file.
    # Returns success or failure.
    def cluster_to_file(self, clust, clust_num, label, fname=None):

        pathname = self.__get_file_path()
        if not pathname:
            return False
        if label < 1 or label > 3:
            return False

        self.__make_path()

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

        self.saved_mean.add((self.subject, self.session, self.fname, clust_num))
        return True


    # saves labeled cluster members to file
    def members_to_file(self, ss, clust, clust_num, label, fname=None):
        pathname = self.__get_file_path()
        if not pathname:
            return False
        if label < 1 or label > 3:
            return False
        
        self.__make_path()

        clust_spikes = ss.spikes[clust.member]
        rows = []
        for i in range(clust_spikes.shape[0]):
            for c in range(clust_spikes.shape[2]):
                row = clust_spikes[i, :, c]
                listrow = row.tolist()
                listrow.append(float(label))
                row = np.array(listrow)
                row = np.roll(row, 1)  # make label show first
                rows.append(row)
        rows = np.array(rows)

        if fname == None:
            fpathname = os.path.join(pathname, self.fname + '_members.csv')

        with open(fpathname, 'a') as f:
            if os.path.exists(fpathname):
                np.savetxt(f, rows, fmt='%g', delimiter=',')
            else:
                np.savetxt(f, rows, fmt='%g', delimiter=',', header='label,waveform')

        self.saved_members.add((self.subject, self.session, self.fname, clust_num))
        return True


# Load all data from PyClust/data/clf_data
# returns (X, y) where X is n x d matrix of attributes and y is n x 1 vector of labels
def load_data(csvs=None):
    root = os.path.join(os.path.dirname(__file__), '..', 'data', 'clf_data')
    data = None

    if csvs is None:
        load_all = True

    # look through entire directory tree rooted at 'clf_data'
    for dirpath, dirnames, filenames in os.walk(root):
        if load_all:
            csvs = filter(lambda f: f.endswith('.csv') and not f.endswith('_members.csv'),
                               filenames)
        for fname in csvs:
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

# load _members.csv data
def load_spikes_data():
    root = os.path.join(os.path.dirname(__file__), '..', 'data', 'clf_data')
    data = None
    for dirpath, dirnames, filenames in os.walk(root):
        for fname in filenames:
            if fname.endswith('_members.csv'):
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
def get_opt_c(X, y):
    c_range = [10.0 ** i for i in range(-5, 5)]

    try:
        min_error = float('inf')
    except:
        min_error = 1e30000

    for c in c_range:
        clf = SVC(C=c, kernel='linear')
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


if __name__ == '__main__':
    X_train, y_train = load_data()

    opt_c, min_error = get_opt_c(X_train, y_train)

    clf = SVC(C=opt_c, kernel='linear')
    clf.fit(X_train, y_train)

    X_test, y_test = load_spikes_data()

    # total error for random sample members
    error = get_error(X_test, y_test, clf)
    print('error: ', error)

    # total error for random exclusive P, I sample members
    get_p_indices = np.vectorize(lambda x: x == 1)
    get_i_indices = np.vectorize(lambda x: x == 2)
    i_indices = get_p_indices(y_test)
    p_indices = get_i_indices(y_test)
    X_test_p = X_test[p_indices]
    X_test_i = X_test[i_indices]
    y_test_p = y_test[p_indices]
    y_test_i = y_test[i_indices]

    error_p = get_error(X_test_p, y_test_p, clf)
    error_i = get_error(X_test_i, y_test_i, clf)

    print('p error: ', error_p)
    print('i error: ', error_i)

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
