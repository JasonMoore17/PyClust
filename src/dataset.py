""" 
    This module handles training data obtained from PyClust.
    labeled data is stored in a directory tree structure rooted at 
    PyClust/data/clf_data.
"""

import os
import numpy as np

class Dataset:
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
    def cluster_to_file(self, ss, clust, clust_num, label, fname=None):
        pathname = self.__get_file_path()
        if not pathname:
            return False
        if label < 1 or label > 3:
            return False
        if not os.path.exists(pathname):
            os.makedirs(pathname)

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
            fpathname = os.path.join(pathname, self.fname + '.csv')

        with open(fpathname, 'a') as f:
            np.savetxt(f, rows, fmt='%g', delimiter=',', header='label,waveform')

        self.saved.add((self.subject, self.session, self.fname, clust_num))
        return True
