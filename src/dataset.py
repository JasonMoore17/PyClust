""" 
    This module handles training data obtained from PyClust.
    labeled data is stored in a directory tree structure rooted at 
    PyClust/data/clf_data.
"""

import os
import numpy as np
import sys
import spikeset


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
        self.added = set()

    # returns the directory path for the new data
    def get_file_path(self):
        if self.subject == None or self.session == None:
            return None
        return os.path.join(self.root, self.subject, self.session)

    # Creates path for new file if it does not exist
    def make_path(self):
        path = self.get_file_path()
        if not os.path.exists(path):
            os.makedirs(path)

    # saves labeled cluster members to file ; returns success or failure
    def cluster_to_file(self, ss, clust, label, fname=None):
        pathname = self.get_file_path()
        if not pathname:
            return False

        if label < 1 or label > 3:
            return False

        cur_path = os.path.dirname(__file__)
        raw_data_path = os.path.join(cur_path, '..', '..', 'data')
        labeled_data_path = os.path.join(cur_path, 'classifier', 'data')

        if fname == None:
            fname = os.path.join(pathname, self.fname + '.csv')

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

        with open(fname, 'a') as f:
            np.savetxt(f, rows, fmt='%.1e', delimiter=',', header='label,waveform')
        return True
