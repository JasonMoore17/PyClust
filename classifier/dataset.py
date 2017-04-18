""" 
    This module handles training data obtained from PyClust
"""

import os
import numpy as np

class DataSet:
    def __init__(self):
        # 'PyClust/classifier/data'
        self.root = os.path.join(os.path.dirname(__file__), '..', 'classifier', 'data')

        # used for directory structure of data files under classifier/data
        self.subject = None
        self.session = None
        self.file = None

        # keeps track of which file and cluster has been added
        self.added = set()


# saves cluster members to file
def cluster_to_file(ss, clust, label, fname):
    labels = {'P': 0, 'I': 1, 'J': 2}
    cur_path = os.path.dirname(__file__)
    raw_data_path = os.path.join(cur_path, '..', '..', 'data')
    labeled_data_path = os.path.join(cur_path, 'classifier', 'data')

    file = open(fname, 'a')

    clust_spikes = ss.spikes[clust.member]
    rows = []
    for i in range(clust_spikes.shape[0]):
        for c in range(clust_spikes.shape[2]):
            row = clust_spikes[i, :, c].tolist().add(label)
            row = np.roll(row, 1)  # make label show first
            rows.add(row)
    rows = np.array(rows)

    np.savetxt(fname, delimiter=',', header='label,waveform')

