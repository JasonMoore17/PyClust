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

class DataSaver:

    def __init__(self, subj=None, sess=None, tt=None):
        # 'PyClust/data/clf_data'
        self.root = os.path.join(os.path.dirname(__file__), '..', 'data',
                'clf_data')

        # used for directory structure of data files under classifier/data
        self.subj = subj
        self.sess = sess
        self.tt = tt 

        # keeps track of which file and cluster has been added
        self.saved= set()

    # clust is the cluster index
    def is_saved(self, clust_num):
        return (self.subj, self.sess, self.tt, clust_num) in self.saved

    def __get_attrs(self, clust, ss):
        raise NotImplementedError('subclass must override this method')

    def get_attr_type(self):
        raise NotImplementedError('subclass must override this method')

    def save_cluster(self, clust, clust_num, ss, label):
        if label not in [CLASS_P, CLASS_I, CLASS_J]:
            raise ValueError('invalid label')

        if label not in [CLASS_P, CLASS_I, CLASS_J]:
            raise ValueError('invalid label')

        path = os.path.join(self.root, self.get_attr_type(), self.subj, self.sess)
        if not os.path.exists(path):
            os.makedirs(path)

        fpath = os.path.join(path, self.tt + '.csv')
        attrs = self.__get_attrs(clust, ss)
        with open(fpath, 'a') as f:
            if os.path.exists(fpath):
                np.savetxt(f, attrs, fmt='%g', delimiter=',')
            else:
                np.savetxt(f, attrs, fmt='%g', delimiter=',', header='label,waveform')

        self.saved.add((self.subj, self.sess, self.tt, clust_num))


class DataSaver_ClusterMeans(DataSaver):
    def get_attr_type(self):
        return 'cluster means'

    def __get_attrs(self, clust, ss):
        wv_mean = clust.wv_mean                                             
        rows = []                                                           
        for chan in range(wv_mean.shape[1]):                                
            row = wv_mean[:, chan]
            rows.append(row)
        return np.array(rows)

class DataSaver_ClusterMembers(DataSaver):
    def get_attr_type(self):
        return 'cluster members'

    def __get_attrs(self, clust, ss):
        clust_spikes = ss.spikes[clust.member]
        rows = []
        for i in range(clust_spikes.shape[0]):
            for c in range(clust_spikes.shape[2]):
                row = clust_spikes[i, :, c]
                rows.append(row)
        return np.array(rows)

class DataSaver_Features(DataSaver):
    def __init__(self, ss, subj=None, sess=None, tt=None):
        DataSaver.__init__(self, subj, sess, tt)
        self.features = []
        self.features.append(features.Feature_Peak(ss))
        self.features.append(features.Feature_Energy(ss))
        self.features.append(features.Feature_Valley(ss))
        self.features.append(features.Feature_Trough(ss))
        #self.features.append(features.Feature_Fwhm(ss))
        #self.features.append(features.Feature_P2vt(ss))


class DataSaver_ClusterMeans_Features(DataSaver_Features):
    def get_attr_type(self):
        return 'cluster mean features'

    def __get_attrs(self, clust, ss):
        raise NotImplementedError


class DataSaver_ClusterMembers_Features(DataSaver_Features):
    def get_attr_type(self):
        return 'cluster member features'

    def __get_attrs(self, clust, ss):
        raise NotImplementedError


class DataSaverSet:
    def __init__(self, ss, subj=None, sess=None, tt=None):
        self.saver_list = []
        self.saver_list.append(DataSaver_ClusterMeans(subj, sess, tt))
        self.saver_list.append(DataSaver_ClusterMembers(subj, sess, tt))
        self.saver_list.append(DataSaver_ClusterMeans_Features(ss, subj, sess, tt))
        self.saver_list.append(DataSaver_ClusterMembers_Features(ss, subj, sess, tt))
        self.saver_dict = dict()
        for saver in self.saver_list:
            self.saver_dict[saver.get_attr_type()] = saver

    def get_saver(self, attrType):
        return self.saver_dict[attrType]

    def get_savers(self):
        return self.saver_list
