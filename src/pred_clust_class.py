# this module is used to help predict the label to make things slightly easier
# to save data

import numpy as np

import features2
import classifier

CLASS_P = classifier.CLASS_P
CLASS_I = classifier.CLASS_I
CLASS_J = classifier.CLASS_J


# predict the label to save
def predict_label(cluster, dt_ms):
    chans = cluster.wv_mean.shape[1]
    counts = np.zeros(3)

    for chan in range(chans):
        wv = cluster.wv_mean[:, chan]
        wv2xres, dt_ms2xres = features2.double_resolution(wv, dt_ms)
        fwhm = features2.calc_fwhm(wv2xres, dt_ms2xres)
        p2vt = features2.calc_p2vt(wv2xres, dt_ms2xres)
        if cluster.stats['csi'] == np.NAN:
            return 0  # unlabeled
        elif cluster.stats['csi'] > 10:  # Pyramidal
            if 1.6 * fwhm + p2vt > 0.95:
                counts[CLASS_P] += 1  # Pyramidal
            else:
                counts[0] += 1  # unlabeled
        else:
            if 1.6 * fwhm + p2vt < 0.95:
                counts[CLASS_I] += 1  # Interneuron
            else:
                counts[0] += 1  # unlabeled

    if counts[CLASS_P] > counts[CLASS_I]:
        return CLASS_P 
    elif counts[CLASS_I] > counts[CLASS_P]:
        return CLASS_I 
    else:
        return 0  # Neither
