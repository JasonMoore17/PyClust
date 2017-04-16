import spikeset_io
import spikeset
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# returns 'P' (Pyramidal), 'I' (Interneuron), or 'J' (Junk)
def compute_label(cluster):
    return
"""
# generate csv from spikeset .spike and .spike.bounds files
# dotSpikeFilePath is the file path that ends with '.spike'
def dotspike2csv(dotSpikeFilePath):
    ss = spikeset_io.loadSpikeset(dotSpikeFilePath, verbose=False)
    filePathPrefix, ext = os.path.splitext(dotSpikeFilePath)
    if ext == ".spike":
        csvFilename = filePathPrefix + ".csv"
        waveform = ss.spikes[0,:,0]
        compute_label(None)

        # append rows to csv
"""

# if the original waveform wv has 60 timestamps, the output is
# a waveform with 120 timestamps
def double_resolution(wv):
    wv_inbetween = []
    for i in range(wv.size - 1):
        wv_inbetween.append((wv[i] + wv[i + 1]) / 2.)
    wv_inbetween.append(wv[wv.size - 1])
    wv_inbetween = np.array(wv_inbetween)
    wv_new = np.array([[wv],[wv_inbetween]]).transpose().flatten()
    return wv_new


def load_spikeset(spikeFile):
    if not spikeFile.endswith('.spike'):
        return None

    currentPath = os.path.dirname(__file__)
    dataPath = os.path.join(currentPath, '..', '..', 'data')
    boundsFile = str(spikeFile) + os.extsep + 'bounds'
    boundsPath = os.path.join(dataPath, boundsFile)
    dotSpikeFilePath = os.path.join(dataPath, spikeFile)

    # load spikeset and its clusters
    ss = spikeset_io.loadSpikeset(dotSpikeFilePath, verbose=False)
    if os.path.exists(boundsPath):
        ss.importBounds(boundsPath)
    return ss

# compute full width at half maximum (FWHM)
def get_fwhm(wv):
    max = np.amax(wv)
    argmax = np.argmax(wv)
    right = wv[argmax:]
    left = wv[:argmax]
    argRhm = None
    argLhm = None

    # align waveform to 0 for baseline left of amplitude
    min = np.amin(wv[:argmax])
    voffset = np.vectorize(lambda x: x - min)
    wv = voffset(wv)

    plt.plot(range(120), wv)
    plt.show()

    # find index for right half max
    for i in range(right.size - 2):
        if right[i] >= max / 2. >= right[i + 2]:
            argRhm = i + 1
            break
        if right[i] <= max / 2. <= right[i + 2]:
            argRhm = i + 1
            break

    # find index for left half max
    for i in np.flipud(range(left.size - 2)):
        if left[i + 2] >= max / 2. >= left[i]:
            argLhm = i + 1
            break
        if left[i + 2] <= max / 2. <= left[i]:
            argLhm = i + 1
            break
    if argRhm == None or argLhm == None:
        return None
    fwhm = (argRhm + argmax) - argLhm
    return fwhm


# compute time from peak to trough (valley)
def get_peak2ValTime(wv):
    return None

# 0 = Pyramidal; 1 = Interneuron; 2 = Neither
def get_label(cluster):
    nChans = cluster.wv_mean.shape[1]
    for chan in range(nChans):
        wv = double_resolution(cluster.wv_mean[:,chan])
        fwhm = get_fwhm(wv)
    return 2

def generate_labels(path):
    currentPath = os.path.dirname(__file__)
    dataPath = os.path.join(currentPath, '..', '..', 'data')
    for spikeFile in filter(lambda f: f.endswith('.spike'), os.listdir(path)):
        boundsFile = str(spikeFile) + os.extsep + 'bounds'
        boundsPath = os.path.join(dataPath, boundsFile)
        if not os.path.exists(boundsPath):
            continue
        ss = load_spikeset(spikeFile)
        for cluster in ss.clusters:
            fwhm = get_label(cluster)

        """filePathPrefix, ext = os.path.splitext(dotSpikeFilePath)
        if ext == ".spike":
            csvFilename = filePathPrefix + ".csv"
            waveform = ss.spikes[0, :, 0]
            compute_label(None)"""


if __name__ == "__main__":
    if len(sys.argv) != 2:
        path = os.path.dirname(__file__)
        path = os.path.join(path, '..', '..', 'data')
        print('dir:', path)
        generate_labels(path)
    else:
        generate_labels(sys.argv[1])
