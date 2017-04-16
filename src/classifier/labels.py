import spikeset_io
import spikeset
import os
import sys
import numpy as np

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
    return None


# compute time from peak to trough (valley)
def get_peak2ValTime(wv):
    return None

# 0 = Pyramidal; 1 = Interneuron; 2 = Neither
def get_label(cluster):
    wv_mean = cluster.wv_mean
    C = cluster.wv_mean.shape(1)
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
        waveform = ss.spikes[0, :, 0]

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
