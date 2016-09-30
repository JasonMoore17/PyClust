from __future__ import print_function
import struct
import os

import numpy as np

# handles vairous io tasks including:
# - loading .ntt to spikeset
# - loading .spike to spikeset
# - importing bounds to give a non-gui useful spikeset with cluster membership

#from spikeset import Spikeset
import spikeset
import nlxio


def spikesetFromNtt(filename):
    ts, sp, fs = nlxio.loadNtt(filename)
    return spikeset.Spikeset(sp, ts, 8, fs)


def readStringFromBinary(f):
    strlen, = struct.unpack('<I', f.read(4))
    if strlen:
        return f.read(strlen)
    else:
        return ''


def spikesetFromDotSpike(filename):
    f = open(filename, 'rb')
    # Everything is little endian
    # A .spike record is as folows:
    # Header:
    # uint16 - version no
    # uint64 - num spikes
    # uint16 - num channels
    # uint16 - num samples per waveform
    # uint32 - sampling frequency
    # uint16 - peak align point
    # c * float64 - a2d conversion factor
    # uint32 + n x char - date time string
    # uint32 + n x char - subject string
    # uint32 + n x char - filter description

    version_no, = struct.unpack('<H', f.read(2))
    if version_no != 1:
        f.close()
        return

    print('')
    print('Loading', filename)
    print('Format version #', version_no)
    num_spikes, = struct.unpack('<Q', f.read(8))
    print('Num spikes', num_spikes)
    num_chans, = struct.unpack('<H', f.read(2))
    print('Num channels', num_chans)
    num_samps, = struct.unpack('<H', f.read(2))
    print('Num samples', num_samps)
    fs, = struct.unpack('<I', f.read(4))
    print('Sampling frequency', fs)
    peak_align, = struct.unpack('<H', f.read(2))
    print('Peak alignment point', peak_align)
    uvolt_conversion = np.array(struct.unpack('<' + ('d' * num_chans), f.read(8 * num_chans)))
    print('Microvolt conversion factor', uvolt_conversion)
    subjectstr = readStringFromBinary(f)
    print('Subject string', subjectstr)
    datestr = readStringFromBinary(f)
    print('Date string', datestr)
    filterstr = readStringFromBinary(f)
    print('Filter string', filterstr)

    # Records:
    # uint64 - timestamp                    bytes 0:8
    # numsample x numchannel x int16 - waveform points

    dt = np.dtype([('time', '<Q'),
        ('spikes', np.dtype('<h'), (num_samps, num_chans))])

    temp = np.fromfile(f, dt)

    f.close()

    return spikeset.Spikeset(temp['spikes'] *
                             np.reshape(uvolt_conversion, [1, 1, num_chans]),
                             temp['time'], peak_align, fs, subject=subjectstr,
                             session=datestr)


def loadSpikeset(filename):
    # Load the file
    if filename.endswith('.ntt'):
        ss = spikesetFromNtt(filename)
        featureFile = filename + '.feat'
    elif filename.endswith('.spike'):
        ss = spikesetFromDotSpike(filename)
        featureFile = filename + '.feat'
    else:
        return None

    # check if feature file exists
    if os.path.exists(featureFile):
        # load it up
        ss.loadFeatures(featureFile)
    else:
        # calculate features, and save them
        ss.calculateFeatures()
        ss.saveFeatures(featureFile)

    return ss
