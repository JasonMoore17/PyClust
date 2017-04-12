# -*- coding: utf-8 -*-
"""
Created on Fri Mar 30 04:18:30 2012

@author: Bernard
"""
from __future__ import print_function
import random

import pickle
import hashlib
import os
import sys
import struct

import scipy.stats
import numpy as np
import sklearn.mixture

from PySide import QtGui

import features
import boundaries
import unique_colors
import spikeset_io

# pickle needs this to load the saved bounds
#from boundaries import BoundaryPolygon2D

# rough breakdown as follows
# spikeset contains spikes, timestamps for the whole ntt file
# feature contains a name, channel count and data
# boundary contains a feature name, and polygon boundary
# cluster contains a list of boundaries, along with some summary statistics
# convention: N number of spikes, C number of channels, L length of waveforms


# Spike data is N x L x C
class Spikeset:
    def __init__(self, spikes, timestamps, peak_index, sampling_frequency,
            use_pca=True, subject='Unknown', session='Unknown'):
        self.spikes = spikes
        self.time = timestamps
        self.N = len(timestamps)
        self.C = spikes.shape[2]
        self.peak_index = peak_index
        self.fs = sampling_frequency
        self.dt_ms = 1000.0 / self.fs
        self.use_pca = True
        self.subject = subject
        self.session = session
        self.T = (max(self.time) - min(self.time)) / 1e6
        self.clusters = []

    def saveFeatures(self, filename):
        """Save feature info (PCA coefficients) to file."""
        print("Saving features info, spikeset hash",)
        f = open(filename, 'wb')
        # compute a hash for the spikeset
        b = self.spikes.view(np.uint8)
        hashkey = hashlib.sha1(b).hexdigest()
        print(hashkey, "to file", filename, ".")
        pickle.dump(hashkey, f)
        pickle.dump(self.feature_special, f)

    def loadFeatures(self, filename):
        """Load feature info (PCA coefficients) from filename."""
        f = open(filename, 'rb')
        loadhash = pickle.load(f)
        b = self.spikes.view(np.uint8)
        hashkey = hashlib.sha1(b).hexdigest()

        if loadhash == hashkey:
            print("Spikeset hashes match, loading features info.")
            self.calculateFeatures(pickle.load(f))
        else:
            print("Hashes don't match, features are from a different dataset. Be careful.")
            self.calculateFeatures(pickle.load(f))

    def calculateFeatures(self, special=None):
        """Calculate the standard battery of features."""
        print("Computing features.")
        if not special:
            self.feature_special = dict()
            self.feature_special['fPCA'] = None
            self.feature_special['wPCA'] = None
        else:
            self.feature_special = special
            if not 'wPCA' in special.keys():
                self.feature_special['wPCA'] = None
            if not 'fPCA' in special.keys():
                self.feature_special['fPCA'] = None

        self.features = []
        self.features.append(features.Feature_Peak(self))
        self.features.append(features.Feature_Energy(self))
        self.features.append(features.Feature_Time(self))
        self.features.append(features.Feature_Valley(self))
        self.features.append(features.Feature_Trough(self))

        if self.use_pca:
            self.features.append(
                    features.Feature_PCA(self, self.feature_special['fPCA']))
            self.features.append(
                features.Feature_Waveform_PCA(self,
                                              self.feature_special['wPCA']))
            self.feature_special['fPCA'] = self.featureByName('fPCA').coeff
            self.feature_special['wPCA'] = self.featureByName('wPCA').coeff

    def __del__(self):
        """Destructor."""
        print("Spikeset object being destroyed")

    def featureNames(self):
        """Return list of feature names."""
        return [feature.name for feature in self.features]

    def featureByName(self, name):
        """Retrieve the feature with the given name."""
        for feature in self.features:
            if feature.name == name:
                return feature
        return None

    def addCluster(self, color=None):
        """Adds a new cluster to the existing list. Auto chooses color."""
        new_cluster = Cluster(self)
        if color:
            new_cluster.color = color
        else:
            color_list = [tuple(map(lambda x: x / 255.0, cluster.color)) for
                    cluster in self.clusters]
            if color_list == []:
                new_cluster.color = (50, 170, 170)
            else:
                c = unique_colors.newcolor(color_list)
                new_cluster.color = tuple(map(lambda x: int(x * 255), c))
        self.clusters.append(new_cluster)
        new_cluster._visible = True
        return new_cluster

    def importBounds(self, filename, excludePCA=False):
        """Imports cluster boundaries from an existing .bounds file"""
        if not os.path.exists(filename):
            return

        infile = open(filename, 'rb')

        versionstr = pickle.load(infile)
        if versionstr == "0.0.1":
            print("Saved bounds version 0.0.1")

            dumped = pickle.load(infile)
            #self.calculateFeatures(dumped['feature_special'])

            print("Founds", len(dumped['clusters']),)
            print("bounds to import, creating clusters.")
            for cluster in dumped['clusters']:
                clust = self.addCluster(color=cluster['color'])

                if(excludePCA):
                    clust.bounds = [xxx for xxx in cluster['bounds'] if not xxx.features[0] in ['fPCA','wPCA','Time']]
                else:
                    clust.bounds = cluster['bounds']

                clust.wave_bounds = cluster['wave_bounds']
                clust.add_bounds = cluster['add_bounds']
                clust.del_bounds = cluster['del_bounds']
                if 'mmodel' in cluster.keys():
                    temp = cluster['mmodel'][0]
                    clust.member_base = temp.labels == temp.model_id
                elif 'member_base' in cluster.keys():
                    clust.member_base = cluster['member_base']
                clust.calculateMembership(self)

        else:
            print("Saved bounds version 0.0.0")
            # the old, very inflexible way of doing things
            #special = versionstr
            #special = pickle.load(infile)
            pickle.load(infile)
            #self.spikeset.calculateFeatures(special)
            saved_bounds = pickle.load(infile)
            print("Found", len(saved_bounds),)
            print("bounds to import, creating clusters.")
            for (col, bound, wave_bound, add_bound, del_bound) \
                    in saved_bounds:
                clust = self.addCluster(color=col)
                clust.bounds = bound
                clust.wave_bounds = wave_bound
                clust.add_bounds = add_bound
                clust.del_bounds = del_bound
                clust.calculateMembership(self)

        infile.close()

    def importAckk(self, filename):
        """Imports preclustered spike IDs from .ackk file."""
        if not os.path.exists(filename):
            return

        #print("Found KKwik cluster file", filename, ':',)
        # Everything is little endian
        # A .ackk record is as folows:
        # Header:
        # uint32 + n x char - subject string
        # uint32 + n x char - session date string
        # uint64 - num spikes
        # uint16 x N - spike cluster IDs
        f = open(filename, 'rb')
        subjectstr = spikeset_io.readStringFromBinary(f)
        datestr = spikeset_io.readStringFromBinary(f)
        num_samps, = struct.unpack('<Q', f.read(8))

        if num_samps != self.N:
            print("Sample counts don't match up, invalid .ackk file")
            print(num_samps)
            print(self.spikeset.N)
            f.close()
            return
        elif subjectstr != self.subject:
            print("Subject lines don't match up, invalid .ackk file")
            f.close()
            return
        elif datestr != self.session:
            print("Session date strings don't match up, invalid .ackk file")
            f.close()
            return

        labels = np.fromfile(f, dtype=np.dtype('<h'), count=num_samps)
        f.close()

        # get a list of cluster numbers
        k = np.unique(labels)
        print(np.size(k) - 1, 'components.')
        colors = unique_colors.unique_colors_hsv(np.size(k) - 1)

        # create the clusters for each component in the KK file.
        if np.size(k) > 1:
            print("Importing clusters from .ackk",)
            for i in k:
                if i == 0:
                    continue
                print(i,)
                sys.stdout.flush()
                cluster = self.addCluster(
                    color=tuple([int(a * 255) for a in colors[i - 1]]))

                cluster.member_base = labels == i
                cluster.calculateMembership(self)
            print(".")
        pass


# Clusters have a color, a set of boundaries and some calculation functions
class Cluster:
    def __init__(self, spikeset):
        self.color = (random.randrange(100, 200), random.randrange(100, 200),
                random.randrange(100, 200))
        self.member = np.array([False] * spikeset.N)
        self.bounds = []
        self.wave_bounds = []
        self.add_bounds = []
        self.del_bounds = []
        self.member_base = []
        self.isi = []
        self.refractory = np.array([False] * spikeset.N)
        self.stats = {}
       # self.stats['burst'] = np.NAN
       # self.stats['csi'] = np.NAN
       # self.stats['isolation'] = np.NAN
       # self.stats['mean_rate'] = 0
       # self.stats['num_spikes'] = 0
       # self.stats['refr_count'] = np.NAN
       # self.stats['refr_fp'] = np.NAN
       # self.stats['refr_frac'] = np.NAN
       # self.stats['wv_com'] = np.NAN

        self.isi_bins = np.logspace(np.log10(0.1), np.log10(1e5), 100)
        self.isi_bin_centers = (self.isi_bins[0:-1] + self.isi_bins[1:]) / 2
        self.isi_bin_count = np.zeros(self.isi_bin_centers.shape)

    def __del__(self):
        # print("Cluster object being destroyed")
        pass

    def addBoundary(self, bound):
        self.removeBound(bound.features, bound.chans)
        self.bounds.append(bound)

    def getBoundaries(self, feature_name_x, feature_chan_x, feature_name_y,
            feature_chan_y, boundtype='limits'):
        f = lambda bound: (bound.features == (feature_name_x, feature_name_y))\
                and (bound.chans == (feature_chan_x, feature_chan_y))
        if boundtype == 'add':
            temp = [bound for bound in self.add_bounds if f(bound)]
        elif boundtype == 'del':
            temp = [bound for bound in self.del_bounds if f(bound)]
        elif boundtype == 'limits':
            temp = [bound for bound in self.bounds if f(bound)]

        return temp

    def removeBound(self, featureNames, featureChans, boundtype='limits'):
        f = lambda bound: bound.features != featureNames or \
            bound.chans != featureChans

        if boundtype == 'add':
            self.add_bounds = [bound for bound in self.add_bounds if f(bound)]
        elif boundtype == 'del':
            self.del_bounds = [bound for bound in self.del_bounds if f(bound)]
        elif boundtype == 'limits':
            self.bounds = [bound for bound in self.bounds if f(bound)]

    def calculateMembership(self, spikeset):
        self.mahal_valid = False
        if (self.bounds == []) and (self.add_bounds == []) and \
                (self.member_base == []):
            self.member = np.array([False] * spikeset.N)
            self.isi = []
            self.refractory = np.array([False] * spikeset.N)
            self.stats = {}
            return

        # If we have no member base, its all or add bounds
        if self.member_base == []:
            if self.add_bounds != []:
                self.member = np.zeros((spikeset.N), dtype=np.bool)
                for bound in self.add_bounds:
                    self.member = np.logical_or(self.member,
                        bound.withinBoundary(spikeset))
            else:
                self.member = np.ones((spikeset.N), dtype=np.bool)
        # If we have a member base, start there then add add bounds
        else:
            self.member = np.copy(self.member_base)
            for bound in self.add_bounds:
                self.member = np.logical_or(self.member,
                    bound.withinBoundary(spikeset))

        # now cut down the start
        for bound in self.bounds:
            w = self.member
            self.member[w] = np.logical_and(self.member[w],
                bound.withinBoundary(spikeset, subset=w))

        for (chan, sample, lower_bound, upper_bound) in self.wave_bounds:
            w = np.logical_and(
                spikeset.spikes[:, sample, chan] >= lower_bound,
                spikeset.spikes[:, sample, chan] <= upper_bound)
            self.member = np.logical_and(self.member, w)

        t = spikeset.time[self.member]
        self.refr_period = 1.7
        self.burst_period = 20
        self.isi = (t[1:] - t[0:-1]) / 1e3

        # calculate the isi histogram
        self.isi_bin_count, _ = np.histogram(self.isi, self.isi_bins)

        self.refractory = np.array([False] * spikeset.N)
        #ref = np.logical_and(self.isi < self.refr_period, self.isi> 0.8)
        ref = self.isi < self.refr_period
        self.refractory[self.member] = np.logical_or(np.append(ref, False),
                np.append(False, ref))

        # stats
        self.stats['num_spikes'] = np.sum(self.member)
        self.stats['mean_rate'] = self.stats['num_spikes'] / spikeset.T

        # some waveform stats might be useful to split the different
        # types of clusters - single unit, multi unit, overlap, etc

        if self.stats['num_spikes'] <= 1:
            self.stats['burst'] = np.NAN
            self.stats['refr_count'] = np.NAN
            self.stats['csi'] = np.NAN
            self.stats['refr_fp'] = np.NAN
            self.stats['isolation'] = np.NAN
            self.stats['refr_frac'] = np.NAN
            self.stats['wv_com'] = np.NAN
            self.wv_mean = np.zeros((spikeset.spikes.shape[1],
                                     spikeset.spikes.shape[2]))
            self.wv_std = np.zeros((spikeset.spikes.shape[1],
                                     spikeset.spikes.shape[2]))
        else:
            # calculate mean waveform
            self.wv_mean = np.mean(spikeset.spikes[self.member, :, :], axis=0)
            self.wv_std = np.std(spikeset.spikes[self.member, :, :], axis=0)

            # work out a mean waveform
            u_wv = np.mean(spikeset.spikes[self.member, :, :], axis=0)
            u_wv_2 = u_wv * u_wv
            com_x = np.arange(u_wv.shape[0])
            com_chan = (np.sum(u_wv_2.T * com_x, axis=1) /
                    np.sum(u_wv_2, axis=0) - spikeset.peak_index)
            p = u_wv[spikeset.peak_index, :]
            # peak weighted average of the channels
            self.stats['wv_com'] = sum(com_chan * p) / sum(p) * spikeset.dt_ms
            #print(self.stats['wv_com'])

            self.stats['burst'] = (100.0 * np.sum(self.isi <
                self.burst_period).astype(float)) / \
                (self.stats['num_spikes'] - 1)

            self.stats['refr_count'] = np.sum(self.isi < self.refr_period)

            self.stats['refr_frac'] = (float(self.stats['refr_count']) /
                (self.stats['num_spikes'] - 1))

            alpha = (self.stats['mean_rate'] * self.stats['refr_count'] /
                (self.stats['num_spikes'] * 2.0 * self.refr_period * 1e-3))
            if alpha > 0.25:
                self.stats['refr_fp'] = 100.0
            else:
                self.stats['refr_fp'] = 100 * 0.5 * \
                        (1 - np.sqrt(1 - 4 * alpha))

            # csi and isolation needs peaks
            p = features.Feature_Peak(spikeset).data
            if self.stats['num_spikes'] * 2 > spikeset.N:
                self.stats['isolation'] = np.NAN
            else:
                try:
                    cvi = np.linalg.inv(np.cov(np.transpose(
                        p[self.member, :])))
                    u = p - np.mean(p[self.member, :], axis=0)
                    m = np.sum(np.dot(u, cvi) * u, axis=1)
                    m = np.sort(m)
                    self.stats['isolation'] = m[
                        self.stats['num_spikes'] * 2 - 1]
                except Exception:
                    self.stats['isolation'] = np.NAN

            chan = np.round(np.mean(np.argmax(p[self.member, :], axis=1)))
            delta = np.diff(p[self.member, chan.astype("int")])

            delta = delta[np.logical_and(
                self.isi < self.burst_period,
                self.isi > self.refr_period)]
            if np.size(delta, 0):
                temp = np.sum(delta <= 0) - np.sum(delta > 0)
                temp = temp.astype(np.float)
                self.stats['csi'] = 100.0 * temp / np.size(delta)
            else:
                self.stats['csi'] = np.NAN

    def calculateMahal(self, spikeset):
        if np.all(np.logical_not(self.member)):
            return
        # Work on this as a separate tool, too slow every click
        # Compute mahal distance for all waveforms in cluster
        try:
            #temp = np.concatenate([spikeset.spikes[self.member, :, i] for i in
            #range(np.size(spikeset.spikes, 2))], axis=1)
            temp = spikeset.featureByName('Peak').data[self.member, :]
            cvi = np.linalg.inv(np.cov(np.transpose(temp)))
            u = temp - np.mean(temp, axis=0)
            m = np.sum(np.dot(u, cvi) * u, axis=1)
            self.mahal = m
            self.mahal_valid = True
        except Exception:
            self.mahal = np.NAN

    def autotrim(self, spikeset, fname='Peak', confidence=None, canvas=None):
        chans = spikeset.featureByName(fname).data.shape[1]
        projs = []
        for x in range(0, chans):
            for y in range(x + 1, chans):
                if len(self.getBoundaries(fname, x, fname, y)) == 0:
                    projs.append((x, y))

        combs = scipy.misc.comb(chans, 2)

        plots_x = np.ceil(np.sqrt(combs))
        plots_y = np.ceil(float(combs) / plots_x)

        col = np.array(self.color) / 255.0
        data = spikeset.featureByName(fname).data[self.member, :]
        refr = self.refractory[self.member]

        N = np.sum(self.member)
        ms = 1
        if N > 1000:
            ms = 1
        elif N > 2000:
            ms = 2
        elif ms > 100:
            ms = 3
        else:
            ms = 5

        counter = 0
        for proj_x in range(0, chans):
            for proj_y in range(proj_x + 1, chans):
                counter = counter + 1
                ax = canvas.figure.add_subplot(plots_y, plots_x, counter)
                ax.plot(data[:, proj_x], data[:, proj_y],
                                 marker='o', markersize=ms,
                                 markerfacecolor=col, markeredgecolor=col,
                                 linestyle='None', zorder=0)
                ax.plot(data[refr, proj_x], data[refr, proj_y],
                                marker='o', markersize=ms + 1,
                                markerfacecolor='k', markeredgecolor='k',
                                linestyle='None', zorder=1)
                bounds = self.getBoundaries(fname, proj_x, fname, proj_y)
                for bound in bounds:
                    bound.draw(ax, color='k')
        canvas.draw()
        canvas.repaint()

        QtGui.QApplication.processEvents()
        print("Attempting to autotrim cluster on feature:", fname)

        gmm = sklearn.mixture.DPGMM(n_components=10, covariance_type='full')
        gmm.fit(data)
        label = -1 * np.ones((spikeset.N))
        temp = gmm.predict(data)
        label[self.member] = temp
        ind = np.argmax(np.bincount(temp))
        maincomp = label == ind

        while True:
            fitness = np.zeros((len(projs), 2))
            ellipses = []

            refr = self.refractory[self.member]
            data = spikeset.featureByName(fname).data[self.member, :]

            if len(projs) == 0:
                break

            for iProj, (proj_x, proj_y) in enumerate(projs):
                if np.any(refr):
                    confs = np.array([0.999,
                        1.0 - (10.0 * np.sum(refr)) /
                        np.size(refr), 1.0 - (1.0 * np.sum(refr)) /
                        np.size(refr), 1.0 - (0.5 * np.sum(refr))
                        / np.size(refr), 1.0 - 1.0 / np.size(data, axis=0),
                        1.0 - 0.5 / np.size(data, axis=0)])
                else:
                    confs = np.array([1.0 - 0.01 / np.size(data, axis=0),
                        1.0 - 0.5 / np.size(data, axis=0),
                        1.0 - 0.1 / np.size(data, axis=0)])
                # Sort them so we pick the biggest if fitness funcs are equal
                confs = np.sort(confs)[::-1]
                kvals = scipy.stats.chi2.ppf(confs,  2)  # 2D projections-2 dof
                # Select the data for this projection
                pdata = data[:, [proj_x, proj_y]]
                # Estimate the ellipse for the projection using the main comp
                center, angle, size = boundaries.robustEllipseEstimator(
                        pdata[maincomp[self.member], :])
                # Compute fitness for different confidence intervals
                proj_fitness = np.zeros((len(confs),))
                for i, kval in enumerate(kvals):
                    width = np.sqrt(kval) * size[0]
                    height = np.sqrt(kval) * size[1]
                    ww = np.logical_not(boundaries.pointsInsideEllipse(pdata,
                        center, angle, (width, height)))
                    if np.sum(ww) == 0:
                        proj_fitness[i] = np.nan
                    else:
                        proj_fitness[i] = np.sum(refr[ww]).astype(float) \
                            / np.sum(ww)
                    if np.isinf(width) or np.isinf(height):
                        raise ValueError("Infinite width or height")
                        import PyQt4.QtCore
                        PyQt4.QtCore.pyqtRemoveInputHook()
                        import ipdb
                        ipdb.set_trace()
                # Choose the maximal fitness confidence interval
                ind = np.argmax(proj_fitness)
                fitness[iProj, 0] = proj_fitness[ind]
                fitness[iProj, 1] = confs[ind]
                ellipses.append((center, angle, size * np.sqrt(kvals[ind])))

            # Choose the best projection
            ind = np.argmax(fitness[:, 0])
            #print("Creating boundary on", fname, projs[ind][0], \)
            #       "vs.", fname, projs[ind][1]
            bound = boundaries.BoundaryEllipse2D()
            bound = bound.init2((fname, fname),
                    (projs[ind][0], projs[ind][1]), ellipses[ind][0],
                    ellipses[ind][1], ellipses[ind][2])

            self.addBoundary(bound)

            # Plot the current cluster state
            counter = 0
            for proj_x in range(0, chans):
                for proj_y in range(proj_x + 1, chans):
                    counter = counter + 1
                    if proj_x == projs[ind][0] and proj_y == projs[ind][1]:
                        sindex = counter
            # Figure out which subplot we're on
            proj_x = projs[ind][0]
            proj_y = projs[ind][1]
            ax = canvas.figure.add_subplot(plots_y, plots_x, sindex)
            ax.cla()
            tdata = spikeset.featureByName(fname).data
            ax.plot(tdata[self.member, proj_x], tdata[self.member, proj_y],
                             marker='o', markersize=ms,
                             markerfacecolor=col, markeredgecolor=col,
                             linestyle='None', zorder=0)
            ax.plot(tdata[self.refractory, proj_x],
                    tdata[self.refractory, proj_y],
                            marker='o', markersize=ms + 1,
                            markerfacecolor='k', markeredgecolor='k',
                            linestyle='None', zorder=1)

            bounds = self.getBoundaries(fname, proj_x, fname, proj_y)
            for bound in bounds:
                bound.draw(ax, color='k')

            canvas.draw()
            canvas.repaint()
            QtGui.QApplication.processEvents()

            # remove this from the list of unlimited projections
            self.calculateMembership(spikeset)
            projs.remove(projs[ind])
