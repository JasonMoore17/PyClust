#!/usr/bin/python
from __future__ import print_function, division

#from PyQt4 import QtCore, QtGui
from PySide import QtCore, QtGui

import numpy as np
import scipy.io as sio
import scipy.stats
import scipy.signal
import matplotlib as mpl
import sklearn.mixture as mixture
#import sklearn.cluster

import pickle
import os
import sys
import time

import six

from gui2 import Ui_MainWindow
#from mygui import Ui_MainWindow

import spikeset
import spikeset_io
import featurewidget
import multiplotwidget
import boundaries

# classifier module
import classifier
#import features


def xcorr_ts(t1, t2, demean=True, normed=True, binsize=2, maxlag=None):
    """Computes the xcorr of spiketrains times t1 and t2. Units assumed ms."""
    binary_xo = np.arange(np.min([np.min(t1), np.min(t2)]) - binsize,
                            np.max([np.max(t1), np.max(t2)]) + binsize,
                            binsize)
    binary_t1, _ = np.histogram(t1, binary_xo)
    binary_t2, _ = np.histogram(t2, binary_xo)

    n = binary_t1.size
    xc = xcorr(binary_t1, binary_t2, demean=demean, normed=normed)
    lags = np.arange(- n + 1, n) * binsize

    if maxlag is not None:
        w = np.abs(lags) <= maxlag
        return (lags[w], xc[w])
    else:
        return (lags, xc)


def xcorr(x1, x2, demean=True, normed=True):
    """Computes the xcorr of time series x1 & x2."""
    if demean:
        x1 = x1 - np.mean(x1)
        x2 = x2 - np.mean(x2)
    c = scipy.signal.fftconvolve(x1, x2[::-1], mode='full')
    if normed:
        c /= np.sqrt(np.dot(x1, x1) * np.dot(x2, x2))
    return c


class PyClustMainWindow(QtGui.QMainWindow):

    @QtCore.Slot()
    def on_actionWaveform_Cutter_triggered(self):
        self.ui.stackedWidget.setCurrentIndex(1)
        self.updateWavecutterPlot()

        # predict label to save as training data
        label_pred = classifier.get_label(self.activeClusterRadioButton().cluster_reference,
                                       self.spikeset.dt_ms)
        self.ui.comboBox_labels.setCurrentIndex(label_pred)

        # conditionally disable save-labeled-cluster button
        clust_num_h = filter(lambda (i, c): self.activeClusterRadioButton().cluster_reference
                                            is c, enumerate(self.spikeset.clusters))
        clust_num = clust_num_h[0][0] if not clust_num_h == [] else None
        # save mean
        if self.clf_data_saver.is_saved(self.ui.label_subjectid.text(), self.ui.label_session.text(),
                                 self.ui.label_fname.text(), clust_num):
            self.ui.pushButton_saveLabeledCluster.setEnabled(False)
        else:
            self.ui.pushButton_saveLabeledCluster.setEnabled(True)

        # save members
        if self.clf_data_saver.is_saved(self.ui.label_subjectid.text(), self.ui.label_session.text(),
                                 self.ui.label_fname.text(), clust_num, mode='members'):
            self.ui.pushButton_saveLabeledMembers.setEnabled(False)
        else:
            self.ui.pushButton_saveLabeledMembers.setEnabled(True)


    def switch_to_maindisplay(self):
        self.ui.stackedWidget.setCurrentIndex(0)
        self.updateFeaturePlot()

    @QtCore.Slot()
    def on_actionMerge_Clusters_triggered(self):
        active = self.activeClusterRadioButton()
        if active and active.cluster_reference != self.junk_cluster:
            # get the current cluster 'number'
            clust = active.cluster_reference

            # create a backup cluster to revert to if we cancel
            self.merge_clust = spikeset.Cluster(self.spikeset)
            # dont want the backup in the list for now
            self.merge_clust.color = clust.color
            self.merge_clust.bounds = clust.bounds
            self.merge_clust.member_base = clust.member_base
            self.merge_clust.add_bounds = clust.add_bounds
            self.merge_clust.calculateMembership(self.spikeset)

            id1 = self.spikeset.clusters.index(clust)
            self.merge_id_1 = id1
            self.ui.comboBox_merge_c1.clear()
            self.ui.comboBox_merge_c1.addItem("Cluster %d" % (id1 + 1))
            self.ui.comboBox_merge_c1.setEnabled(False)

            self.ui.comboBox_merge_c2.blockSignals(True)
            self.ui.comboBox_merge_c2.clear()
            self.ui.comboBox_merge_c2.addItem("(Select)")
            for i in range(len(self.spikeset.clusters)):
                if i == id1:
                    continue
                self.ui.comboBox_merge_c2.addItem("Cluster %d" % (i + 1))
            self.ui.comboBox_merge_c2.blockSignals(False)

            self.ui.stackedWidget.setCurrentIndex(3)
            self.merge_cluster_choice_changed(0)

    class CommandMergeClusters(QtGui.QUndoCommand):
        """Wraps the merge cluster logic to provide undo functionality."""

        def __init__(self, mainwindow, cl_mod, cl_rem, bounds, abounds, mbase):
            desc = "merge cluster"
            super(PyClustMainWindow.CommandMergeClusters, self).__init__(desc)
            self.mainwindow = mainwindow
            self.cl_mod = cl_mod
            self.cl_rem = cl_rem
            self.bounds = bounds
            self.abounds = bounds
            self.mbase = mbase
            self.clm_mbase = cl_mod.member_base

        def redo(self):
            self.unsaved = self.mainwindow.unsaved
            self.index = self.mainwindow.spikeset.clusters.index(self.cl_rem)
            self.mainwindow.spikeset.clusters.remove(self.cl_rem)
            self.cl_mod.bounds = []
            self.cl_mod.add_bounds = []
            self.cl_mod.wave_bounds = []
            self.cl_mod.member_base = self.clm_mbase
            self.cl_mod.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.update_ui_cluster_buttons(self.cl_mod)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved
            self.cl_mod.bounds = self.bounds
            self.cl_mod.add_bounds = self.abounds
            self.cl_mod.member_base = self.mbase
            self.cl_mod.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.spikeset.clusters.insert(self.index, self.cl_rem)
            self.mainwindow.update_ui_cluster_buttons(self.cl_mod)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()

    def merge_apply(self):
        """Keep the merge boundaries, and delete other cluster."""
        # if we've decided to apply the merge, should remove the other cluster
        self.ui.stackedWidget.setCurrentIndex(0)
        command = self.CommandMergeClusters(self,
                        self.spikeset.clusters[self.merge_id_1],
                        self.spikeset.clusters[self.merge_id_2],
                        self.merge_clust.bounds,
                        self.merge_clust.add_bounds,
                        self.merge_clust.member_base)
        self.undoStack.push(command)

        self.merge_clust = None
        self.merge_id_1 = None
        self.merge_id_2 = None

    def merge_cancel(self):
        """Cancel the merge, revert to original boundaries."""
        self.ui.stackedWidget.setCurrentIndex(0)
        # revert the active cluster to the original bounds
        clust = self.spikeset.clusters[self.merge_id_1]
        clust.bounds = self.merge_clust.bounds
        clust.member_base = self.merge_clust.member_base
        clust.add_bounds = self.merge_clust.add_bounds
        clust.calculateMembership(self.spikeset)
        self.updateClusterDetailPlots()

        self.merge_clust = None
        self.merge_id_1 = None
        self.merge_id_2 = None

    def merge_redraw(self):
        """Redraws the merge interface, called after UI events."""
        if self.merge_clust is None:
            return

        if self.merge_id_2 is None:
            self.ui.mplwidget_merge_wv.figure.clear()
            self.ui.mplwidget_merge.figure.clear()
            return

        cf = lambda s: s / 255.0
        w1 = self.merge_clust.member
        col1 = tuple(map(cf, self.merge_clust.color))
        w2 = self.spikeset.clusters[self.merge_id_2].member
        col2 = tuple(map(cf, self.spikeset.clusters[self.merge_id_2].color))
        w3 = self.junk_cluster.member
        refr = self.spikeset.clusters[self.merge_id_1].refractory

        #wv1 = np.mean(self.spikeset.spikes[w1, :, :], axis=0)
        wv1 = self.merge_clust.wv_mean
        wv1 = wv1.T.reshape((wv1.size,))
        #wv2 = np.mean(self.spikeset.spikes[w2, :, :], axis=0)
        wv2 = self.spikeset.clusters[self.merge_id_2].wv_mean
        wv2 = wv2.T.reshape((wv2.size,))

        # plot the xcorr
        self.ui.mplwidget_merge_wv.figure.clear()
        ax = self.ui.mplwidget_merge_wv.figure.add_subplot(1, 3, 3)
        ax.plot(self.merge_plot_lags, self.merge_plot_xc)
        ax.set_xlim([-100, 100])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.plot([0, 0], ax.get_ylim(), 'k--')
        ax.set_xticks([])
        ax.set_yticks([])

        # Plot the waveforms
        ax = self.ui.mplwidget_merge_wv.figure.add_subplot(1, 3, 1)
        ax.plot(wv1, color=col1)
        ax.plot(wv2, color=col2)
        ax.set_xticks([])
        ax.set_yticks([])

        # draw the linear discriminant
        ax = self.ui.mplwidget_merge_wv.figure.add_subplot(1, 3, 2)
        ax.plot(self.merge_plot_discrim_bins, self.merge_plot_discrim_h1,
               color=col1, linewidth=2)
        ax.plot(self.merge_plot_discrim_bins, self.merge_plot_discrim_h2,
               color=col2, linewidth=2)
        ax.set_xticks([])
        ax.set_yticks([])

        self.ui.mplwidget_merge_wv.draw()

        # now draw the projections
        self.ui.mplwidget_merge.figure.clear()
        data = self.spikeset.featureByName('Peak').data
        chans = data.shape[1]
        combs = scipy.misc.comb(chans, 2)
        plots_x = np.ceil(np.sqrt(combs))
        plots_y = np.ceil(float(combs) / plots_x)
        counter = 0
        for proj_x in range(0, chans):
            for proj_y in range(proj_x + 1, chans):
                counter = counter + 1
                ax = self.ui.mplwidget_merge.figure.add_subplot(
                    plots_y, plots_x, counter)
                # plot unclustered
                ax.plot(data[w1, proj_x], data[w1, proj_y],
                                 marker='o', markersize=1,
                                 markerfacecolor=col1, markeredgecolor=col1,
                                 linestyle='None', zorder=1)
                ax.plot(data[w2, proj_x], data[w2, proj_y],
                                 marker='o', markersize=1,
                                 markerfacecolor=col2, markeredgecolor=col2,
                                 linestyle='None', zorder=1)
                ax.autoscale()
                if self.ui.checkBox_merge_background.isChecked():
                    xl = ax.get_xlim()
                    yl = ax.get_ylim()
                    not_junk = (~w1) & (~w2) & (~w3)
                    # don't plot things we know to be outside the plot limits
                    within_xl = (data[:, proj_x] >= xl[0]) & \
                                (data[:, proj_x] <= xl[1])
                    within_yl = (data[:, proj_y] >= yl[0]) & \
                                (data[:, proj_y] <= yl[1])
                    ax.plot(data[not_junk & within_xl & within_yl, proj_x],
                            data[not_junk & within_xl & within_yl, proj_y],
                                    marker='o', markersize=0.1,
                                    markerfacecolor='k', markeredgecolor='k',
                                    linestyle='None', zorder=0)
                    ax.set_xlim(xl)
                    ax.set_ylim(yl)
                if ax.get_xlim()[0] < 0:
                    ax.set_xlim([0, ax.get_xlim()[1]])
                if ax.get_ylim()[0] < 0:
                    ax.set_ylim([0, ax.get_ylim()[1]])
                if self.mp_proj.refractory:
                    ax.plot(data[refr, proj_x], data[refr, proj_y],
                                    marker='o', markersize=2,
                                    markerfacecolor='k', markeredgecolor='k',
                                    linestyle='None', zorder=2)
        self.ui.mplwidget_merge.draw()

    def merge_cluster_choice_changed(self, index):
        """Event fires when a different cluster match is chosen for merge."""
        if index == -1:
            return
        elif index == 0:
            self.merge_id_2 = None
            self.merge_redraw()
            return
        elif index - 1 >= self.merge_id_1:
            self.merge_id_2 = index
        else:
            self.merge_id_2 = index - 1

        # Update the 'merged' membership for the details plot
        w1 = self.merge_clust.member
        w2 = self.spikeset.clusters[self.merge_id_2].member
        clust = self.spikeset.clusters[self.merge_id_1]
        clust.bounds = []
        clust.add_bounds = []
        clust.wave_bounds = []
        clust.member_base = (w1 | w2)
        clust.calculateMembership(self.spikeset)

        # compute the linear discrim
        spikes1 = self.spikeset.spikes[w1, :, :]
        spikes1 = spikes1.reshape((spikes1.shape[0], spikes1.shape[1] *
                                    spikes1.shape[2]))
        spikes2 = self.spikeset.spikes[w2, :, :]
        spikes2 = spikes2.reshape((spikes2.shape[0], spikes2.shape[1] *
                                    spikes2.shape[2]))
        wv1 = np.mean(spikes1, axis=0)
        wv2 = np.mean(spikes2, axis=0)
        delta = wv1 - wv2
        spikes1 = spikes1 - delta
        spikes2 = spikes2 - delta

        dnorm = np.dot(delta, delta)
        proj1 = np.dot(spikes1, delta) / dnorm - 0.5
        proj2 = np.dot(spikes2, delta) / dnorm - 0.5
        bscale = np.max([np.max(np.abs(proj1)), np.max(np.abs(proj2))])
        xo = np.linspace(-bscale, bscale, 100)
        f1 = scipy.stats.gaussian_kde(proj1)
        f2 = scipy.stats.gaussian_kde(proj2)
        # get some bounds approximations
        y1 = f1(xo)
        y1 = np.cumsum(y1) / np.sum(y1)
        y2 = f2(xo)
        y2 = np.cumsum(y2) / np.sum(y2)

        try:
            lbound = min([np.max(xo[y1 <= 0.01]), np.max(xo[y2 <= 0.01])])
        except ValueError:
            lbound = xo[0]
        try:
            rbound = max([np.min(xo[y1 >= 0.99]), np.min(xo[y2 >= 0.99])])
        except ValueError:
            rbound = xo[-1]
        xo = np.linspace(lbound, rbound, 100)
        self.merge_plot_discrim_bins = xo
        self.merge_plot_discrim_h1 = f1(xo)
        self.merge_plot_discrim_h2 = f2(xo)

        # compute the xcorr
        t1 = self.spikeset.time[w1]
        t2 = self.spikeset.time[w2]
        self.merge_plot_lags, self.merge_plot_xc = xcorr_ts(t1 / 1e3, t2 / 1e3,
                        maxlag=100, normed=False, demean=False, binsize=1.5)

        self.merge_redraw()
        self.updateClusterDetailPlots()

    class CommandSplitCluster(QtGui.QUndoCommand):
        """Wraps the split cluster logic to provide undo functionality."""

        def __init__(self, mainwindow, old_cluster, labels, n):
            desc = "split cluster"
            super(PyClustMainWindow.CommandSplitCluster, self).__init__(desc)
            self.mainwindow = mainwindow
            self.old_cluster = old_cluster
            self.index = self.mainwindow.spikeset.clusters.index(old_cluster)
            self.labels = labels
            self.new_clusters = {}
            self.n = n

        def redo(self):
            self.unsaved = self.mainwindow.unsaved

            for i in range(self.n):
                new_clust = self.mainwindow.spikeset.addCluster()
                new_clust.member_base = self.labels == i + 1
                new_clust.calculateMembership(self.mainwindow.spikeset)
                self.new_clusters[i] = new_clust

            self.mainwindow.spikeset.clusters.remove(self.old_cluster)
            self.mainwindow.update_ui_cluster_buttons()
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved

            for i in range(self.n):
                self.mainwindow.spikeset.clusters.remove(self.new_clusters[i])
            self.mainwindow.spikeset.clusters.insert(self.index,
                                                     self.old_cluster)

            self.mainwindow.update_ui_cluster_buttons(self.old_cluster)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()

    @QtCore.Slot()
    def on_actionSplit_Cluster_triggered(self):
        n, ok = QtGui.QInputDialog.getInt(self, 'Split clusters',
                'Split into how many clusters?', 2)
        if ok:
            print('Splitting cluster into', n, 'components using wPCA/GMM')
            clust = self.activeClusterRadioButton().cluster_reference
            gmm = mixture.GMM(n_components=n, covariance_type='full')
            #spikes = self.spikeset.spikes[clust.member, :, :]
            #spikes = spikes.reshape((spikes.shape[0], spikes.shape[1] *
            #                          spikes.shape[2]))
            #input_data, _, _ = features.PCA(spikes, 4)
            if self._overviewmode:
                # load all the current overview data
                data_x = self.spikeset.featureByName(self.mp_proj.feature_x)
                input_data = data_x.data[clust.member, :]
            else:
                # load the current projection data if were not in overview mode
                data_x = self.spikeset.featureByName(self.mp_proj.feature_x)
                data_x = data_x.data[clust.member, self.mp_proj.chan_x]
                data_y = self.spikeset.featureByName(self.mp_proj.feature_y)
                data_y = data_y.data[clust.member, self.mp_proj.chan_y]
                input_data = np.vstack((data_x, data_y)).T

                #features.debug_trace()
            gmm.fit(input_data)
            #km = sklearn.cluster.KMeans(n)
            #labels = km.fit_predict(input_data)
            labels = gmm.predict(input_data)
            mlabels = np.zeros((self.spikeset.N,))
            mlabels[clust.member] = labels + 1

            command = self.CommandSplitCluster(self, clust, mlabels, n)
            self.undoStack.push(command)

    @QtCore.Slot()
    def on_actionAutotrim_triggered(self):
        active = self.activeClusterRadioButton()
        if active and active.cluster_reference != self.junk_cluster:
            clust = active.cluster_reference
            self.trim_cluster = spikeset.Cluster(self.spikeset)
            self.trim_cluster.color = clust.color
            self.trim_cluster.bounds = clust.bounds
            self.trim_cluster.calculateMembership(self.spikeset)

            self.ui.stackedWidget.setCurrentIndex(2)

            self.ui.pushButton_autotrim_apply.setEnabled(False)
            self.ui.pushButton_autotrim_cancel.setEnabled(False)
            self.ui.mplwidget_autotrim.figure.clear()

            current_x = str(self.ui.comboBox_feature_x.currentText())
            clust.autotrim(
                self.spikeset, fname=current_x,
                canvas=self.ui.mplwidget_autotrim)

            self.updateClusterDetailPlots()

            self.ui.pushButton_autotrim_apply.setEnabled(True)
            self.ui.pushButton_autotrim_cancel.setEnabled(True)

    @QtCore.Slot(bool)
    def on_actionRefractory_triggered(self, checked=None):
        self.mp_proj.setRefractory(checked)
        self.mp_proj_multi.setRefractory(checked)
        self.ui.checkBox_refractory.blockSignals(True)
        self.ui.checkBox_merge_refractory.blockSignals(True)
        self.ui.checkBox_refractory.setChecked(checked)
        self.ui.checkBox_merge_refractory.setChecked(checked)
        self.ui.checkBox_refractory.blockSignals(False)
        self.ui.checkBox_merge_refractory.blockSignals(False)
        self.merge_redraw()

    @QtCore.Slot(bool)
    def on_actionOverview_Mode_triggered(self, checked=None):
        self.ui.checkBox_overview.blockSignals(True)
        self.ui.checkBox_overview.setChecked(checked)
        self.ui.checkBox_overview.blockSignals(False)

        self._overviewmode = checked
        # rather than typing not checked all the time lets just negate it
        checked = not checked
        self.mp_proj.blockSignals(self._overviewmode)
        self.mp_proj_multi.blockSignals(not self._overviewmode)
        self.mp_proj_multi.setVisible(self._overviewmode)
        self.mp_proj.setVisible(not self._overviewmode)

        # Disable the add limit/channel dialogs
        self.ui.pushButton_next_projection.setEnabled(checked)
        self.ui.pushButton_previous_projection.setEnabled(checked)
        self.ui.comboBox_feature_y_chan.setEnabled(checked)
        self.ui.comboBox_feature_x_chan.setEnabled(checked)
        #self.ui.comboBox_feature_y.setEnabled(checked)

        self.ui.pushButton_addLimit.setEnabled(checked)
        self.ui.actionAdd_Limit.setEnabled(checked)

        self.ui.pushButton_deleteLimit.setEnabled(checked)
        self.ui.actionDelete_Limit.setEnabled(checked)

        self.ui.checkBox_ellipse.setEnabled(checked)
        self.ui.actionEliptical.setEnabled(checked)

        self.updateFeaturePlot()

    @QtCore.Slot(bool)
    def on_actionScatter_triggered(self, checked=None):
        if checked:
            self.mp_proj.setPlotType(-2)
            self.mp_proj_multi.setPlotType(-2)
            self.ui.radioButton_scatter.blockSignals(True)
            self.ui.radioButton_scatter.setChecked(True)
            self.ui.radioButton_scatter.blockSignals(False)

    @QtCore.Slot(bool)
    def on_actionDensity_triggered(self, checked=None):
        if checked:
            self.mp_proj.setPlotType(-3)
            self.ui.radioButton_density.blockSignals(True)
            self.ui.radioButton_density.setChecked(True)
            self.ui.radioButton_density.blockSignals(False)

    @QtCore.Slot(bool)
    def on_actionLog_Density_triggered(self, checked=None):
        if checked:
            self.mp_proj.setPlotType(-4)
            self.ui.radioButton_log_density.blockSignals(True)
            self.ui.radioButton_log_density.setChecked(True)
            self.ui.radioButton_log_density.blockSignals(False)

    @QtCore.Slot(bool)
    def on_actionMarker_Size1_triggered(self, checked=None):
        if checked:
            self.mp_proj.setMarkerSize(1)
            self.ui.spinBox_markerSize.blockSignals(True)
            self.ui.spinBox_markerSize.setValue(1)
            self.ui.spinBox_markerSize.blockSignals(False)

    @QtCore.Slot(bool)
    def on_actionMarker_Size3_triggered(self, checked=None):
        if checked:
            self.mp_proj.setMarkerSize(3)
            self.ui.spinBox_markerSize.blockSignals(True)
            self.ui.spinBox_markerSize.setValue(3)
            self.ui.spinBox_markerSize.blockSignals(False)

    @QtCore.Slot(bool)
    def on_actionMarker_Size5_triggered(self, checked=None):
        if checked:
            self.mp_proj.setMarkerSize(5)
            self.ui.spinBox_markerSize.blockSignals(True)
            self.ui.spinBox_markerSize.setValue(5)
            self.ui.spinBox_markerSize.blockSignals(False)

    @QtCore.Slot(bool)
    def on_actionEliptical_triggered(self, checked=None):
        self.mp_proj.setBoundaryElliptical(checked)
        self.ui.checkBox_ellipse.blockSignals(True)
        self.ui.checkBox_ellipse.setChecked(checked)
        self.ui.checkBox_ellipse.blockSignals(False)

    class CommandAutotrimClusters(QtGui.QUndoCommand):
        """Wraps the final autotrim logic to provide undo functionality."""

        def __init__(self, mainwindow, cluster, old_bounds):
            desc = "autotrim cluster"
            super(PyClustMainWindow.CommandAutotrimClusters,
                  self).__init__(desc)
            self.mainwindow = mainwindow
            self.cluster = cluster
            self.old_bounds = old_bounds
            self.new_bounds = cluster.bounds

        def redo(self):
            self.unsaved = self.mainwindow.unsaved
            self.cluster.bounds = self.new_bounds
            self.cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.updateFeaturePlot()
            self.mainwindow.updateClusterDetailPlots()
            self.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved
            self.cluster.bounds = self.old_bounds
            self.cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.updateFeaturePlot()
            self.mainwindow.updateClusterDetailPlots()

    def autotrim_apply(self):
        command = self.CommandAutotrimClusters(self,
                        self.activeClusterRadioButton().cluster_reference,
                        self.trim_cluster.bounds)
        self.undoStack.push(command)
        self.trim_cluster = None
        self.ui.stackedWidget.setCurrentIndex(0)

    def autotrim_cancel(self):
        clust = self.activeClusterRadioButton().cluster_reference
        clust.bounds = self.trim_cluster.bounds
        clust.calculateMembership(self.spikeset)
        self.trim_cluster = None
        self.updateFeaturePlot()
        self.updateClusterDetailPlots()
        self.ui.stackedWidget.setCurrentIndex(0)

    def __init__(self, parent=None):
        QtGui.QMainWindow.__init__(self, parent)

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # There is no spikeset or feature set on load
        self.spikeset = None
        self.current_feature = None
        self.current_filename = None
        self.junk_cluster = None
        self.merge_clust = None

        self.limit_mode = False
        self.unsaved = False

        # data saver and loader object for classifier machine learning
        self.clf_data_saver = None
        self.ui.comboBox_labels.addItems(['', 'Pyramidal', 'Interneuron', 'Junk'])

        # Create the undo stack and actions
        self.undoStack = QtGui.QUndoStack(self)
        self.ui.actionUndo = self.undoStack.createUndoAction(self)
        self.ui.actionRedo = self.undoStack.createRedoAction(self)
        self.ui.menuEdit.addAction(self.ui.actionUndo)
        self.ui.menuEdit.addAction(self.ui.actionRedo)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence.Undo, self)
        shortcut.activated.connect(self.ui.actionUndo.trigger)
        shortcut = QtGui.QShortcut(QtGui.QKeySequence.Redo, self)
        shortcut.activated.connect(self.ui.actionRedo.trigger)

        # Create action groups for the mnu items
        self.ui.agroup_marker = QtGui.QActionGroup(self, exclusive=True)
        self.ui.agroup_marker.addAction(self.ui.actionMarker_Size1)
        self.ui.agroup_marker.addAction(self.ui.actionMarker_Size3)
        self.ui.agroup_marker.addAction(self.ui.actionMarker_Size5)

        self.ui.agroup_scatter = QtGui.QActionGroup(self, exclusive=True)
        self.ui.agroup_scatter.addAction(self.ui.actionScatter)
        self.ui.agroup_scatter.addAction(self.ui.actionDensity)
        self.ui.agroup_scatter.addAction(self.ui.actionLog_Density)

        # action group for autozoom
        self.ui.agroup_autozoom = QtGui.QActionGroup(self, exclusive=True)
        self.ui.agroup_autozoom.addAction(self.ui.actionAutozoom_None)
        self.ui.agroup_autozoom.addAction(self.ui.actionAutozoom_10)
        self.ui.agroup_autozoom.addAction(self.ui.actionAutozoom_25)
        self.ui.agroup_autozoom.addAction(self.ui.actionAutozoom_33)
        self.ui.agroup_autozoom.addAction(self.ui.actionAutozoom_50)
        self.ui.agroup_autozoom.triggered.connect(self.actionAutozoom_triggered)

        # Connect the handlers
        self.ui.checkBox_merge_background.clicked.connect(
                self.merge_redraw)

        self.ui.comboBox_feature_x_chan.currentIndexChanged.connect(
            self.feature_channel_x_changed)

        self.ui.comboBox_feature_x.currentIndexChanged.connect(
            self.feature_x_changed)

        self.ui.comboBox_feature_y_chan.currentIndexChanged.connect(
            self.feature_channel_y_changed)

        self.ui.comboBox_feature_y.currentIndexChanged.connect(
            self.feature_y_changed)

        self.ui.comboBox_merge_c2.currentIndexChanged.connect(
            self.merge_cluster_choice_changed)

        self.ui.pushButton_next_projection.clicked.connect(
            self.button_next_feature_click)

        self.ui.pushButton_previous_projection.clicked.connect(
            self.button_prev_feature_click)

        self.ui.pushButton_wavecutter_done.clicked.connect(
            self.switch_to_maindisplay)

        self.ui.pushButton_wavecutter_redraw.clicked.connect(
            self.updateWavecutterPlot)

        self.ui.lineEdit_wavecutter_count.editingFinished.connect(
            self.updateWavecutterPlot)

        self.ui.checkBox_wavecutter_refractory.stateChanged.connect(
            self.updateWavecutterPlot)

        self.ui.spinBox_wavecutter_channel.valueChanged.connect(
            self.updateWavecutterPlot)

        self.ui.pushButton_wavecutter_add_limit.clicked.connect(
            self.wavecutter_add_limit)

        self.ui.pushButton_wavecutter_remove_limit.clicked.connect(
            self.wavecutter_remove_limits)

        self.ui.pushButton_hide_all.clicked.connect(
            lambda: self.hide_show_all_clusters(True))

        self.ui.pushButton_show_all.clicked.connect(
            lambda: self.hide_show_all_clusters(False))

        self.ui.buttonGroup_trimmer.buttonClicked.connect(
            lambda x: self.ui.stackedWidget_trimmer.setCurrentIndex(-x - 2))

        self.ui.stackedWidget_trimmer.currentChanged.connect(
            lambda x: self.updateWavecutterPlot() if x == 0
                else self.updateOutlierPlot())

        self.ui.actionQuit.triggered.connect(self.close)

        self.ui.pushButton_autotrim_apply.clicked.connect(self.autotrim_apply)
        self.ui.pushButton_autotrim_cancel.clicked.connect(
            self.autotrim_cancel)

        self.ui.pushButton_merge_apply.clicked.connect(self.merge_apply)
        self.ui.pushButton_merge_cancel.clicked.connect(self.merge_cancel)

        # training data for classifier
        self.ui.pushButton_saveLabeledCluster.clicked.connect(
                self.action_saveLabeledCluster)
        self.ui.pushButton_saveLabeledMembers.clicked.connect(
                self.action_saveLabeledMembers)

        self.ui.label_subjectid.setText('')
        self.ui.label_session.setText('')
        self.ui.label_fname.setText('')

        # Set up the cluster list area
        self.labels_container = QtGui.QWidget()
        self.ui.scrollArea_cluster_list.setWidget(self.labels_container)

        layout = QtGui.QVBoxLayout(self.labels_container)
        self.buttonGroup_cluster = QtGui.QButtonGroup()
        self.buttonGroup_cluster.setExclusive(True)
        self.buttonGroup_cluster.buttonClicked.connect(
             self.update_active_cluster)

        layout.addItem(QtGui.QSpacerItem(20, 40, QtGui.QSizePolicy.Minimum,
            QtGui.QSizePolicy.Expanding))

        # Add the unclustered entry
        layout = self.labels_container.layout()

        hlayout = QtGui.QHBoxLayout()

        self.ui.checkBox_show_unclustered = QtGui.QCheckBox(
                self.labels_container)
        self.ui.checkBox_show_unclustered.setChecked(True)
        hlayout.addWidget(self.ui.checkBox_show_unclustered)

        hlayout.addWidget(QtGui.QLabel('Unclustered',
            parent=self.labels_container))
        hlayout.addItem(QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Minimum))

        self.ui.checkBox_show_unclustered_exclusive = QtGui.QCheckBox(
                self.labels_container)
        self.ui.checkBox_show_unclustered_exclusive.setChecked(True)
        hlayout.addWidget(self.ui.checkBox_show_unclustered_exclusive)

        layout.insertLayout(0, hlayout)

        # Create the projection widget
        self.mp_proj = featurewidget.ProjectionWidget()
        self.mp_proj.setAutoFillBackground(True)
        self.mp_proj.setObjectName("mplwidget_projection")
        self.ui.verticalLayout_3.addWidget(self.mp_proj)

        self.mp_proj_multi = multiplotwidget.MultiplotWidget()
        self.mp_proj_multi.setAutoFillBackground(True)
        self.mp_proj_multi.setObjectName("mplwidget_projection_multi")
        self.ui.verticalLayout_3.addWidget(self.mp_proj_multi)
        self._overviewmode = False
        self.mp_proj_multi.setVisible(False)

        # Connect the relevant signals/slots
        self.mp_proj.featureRedrawRequired.connect(self.updateFeaturePlot)
        self.mp_proj_multi.featureRedrawRequired.connect(
            self.updateFeaturePlot)
        self.mp_proj_multi.selectFromOverview_signal.connect(self.selectProjectionFromOverview)

        # Signals for unclustered show checkbox
        self.ui.checkBox_show_unclustered.stateChanged.connect(
                self.mp_proj.setShowUnclustered)
        self.ui.checkBox_show_unclustered.stateChanged.connect(
                self.mp_proj_multi.setShowUnclustered)

        # Signals for unclustered exclusive checkbox
        self.ui.checkBox_show_unclustered_exclusive.stateChanged.connect(
                self.mp_proj.setUnclusteredExclusive)
        self.ui.checkBox_show_unclustered_exclusive.stateChanged.connect(
                self.mp_proj_multi.setUnclusteredExclusive)

        # Signals for marker size
        self.ui.spinBox_markerSize.valueChanged.connect(
                self.mp_proj.setMarkerSize)
        self.ui.spinBox_markerSize.valueChanged.connect(
                self.mp_proj_multi.setMarkerSize)

        # Signal for polygon boundary drawn
        self.mp_proj.polygonBoundaryDrawn.connect(self.addBoundary)
        self.mp_proj.ellipseBoundaryDrawn.connect(self.addBoundary)

        # Create shorter handles to important widgets
        self.mp_wave = self.ui.mplwidget_waveform
        self.mp_isi = self.ui.mplwidget_isi
        self.mp_wavecutter = self.ui.mplwidget_wavecutter
        self.mp_outlier = self.ui.mplwidget_outliers
        self.mp_drift = self.ui.mplwidget_drift

        pal = self.palette().window().color()
        bgcolor = (pal.red() / 255.0, pal.blue() / 255.0, pal.green() / 255.0)

        # Set the window background color on plots
        self.mp_wave.figure.clear()
        self.mp_wave.figure.set_facecolor(bgcolor)

        self.mp_isi.figure.clear()
        self.mp_isi.figure.set_facecolor(bgcolor)

        self.mp_wavecutter.figure.clear()
        self.mp_wavecutter.figure.set_facecolor(bgcolor)

        self.mp_outlier.figure.clear()
        self.mp_outlier.figure.set_facecolor(bgcolor)

        self.mp_drift.figure.clear()
        self.mp_drift.figure.set_facecolor(bgcolor)

        # Set up autotrim and merge plots
        self.ui.mplwidget_autotrim.figure.clear()
        self.ui.mplwidget_autotrim.figure.set_facecolor(bgcolor)

        self.ui.mplwidget_merge.figure.clear()
        self.ui.mplwidget_merge.figure.set_facecolor(bgcolor)

        self.ui.mplwidget_merge_wv.figure.clear()
        self.ui.mplwidget_merge_wv.figure.set_facecolor(bgcolor)

        self.wave_limit_mode = False
        self.mp_wavecutter.mpl_connect('button_press_event',
            self.wavecutter_onMousePress)
        self.mp_wavecutter.mpl_connect('button_release_event',
            self.wavecutter_onMouseRelease)
        self.mp_wavecutter.mpl_connect('motion_notify_event',
            self.wavecutter_onMouseMove)

        self.ui.comboBox_feature_x.clear()
        self.ui.comboBox_feature_y.clear()
        self.ui.comboBox_feature_x_chan.clear()
        self.ui.comboBox_feature_y_chan.clear()

        self.activeClusterRadioButton = self.buttonGroup_cluster.checkedButton

        # Set up ISI plot axes
        self.mp_isi.figure.clear()
        self.mp_isi.figure.subplots_adjust(hspace=0.0001, wspace=0.0001,
            bottom=0.15, top=1, left=0.0, right=1)
        self.mp_isi.axes = self.mp_isi.figure.add_subplot(1, 1, 1)
        self.mp_isi.axes.hold(False)
        self.mp_isi.axes.set_xscale('log')
        self.mp_isi.axes.set_xlim([0.1, 1e5])
        refractory_line = mpl.lines.Line2D([2, 2], self.mp_isi.axes.get_ylim(),
            color='r', linestyle='--')
        self.mp_isi.axes.add_line(refractory_line)
        burst_line = mpl.lines.Line2D([20, 20], self.mp_isi.axes.get_ylim(),
            color='b', linestyle='--')
        self.mp_isi.axes.add_line(burst_line)
        theta_line = mpl.lines.Line2D([125, 125], self.mp_isi.axes.get_ylim(),
            color='g', linestyle='--')
        self.mp_isi.axes.add_line(theta_line)
        self.mp_isi.axes.set_xticks([1e1, 1e2, 1e3, 1e4])
        self.mp_isi.draw()

        # Set up the drift axes
        self.mp_drift.figure.clear()
        self.mp_drift.figure.subplots_adjust(hspace=0.0001, wspace=0.0001,
            bottom=0, top=1, left=0.15, right=1)
        self.mp_drift.axes = self.mp_drift.figure.add_subplot(1, 1, 1)
        self.mp_drift.axes.set_xticks([])
        self.mp_drift.axes.set_yticks([])
        self.mp_drift.draw()

        # Set up the outlier axes
        self.mp_outlier.figure.clear()
        self.mp_outlier.axes = self.mp_outlier.figure.add_subplot(1, 1, 1)
        self.mp_outlier.draw()

        # Set up autotrim plot and merge plots
        self.ui.mplwidget_autotrim.figure.subplots_adjust(bottom=0, top=1,
                left=0, right=1, hspace=0.01, wspace=0.01)
        self.ui.mplwidget_merge.figure.subplots_adjust(bottom=0, top=1,
                left=0, right=1, hspace=0.01, wspace=0.01)
        self.ui.mplwidget_merge_wv.figure.subplots_adjust(bottom=0, top=1,
                left=0, right=1, hspace=0.01, wspace=0.01)

        # Clear the stats labels
        self.ui.label_spike_count.setText('')
        self.ui.label_mean_rate.setText('')
        self.ui.label_burst.setText('')
        self.ui.label_csi.setText('')
        self.ui.label_refr_count.setText('')
        self.ui.label_refr_fp.setText('')
        self.ui.label_refr_frac.setText('')
        self.ui.label_isolation.setText('')

        # Set up the waveform axes
        self.mp_wavecutter.figure.clear()
        self.mp_wavecutter.figure.subplots_adjust(hspace=0.0001, wspace=0.0001,
            bottom=0, top=1, left=0.0, right=1)
        self.mp_wavecutter.axes = \
            self.mp_wavecutter.figure.add_subplot(1, 1, 1)
        self.mp_wavecutter.axes.hold(False)
        self.mp_wavecutter.axes.set_xticks([])
        self.mp_wavecutter.axes.set_yticks([])
        self.mp_wavecutter.draw()

        self.ui.stackedWidget.setCurrentIndex(0)

        layout = self.labels_container.layout()

        hlayout = QtGui.QHBoxLayout()

        self.checkBox_junk = QtGui.QCheckBox()
        self.checkBox_junk.setChecked(True)
        self.checkBox_junk.stateChanged.connect(self.updateFeaturePlot)

        self.radioButton_junk = QtGui.QRadioButton()
        self.radioButton_junk.setChecked(True)
        spacer = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Minimum)

        hlayout.addWidget(self.checkBox_junk)
        hlayout.addItem(spacer)
        hlayout.addWidget(self.radioButton_junk)
        hlayout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
        self.buttonGroup_cluster.addButton(self.radioButton_junk)

        label = QtGui.QLabel()
        label.setText('Junk')

        hlayout.addItem(QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
            QtGui.QSizePolicy.Minimum))

        hlayout.addWidget(label)
        layout.insertLayout(layout.count() - 1, hlayout)

        # list containing the cluster check/radio/color buttons
        self.cluster_ui_buttons = []


    def hide_show_all_clusters(self, hidden):
        radio = self.activeClusterRadioButton()
        for cluster, ui in zip(self.spikeset.clusters,
                               self.cluster_ui_buttons):
            if ui[0] == radio:
                continue

            ui[3].blockSignals(True)
            ui[3].setChecked(not hidden)
            ui[3].cluster_reference._visible = not hidden
            ui[3].blockSignals(False)

            if hidden:
                # dont show this on show all
                self.checkBox_junk.blockSignals(True)
                self.checkBox_junk.setChecked(False)
                self.checkBox_junk.blockSignals(False)

        if self._overviewmode:
            self.mp_proj_multi.blockSignals(True)
        else:
            self.mp_proj.blockSignals(True)

        self.ui.checkBox_show_unclustered.setChecked(not hidden)

        if self._overviewmode:
            self.mp_proj_multi.blockSignals(False)
        else:
            self.mp_proj.blockSignals(False)

        self.updateFeaturePlot()

    # When we switch clusters, correctly enabled/disable cluster
    # checkboxes, and tell the projection widget to stop drawing
    def update_active_cluster(self):
        self.updateClusterDetailPlots()
        for ui in self.cluster_ui_buttons:
            ui[3].setEnabled(True)  # enable all check boxes
        if self.activeClusterRadioButton() == self.radioButton_junk:
            return
        for ui in self.cluster_ui_buttons:
            if ui[0] == self.activeClusterRadioButton():
                check = ui[3]
                check.setEnabled(False)
                check.setChecked(True)
        self.mp_proj.stopMouseAction()

    # Add a new cluster by generating a color, creating GUI elements, etc.

    def cluster_checkbox_check_event(self, state):
        """Called each time a cluster visible checkbox state changes."""
        checkbox = self.sender()
        checkbox.cluster_reference._visible = checkbox.isChecked()
        self.updateFeaturePlot()

    def update_ui_cluster_buttons(self, selected_cluster=None):
        # remove all the UI buttons for clusters
        for ui_container in self.cluster_ui_buttons:
            radio = ui_container[0]
            layout = ui_container[1]
            cbut = ui_container[2]
            check = ui_container[3]

            self.buttonGroup_cluster.removeButton(radio)

            # for reasons unclear to me using removeitem segfaults
            labels_cont = self.labels_container.layout()
            for  i in range(labels_cont.count()):
                if labels_cont.itemAt(i) is layout:
                    labels_cont.takeAt(i)
            #labels_cont.removeItem(layout)

            for i in range(layout.count()):
                if layout.itemAt(i).widget():
                    layout.itemAt(i).widget().close()
                    layout.itemAt(i).widget().deleteLater()

            radio.cluster_reference = None
            cbut.cluster_reference = None
            check.cluster_reference = None
        self.cluster_ui_buttons = []

        # Now add buttons for each cluster
        layout = self.labels_container.layout()
        for i, new_cluster in enumerate(self.spikeset.clusters):
            hlayout = QtGui.QHBoxLayout()

            check = QtGui.QCheckBox()
            if i != len(self.spikeset.clusters) - 1:
                check.setChecked(new_cluster._visible)
            else:
                check.setChecked(True)
                new_cluster._visible = True
            check.stateChanged.connect(self.cluster_checkbox_check_event)

            radio = QtGui.QRadioButton()
            radio.setChecked(True)
            spacer = QtGui.QSpacerItem(40, 20, QtGui.QSizePolicy.Expanding,
                QtGui.QSizePolicy.Minimum)

            hlayout.addWidget(check)
            hlayout.addItem(spacer)
            hlayout.addWidget(radio)
            hlayout.setSizeConstraint(QtGui.QLayout.SetMaximumSize)
            self.buttonGroup_cluster.addButton(radio)

            cbut = QtGui.QPushButton()
            cbut.setMaximumSize(20, 20)
            cbut.setText(str(i + 1))
            cbut.setStyleSheet(
                    "QPushButton {background-color: rgb(%d, %d, %d);"
                    % new_cluster.color + " font: bold;" +
                    " color: rgb(255, 255, 255);}")
            cbut.clicked.connect(self.button_cluster_color)

            hlayout.addItem(QtGui.QSpacerItem(40, 20,
                    QtGui.QSizePolicy.Expanding,
                    QtGui.QSizePolicy.Minimum))
            hlayout.addWidget(cbut)
            layout.insertLayout(layout.count() - 1, hlayout)

            # add the gui elements to the cluster reference
            # so we can access them when we need to
            cbut.cluster_reference = new_cluster
            radio.cluster_reference = new_cluster
            check.cluster_reference = new_cluster

            self.cluster_ui_buttons.append((radio, hlayout, cbut, check))
        if selected_cluster is not None:
            for (radio, hlayout, cbut, check) in self.cluster_ui_buttons:
                if radio.cluster_reference == selected_cluster:
                    radio.setChecked(True)
                    break

    class CommandDeleteCluster(QtGui.QUndoCommand):
        """Wraps the delete cluster logic to provide undo functionality."""

        def __init__(self, mainwindow):
            desc = "Delete cluster"
            super(PyClustMainWindow.CommandDeleteCluster, self).__init__(desc)
            self.mainwindow = mainwindow
            self.cluster = \
                self.mainwindow.activeClusterRadioButton().cluster_reference
            self.index = self.mainwindow.spikeset.clusters.index(self.cluster)

        def redo(self):
            self.unsaved = self.mainwindow.unsaved
            self.mainwindow.delete_cluster(self.cluster)
            self.mainwindow.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved
            self.mainwindow.spikeset.clusters.insert(self.index, self.cluster)
            self.mainwindow.update_ui_cluster_buttons(self.cluster)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.mainwindow.updateClusterDetailPlots()

    @QtCore.Slot()
    def on_actionDelete_Cluster_triggered(self):
        command = self.CommandDeleteCluster(self)
        self.undoStack.push(command)

    def delete_cluster(self, cluster=None):
        if cluster is None:
            if ((not self.activeClusterRadioButton()) or
                    self.activeClusterRadioButton() == self.radioButton_junk):
                return
            cluster = self.activeClusterRadioButton().cluster_reference

        self.spikeset.clusters.remove(cluster)
        self.update_ui_cluster_buttons()
        self.updateFeaturePlot()
        self.updateClusterDetailPlots()
        self.unsaved = True

    # Tell the projection widget to start drawing a boundary
    @QtCore.Slot()
    def on_actionAdd_Limit_triggered(self):
        if not self.spikeset:
            return

        if self.activeClusterRadioButton():
            self.mp_proj.drawBoundary(
                    self.activeClusterRadioButton().cluster_reference.color)

    class CommandAddBoundary(QtGui.QUndoCommand):
        """Wraps the add boundary logic to provide undo functionality."""

        def __init__(self, mainwindow, cluster, bound):
            desc = "add boundary"
            super(PyClustMainWindow.CommandAddBoundary, self).__init__(desc)
            self.mainwindow = mainwindow
            self.bound = bound
            self.cluster = cluster

        def redo(self):
            self.unsaved = self.mainwindow.unsaved

            if self.cluster != self.mainwindow.junk_cluster:
                self.bounds_before = self.cluster.bounds
                self.cluster.addBoundary(self.bound)
                self.cluster.calculateMembership(self.mainwindow.spikeset)
            else:
                self.cluster.add_bounds.append(self.bound)
                self.cluster.calculateMembership(self.mainwindow.spikeset)
                self.mainwindow.mp_proj.resetLimits()
                self.mainwindow.mp_proj_multi.resetLimits()

            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.mainwindow.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved

            if self.cluster != self.mainwindow.junk_cluster:
                self.cluster.bounds = self.bounds_before
                self.cluster.calculateMembership(self.mainwindow.spikeset)
            else:
                self.cluster.add_bounds.remove(self.bound)
                self.cluster.calculateMembership(self.mainwindow.spikeset)
                self.mainwindow.mp_proj.resetLimits()
                self.mainwindow.mp_proj_multi.resetLimits()

            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()

    def addBoundary(self, bound):
        """Takes the bound returned by the plot widget and implements it."""
        if not self.activeClusterRadioButton():
            return

        cluster = self.activeClusterRadioButton().cluster_reference
        command = self.CommandAddBoundary(self, cluster, bound)
        self.undoStack.push(command)

    # Delete the active cluster boundary on the current projection

    class CommandDeleteBoundary(QtGui.QUndoCommand):
        """Wraps the add boundary logic to provide undo functionality."""

        def __init__(self, mainwindow, cluster, features, channels):
            desc = "delete boundary"
            super(PyClustMainWindow.CommandDeleteBoundary, self).__init__(desc)
            self.mainwindow = mainwindow
            self.cluster = cluster
            self.features = features
            self.channels = channels

        def redo(self):
            self.unsaved = self.mainwindow.unsaved

            if self.cluster != self.mainwindow.junk_cluster:
                self.bounds_before = self.cluster.bounds
                self.cluster.removeBound(self.features, self.channels)
                self.cluster.calculateMembership(self.mainwindow.spikeset)
            else:
                self.bounds_before = self.cluster.add_bounds
                self.cluster.removeBound(self.features, self.channels, 'add')
                self.cluster.calculateMembership(self.mainwindow.spikeset)
                self.mainwindow.mp_proj.resetLimits()
                self.mainwindow.mp_proj_multi.resetLimits()

            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.mainwindow.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved

            if self.cluster != self.mainwindow.junk_cluster:
                self.cluster.bounds = self.bounds_before
                self.cluster.calculateMembership(self.mainwindow.spikeset)
            else:
                self.cluster.add_bounds = self.bounds_before
                self.cluster.calculateMembership(self.mainwindow.spikeset)
                self.mainwindow.mp_proj.resetLimits()
                self.mainwindow.mp_proj_multi.resetLimits()

            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()

    @QtCore.Slot()
    def on_actionDelete_Limit_triggered(self):
        if (not self.spikeset) or (not self.activeClusterRadioButton()):
            return

        feature_x = self.ui.comboBox_feature_x.currentText()
        feature_y = self.ui.comboBox_feature_y.currentText()
        feature_x_chan = int(self.ui.comboBox_feature_x_chan.currentText()) - 1
        feature_y_chan = int(self.ui.comboBox_feature_y_chan.currentText()) - 1

        cluster = self.activeClusterRadioButton().cluster_reference
        # check if there actually is a boundary first
        if cluster == self.junk_cluster:
            btype = 'add'
        else:
            btype = 'limits'
        bns = cluster.getBoundaries(feature_x, feature_x_chan, feature_y,
                                feature_y_chan, btype)
        if bns == []:
            return

        command = self.CommandDeleteBoundary(self, cluster,
                (feature_x, feature_y),  (feature_x_chan, feature_y_chan))
        self.undoStack.push(command)

    def button_cluster_color(self):
        color = QtGui.QColorDialog.getColor(
            QtGui.QColor(*self.sender().cluster_reference.color),
            self, "ColorDialog")

        self.sender().setStyleSheet(
            "QPushButton {background-color: rgb(%d, %d, %d)}"
            % (color.red(), color.green(), color.blue()))

        self.sender().cluster_reference.color = (color.red(),
            color.green(), color.blue())

        self.updateFeaturePlot()

    class CommandAddCluster(QtGui.QUndoCommand):
        """Wraps the add cluster logic to provide undo functionality."""

        def __init__(self, mainwindow):
            desc = "Add cluster"
            super(PyClustMainWindow.CommandAddCluster, self).__init__(desc)
            self.mainwindow = mainwindow
            self.cluster = None

        def redo(self):
            rdio = self.mainwindow.activeClusterRadioButton()
            if rdio:
                self.old_selection = rdio.cluster_reference
            else:
                self.old_selection = None
            self.unsaved = self.mainwindow.unsaved

            if self.cluster is None:
                self.cluster = self.mainwindow.spikeset.addCluster()
                self.cluster._visible = True
            else:
                self.mainwindow.spikeset.clusters.append(self.cluster)

            self.mainwindow.update_ui_cluster_buttons(self.cluster)
            self.mainwindow.update_active_cluster()
            self.mainwindow.unsaved = True

        def undo(self):
            self.mainwindow.spikeset.clusters.remove(self.cluster)
            self.mainwindow.unsaved = self.unsaved
            self.mainwindow.update_ui_cluster_buttons(self.old_selection)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.mainwindow.updateClusterDetailPlots()

    @QtCore.Slot()
    def on_actionAdd_Cluster_triggered(self):
        command = self.CommandAddCluster(self)
        self.undoStack.push(command)

    def feature_x_changed(self, index):
        if index == -1:
            return

        # Set up blocks to things that get changed
        self.mp_proj.blockSignals(True)
        self.ui.comboBox_feature_y.blockSignals(True)
        self.ui.comboBox_feature_y_chan.blockSignals(True)
        self.ui.comboBox_feature_x_chan.blockSignals(True)

        # Set the options in the y feature box accordingly
        current_x = str(self.ui.comboBox_feature_x.currentText())

        # Possible y features
        # is always going to be 0 first
        valid_y = self.spikeset.featureByName(current_x).valid_y_features(0)
        self.ui.comboBox_feature_y.clear()
        for (name, chans) in valid_y:
            self.ui.comboBox_feature_y.addItem(name)
        current_y = str(self.ui.comboBox_feature_y.currentText())

        # Possible x channels
        valid_x = self.spikeset.featureByName(current_x).valid_x_features()

        self.ui.comboBox_feature_x_chan.clear()
        for i in valid_x:
            self.ui.comboBox_feature_x_chan.addItem(str(i + 1))

        # Possible y channels
        valid_y_chans = valid_y[0][1]
        # y_chans is None if it should be all channels for a different feature
        # type, since we have no way of knowing how many there will be apriori,
        #we pass none
        if valid_y_chans is None:
            valid_y_chans = range(0,
                self.spikeset.featureByName(valid_y[0][0]).channels)

        self.ui.comboBox_feature_y_chan.clear()
        for i in valid_y_chans:
            self.ui.comboBox_feature_y_chan.addItem(str(i + 1))

        # If we're plotting the same thing on X and Y remove the last chan in X
        chans = self.spikeset.featureByName(current_x).channels
        index = self.ui.comboBox_feature_x_chan.findText(str(chans))

        if current_x == current_y and index >= 0:
            if self.ui.comboBox_feature_x_chan.currentText() == str(chans):
                self.ui.comboBox_feature_x_chan.setCurrentIndex(chans - 2)
            self.ui.comboBox_feature_x_chan.removeItem(index)

        # Update the feature widget
        self.mp_proj.setFeatureX(current_x)
        self.mp_proj.setChanX(int(
                self.ui.comboBox_feature_x_chan.currentText()) - 1)
        self.mp_proj.setFeatureY(
                self.ui.comboBox_feature_y.currentText())
        self.mp_proj.setChanY(int(
                self.ui.comboBox_feature_y_chan.currentText()) - 1)
        # and the multi plot widget
        self.mp_proj_multi.setFeatureX(current_x)

        # Remove the blocks
        self.mp_proj.blockSignals(False)
        self.ui.comboBox_feature_y.blockSignals(False)
        self.ui.comboBox_feature_y_chan.blockSignals(False)
        self.ui.comboBox_feature_x_chan.blockSignals(False)
        # And redraw the widget
        self.updateFeaturePlot()

    def feature_channel_x_changed(self, index, setYtoLast=False):
        """Event fires only when X channel directly changes."""
        if index == -1:
            return

        # Set up blocks to things that get changed
        self.mp_proj.blockSignals(True)
        self.ui.comboBox_feature_y_chan.blockSignals(True)

        # Get the valid y channel options based on px, cx, py
        current_x = str(self.ui.comboBox_feature_x.currentText())
        current_y = str(self.ui.comboBox_feature_y.currentText())

        valid_y = self.spikeset.featureByName(current_x).valid_y_features(
            int(self.ui.comboBox_feature_x_chan.currentText()) - 1)
        valid_y_chans = [chans for (name, chans) in valid_y
            if name == current_y][0]

        if setYtoLast:
            current = len(valid_y_chans) - 1
        else:
            current = 0

        # y_chans is None if it should be all channels for a different
        # feature type, since we have no way of knowing how many there
        # will be apriori, we pass none
        if valid_y_chans is None:
            valid_y_chans = range(0,
                self.spikeset.featureByName(current_y).channels)

        self.ui.comboBox_feature_y_chan.clear()
        for i in valid_y_chans:
            self.ui.comboBox_feature_y_chan.addItem(str(i + 1))
        self.ui.comboBox_feature_y_chan.setCurrentIndex(current)

        self.mp_proj.setChanX(int(
                self.ui.comboBox_feature_x_chan.currentText()) - 1)
        self.mp_proj.setChanY(int(
                self.ui.comboBox_feature_y_chan.currentText()) - 1)

        # Remove the blocks
        self.mp_proj.blockSignals(False)
        self.ui.comboBox_feature_y_chan.blockSignals(False)
        # and redraw feature widget
        self.updateFeaturePlot()

    def feature_y_changed(self, index):
        """Event fires when the Y feature directly changes."""
        if index == -1:
            return

        # Blocks for things that change
        self.mp_proj.blockSignals(True)
        self.ui.comboBox_feature_y_chan.blockSignals(True)
        self.ui.comboBox_feature_x_chan.blockSignals(True)

        # Set the options in the y channel box accordingly
        current_x = str(self.ui.comboBox_feature_x.currentText())
        current_y = str(self.ui.comboBox_feature_y.currentText())

        # If x and y features are the same, remove the maximal channel number
        chans = self.spikeset.featureByName(current_x).channels
        index = self.ui.comboBox_feature_x_chan.findText(str(chans))

        if current_x == current_y and index >= 0:
            if self.ui.comboBox_feature_x_chan.currentText() == str(chans):
                self.ui.comboBox_feature_x_chan.setCurrentIndex(chans - 2)
            self.ui.comboBox_feature_x_chan.removeItem(index)
        # If they're not the same we might want it back in
        if current_x != current_y and index == -1:
            self.ui.comboBox_feature_x_chan.insertItem(chans - 1, str(chans))

        valid_y = self.spikeset.featureByName(current_x).valid_y_features(
            int(self.ui.comboBox_feature_x_chan.currentText()) - 1)
        valid_y_chans = [channels for (name, channels) in valid_y
            if name == current_y][0]
        # y_chans is None if it should be all channels for a different feature
        # type, since we have no way of knowing how many there will be apriori,
        #we pass none
        if valid_y_chans is None:
            valid_y_chans = range(0,
                self.spikeset.featureByName(current_y).channels)

        self.ui.comboBox_feature_y_chan.clear()
        for i in valid_y_chans:
            self.ui.comboBox_feature_y_chan.addItem(str(i + 1))

        self.mp_proj.setFeatureY(current_y)
        self.mp_proj.setChanX(int(
                self.ui.comboBox_feature_x_chan.currentText()) - 1)
        self.mp_proj.setChanY(int(
                self.ui.comboBox_feature_y_chan.currentText()) - 1)
        self.mp_proj_multi.setFeatureY(current_y)

        # Remove the blocks
        self.mp_proj.blockSignals(False)
        self.ui.comboBox_feature_y_chan.blockSignals(False)
        self.ui.comboBox_feature_x_chan.blockSignals(False)
        # and draw the feature plot
        self.updateFeaturePlot()

    def feature_channel_y_changed(self, index):
        """Event fires when Y channel directly changes."""
        self.mp_proj.blockSignals(True)

        if index == -1:
            return
        self.mp_proj.setChanY(int(
            self.ui.comboBox_feature_y_chan.currentText()) - 1)

        self.mp_proj.blockSignals(False)
        self.updateFeaturePlot()

    def button_next_feature_click(self):
        """Steps forward to the next feature, looping as needed."""
        # if we're at the end of the current y channel range
        if (self.ui.comboBox_feature_y_chan.currentIndex()
                == self.ui.comboBox_feature_y_chan.count() - 1):
            # but not at the end of the current x channel range
            if not (self.ui.comboBox_feature_x_chan.currentIndex()
                    == self.ui.comboBox_feature_x_chan.count() - 1):
                # step x to the next projection
                self.ui.comboBox_feature_x_chan.setCurrentIndex(
                    self.ui.comboBox_feature_x_chan.currentIndex() + 1)
        else:
            self.ui.comboBox_feature_y_chan.setCurrentIndex(
                self.ui.comboBox_feature_y_chan.currentIndex() + 1)

    def button_prev_feature_click(self):
        """Steps back to the previous feature channel, looping as needed."""
        if self.ui.comboBox_feature_y_chan.currentIndex() == 0:
            if not self.ui.comboBox_feature_x_chan.currentIndex() == 0:
                # step x to the previous projection
                # but this time we do want to loop around the y projection

                self.ui.comboBox_feature_x_chan.blockSignals(True)
                self.ui.comboBox_feature_x_chan.setCurrentIndex(
                        self.ui.comboBox_feature_x_chan.currentIndex() - 1)
                self.ui.comboBox_feature_x_chan.blockSignals(False)
                self.feature_channel_x_changed(
                        self.ui.comboBox_feature_x_chan.currentIndex(),
                        setYtoLast=True)
        else:
            self.ui.comboBox_feature_y_chan.setCurrentIndex(
                self.ui.comboBox_feature_y_chan.currentIndex() - 1)

    def load_spikefile(self, fname):
        """Loads a spikeset from the given filename, imports clusters, etc."""
        if self.unsaved:
            reply = QtGui.QMessageBox.question(self, 'Save',
                "Do you want to save before loading?", QtGui.QMessageBox.Yes |
                QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel,
                QtGui.QMessageBox.Yes)

            if reply == QtGui.QMessageBox.Cancel:
                return
            if reply == QtGui.QMessageBox.Yes:
                self.on_actionSave_triggered()
            if reply == QtGui.QMessageBox.No:
                pass

        print('')
        print('Trying to load file: ', fname)
        print('Clearing current clusters')
        if self.spikeset is not None:
            for cluster in self.spikeset.clusters[:]:
                self.delete_cluster(cluster)

        self.spikeset = None
        self.current_feature = None
        self.junk_cluster = None
        self.current_filename = None
        self.limit_mode = False
        self.unsaved = False
        self.ui.checkBox_show_unclustered.setChecked(True)
        self.ui.checkBox_show_unclustered_exclusive.setChecked(True)

        print('Loading file', fname)
        t1 = time.clock()
        self.spikeset = spikeset_io.loadSpikeset(fname)
        t2 = time.clock()
        print('Loaded', self.spikeset.N, 'spikes in ', (t2 - t1), 'seconds')

        self.ui.label_subjectid.setText(self.spikeset.subject.decode())
        self.ui.label_session.setText(self.spikeset.session.decode())
        self.ui.label_fname.setText(
            os.path.splitext(os.path.split(fname)[1])[0])
        self.curfile = fname

        # Load info to keep track of which clusters have been added to DataSaver for ML
        if self.clf_data_saver == None:
            self.clf_data_saver = classifier.DataSaver(self.ui.label_subjectid.text(),
                    self.ui.label_session.text(), self.ui.label_fname.text())
        else:
            self.clf_data_saver.subject = self.ui.label_subjectid.text()
            self.clf_data_saver.session = self.ui.label_session.text()
            self.clf_data_saver.fname = self.ui.label_fname.text()

        # conditionally disable save-labeled-cluster button
        clust_num_h = filter(lambda (i, c): self.activeClusterRadioButton().cluster_reference
                                            is c, enumerate(self.spikeset.clusters))
        clust_num = clust_num_h[0][0] if not clust_num_h == [] else None
    
        # mean button
        if self.clf_data_saver.is_saved(self.ui.label_subjectid.text(), self.ui.label_session.text(),
                                 self.ui.label_fname.text(), clust_num):
            self.ui.pushButton_saveLabeledCluster.setEnabled(False)
        else:
            self.ui.pushButton_saveLabeledCluster.setEnabled(True)

        # members button
        if self.clf_data_saver.is_saved(self.ui.label_subjectid.text(), self.ui.label_session.text(),
                                 self.ui.label_fname.text(), clust_num, mode='members'):
            self.ui.pushButton_saveLabeledMembers.setEnabled(False)
        else:
            self.ui.pushButton_saveLabeledMembers.setEnabled(True)

        self.undoStack.clear()

        self.t_bins = np.arange(self.spikeset.time[0],
            self.spikeset.time[-1], 60e6)
        self.t_bin_centers = ((self.t_bins[0:-1] + self.t_bins[1:]) / 2
             - self.spikeset.time[0]) / 60e6

        # Initiate the waveform axes
        self.mp_wave.figure.clear()
        self.mp_wave.figure.subplots_adjust(hspace=0.0001, wspace=0.0001,
            bottom=0, top=1, left=0.0, right=1)
        self.mp_wave.axes = [None] * self.spikeset.C
        subplots_x = int(np.ceil(np.sqrt(self.spikeset.C)))
        subplots_y = int(np.ceil(float(self.spikeset.C) / subplots_x))
        for i in range(0, self.spikeset.C):
            if i == 0:
                self.mp_wave.axes[i] = self.mp_wave.figure.add_subplot(
                        subplots_y, subplots_x, i + 1)
            else:
                self.mp_wave.axes[i] = self.mp_wave.figure.add_subplot(
                        subplots_y, subplots_x, i + 1,
                        sharey=self.mp_wave.axes[0])
            self.mp_wave.axes[i].hold(False)
            self.mp_wave.axes[i].set_xticks([])
            self.mp_wave.axes[i].set_yticks([])

        # Reset the limits
        self.junk_cluster = spikeset.Cluster(self.spikeset)
        self.junk_cluster.color = (255, 0, 0)
        self.junk_cluster.check = self.checkBox_junk
        self.checkBox_junk.blockSignals(True)
        self.checkBox_junk.setChecked(False)
        self.checkBox_junk.blockSignals(False)
        self.radioButton_junk.cluster_reference = self.junk_cluster

        # Add some points to the junk cluster since true spikes are rarely >1mv
        d = np.max(self.spikeset.featureByName('Peak').data, axis=0) + 10
        # similarly check for peak < 10
        d2 = np.min(self.spikeset.featureByName('Peak').data, axis=0)
        d2[d2 >= 0] = -0.001
        jthresh = 1e3
        for i in range(d.shape[0]):
            for j in range(i + 1, d.shape[0]):
                jbounds = np.zeros((4, 2))
                # bounds between 1mv and max peak
                jbounds[0, :] = [min([d[i], jthresh]), min([d[j], jthresh])]
                jbounds[1, :] = [max([d[i], jthresh]), min([d[j], jthresh])]
                jbounds[2, :] = [max([d[i], jthresh]), max([d[j], jthresh])]
                jbounds[3, :] = [min([d[i], jthresh]), max([d[j], jthresh])]
                jbound = boundaries.BoundaryPolygon2D()
                jbound = jbound.init2(('Peak', 'Peak'),
                        (i, j), jbounds)

                # bounds between negative peaks and zero
                jbounds = np.zeros((6, 2))
                jbounds[0, :] = [d2[i], d2[j]]  # bottom left
                jbounds[1, :] = [d[i], d2[j]]   # bottom right
                jbounds[2, :] = [d[i], 0]       # top right
                jbounds[3, :] = [0, 0]          # origin
                jbounds[4, :] = [0, d[j]]       # top middle
                jbounds[5, :] = [d2[i], d[j]]   # top left
                jbound = boundaries.BoundaryPolygon2D()
                jbound = jbound.init2(('Peak', 'Peak'),
                        (i, j), jbounds)

                self.junk_cluster.add_bounds.append(jbound)

        self.junk_cluster.calculateMembership(self.spikeset)
        self.mp_proj.resetLimits()
        self.mp_proj_multi.resetLimits()

        # Autoload any existing boundary files for this data set
        imported_bounds = False
        boundfilename = str(fname) + os.extsep + 'bounds'
        if os.path.exists(boundfilename) and (not imported_bounds):
            print("Found boundary file", boundfilename)
            self.spikeset.importBounds(boundfilename)
            imported_bounds = True
        (root, ext) = os.path.splitext(str(fname))
        boundfilename = root + os.extsep + 'bounds'
        if os.path.exists(boundfilename):
            print("Found boundary file", boundfilename)
            self.spikeset.importBounds(boundfilename)
            imported_bounds = True

        # see if there is a modified klustakwik cluster file to load
        ackkfilename = fname + os.extsep + 'ackk'
        if (not imported_bounds) and os.path.exists(ackkfilename):
            self.spikeset.importAckk(ackkfilename)

        # Set the combo boxes to the current feature for now
        self.ui.comboBox_feature_x.clear()
        for name in self.spikeset.featureNames():
            self.ui.comboBox_feature_x.addItem(name)

        self.current_filename = str(fname)
        self.ui.stackedWidget.setCurrentIndex(0)

        self.update_ui_cluster_buttons()
        self.updateClusterDetailPlots()
        self.updateFeaturePlot()

    @QtCore.Slot()
    def on_actionOpen_triggered(self):
        fname, _ = QtGui.QFileDialog.getOpenFileName(self,
            'Open ntt file',
            filter='DotSpike (*.spike);;Neuralynx NTT (*.ntt)')

        print(type(fname))
        if fname:
            self.load_spikefile(fname)

    def updateFeaturePlot(self):
        if not self.spikeset or not self.junk_cluster:
            return

        self.junk_cluster._visible = self.junk_cluster.check.isChecked()

        if self._overviewmode:
            self.mp_proj_multi.updatePlot(self.spikeset, self.junk_cluster)
            self.mp_proj_multi.draw()
        else:
            self.mp_proj.updatePlot(self.spikeset, self.junk_cluster)
            self.mp_proj.draw()

    def updateClusterDetailPlots(self):
        if self.spikeset is None:
            return

        if self.activeClusterRadioButton():
            w = self.activeClusterRadioButton().cluster_reference.member
            cluster = self.activeClusterRadioButton().cluster_reference
        else:
            w = np.array([False] * self.spikeset.N)
        N = np.sum(w)

        # Draw average waveforms
        for i in range(0, self.spikeset.C):
            if N:
                #self.mp_wave.axes[i].errorbar(range(0,
                #    np.size(self.spikeset.spikes, 1)),
                #    cluster.wv_mean[:, i], cluster.wv_std[:, i], color='k')
                self.mp_wave.axes[i].hold(False)
                self.mp_wave.axes[i].plot(cluster.wv_mean[:,i], 'k',
                        linewidth=2)
                self.mp_wave.axes[i].hold(True)
                self.mp_wave.axes[i].plot(cluster.wv_std[:,i] +
                        cluster.wv_mean[:,i], 'k-',
                        linewidth=0.5)
                self.mp_wave.axes[i].plot(-cluster.wv_std[:,i] +
                        cluster.wv_mean[:,i], 'k-',
                        linewidth=0.5)
            else:
                self.mp_wave.axes[i].cla()

            self.mp_wave.axes[i].set_xticks([])
            self.mp_wave.axes[i].set_yticks([])
        self.mp_wave.draw()

        # Draw ISI distribution
        if N:
            self.mp_isi.axes.plot(cluster.isi_bin_centers,
                                  cluster.isi_bin_count)
        else:
            self.mp_isi.axes.cla()

        self.mp_isi.axes.set_xscale('log')
        self.mp_isi.axes.set_xlim([0.1, 1e5])

        refractory_line = mpl.lines.Line2D([2, 2],
            self.mp_isi.axes.get_ylim(), color='r', linestyle='--')
        self.mp_isi.axes.add_line(refractory_line)

        burst_line = mpl.lines.Line2D([20, 20], self.mp_isi.axes.get_ylim(),
            color='b', linestyle='--')
        self.mp_isi.axes.add_line(burst_line)

        theta_line = mpl.lines.Line2D([125, 125], self.mp_isi.axes.get_ylim(),
            color='g', linestyle='--')
        self.mp_isi.axes.add_line(theta_line)

        self.mp_isi.axes.set_xticks([1e1, 1e2, 1e3, 1e4])
        self.mp_isi.draw()

        # Now update the stats display
        if N:
            self.ui.label_spike_count.setText('%d' %
                cluster.stats['num_spikes'])

            self.ui.label_mean_rate.setText('%3.2f' %
                cluster.stats['mean_rate'])

            self.ui.label_burst.setText('%3.2f' %
                cluster.stats['burst'])

            self.ui.label_csi.setText('%3.0f' %
                cluster.stats['csi'])

            self.ui.label_refr_count.setText('%3.0f' %
                cluster.stats['refr_count'])

            self.ui.label_refr_fp.setText('%3.2f%%' %
                cluster.stats['refr_fp'])

            self.ui.label_refr_frac.setText('%3.3f%%' %
                (100.0 * cluster.stats['refr_frac']))

            self.ui.label_isolation.setText('%3.0f' %
                cluster.stats['isolation'])
        else:
            self.ui.label_spike_count.setText('')
            self.ui.label_mean_rate.setText('')
            self.ui.label_burst.setText('')
            self.ui.label_csi.setText('')
            self.ui.label_refr_count.setText('')
            self.ui.label_refr_fp.setText('')
            self.ui.label_refr_frac.setText('')
            self.ui.label_isolation.setText('')

        # Drift plot
        if N:
            t = self.spikeset.time[cluster.member]
            p = self.spikeset.featureByName('Peak').data
            p = p[cluster.member, :]

            # this is sort of hacky, wish i had histC
            countsPerBin = np.histogram(t, self.t_bins)[0]

            P = np.zeros((np.size(self.t_bin_centers), np.size(p, 1)))
            for i in range(np.size(p, 1)):
                P[:, i] = np.histogram(t, self.t_bins, weights=p[:, i])[0]
                P[countsPerBin == 0, i] = np.NAN
                P[countsPerBin != 0, i] = (P[countsPerBin != 0, i]
                    / countsPerBin[countsPerBin != 0])

            self.mp_drift.axes.hold(False)
            self.mp_drift.axes.plot(self.t_bin_centers, P)
            ylim = self.mp_drift.axes.get_ylim()
            self.mp_drift.axes.set_ylim([0, ylim[1] * 1.25])
            self.mp_drift.axes.set_xticks([])
        else:
            self.mp_drift.axes.cla()
            self.mp_drift.axes.set_yticks([])
            self.mp_drift.axes.set_xticks([])

        self.mp_drift.draw()

    @QtCore.Slot()
    def on_actionSave_triggered(self):
        if not (self.current_filename):
            return

        if not (len([cluster for cluster in self.spikeset.clusters if len(cluster.stats)>0])>0):
            bounds = self.current_filename + os.extsep + 'bounds'
            mat = self.current_filename + os.extsep + 'mat'
            pkl = self.current_filename + os.extsep + 'pkl'

            try:
                os.remove(bounds)
            except WindowsError:
                pass
            try:
                os.remove(mat)
            except WindowsError:
                pass
            try:
                os.remove(pkl)
            except WindowsError:
                pass

            return

        (root, ext) = os.path.splitext(self.current_filename)

        # Save the bounds to a format we can easily read back in later
        #outfilename = root + os.extsep + 'bounds'
        outfilename = self.current_filename + os.extsep + 'bounds'

        outfile = open(outfilename, 'wb')
        versionstr = "0.0.1"
        if versionstr == "0.0.1":
            dumping = dict()
            dumping['feature_special'] = self.spikeset.feature_special

            sb = [{'color': cluster.color, 'bounds': cluster.bounds,
                'wave_bounds': cluster.wave_bounds,
                'add_bounds':  cluster.add_bounds,
                'del_bounds': cluster.del_bounds,
                'member_base': cluster.member_base}
                        for cluster in self.spikeset.clusters if len(cluster.stats)>0]
            dumping['clusters'] = sb

            pickle.dump(versionstr, outfile)
            pickle.dump(dumping, outfile)
        else:
            # save special info about the features, such as PCA coeffs
            pickle.dump(self.spikeset.feature_special, outfile)
            #pickle.dump(self.matches, outfile)
            pickle.dump(None)

            save_bounds = [(cluster.color, cluster.bounds, cluster.wave_bounds,
                cluster.add_bounds, cluster.del_bounds,
                cluster.member_base)
                for cluster in self.spikeset.clusters if len(cluster.stats)>0]

            pickle.dump(save_bounds, outfile)
            print("Saved bounds to", outfilename)
        outfile.close()
        print("Saved cluster membership to", outfilename)

        # Save the cluster membership vectors in matlab format
        #outfilename = root + os.extsep + 'mat'
        outfilename = self.current_filename + os.extsep + 'mat'

        cluster_member = np.column_stack(tuple([cluster.member
            for cluster in self.spikeset.clusters if len(cluster.stats)>0]))
        cluster_stats = [cluster.stats for cluster in self.spikeset.clusters if len(cluster.stats)>0]

        save_data = {'cluster_id': cluster_member,
            'spike_time': self.spikeset.time,
            'subject': self.spikeset.subject,
            'session': self.spikeset.session}

        for key in cluster_stats[0].keys():
            save_data[key] = [stat[key] for stat in cluster_stats]

        sio.savemat(outfilename, save_data, oned_as='column', appendmat=False)
        print("Saved summary data (Matlab) to" + outfilename)

        # now that we've saved in .mat for matlab users, do a
        # pickle dump for python people. TODO: hdf support in future?
        outfilename = self.current_filename + os.extsep + 'pkl'
        outfile = open(outfilename, 'wb')
        pickle.dump(save_data, outfile)
        outfile.close()
        print("Saved summary data (Python) to" + outfilename)

        self.unsaved = False

    @QtCore.Slot()
    def on_actionImport_Bound_triggered(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(self,
            'Open ntt file', filter='*.bounds')
        if not filename:
            return
        if os.path.exists(filename):
            self.spikeset.importBounds(filename, True)
            self.update_ui_cluster_buttons()
            self.updateClusterDetailPlots()
            self.updateFeaturePlot()

    class CommandCopyClusters(QtGui.QUndoCommand):
        """Wraps the copy cluster logic to provide undo functionality."""

        def __init__(self, mainwindow, cluster):
            desc = "copy cluster"
            super(PyClustMainWindow.CommandCopyClusters, self).__init__(desc)
            self.mainwindow = mainwindow
            self.cluster = cluster
            self.new_cluster = None

        def redo(self):
            self.unsaved = self.mainwindow.unsaved
            if self.new_cluster:
                self.mainwindow.spikeset.clusters.append(self.new_cluster)
            else:
                self.new_cluster = self.mainwindow.spikeset.addCluster()
                self.new_cluster.bounds = self.cluster.bounds
                self.new_cluster.add_bounds = self.cluster.add_bounds
                self.new_cluster.member_base = self.cluster.member_base
                self.new_cluster.wave_bounds = self.cluster.wave_bounds
                self.new_cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.update_ui_cluster_buttons(self.new_cluster)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()
            self.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved
            self.mainwindow.spikeset.clusters.remove(self.new_cluster)
            self.mainwindow.update_ui_cluster_buttons(self.cluster)
            self.mainwindow.update_active_cluster()
            self.mainwindow.updateFeaturePlot()

    @QtCore.Slot()
    def on_actionCopy_Cluster_triggered(self):
        if self.activeClusterRadioButton():
            cluster = self.activeClusterRadioButton().cluster_reference
            command = self.CommandCopyClusters(self, cluster)
            self.undoStack.push(command)

    def closeEvent(self, event):
        if not self.unsaved:
            event.accept()

        reply = QtGui.QMessageBox.question(self, 'Save',
            "Do you want to save before quitting?", QtGui.QMessageBox.Yes |
            QtGui.QMessageBox.No | QtGui.QMessageBox.Cancel,
            QtGui.QMessageBox.Yes)

        if reply == QtGui.QMessageBox.Cancel:
            event.ignore()
        if reply == QtGui.QMessageBox.Yes:
            event.accept()
            self.on_actionSave_triggered()
        if reply == QtGui.QMessageBox.No:
            event.accept()

    def updateWavecutterPlot(self):
        if self.spikeset is None:
            return
        if not self.activeClusterRadioButton():
            return

        self.wave_limit_mode = False
        self.wave_limit_data = []

        w = self.activeClusterRadioButton().cluster_reference.member
        cluster = self.activeClusterRadioButton().cluster_reference

        chan_no = int(self.ui.spinBox_wavecutter_channel.value()) - 1

        #print("Trying to plot channel", chan_no)

        wf_perm = np.random.permutation(np.nonzero(w)[0])
        num_wave = np.min([int(self.ui.lineEdit_wavecutter_count.text()),
            np.size(wf_perm)])

        #print("Trying to plot", num_wave, "waveforms")

        wf_perm = wf_perm[:num_wave]

        if not np.size(wf_perm):
            self.mp_wavecutter.axes.clear()
            self.mp_wavecutter.draw()
            return

        self.mp_wavecutter.axes.hold(False)
        self.mp_wavecutter.axes.plot(np.transpose(
            self.spikeset.spikes[wf_perm, :, chan_no]),
            linewidth=0.5, color=(0.7, 0.7, 0.7))

        if self.ui.checkBox_wavecutter_refractory.isChecked():
            # Overlay refractory spikes
            w = np.logical_and(cluster.refractory, cluster.member)
            wf_perm = np.random.permutation(np.nonzero(w))
            num_wave = np.min([int(self.ui.lineEdit_wavecutter_count.text()),
                np.size(wf_perm)])

            #print("Trying to plot", num_wave, "refractory waveforms")
            wf_perm = wf_perm[0, :num_wave]
            if np.size(wf_perm):
                self.mp_wavecutter.axes.hold(True)
                self.mp_wavecutter.axes.plot(np.transpose(
                    self.spikeset.spikes[wf_perm, :, chan_no]),
                    color=(0.3, 0.3, 0.3))

        # Plot boundaries
        if cluster.wave_bounds:
            for (chan, sample, lower, upper) in cluster.wave_bounds:
                if chan != chan_no:
                    continue
                line = mpl.lines.Line2D([sample, sample],
                    [lower, upper], color=(1, 0, 0),
                    linestyle='-', linewidth=4)

                self.mp_wavecutter.axes.add_line(line)

        self.mp_wavecutter.axes.set_xticks([])
        self.mp_wavecutter.axes.set_yticks([])
        ylim = self.mp_wavecutter.axes.get_ylim()
        ylim = np.max(np.abs(ylim))
        self.mp_wavecutter.axes.set_ylim([-ylim, ylim])

        line = mpl.lines.Line2D(self.mp_wavecutter.axes.get_xlim(),
                [0, 0], color=(0, 0, 0), linestyle='-', linewidth=1)
        self.mp_wavecutter.axes.add_line(line)

        self.mp_wavecutter.axes.set_xlim([-0.25, -0.75 +
            np.size(self.spikeset.spikes, 1)])
        self.mp_wavecutter.draw()

    class CommandAddWvBoundary(QtGui.QUndoCommand):
        """Wraps the add wave boundary logic to provide undo functionality."""

        def __init__(self, mainwindow, cluster, bound):
            desc = "add waveform boundary"
            super(PyClustMainWindow.CommandAddWvBoundary, self).__init__(desc)
            self.mainwindow = mainwindow
            self.bound = bound
            self.cluster = cluster

        def redo(self):
            self.unsaved = self.mainwindow.unsaved
            self.cluster.wave_bounds.append(self.bound)
            self.cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.updateClusterDetailPlots()
            self.mainwindow.updateWavecutterPlot()
            self.mainwindow.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved
            self.cluster.wave_bounds.remove(self.bound)
            self.cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.updateWavecutterPlot()
            self.mainwindow.updateClusterDetailPlots()

    def wavecutter_onMousePress(self, event):
        pass

    def wavecutter_onMouseRelease(self, event):
        if not self.spikeset:
            return
        if not self.activeClusterRadioButton():
            return

        if event.button == 1 and (event.xdata is not None and
                                  event.ydata is not None):
            if not self.wave_limit_mode:
                return

            if not self.wave_limit_data:
                self.wave_limit_data = (event.xdata, event.ydata,
                    event.x, event.y)
            else:
                self.wave_limit_mode = False
                chan_no = int(self.ui.spinBox_wavecutter_channel.value()) - 1

                bound = (chan_no, int(np.round(self.wave_limit_data[0])),
                    min(self.wave_limit_data[1], event.ydata),
                    max(self.wave_limit_data[1], event.ydata))

                self.wave_limit_data = []

                cluster = self.activeClusterRadioButton().cluster_reference

                command = self.CommandAddWvBoundary(self, cluster, bound)
                self.undoStack.push(command)

        if event.button == 3:
            pass

    def wavecutter_onMouseMove(self, event):
        if not (self.wave_limit_mode and self.wave_limit_data):
            return

        height = self.mp_wavecutter.figure.bbox.height
        width = self.mp_wavecutter.figure.bbox.width
        #x0 = self.wave_limit_data[2]
        offset = width * 0.5 / (np.size(self.spikeset.spikes, 1) - 0.5)
        width = width - offset
        x0 = np.round(self.wave_limit_data[0]) * width / \
                (np.size(self.spikeset.spikes, 1) - 1) + offset
        y0 = height - self.wave_limit_data[3]
        x1 = x0
        y1 = height - event.y
        rect = [int(val) for val in (min(x0, x1),
            min(y0, y1), abs(x1 - x0), abs(y1 - y0))]
        self.mp_wavecutter.drawRectangle(rect)

    class CommandDeleteWvBoundary(QtGui.QUndoCommand):
        """Wraps the delete waveform boundary logic for undo functionality."""

        def __init__(self, mainwindow, cluster, chan):
            desc = "delete waveform boundary"
            super(PyClustMainWindow.CommandDeleteWvBoundary, self).__init__(
                desc)
            self.mainwindow = mainwindow
            self.cluster = cluster
            self.chan = chan

        def redo(self):
            self.unsaved = self.mainwindow.unsaved
            self.bounds_before = self.cluster.wave_bounds
            self.cluster.wave_bounds = [bound for bound in
                            self.cluster.wave_bounds if bound[0] != self.chan]
            self.cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.updateClusterDetailPlots()
            self.mainwindow.updateWavecutterPlot()
            self.mainwindow.unsaved = True

        def undo(self):
            self.mainwindow.unsaved = self.unsaved
            self.cluster.wave_bounds = self.bounds_before
            self.cluster.calculateMembership(self.mainwindow.spikeset)
            self.mainwindow.updateWavecutterPlot()
            self.mainwindow.updateClusterDetailPlots()

    def wavecutter_remove_limits(self):
        if not self.spikeset:
            return
        if not self.activeClusterRadioButton():
            return

        chan_no = int(self.ui.spinBox_wavecutter_channel.value()) - 1
        cluster = self.activeClusterRadioButton().cluster_reference

        command = self.CommandDeleteWvBoundary(self, cluster, chan_no)
        self.undoStack.push(command)

    def wavecutter_add_limit(self):
        self.wave_limit_mode = True

    def updateOutlierPlot(self):
        if self.spikeset is None:
            return
        if not self.activeClusterRadioButton():
            return

        cluster = self.activeClusterRadioButton().cluster_reference
        if not cluster.mahal_valid:
            cluster.calculateMahal(self.spikeset)

        if cluster.mahal_valid:
            # Mahal histogram
            m1 = np.min(cluster.mahal)
            m2 = np.max(cluster.mahal)
            bins = np.logspace(np.log10(m1), np.log10(m2))
            centers = (bins[0:-1] + bins[1:]) / 2

            count = np.histogram(cluster.mahal, bins)[0]
            count = count.astype(float) / np.sum(count)

            self.mp_outlier.axes.hold(False)
            self.mp_outlier.axes.plot(centers, count)
            self.mp_outlier.axes.hold(True)

            m_ref = cluster.mahal[cluster.refractory[cluster.member]]
            if np.size(m_ref):
                count_ref = np.histogram(
                    cluster.mahal[cluster.refractory[cluster.member]], bins)[0]

                count_ref = (count_ref.astype(float) * np.max(count) /
                        np.max(count_ref))

                self.mp_outlier.axes.plot(centers, count_ref, 'k')

            chi = scipy.stats.chi2.pdf(centers, 4)
            chi = chi / np.sum(chi)

            self.mp_outlier.axes.plot(centers, chi, 'r--')
            self.mp_outlier.axes.set_xscale('log')
            self.mp_outlier.axes.set_xlim([m1, m2])

            endpoint = scipy.stats.chi2.ppf(
                ((float(cluster.stats['num_spikes']) - 1.0) /
                cluster.stats['num_spikes']), 4)

            endpoint_line = mpl.lines.Line2D([endpoint, endpoint],
                self.mp_outlier.axes.get_ylim(), color='g', linestyle='--')

            self.mp_outlier.axes.add_line(endpoint_line)
        else:
            self.mp_outlier.axes.cla()

        self.mp_outlier.draw()

    def selectProjectionFromOverview(self):
        # self.mp_proj.feature_x = self.mp_proj_multi.feature
        if self._overviewmode:
            self.ui.checkBox_overview.setChecked(False)

            feature_x = self.mp_proj.feature_x
            feature_y = self.mp_proj.feature_y

            self.ui.comboBox_feature_x_chan.setCurrentIndex(0)
            self.ui.comboBox_feature_y_chan.setCurrentIndex(0)
            # start at first projection so that we can get the correct number of channels

            # calculate the correct projections
            if feature_x == feature_y:  # execute code for projection against different features
                max_chan = self.ui.comboBox_feature_x_chan.count() + 1
                count = 0
                breakPlease = False

                x_proj = 1
                y_proj = 2
                while count != self.mp_proj_multi.selectedSubplot:
                    count += 1
                    if y_proj < max_chan:
                        y_proj += 1
                    else: # y_proj is max
                        x_proj += 1
                        y_proj = x_proj + 1

                index = self.ui.comboBox_feature_x_chan.findText(str(x_proj))
                self.ui.comboBox_feature_x_chan.setCurrentIndex(index)
                index = self.ui.comboBox_feature_y_chan.findText(str(y_proj))
                self.ui.comboBox_feature_y_chan.setCurrentIndex(index)

            else:   # execute code for projection against same feature
                    # if self.mp_proj.feature_y in self.spikeset.featureByName(feature_x).valid_y_same_chan:
                index = self.mp_proj_multi.selectedSubplot
                self.ui.comboBox_feature_x_chan.setCurrentIndex(index)
                self.ui.comboBox_feature_y_chan.setCurrentIndex(0)


    def actionAutozoom_triggered(self):
        if self.ui.agroup_autozoom.checkedAction() == self.ui.actionAutozoom_None:
            self.mp_proj.autozoom_mode = False
        else:
            self.mp_proj.autozoom_mode = True
            if self.ui.agroup_autozoom.checkedAction() == self.ui.actionAutozoom_10:
                self.mp_proj.autozoom_factor = 0.10
            elif self.ui.agroup_autozoom.checkedAction() == self.ui.actionAutozoom_25:
                self.mp_proj.autozoom_factor = 0.25
            elif self.ui.agroup_autozoom.checkedAction() == self.ui.actionAutozoom_33:
                self.mp_proj.autozoom_factor = 0.33
            elif self.ui.agroup_autozoom.checkedAction() == self.ui.actionAutozoom_50:
                self.mp_proj.autozoom_factor = 0.50
        self.mp_proj.featureRedrawRequired.emit()


    # save labeled cluster to file in csv format
    def action_saveLabeledCluster(self):
        if self.clf_data_saver == None:
            self.clf_data_saver = classifier.DataSaver(self.ui.label_subjectid.text(),
                    self.ui.label_session.text(), self.ui.label_fname.text())  
        clust = self.activeClusterRadioButton().cluster_reference

        # get cluster number
        index_helper = filter(lambda (i, c): clust is c, enumerate(self.spikeset.clusters))
        if not index_helper == []:
            clust_num = index_helper[0][0]
        else:
            clust_num = None
        label = self.ui.comboBox_labels.currentIndex()
        self.clf_data_saver.cluster_to_file(clust, clust_num, label)
        self.ui.pushButton_saveLabeledCluster.setEnabled(False)


    # save labeled cluster members to file in csv format
    def action_saveLabeledMembers(self):
        if self.clf_data_saver == None:
            self.clf_data_saver = classifier.DataSaver(self.ui.label_subjectid.text(),
                    self.ui.label_session.text(), self.ui.label_fname.text())  
        clust = self.activeClusterRadioButton().cluster_reference

        # get cluster number
        index_helper = filter(lambda (i, c): clust is c, enumerate(self.spikeset.clusters))
        if not index_helper == []:
            clust_num = index_helper[0][0]
        else:
            clust_num = None
        label = self.ui.comboBox_labels.currentIndex()
        self.clf_data_saver.members_to_file(self.spikeset, clust, clust_num, label)
        self.ui.pushButton_saveLabeledMembers.setEnabled(False)



    def keyPressEvent(self, e): 
        ''' Ugly keybindings...but it works. '''
        
        key = e.key()       
        if key == QtCore.Qt.Key_Escape:
            self.close()
        elif key == QtCore.Qt.Key_A:
            self.on_actionAdd_Cluster_triggered()
        elif key == QtCore.Qt.Key_S:
            self.on_actionAdd_Limit_triggered()
        elif key == QtCore.Qt.Key_C:
            self.on_actionCopy_Cluster_triggered()
        elif key == QtCore.Qt.Key_X:
            self.on_actionSplit_Cluster_triggered()
        elif key == QtCore.Qt.Key_H:
            self.hide_show_all_clusters(True)
        elif key == QtCore.Qt.Key_J:
            self.hide_show_all_clusters(False)
        elif key == QtCore.Qt.Key_U:
            self.ui.checkBox_show_unclustered.setChecked(not self.ui.checkBox_show_unclustered.isChecked())
        elif key == QtCore.Qt.Key_G:
            self.hide_show_all_clusters(True)
            self.ui.checkBox_show_unclustered.setChecked(True)
            self.radioButton_junk.setChecked(True)
        elif key == QtCore.Qt.Key_2:    
            self.button_next_feature_click()
        elif key == QtCore.Qt.Key_1:
            self.button_prev_feature_click()
        elif key == QtCore.Qt.Key_W:    
            self.button_next_feature_click()
        elif key == QtCore.Qt.Key_Q:
            self.button_prev_feature_click()
        elif key == QtCore.Qt.Key_O:    
            self.on_actionOverview_Mode_triggered(checked=(not self.ui.checkBox_overview.isChecked()))
        elif key == QtCore.Qt.Key_E:    
            self.on_actionOverview_Mode_triggered(checked=(not self.ui.checkBox_overview.isChecked()))
        elif key == QtCore.Qt.Key_T:
            self.on_actionAutotrim_triggered()
        elif key == QtCore.Qt.Key_M:
            self.on_actionMerge_Clusters_triggered()
        
        elif key == QtCore.Qt.Key_Period:            
            for i,ui in enumerate(self.buttonGroup_cluster.buttons()):
                if ui == self.buttonGroup_cluster.checkedButton():
                    if(i < len(self.buttonGroup_cluster.buttons())-1):
                        self.buttonGroup_cluster.buttons()[i+1].blockSignals(True)
                        self.buttonGroup_cluster.buttons()[i+1].setChecked(True)
                        self.buttonGroup_cluster.buttons()[i+1].blockSignals(False)
                        self.update_active_cluster()
                    else:
                        self.buttonGroup_cluster.buttons()[0].blockSignals(True)
                        self.buttonGroup_cluster.buttons()[0].setChecked(True)
                        self.buttonGroup_cluster.buttons()[0].blockSignals(False)
                        self.update_active_cluster()
                    break
                    
        elif key == QtCore.Qt.Key_Comma:
            for i,ui in enumerate(self.buttonGroup_cluster.buttons()):
                if ui == self.buttonGroup_cluster.checkedButton():
                    if(i > 0):
                        self.buttonGroup_cluster.buttons()[i-1].blockSignals(True)
                        self.buttonGroup_cluster.buttons()[i-1].setChecked(True)
                        self.buttonGroup_cluster.buttons()[i-1].blockSignals(False)
                        self.update_active_cluster()
                    else:
                        length = len(self.buttonGroup_cluster.buttons())
                        self.buttonGroup_cluster.buttons()[length-1].blockSignals(True)
                        self.buttonGroup_cluster.buttons()[length-1].setChecked(True)
                        self.buttonGroup_cluster.buttons()[length-1].blockSignals(False)
                        self.update_active_cluster()
                    break
        
        elif key == QtCore.Qt.Key_Delete:
            self.on_actionDelete_Cluster_triggered()
        elif key == QtCore.Qt.Key_D:
            self.on_actionDelete_Limit_triggered()     

if __name__ == "__main__":
    #app = QtGui.QApplication.instance()
    app = QtGui.QApplication(sys.argv)
    myapp = PyClustMainWindow()
    myapp.show()
    sys.exit(app.exec_())
