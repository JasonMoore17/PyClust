
__version__ = "1.0.0"

from PySide.QtGui import *
from PySide.QtCore import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

import numpy as np
import matplotlib as mpl
from scipy import ndimage
import scipy

from matplotlib import rcParams
rcParams['font.size'] = 9

import boundaries  # Need to know how to create boundaries


class MultiplotWidget(Canvas):
    """A widget to plot all permutation of feature channels as an overview."""

    #__pyqtSignals__ = ("featureRedrawRequired()")
    # define the signal
    featureRedrawRequired = Signal()

    def __init__(self, parent=None):
        self.figure = Figure()
        Canvas.__init__(self, self.figure)
        self.setParent(parent)

        pal = self.palette().window().color()
        bgcolor = (pal.red() / 255.0, pal.blue() / 255.0, pal.green() / 255.0)

        self.figure.clear()
        self.figure.set_facecolor(bgcolor)
        # Set up the feature axes
        self.figure.subplots_adjust(hspace=0.000, wspace=0.000,
            bottom=0.0, top=1, left=0.0, right=1)

        Canvas.setSizePolicy(self, QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.prof_limits = {}
        # Properties describing how to draw the clusters
        self.unclustered = True
        self.refractory = True
        self.exclusive = True
        self.markersize = 1
        self.ptype = -2

        self.feature = None


    def setFeatureX(self, feature):
        self.setFeature(feature)

    def setFeatureY(self, feature):
        pass

    def setChanX(self, chan):
        pass

    def setChanY(self, chan):
        pass

    @Slot(bool)
    def setShowUnclustered(self, show):
        self.unclustered = show
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    @Slot(bool)
    def setUnclusteredExclusive(self, excl):
        self.exclusive = excl
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    @Slot(bool)
    def setRefractory(self, show):
        self.refractory = show
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    @Slot(int)
    def setMarkerSize(self, size):
        self.markersize = size
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    @Slot(int)
    def setPlotType(self, ptype):
        self.ptype = ptype
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    def resetLimits(self):
        #self.prof_limits_reference = None
        self.prof_limits_reference = {}
        ##self.emit(SIGNAL("featureRedrawRequired()"))
        #self.featureRedrawRequired.emit()

    # These are not provided as signals since there is some non trivial
    # logic the main form needs to perform first
    def setFeature(self, name):
        self.feature = name
        self.figure.clear()
        self.axes = {}
        self.resetLimits()
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    # Given a set of spike and cluster data, create the figure
    def updatePlot(self, spikeset, junk):
        if self.feature is None:
            return

        clusters = spikeset.clusters

        data = spikeset.featureByName(self.feature).data
        chans = data.shape[1]
        combs = scipy.misc.comb(chans, 2)
        plots_x = np.ceil(np.sqrt(combs))
        plots_y = np.ceil(float(combs) / plots_x)

        cl_list = clusters + [junk]

        count = 0
        # Iterate over projections
        for proj_x in range(0, chans):
            for proj_y in range(proj_x + 1, chans):
                count += 1
                tw = (proj_x, proj_y)

                if tw not in self.axes.keys():
                    self.axes[(proj_x, proj_y)] = self.figure.add_subplot(
                        plots_y, plots_x, count)

                xdata = data[:, proj_x]
                ydata = data[:, proj_y]

                self.axes[tw].hold(False)

                if not tw in self.prof_limits_reference.keys():
                    w = np.array([True] * spikeset.N)
                    if not junk._visible:
                        w[junk.member] = False

                    temp = ([np.min(xdata[w]), np.max(xdata[w])],
                            [np.min(ydata[w]), np.max(ydata[w])])

                    w_x = (temp[0][1] - temp[0][0]) * 0.01
                    w_y = (temp[1][1] - temp[1][0]) * 0.01
                    self.prof_limits_reference[tw] = (
                        [temp[0][0] - w_x, temp[0][1] + w_x],
                        [temp[1][0] - w_y, temp[1][1] + w_y])

                    self.prof_limits[tw] = self.prof_limits_reference[tw]

                # Scatter Plot
                if self.ptype == -2:
                    #Plot the unclustered spikes
                    if self.unclustered:
                        w = np.array([True] * spikeset.N)

                        if self.exclusive:
                            for cluster in clusters + [junk]:
                                w[cluster.member] = False

                        self.axes[tw].plot(xdata[w], ydata[w],
                            linestyle='None',
                            marker='o', markersize=self.markersize,
                            markerfacecolor='k',  markeredgecolor='k',
                            alpha=1.0)

                        self.axes[tw].hold(True)

                    # Iterate over clusters
                    for cluster in cl_list:
                        if not cluster._visible:
                            continue

                        col = tuple(map(lambda s: s / 255.0, cluster.color))
                        # Plot the cluster spikes
                        self.axes[tw].plot(xdata[cluster.member],
                                       ydata[cluster.member],
                                marker='o', markersize=self.markersize,
                                markerfacecolor=col, markeredgecolor=col,
                                linestyle='None', alpha=0.99)
                        self.axes[tw].hold(True)

                        # Plot refractory spikes
                        if self.refractory:
                            self.axes[tw].plot(xdata[cluster.refractory],
                                ydata[cluster.refractory], marker='o',
                                markersize=5, markerfacecolor='k',
                                markeredgecolor='k', linestyle='None')

                # Do a density plot
                else:
                    w = np.array([False] * spikeset.N)
                    if self.unclustered:
                        w = np.array([True] * spikeset.N)
                        if self.exclusive:
                                for cluster in clusters + [junk]:
                                        w[cluster.member] = False
                    for cluster in [cluster for cluster in cl_list if
                                    cluster._visible]:
                        w[cluster.member] = True

                    if not np.any(w):
                        w[0] = True  # Histogram routine fails with empty data

                    bins_x = np.linspace(self.prof_limits[tw][0][0],
                        self.prof_limits[tw][0][1], 100)
                    bins_y = np.linspace(self.prof_limits[tw][1][0],
                        self.prof_limits[tw][1][1], 100)

                    self.axes[tw].cla()
                    self.axes[tw].set_xlim(self.prof_limits[tw][0])
                    self.axes[tw].set_ylim(self.prof_limits[tw][1])

                    count = np.histogram2d(xdata[w],
                        ydata[w], [bins_x, bins_y])[0]

                    count = ndimage.filters.gaussian_filter(count, 0.5)

                    if self.ptype == -3:
                        self.axes[tw].imshow(count.T, cmap=mpl.cm.gist_earth_r,
                                aspect='auto',
                                extent=self.axes[tw].get_xlim() +
                                self.axes[tw].get_ylim()[::-1])
                    else:
                        self.axes[tw].imshow(np.log(count + 1).T,
                                cmap=mpl.cm.gist_earth_r,
                                aspect='auto',
                                extent=self.axes[tw].get_xlim() +
                                self.axes[tw].get_ylim()[::-1])

                    # Iterate over clusters for refractory spikes
                    if self.refractory:
                        self.axes[tw].hold(True)
                        for cluster in cl_list:
                            if not cluster._visible:
                                continue

                            # Plot refractory spikes
                            self.axes[tw].plot(xdata[cluster.refractory],
                                ydata[cluster.refractory], marker='o',
                                markersize=3,
                                markerfacecolor='k', markeredgecolor='k',
                                linestyle='None')

                self.axes[tw].set_xlim(self.prof_limits[tw][0])
                self.axes[tw].set_ylim(self.prof_limits[tw][1])
                for tick in self.axes[tw].xaxis.get_major_ticks():
                    tick.set_pad(-15)
                for tick in self.axes[tw].yaxis.get_major_ticks():
                    tick.set_pad(-20)
                    tick.label2.set_horizontalalignment('left')

                # Now draw the boundaries
                for cluster in cl_list:
                    if not cluster._visible:
                        continue

                    # Limit boundaries with solid line
                    bounds = cluster.getBoundaries(self.feature, proj_x,
                        self.feature, proj_y)
                    for bound in bounds:
                        col = tuple(map(lambda s: s / 255.0, cluster.color))
                        bound.draw(self.axes[tw], color=col, linestyle='-')

                    # Addition boundaries with dashed line
                    bounds = cluster.getBoundaries(self.feature, proj_x,
                        self.feature, proj_y, 'add')
                    for bound in bounds:
                        col = tuple(map(lambda s: s / 255.0, cluster.color))
                        bound.draw(self.axes[tw], color=col, linestyle='--')
