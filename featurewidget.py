
__version__ = "1.0.0"

from PySide.QtGui import *
from PySide.QtCore import *

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as Canvas
from matplotlib.figure import Figure

import numpy as np
import matplotlib as mpl
from scipy import ndimage

from matplotlib import rcParams
rcParams['font.size'] = 9

import boundaries  # Need to know how to create boundaries

#AUTOZOOM_FACTOR = 0.25

class ProjectionWidget(Canvas):

    #__pyqtSignals__ = ("featureRedrawRequired()",
    #"polygonBoundaryDrawn(PyQt_PyObject)",
    #"ellipseBoundaryDrawn(PyQt_PyObject)")
    featureRedrawRequired = Signal()
    polygonBoundaryDrawn = Signal(object)
    ellipseBoundaryDrawn = Signal(object)

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
        self.axes = self.figure.add_subplot(1, 1, 1)

        Canvas.setSizePolicy(self, QSizePolicy.Expanding,
                QSizePolicy.Expanding)
        Canvas.updateGeometry(self)

        self.prof_limits = None
        # Properties describing how to draw the clusters
        self.unclustered = True
        self.refractory = True
        self.exclusive = True
        self.markersize = 1
        self.ptype = -2

        self.feature_x = 'Peak'
        self.feature_y = 'Peak'
        self.chan_x = 0
        self.chan_y = 1

        # Mouse handlers
        self.zoom_active = False
        self.limit_mode = False
        self.limit_cluster = None
        self.limit_type = 2  # 1 = polygon, 2 = ellipse
        self.limit_data = []
        self.mpl_connect('button_press_event', self.onMousePress)
        self.mpl_connect('button_release_event', self.onMouseRelease)
        self.mpl_connect('motion_notify_event', self.onMouseMove)

        self.autozoom_mode = False
        self.autozoom_factor = None
        self.autozoom_limits = np.array([None for i in range(2)], dtype=object)

        # used for paint bucket
        self.paintBucket_mode = False

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

    @Slot(bool)
    def setBoundaryElliptical(self, iselliptical):
        if iselliptical:
            self.limit_type = 2
        else:
            self.limit_type = 1
        self.stopMouseAction()

    def resetLimits(self):
        self.prof_limits_reference = None
        ##self.emit(SIGNAL("featureRedrawRequired()"))
        #self.featureRedrawRequired.emit()

    # These are not provided as signals since there is some non trivial
    # logic the main form needs to perform first
    def setFeatureX(self, name):
        self.feature_x = name
        self.resetLimits()
        self.stopMouseAction()
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    def setFeatureY(self, name):
        self.feature_y = name
        self.resetLimits()
        self.stopMouseAction()
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    def setChanX(self, chan):
        self.chan_x = chan
        self.resetLimits()
        self.stopMouseAction()
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    def setChanY(self, chan):
        self.chan_y = chan
        self.resetLimits()
        self.stopMouseAction()
        #self.emit(SIGNAL("featureRedrawRequired()"))
        self.featureRedrawRequired.emit()

    # Given a set of spike and cluster data, create the figure
    def updatePlot(self, spikeset, junk):
        clusters = spikeset.clusters
        xdata = spikeset.featureByName(self.feature_x).data[:, self.chan_x]
        ydata = spikeset.featureByName(self.feature_y).data[:, self.chan_y]
        self.axes.hold(False)

        cl_list = clusters + [junk]
        if self.prof_limits_reference is None:
            w = np.array([True] * spikeset.N)
            if not junk._visible:
                w[junk.member] = False

            temp = ([np.min(xdata[w]), np.max(xdata[w])],
                    [np.min(ydata[w]), np.max(ydata[w])])

            w_x = (temp[0][1] - temp[0][0]) * 0.01
            w_y = (temp[1][1] - temp[1][0]) * 0.01
            self.prof_limits_reference = ([temp[0][0] - w_x, temp[0][1] + w_x],
                [temp[1][0] - w_y, temp[1][1] + w_y])

            self.prof_limits = self.prof_limits_reference

        # Scatter Plot
        if self.ptype == -2:
            #Plot the unclustered spikes
            if self.unclustered:
                w = np.array([True] * spikeset.N)

                if self.exclusive:
                    for cluster in clusters + [junk]:
                        w[cluster.member] = False

                self.axes.plot(xdata[w], ydata[w], linestyle='None',
                    marker='o', markersize=self.markersize,
                    markerfacecolor='k',  markeredgecolor='k', alpha=1.0)

                self.axes.hold(True)

            # Iterate over clusters
            for cluster in cl_list:
                if not cluster._visible:
                    continue

                col = tuple(map(lambda s: s / 255.0, cluster.color))
                # Plot the cluster spikes
                self.axes.plot(xdata[cluster.member], ydata[cluster.member],
                        marker='o', markersize=self.markersize,
                        markerfacecolor=col, markeredgecolor=col,
                        linestyle='None', alpha=0.99)
                self.axes.hold(True)

                # Plot refractory spikes
                if self.refractory:
                    self.axes.plot(xdata[cluster.refractory],
                        ydata[cluster.refractory], marker='o', markersize=5,
                        markerfacecolor='k', markeredgecolor='k',
                        linestyle='None')

        # Do a density plot
        else:
            w = np.array([False] * spikeset.N)
            if self.unclustered:
                w = np.array([True] * spikeset.N)
                if self.exclusive:  # unclustered only
                        for cluster in clusters + [junk]:
                                w[cluster.member] = False
            for cluster in [cluster for cluster in cl_list if cluster._visible]:
                w[cluster.member] = True

            if not np.any(w):
                w[0] = True  # Histogram routine fails with empty data

            bins_x = np.linspace(self.prof_limits[0][0],
                self.prof_limits[0][1], 100)
            bins_y = np.linspace(self.prof_limits[1][0],
                self.prof_limits[1][1], 100)

            self.axes.cla()
            self.axes.set_xlim(self.prof_limits[0])
            self.axes.set_ylim(self.prof_limits[1])

            count = np.histogram2d(xdata[w],
                ydata[w], [bins_x, bins_y])[0]

            count = ndimage.filters.gaussian_filter(count, 0.5)

            if self.ptype == -3:
                self.axes.imshow(count.T, cmap=mpl.cm.gist_earth_r,
                        aspect='auto',  extent=self.axes.get_xlim() +
                        self.axes.get_ylim()[::-1])
            else:
                self.axes.imshow(np.log(count + 1).T,
                        cmap=mpl.cm.gist_earth_r,
                        aspect='auto',  extent=self.axes.get_xlim() +
                        self.axes.get_ylim()[::-1])

            # Iterate over clusters for refractory spikes
            if self.refractory:
                self.axes.hold(True)
                for cluster in cl_list:
                    if not cluster._visible:
                        continue

                    # Plot refractory spikes
                    self.axes.plot(xdata[cluster.refractory],
                        ydata[cluster.refractory], marker='o', markersize=3,
                        markerfacecolor='k', markeredgecolor='k',
                        linestyle='None')

        ######################################################
        # autozoom code

        if self.autozoom_mode:
            N = spikeset.N
            superclust = np.array([False] * N)
            # members of all clusters, including "unclustered" cluster

            superclust_isEmpty = True

            # add unclustered data to superclust
            if self.unclustered:
                superclust = np.array([True] * N)
                for c in clusters:
                    for i in range(N):
                        if c.member[i]:
                            superclust[i] = False

            for c in clusters:
                if c._visible:
                    for i in range(N):
                        if c.member[i]:
                            superclust[i] = True

            for i in range(N):
                if superclust[i]:
                    superclust_isEmpty = False
                    break

            if not superclust_isEmpty:
                # compute the x and y limits for autozoom
                min_x = min(xdata[superclust])
                max_x = max(xdata[superclust])
                min_y = min(ydata[superclust])
                max_y = max(ydata[superclust])

                xdist = max_x - min_x
                ydist = max_y - min_y

                xlims = (min_x - xdist * self.autozoom_factor, max_x + xdist * self.autozoom_factor)
                ylims = (min_y - ydist * self.autozoom_factor, max_y + ydist * self.autozoom_factor)

                self.autozoom_limits[0] = xlims
                self.autozoom_limits[1] = ylims

        if self.autozoom_mode and self.autozoom_limits[0] and self.autozoom_limits[0]:
            # first check that autozoom indeed has more zoom
            if (self.autozoom_limits[0][1] - self.autozoom_limits[0][0]
                    < self.prof_limits[0][1] - self.prof_limits[0][0]):
                self.axes.set_xlim(self.autozoom_limits[0])
            else:
                self.axes.set_xlim(self.prof_limits[0])

            if (self.autozoom_limits[1][1] - self.autozoom_limits[1][0]
                    < self.prof_limits[1][1] - self.prof_limits[1][0]):
                self.axes.set_ylim(self.autozoom_limits[1])
            else:
                self.axes.set_ylim(self.prof_limits[1])

        # end of autozoom code
        ######################################################

        else:
            self.axes.set_xlim(self.prof_limits[0])
            self.axes.set_ylim(self.prof_limits[1])


        for tick in self.axes.xaxis.get_major_ticks():
            tick.set_pad(-15)
        for tick in self.axes.yaxis.get_major_ticks():
            tick.set_pad(-20)
            tick.label2.set_horizontalalignment('left')

        # Now draw the boundaries
        for cluster in cl_list:
            if not cluster._visible:
                continue

            # Limit boundaries with solid line
            bounds = cluster.getBoundaries(self.feature_x, self.chan_x,
                self.feature_y, self.chan_y)
            for bound in bounds:
                col = tuple(map(lambda s: s / 255.0, cluster.color))
                bound.draw(self.axes, color=col, linestyle='-')

            # Addition boundaries with dashed line
            bounds = cluster.getBoundaries(self.feature_x, self.chan_x,
                self.feature_y, self.chan_y, 'add')
            for bound in bounds:
                col = tuple(map(lambda s: s / 255.0, cluster.color))
                bound.draw(self.axes, color=col, linestyle='--')

    def finalizeZoom(self, event):
        self.zoom_active = False

        temp = (np.sort([event.xdata, self.zoom_startpos[0]]),
            np.sort([event.ydata, self.zoom_startpos[1]]))

        # do a sanity check, we cant zoom to an area smaller than
        # say 1/20 of the display
        delta_x = ((temp[0][1] - temp[0][0])
                / (self.prof_limits[0][1] - self.prof_limits[0][0]))
        delta_y = ((temp[1][1] - temp[1][0])
                / (self.prof_limits[1][1] - self.prof_limits[1][0]))

        if delta_x < 0.025 or delta_y < 0.025:
            return

        self.prof_limits = temp
        # prof_limits[0] is the 2-tuple boundary along the x-axis
        # prof_limits[1] is the 2-tuple boundary along the y-axis

        if self.ptype == -2: # if scatter plot
            self.axes.set_xlim(self.prof_limits[0])
            self.axes.set_ylim(self.prof_limits[1])
            self.draw()
        else:  # if its a density plot we should rebin for now
            #self.emit(SIGNAL("featureRedrawRequired()"))
            self.featureRedrawRequired.emit()

        self.repaint()

    def onMousePress(self, event):
        if self.limit_mode:
            return
        if (event.xdata is not None and event.ydata is not None) \
                and event.button == 1:
            if not self.prof_limits:
                return
            if not self.zoom_active:
                self.zoom_active = True
                self.zoom_startpos = (event.xdata, event.ydata)
            else:
                self.finalizeZoom(event)

    def onMouseRelease(self, event):
        # Left mouse button
        if event.button == 1 and \
                (event.xdata is not None and event.ydata is not None):
            if self.zoom_active:
                self.finalizeZoom(event)
                rect = [0, 0, 0, 0]
                self.drawRectangle(rect)
            if self.limit_mode and self.limit_type == 1:
                self.limit_data.append((event.xdata, event.ydata))
            if self.limit_mode and self.limit_type == 2:
                if len(self.limit_data) == 1:
                    self.limit_data.append((event.xdata, event.ydata))
                    self.repaint()
                elif len(self.limit_data) == 0:
                    # set the center of the ellipse
                    self.limit_data = [(event.xdata, event.ydata)]
                    self.repaint()

        # complete the polygon boundary
        if event.button == 3 and self.limit_mode and self.limit_type == 1:
            if event.xdata is not None and event.ydata is not None:
                self.limit_data.append((event.xdata, event.ydata,
                    event.x, event.y))
                self.limit_mode = False

                # create an array to describe the bound
                temp = np.array([np.array([point[0], point[1]]) for
                    point in self.limit_data])

                print("Adding boundary on", self.feature_x, self.chan_x + 1, \
                    self.feature_y, self.chan_y + 1)

                bound = boundaries.BoundaryPolygon2D()
                bound = bound.init2(
                        (self.feature_x, self.feature_y),
                        (self.chan_x, self.chan_y), temp)

                # With only two points there isnt really a polyogon
                if len(temp) > 2:
                    self.polygonBoundaryDrawn.emit(bound)
                    #self.emit(SIGNAL("polygonBoundaryDrawn(PyQt_PyObject)"),
                    #          bound)
                self.stopMouseAction()

        # complete the ellipse boundary
        if event.button == 3 and self.limit_mode and self.limit_type == 2:
            t = lambda s: np.array(self.axes.transData.transform(s))

            # We need to map the ellipse from display coordinates to
            # data coordinates, do this through the matrix form of the
            # ellipse equation
            center = tuple(0.5 * np.array(self.limit_data[0]) +
                    0.5 * np.array(self.limit_data[1]))
            vc = 0.5 * (t(self.limit_data[0]) + t(self.limit_data[1]))
            va = t(self.limit_data[1])
            vm = t((event.xdata, event.ydata))

            angvec = va - vc
            angle = np.arctan2(angvec[1], angvec[0])
            ewidth = np.linalg.norm(angvec)
            angvec = angvec / ewidth
            mvec = vm - vc
            eheight = np.linalg.norm(mvec - np.dot(mvec, angvec) * angvec)

            # define the mapping from view to data coordinate scaling
            xlim = self.axes.get_xlim()
            ylim = self.axes.get_ylim()

            xlimv = [x for x, y in t([(xlim[0], ylim[0]), (xlim[1], ylim[0])])]
            ylimv = [y for x, y in t([(xlim[0], ylim[0]), (xlim[0], ylim[1])])]

            scaleX = np.diff(xlim) / np.diff(xlimv)
            scaleY = np.diff(ylim) / np.diff(ylimv)

            mapping = np.array([[1.0 / scaleX, 0.0], [0.0, 1.0 / scaleY]])

            cdsp = np.linalg.inv(
                    boundaries.covarianceFromEllipse(angle, ewidth, eheight))
            cdat = np.dot(np.dot(mapping, cdsp), mapping.T)

            (angle, width, height) = boundaries.ellipseFromCovariance(
                    np.linalg.inv(cdat.astype("double")))
            bound = boundaries.BoundaryEllipse2D()
            bound = bound.init2(
                    (self.feature_x, self.feature_y),
                    (self.chan_x, self.chan_y), center, angle,
                    (width, height))

            #self.emit(SIGNAL("ellipseBoundaryDrawn(PyQt_PyObject)"), bound)
            self.ellipseBoundaryDrawn.emit(bound)

            self.limit_data = []
            self.limit_mode = False
            self.repaint()

        # Right click resets to original bounds
        elif event.button == 3 and (not self.limit_mode):
            self.prof_limits = self.prof_limits_reference
            if self.ptype == -2:  # scatter plot
                # decide whether to return to default bounds or autozoom bounds
                if self.autozoom_mode and self.autozoom_limits[0] and self.autozoom_limits[1]:
                    if (self.autozoom_limits[0][1] - self.autozoom_limits[0][0]
                            < self.prof_limits[0][1] - self.prof_limits[0][0]):
                        self.axes.set_xlim(self.autozoom_limits[0])
                    else:
                        self.axes.set_xlim(self.prof_limits[0])

                    if (self.autozoom_limits[1][1] - self.autozoom_limits[1][0]
                            < self.prof_limits[1][1] - self.prof_limits[1][0]):
                        self.axes.set_ylim(self.autozoom_limits[1])
                    else:
                        self.axes.set_ylim(self.prof_limits[1])

                else:
                    self.axes.set_xlim(self.prof_limits[0])
                    self.axes.set_ylim(self.prof_limits[1])
                self.draw()
            else:  # density plot
                #self.emit(SIGNAL("featureRedrawRequired()"))
                self.featureRedrawRequired.emit()

    def onMouseMove(self, event):
        if self.limit_mode:
            self._mouse_move_event = event
            self.draw()
            self.repaint()
        if (not self.limit_mode) and self.zoom_active and \
                (event.xdata is not None and event.ydata is not None):
            # draw rectangle is in the qt coordinates,
            # so we need a little conversion
            height = self.figure.bbox.height
            start = self.axes.transData.transform(self.zoom_startpos)
            end = self.axes.transData.transform((event.xdata, event.ydata))
            x0 = start[0]
            y0 = height - start[1]
            x1 = end[0]
            y1 = height - end[1]
            rect = [int(val) for val in (min(x0, x1),
                min(y0, y1), abs(x1 - x0), abs(y1 - y0))]
            self.drawRectangle(rect)

    # Paint using standard Canvas and then draw lines during
    # boundary creation as required
    def paintEvent(self, event):
        Canvas.paintEvent(self, event)
        if not self.limit_mode:
            return

       # convert to data units
        height = self.figure.bbox.height
        t = lambda s: np.array(self.axes.transData.transform(s))

        qp = QPainter()
        qp.begin(self)
        qp.setPen(QColor(*self.limit_color))

        # Draw polygon
        if self.limit_mode and self.limit_type == 1:
            for i in range(len(self.limit_data) - 1):
                start = t(self.limit_data[i])
                end = t(self.limit_data[i + 1])
                qp.drawLine(start[0], height - start[1],
                        end[0], height - end[1])
            if self.limit_data:
                start = t(self.limit_data[-1])
                end = t((self._mouse_move_event.xdata,
                    self._mouse_move_event.ydata))
                qp.drawLine(start[0], height - start[1],
                        end[0], height - end[1])

        # Draw an ellipse
        if self.limit_mode and self.limit_type == 2 and self.limit_data:
            # convert to display coordinates
            startp = t(self.limit_data[0])
            mouse = t((self._mouse_move_event.xdata,
                    self._mouse_move_event.ydata))
            center = 0.5 * (startp + mouse)

            if len(self.limit_data) == 1:  # we've set the center, draw line
                eheight = 30
                angvec = mouse - center
                angle = np.arctan2(angvec[1], angvec[0])
                ewidth = np.linalg.norm(angvec)

            elif len(self.limit_data) > 1:  # we've also fixed the angle
                angleline = t(self.limit_data[1])
                center = 0.5 * (startp + angleline)
                angvec = angleline - center
                angle = np.arctan2(angvec[1], angvec[0])
                ewidth = np.linalg.norm(angvec)
                angvec = angvec / ewidth

                mvec = mouse - center
                eheight = np.linalg.norm(mvec - np.dot(mvec, angvec) * angvec)

            if self.limit_data:
                qp.translate(center[0], height - center[1])
                qp.rotate(-angle * 180.0 / np.pi)
                qp.drawEllipse(QPoint(0, 0), ewidth, eheight)
                qp.drawLine(-ewidth, 0, ewidth, 0)
                qp.drawLine(0, -eheight, 0, eheight)

        qp.end()

    # Start drawing a boundary for a cluster
    def drawBoundary(self, color):
        self.limit_color = color
        self.limit_mode = True
        self.limit_data = []

    def stopMouseAction(self):
        self.limit_mode = False
        self.limit_color = None
        self.zoom_active = False
        self.repaint()
