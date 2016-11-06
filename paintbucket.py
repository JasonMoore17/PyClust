import numpy as np
import math
from collections import deque
import time
from scipy.spatial import distance

class PaintBucket:
    """ A tool that automatically adds membership to a cluster """

    def __init__(self, spikeset, mp_proj):
        self.active = False

        # spikeset size
        self.spikeset = spikeset
        self.N = spikeset.N
        self.spikes = np.array([True] * self.N)
            # Starting spikes

        """ Parameters """
        self.binSize = 5
        self.minPtsFactors = {'Peak': 0.08, 'Valley': 0.04, 
                            'Trough': 0.04, 'fPCA': 0.04, 'wPCA': 0.04}

        self.mp_proj = mp_proj

    def set_spikes(self, spikes):
        """ set the set of spikes to be considered by DBSCAN.
            USAGE: 
            if cluster already has members and is
            underclustered, start from that set of members """

        self.spikes = np.copy(spikes)
        return

    def get_proj_data(self, projection):
        """ Returns the data from the 
            projection of feature on channels xChan and yChan """
        
        (feature, xChan, yChan) = projection
        xData = self.spikeset.featureByName(feature).data[:, xChan]
        yData = self.spikeset.featureByName(feature).data[:, yChan]
        return xData, yData

    def set_bin_size(self, size):
        self.binSize = size
        return

    def switch_on(self):
        self.active = True
        return

    def switch_off(self):
        self.active = False
        return

    def get_bin_count(self, projection):
        """ returns number of partitions along x- and y-
            axes on the grid """
        
        (feature, xChan, yChan) = projection
        xData, yData = self.get_proj_data(projection)

        min_x = min(xData)
        min_y = min(yData)

        # number of x- and y- indices for bins
        xBins = int(math.ceil((max(xData) - min_x + 1) / self.binSize))
        yBins = int(math.ceil((max(yData) - min_y + 1) / self.binSize))
            # add 1 extra bin in calculation to prevent 
            # segmentation fault

        return xBins, yBins

    def create_member_bins(self, projection):
        """ Returns a 2-D np.array of bins, each containing
            a list of spikes """

        (feature, xChan, yChan) = projection
        xData, yData = self.get_proj_data(projection)
        xBins, yBins = self.get_bin_count(projection)

        bins = np.array([[None] * yBins] * xBins)
        for i in range(xBins):
            for j in range(yBins):
                bins[i][j] = []

        xMin = min(xData)
        yMin = min(yData)
        for p in range(self.N):
            xBin = int((xData[p] - xMin)/self.binSize)
            yBin = int((yData[p] - yMin)/self.binSize)
            bins[xBin][yBin].append(p)

        return bins

    def create_count_bins(self, projection):
        """ Returns a 2-D array of bins, each containing
            a count of spikes in that bin """
        
        (feature, xChan, yChan) = projection
        xData, yData = self.get_proj_data(projection)
        xBins, yBins = self.get_bin_count(projection)

        countBins = np.array([[0] * yBins] * xBins)

        xMin = min(xData)
        yMin = min(yData)
        for p in range(self.N):
            xBin = int((xData[p] - xMin) / self.binSize)
            yBin = int((yData[p] - yMin) / self.binSize)
            countBins[xBin][yBin] += 1

        return countBins

    def get_bin_coord(self, spikeCoord, projection):
        """ Return the bin coordinates that contains the spike
            at coordinate spikeCoord """

        xData, yData = self.get_proj_data(projection)

        xMin = min(xData)
        yMin = min(yData)
        return ((int((spikeCoord[0] - xMin) / self.binSize)), 
                int((spikeCoord[1] - yMin) / self.binSize))

    def get_bin_neighbors(self, bin, projection):
        """ Returns a list of bin coordinates neighboring
            the center coordinate bin, including the center 
            """

        xBins, yBins = self.get_bin_count(projection)

        binNeighbors = []
        binNeighbors.append((bin[0], bin[1]))  # center
        if bin[0] - 1 >= 0 and bin[1] + 1 < yBins:
            binNeighbors.append((bin[0] - 1, bin[1] + 1))  # top left
        if bin[1] + 1 < yBins:
            binNeighbors.append((bin[0], bin[1] + 1))  # top
        if bin[0] + 1 < xBins and bin[1] + 1 < yBins:
            binNeighbors.append((bin[0] + 1, bin[1] + 1))  # top right
        if bin[0] + 1 < xBins:
            binNeighbors.append((bin[0] + 1, bin[1]))  # right
        if bin[0] + 1 < xBins and bin[1] - 1 >= 0:
            binNeighbors.append((bin[0] + 1, bin[1] - 1))  # bot right
        if bin[1] - 1 >= 0:
            binNeighbors.append((bin[0], bin[1] - 1))  # bot
        if bin[0] - 1 >= 0 and bin[1] - 1 >= 0:
            binNeighbors.append((bin[0] - 1, bin[1] - 1))  # bot left
        if bin[0] - 1 >= 0:
            binNeighbors.append((bin[0] - 1, bin[1]))  # left

        return binNeighbors

    def limit_spikes_to_window(self, spikes, projection):
        """ Cut down spikes to those within current plot limits.
            This is used to cut down running time in the case
            that paint bucket "leaks" out. """

        if self.mp_proj.autozoom_mode:
            limits = self.mp_proj.autozoom_limits
        else:
            limits = self.mp_proj.prof_limits

        xData, yData = self.get_proj_data(projection)
        for i in range(self.N):
            if spikes[i]:
                spikes[i] = (limits[0][0] <= xData[i] <= limits[0][1]
                             and limits[1][0] <= yData[i] <= limits[1][1])
        return spikes

    def DBSCAN_bins(self, s, minPts, spikes, projection):
        """ DBSCAN_bins runs a modified DBSCAN algorithm 
            on bins of spikes on a given projection.

            PARAMS:
            s - source spike 
            minPts - lower bound for bin count
            spikes - set of spikes to be considered 

            Returns set of spikes admitted by DBSCAN_bins """

        (feature, xChan, yChan) = projection
        xData, yData = self.get_proj_data(projection)

        memberBins = self.create_member_bins(projection)
        countBins = self.create_count_bins(projection)
        binCount = self.get_bin_count(projection)
        xBins = binCount[0]
        yBins = binCount[1]

        visited = np.array([[False] * yBins] * xBins)

        sCoord = (xData[s], yData[s])
        print("calling get_bin_coord in DBSCAN_bins")
        sBin = self.get_bin_coord(sCoord, projection)

        # run BFS on bins, where edges are determined by
        # DBSCAN algorithm
        queue = deque([sBin])
        visited[sBin[0]][sBin[1]] = True
        binsClustered = []  # bins that are added to the cluster
        binsClustered.append(sBin)
        while len(queue) != 0:
            bin = queue.popleft()
            binNeighborhood = self.get_bin_neighbors(bin, projection)
            count = 0
            for b in binNeighborhood:
                count += countBins[b[0]][b[1]]
            if count >= minPts:  
                # bin is a core bin
                for b in binNeighborhood:
                    if visited[b[0]][b[1]]:
                        continue
                    visited[b[0]][b[1]] = True
                    binsClustered.append(b)
                    queue.append(b)

        # add membership on all spikes in bins visited by BFS
        members = np.array([False] * self.N)  
            # spikes that are added to the cluster

        for b in binsClustered:
            for p in memberBins[b[0]][b[1]]:
                if spikes[p]:
                    members[p] = True

        return members

    def get_source(self, cursorpos, projection):
        """ returns source member, which is the member
            closest to cursor position """

        (xData, yData) = self.get_proj_data(projection)
        print(cursorpos)

        print("calling get_bin_coord from get_source")
        sBin = self.get_bin_coord(cursorpos, projection)
        sBinNeighbors = self.get_bin_neighbors(sBin, projection)
        memberBins = self.create_member_bins(projection)
        candidates = []
        for b in sBinNeighbors:
            for p in memberBins[b[0]][b[1]]:
                v = (xData[p], yData[p])
                dist = distance.euclidean(cursorpos, v)
                candidates.append((p, dist))
        s = min(candidates, key=lambda x: x[1])[0]

        return s

    def get_minPts(self, s, projection):
        """ Returns DBSCAN parameter minPts, using a fixed
            percentage of starting density, centered at the
            bin containing source member s """

        feature = projection[0]
        xData, yData = self.get_proj_data(projection)
        sCoord = (xData[s], yData[s])

        minPts = 0
        print("calling get_bin_coord from get_minPts")
        sBin = self.get_bin_coord(sCoord, projection)
        sBinNeighbors = self.get_bin_neighbors(sBin, projection)
        countBins = self.create_count_bins(projection)
        for b in sBinNeighbors:
            minPts += countBins[b[0]][b[1]]

        minPts = int(minPts * self.minPtsFactors[feature])
        return minPts
            
        
    def cluster_projections(self, event):
        """ TOP LEVEL FUNCTION: 
            For all projections of the current feature,
            run DBSCAN to add members to the selected 
            cluster """

        feature = self.mp_proj.feature_x
        curXChan = self.mp_proj.chan_x
        curYChan = self.mp_proj.chan_y
        curProj = (feature, curXChan, curYChan)
            # current projection

        cursorpos = (event.xdata, event.ydata)

        s = self.get_source(cursorpos, curProj)

        print("s: ")
        print(s)
        minPts = self.get_minPts(s, curProj)
        spikes = np.copy(self.spikes)
            # spikes to be considered in DBSCAN
            
        spikes = self.limit_spikes_to_window(spikes, curProj)

        spikes = self.DBSCAN_bins(s, minPts, spikes, curProj)
            # members added by DBSCAN


        # repeat DBSCAN for all projections of current feature
        chans = self.spikeset.featureByName(feature).data.shape[1]
        for xChan in range(chans - 1):
            for yChan in range(xChan + 1, chans):
                if xChan == curXChan and yChan == curYChan:
                    # skip the projection we started with
                    continue
                
                proj = (feature, xChan, yChan)
                xData, yData = self.get_proj_data(proj)

                # re-calculate minPts
                minPts = self.get_minPts(s, proj)

                # cut down with another run of DBSCAN
                spikes = self.DBSCAN_bins(s, minPts, spikes, proj)
        
        return spikes
