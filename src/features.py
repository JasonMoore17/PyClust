from __future__ import print_function
import time

import numpy as np

# Feature data is N x C


class Feature:

    def __init__(self, name, spikeset):
        self.name = name
        self.data = self.calculate(spikeset)
        shape = np.shape(self.data)
        if len(shape) == 1:     # We need an (N,1) array rather than (N,)
            self.data = np.reshape(self.data, (self.data.size, 1))
        self.channels = np.size(self.data, 1)
        self.valid_y_all_chans = []
        self.valid_y_same_chan = []
        #self.channels = np.size(self.data,1)

    def calculate(self, spikeset):
        pass

    def valid_y_features(self, feature_chan):
        ret_val = []
        if feature_chan < self.channels - 1:
            ret_val.append((self.name, range(feature_chan + 1, self.channels)))
        for item in self.valid_y_all_chans:
            ret_val.append((item, None))
        for item in self.valid_y_same_chan:
            ret_val.append((item, [feature_chan]))
        return ret_val

    def valid_x_features(self):
        if self.channels == 1:
            return [0]
        else:
            if self.valid_y_all_chans or self.valid_y_same_chan:
                return range(0, self.channels)
            else:
                return range(0, self.channels - 1)


class Feature_Valley(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Valley', spikeset)

    def calculate(self, spikeset):
        # fall time is typically around 0.4ms for interneuron, so
        # search in that range
        minrange = int(round(0.15 / spikeset.dt_ms)) + spikeset.peak_index
        maxrange = int(round(0.25 / spikeset.dt_ms)) + spikeset.peak_index

        return np.min(spikeset.spikes[:, minrange:maxrange + 1, :], axis=1)


class Feature_Trough(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Trough', spikeset)
        self.valid_y_same_chan.append('Valley')

    def calculate(self, spikeset):
        return np.min(spikeset.spikes[:, :spikeset.peak_index + 1, :], axis=1)


class Feature_Energy(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Energy', spikeset)
        self.valid_y_same_chan.append('Peak')

    def calculate(self, spikeset):
        return np.sum(spikeset.spikes * spikeset.spikes, axis=1)


class Feature_Time(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Time', spikeset)
        self.valid_y_all_chans.append('Peak')

    def calculate(self, spikeset):
        #temp = np.zeros([spikeset.N, 1])
        #temp[:, 0] =
        return np.array(spikeset.time - spikeset.time[0], dtype=np.float) / 1e6
        #return temp


class Feature_Peak(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Peak', spikeset)
        self.valid_y_same_chan.append('Energy')
        self.valid_y_same_chan.append('Valley')

    def calculate(self, spikeset):
        searchrange = range(spikeset.peak_index - 3, spikeset.peak_index + 4)
        return np.max(spikeset.spikes[:, searchrange, :], axis=1)


class Feature_Barycenter(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Barycenter', spikeset)

    def calculate(self, spikeset):
        p = Feature_Peak(spikeset)
        p = p.data

        p = p - np.min(np.max(p, axis=1))

        x = p[:, 0] - p[:, 2]
        y = p[:, 1] - p[:, 3]
        angle = np.arctan2(y, x)
        retval = np.zeros((np.size(p, 0), 2))
        retval[:, 0] = np.cos(angle)
        w = retval[:, 0] > 0
        retval[w, 0] = retval[w, 0] * p[w, 0]
        w = np.logical_not(w)
        retval[w, 0] = retval[w, 0] * p[w, 2]

        retval[:, 1] = np.sin(angle)
        w = retval[:, 1] > 0
        retval[w, 1] = retval[w, 1] * p[w, 1]
        w = np.logical_not(w)
        retval[w, 1] = retval[w, 1] * p[w, 3]

        return retval


class Feature_FallArea(Feature):
    def __init__(self, spikeset):
        Feature.__init__(self, 'Fall Area/Time', spikeset)
#        self.valid_y_all_chans.append('Fall Time')

    def calculate(self, spikeset):
        wv = np.mean(spikeset.spikes, axis=2)  # Average over channels
        ret = np.sum(wv[:, spikeset.peak_index:], axis=1)
        ind = np.argmin(wv[:, spikeset.peak_index:], axis=1)
        rnd = np.random.rand(np.size(ind, 0)) - 0.5
        retval = np.zeros((np.size(wv, 0), 2))
        retval[:, 0] = ret
        retval[:, 1] = (ind + rnd) * 1000.0 / spikeset.fs
        return retval


# data is N x K
def PCA(data, numcomponents=3):
    A = numcomponents
    N = np.size(data, 0)
    K = np.size(data, 1)
    nipals_T = np.zeros((N, A))
    nipals_P = np.zeros((K, A))

    tolerance = 1E-10
    for a in range(A):
        t_a_guess = np.random.rand(N, 1) * 2
        t_a = t_a_guess + 1.0
        itern = 0

        # Repeat until the score, t_a, converges, or until a maximum number of
        # iterations has been reached
        while np.linalg.norm(t_a_guess - t_a) > tolerance or itern < 500:
            # 0: starting point for convergence checking on next loop
            t_a_guess = t_a
            # 1: Regress the scores, t_a, onto every column in X; compute the
            #    regression coefficient and store it in the loadings, p_a
            #    i.e. p_a = (X' * t_a)/(t_a' * t_a)
            p_a = np.dot(data.T, t_a) / np.dot(t_a.T, t_a)
            # 2: Normalize loadings p_a to unit length
            p_a = p_a / np.linalg.norm(p_a)
            # 3: Now regress each row in X onto the loading vector; store the
            #    regression coefficients in t_a.
            #    i.e. t_a = X * p_a / (p_a.T * p_a)
            t_a = np.dot(data, p_a) / np.dot(p_a.T, p_a)

            itern += 1

        #  We've converged, or reached the limit on the number of iteration
        # Deflate the part of the data in X thats explained with t_a and p_a
        data = data - np.dot(t_a, p_a.T)
        # Store result before computing the next component
        nipals_T[:, a] = t_a.ravel()
        nipals_P[:, a] = p_a.ravel()

    spe_x = np.sum(data ** 2, axis=1)
    return (nipals_T, nipals_P, spe_x)


class Feature_PCA(Feature):
    def __init__(self, spikeset, coeff=None):
        self.coeff = coeff
        Feature.__init__(self, 'fPCA', spikeset)

    def calculate(self, spikeset):
        p = spikeset.featureByName('Peak').data
        v = spikeset.featureByName('Valley').data
        t = spikeset.featureByName('Trough').data
        e = np.sqrt(spikeset.featureByName('Energy').data)
        inputdata = np.hstack((p, np.sqrt(e), v, t))
        temp = inputdata - np.mean(inputdata, axis=0)
        sigma = np.dot(temp.T, temp)
        # demeaning really just shifts by a constant, not worth breaking
        # existing cluster files
        # inputdata = inputdata - np.mean(inputdata, axis=0)
        if self.coeff is not None:  # See if we were given projectioncomponents
            scores = np.dot(inputdata, self.coeff)
        else:
            print("Calculating feature based PCA",)
            t1 = time.clock()
            u, s, d = np.linalg.svd(sigma)
            K = 6
            self.coeff = u[:, :K]
            scores = np.dot(inputdata, self.coeff)
            #M = 100000
            #if spikeset.N > M:
            #    perm = np.random.permutation(spikeset.N)
            #    scores, coeff, stx = PCA(inputdata[perm[0:M], :], 6)
            #    scores = np.dot(inputdata, coeff)
            #else:
            #    scores, coeff, stx = PCA(inputdata, 6)
            #self.coeff = coeff
            t2 = time.clock()
            print("took", (t2 - t1), "seconds.")
        return scores


def debug_trace():
    from PyQt4.QtCore import pyqtRemoveInputHook
    from ipdb import set_trace
    pyqtRemoveInputHook()
    set_trace()


class Feature_Waveform_PCA(Feature):
    def __init__(self, spikeset, coeff=None):
        self.coeff = coeff
        Feature.__init__(self, 'wPCA', spikeset)

    def calculate(self, spikeset):
        print("Calculating waveform based PCA",)
        t1 = time.clock()
        temp = spikeset.spikes.reshape((spikeset.N, spikeset.spikes.shape[1]
                                        * spikeset.spikes.shape[2]))
        temp = temp - np.mean(temp, axis=0)
        if self.coeff is not None:
            scores = np.dot(temp, self.coeff)
        else:
            sigma = np.dot(temp.T, temp)
            u, s, d = np.linalg.svd(sigma)
            K = 8
            self.coeff = u[:, :K]
            scores = np.dot(temp, self.coeff)
            t2 = time.clock()
            print("took", (t2 - t1), "seconds.")
        return scores
