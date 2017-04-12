
import numpy as np
import matplotlib as mpl
import scipy.stats


# A simple boundary base class
class Boundary(object):

    def withinBoundary(self, spikeset):
        return np.array([True] * spikeset.N)

    def draw(self, axes, color, linestyle):
        return


# 2D Polygon boundary
class BoundaryPolygon2D(Boundary):

    # featureNames should be a length 2 tuple (featureX, featureY)
    # bounds is a mx2 set of polygon coordinates
    def __init__(self):
        self.features = None
        self.chans = None
        self.bounds = None
        self.special_info = None  # might need things like PCA coefficients

    def init2(self, featureNames, featureChans, bounds):
        self.features = featureNames
        self.chans = featureChans
        self.bounds = bounds
        self.special_info = None  # might need things like PCA coefficients

        return self


    def withinBoundary(self, spikeset, subset=None):
        if subset is None:
            subset = np.array([True] * spikeset.N)
        px = spikeset.featureByName(self.features[0]).data[subset,
                self.chans[0]]
        py = spikeset.featureByName(self.features[1]).data[subset,
                self.chans[1]]

        data = np.column_stack((px, py))

        #return nx.points_inside_poly(data, self.bounds)
        path = mpl.path.Path(self.bounds)
        return path.contains_points(data)

    def draw(self, axes, color='k', linestyle='-'):
        bound = np.vstack((self.bounds, self.bounds[0, :]))
        for i in range(np.size(bound, 0)):
            line = mpl.lines.Line2D(bound[i:i + 2, 0],
                bound[i:i + 2, 1], color=color, linestyle=linestyle)
            axes.add_line(line)


def robustEllipseEstimator(data):
    # data is N x 2
    center = np.median(data, axis=0)
    cdata = data - center

    # Estimate the angle using 'inner' data
    covar = np.cov(data.T)
    angle, rx, ry = ellipseFromCovariance(covar)
#    angle = np.pi/2 - angle

    rotmat = np.array([[np.cos(angle), - np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])

    temp = np.dot(cdata, rotmat)
    rx = np.median(np.abs(temp[:, 0]))
    ry = np.median(np.abs(temp[:, 1]))
    return (center, angle, np.array([rx, ry]))


# Helper functions
def ellipseFromCovariance(covar, conf=None):
    if conf is None:
        kval = 1
    else:
        kval = scipy.stats.chi2.ppf(conf, covar.shape[0] - 1)
    vals, vecs = np.linalg.eigh(covar)
    ind = np.argsort(vals)[::-1]  # descending sort
    vals = vals[ind]
    vecs = vecs[ind, :]
    projv = vecs[0] / np.linalg.norm(vecs[0])
    angle = np.arctan2(projv[1], projv[0])
    radiusx = np.sqrt(vals[0] * kval)
    radiusy = np.sqrt(vals[1] * kval)
    return (angle, radiusx, radiusy)


def covarianceFromEllipse(angle, radiusx, radiusy):
    # eigen vectors as columns
    u = np.array([[np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])
    d = np.array([[radiusx ** 2, 0], [0, radiusy ** 2]])
    return np.dot(np.dot(u, d), np.linalg.inv(u))


def pointsInsideEllipse(data, center, angle, size):
    data = data - center
    rotmat = np.array([[np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]])
    data = np.dot(rotmat, data.T).T
    data = data / size
    return np.sum(np.power(data, 2), axis=1) <= 1


# 2D Elliptical boundary
class BoundaryEllipse2D(Boundary):
    # size should be (rwidth x rheight) in rotated space
    def __init__(self):
        self.features = None
        self.chans = None
        self.center = None
        self.angle = None
        self.size = None
        self.special_info = None


    # size should be (rwidth x rheight) in rotated space
    def init2(self, featureNames, featureChans, center, angle, size):
        self.features = featureNames
        self.chans = featureChans
        self.center = center
        self.angle = angle
        self.size = size
        self.special_info = None

        return self


    def withinBoundary(self, spikeset, subset=None):
        if subset is None:
            subset = np.array([True] * spikeset.N)
        px = spikeset.featureByName(self.features[0]).data[subset,
                self.chans[0]]
        py = spikeset.featureByName(self.features[1]).data[subset,
                self.chans[1]]

        data = np.column_stack((px, py))

        return pointsInsideEllipse(data, self.center, self.angle, self.size)

    def draw(self, axes, color='k', linestyle='-'):
        if linestyle == '-':
            linestyle = 'solid'
        elif linestyle == '--':
            linestyle = 'dashed'
        elif linestyle == '-.':
            linestyle = 'dashdot'
        else:
            linestyle = 'solid'

        ell = mpl.patches.Ellipse(self.center, self.size[0] * 2.0,
                self.size[1] * 2.0, 180 * self.angle / np.pi, color=color,
                linestyle=linestyle,  fill=False, linewidth=2)
        axes.add_artist(ell)
