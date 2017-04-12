# -*- coding: utf-8 -*-
""" Generating n distinct colors. Obviously as n increases the colors become
    less visually distinct but at least the first two functions for n ~15
    will generte colors that one would pick manually."""
import colorsys
import numpy as np
import math
from matplotlib import mlab


def unique_colors_hsv(n):
    """ hsv cylinder"""
    colors = []
    for i in range(n):
        colors.append(colorsys.hsv_to_rgb(i * 1.0 / n,
        1 - np.random.random_sample() / 5, 1 - np.random.random_sample() / 2))
    return colors


#=======================================================
def unique_colors_hsl(n):
    """ hsl cylinder"""
    colors = []
    for i in np.arange(0., 360., 360. / n):
        hue = i / 360.
        lightness = (30 + np.random.rand() * 25) / 100.
        saturation = (80 + np.random.rand() * 20) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return colors


#======================================================
def unique_colors_rgb(n):
    """Compute a list of distinct colors, each of which
       is represented as an RGB 3-tuple."""
    hues = []
    # i is in the range 0, 1, ..., n - 1
    for i in range(1, n + 1):
        hues.append(360.0 / i)

    hs = []
    for hue in hues:
        h = math.floor(hue / 60) % 6
        hs.append(h)

    fs = []
    for hue in hues:
        f = hue / 60 - math.floor(hue / 60)
        fs.append(f)

    rgbcolors = []
    for h, f in zip(hs, fs):
        v = 1
        p = 0
        q = 1 - f
        t = f
        if h == 0:
            color = v, t, p
        elif h == 1:
            color = q, v, p
        elif h == 2:
            color = p, v, t
        elif h == 3:
            color = p, q, v
        elif h == 4:
            color = t, p, v
        elif h == 5:
            color = v, p, q
        rgbcolors.append(color)

    return rgbcolors


#=============================================================================
def circ_mean(angles, radians=True):
    """ This function takes a list of angles and calculates their circular mean
        in degrees"""
    if radians is False:
        for i in range(len(angles)):
            angles[i] = angles[i] * np.pi / 180
    cthetas = []
    for i in range(len(angles)):
        cthetas.append(complex(np.cos(angles[i]), np.sin(angles[i])))
    cthetas = np.array(cthetas)
    temp = np.sum(cthetas) / np.absolute(np.sum(cthetas))
    mean_angle = np.angle(temp, deg=True) % 360
    #mean_angle = (np.arctan2(temp.imag,temp.real)+np.pi)*180/np.pi
    return mean_angle


def circ_dist(angles, radians=True):
    """This function computes the circular distance between all input angles"""
    D = []
    if radians is False:
        for i in range(len(angles)):
            angles[i] = angles[i] * np.pi / 180
    for i in range(len(angles)):
        for j in range(i + 1, len(angles)):
            D.append(.5 * (1 + np.cos(angles[i] - angles[j])))
    return D


def newcolor(currentlist):
    hsvcolors, angles, z, r = [], [], [], []
    for i in range(len(currentlist)):
        hsvcolors.append(colorsys.rgb_to_hsv(currentlist[i][0],
                    currentlist[i][1], currentlist[i][2]))
        angles.append(360 * hsvcolors[i][0])
        z.append(hsvcolors[i][1])
        r.append(hsvcolors[i][2])

    new_z = .4 * (1 - np.average(z)) + .6 - .1 * np.random.rand()
    new_r = .3 * (1 - np.average(r)) + .7 - .3 * np.random.rand()

    i = 1

    (n, edges) = np.histogram(angles, 1, (0, 360))
    while np.size(np.transpose(n.nonzero())) >= len(edges) - 1 \
          or len(mlab.find(n > 1)) > 0:
        i = i + 1
        (n, edges) = np.histogram(angles, i, (0, 360))
    valid = mlab.find(n == 0)
    D = np.zeros(len(angles))
    for i in range(len(valid)):
        new_angle = circ_mean([edges[valid[i]], edges[valid[i] + 1]],
                              radians=False)
        for j in range(len(angles)):
            D[j] = np.array(circ_dist([new_angle, angles[j]], radians=False))
        if np.maximum(D, .06 * np.ones(len(angles))).all == D.all:
            break
    #other useless method, maximizing distance on a circle
    #sumcos =[]
    #for theta in bins:
    #  sumcos.append(np.sum(np.cos(np.pi*(theta-angles)/180)))
    #new_angle=bins[sumcos.index(min(sumcos))]
    new_color = colorsys.hsv_to_rgb(new_angle / 360, new_z, new_r)

    return new_color
