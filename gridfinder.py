from itertools import product

import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io)

from toolz.curried import *

import imgz

#  _____     _   _                 _                    _     _  
# | ____|___| |_(_)_ __ ___   __ _| |_ ___    __ _ _ __(_) __| | 
# |  _| / __| __| | '_ ` _ \ / _` | __/ _ \  / _` | '__| |/ _` | 
# | |___\__ \ |_| | | | | | | (_| | ||  __/ | (_| | |  | | (_| | 
# |_____|___/\__|_|_| |_| |_|\__,_|\__\___|  \__, |_|  |_|\__,_| 
#                                            |___/               


def find_grid(img, nrows, ncols,
              threshold_func = imgz.threshold_otsu,
              selem_size = None,
              min_gap = None, min_n = None):

    rdim, cdim = img.shape
    min_dim, max_dim = min(img.shape), max(img.shape)
    
    if selem_size is None:
        selem_size = int(round(max(1, min_dim/100.)))

    if min_gap is None:
        min_gap = int(min(rdim/nrows * 0.2, cdim/ncols * 0.2))

    if min_n is None:
        min_n = int(0.5 * min(nrows, ncols))

    # threshold image
    binary_img = threshold_func(img)

    # apply morphological opening
    binary_img = imgz.disk_opening(selem_size, binary_img)

    bboxes, centers = find_grid_bboxes(binary_img, min_gap, min_n)
        
    
    


def find_grid_bboxes(binary_img, min_gap, min_n):
    labeled_img = morphology.label(binary_img)
    regions = measure.regionprops(labeled_img)
    centroids = np.vstack([r.centroid for r in regions])
    row_centers, col_centers = estimate_grid_centers(centroids, min_gap, min_n)
    grid_centers = list(product(row_centers, col_centers))
    grid_bboxes = grid_bboxes_from_centers(row_centers, col_centers, labeled_img.shape)
    return grid_bboxes, grid_centers
    

def connected_intervals(vals, min_gap):
    """Find intervals where distance between successive values > min_gap
    """
    vals = np.sort(vals)
    dist = vals[1:] - vals[:-1]
    rbreaks = np.argwhere(dist > min_gap).flatten() + 1
    lbreaks = rbreaks + 1
    lbreaks = [0] + lbreaks.tolist()
    rbreaks = rbreaks.tolist() + [len(vals)-1]
    intervals = zip(lbreaks, rbreaks)
    return vals, intervals

def interval_means(vals, intervals):
    """Find means of connected intervals.
    """
    return [np.mean(vals[i[0]:i[1]]) for i in intervals]
    

def estimate_grid_centers(centroids, min_gap, min_n):
    row_vals, row_int = connected_intervals(centroids[:,0], min_gap)
    col_vals, col_int = connected_intervals(centroids[:,1], min_gap)

    valid_rows = [i for i in row_int if (i[1] - i[0]) > min_n]
    valid_cols = [i for i in col_int if (i[1] - i[0]) > min_n]

    row_centers = interval_means(row_vals, valid_rows)
    col_centers = interval_means(col_vals, valid_cols)

    return row_centers, col_centers

def grid_bboxes_from_centers(row_centers, col_centers, shape):
    row_centers = np.asarray(row_centers)
    col_centers = np.asarray(col_centers)
    maxr, maxc = shape
    
    row_dists = 0.5 * (row_centers[1:] - row_centers[:-1])
    col_dists = 0.5 * (col_centers[1:] - col_centers[:-1])

    rowFirst = [max(0, row_centers[0] - row_dists[0])]
    rowLast =  [min(maxr, row_centers[-1] + row_dists[-1])]
    row_borders = np.concatenate((rowFirst,
                                  row_centers[:-1] + row_dists,
                                  rowLast))

    colFirst = [max(0, col_centers[0] - col_dists[0])]
    colLast =  [min(maxc, col_centers[-1] + col_dists[-1])]
    col_borders = np.concatenate((colFirst,
                                  col_centers[:-1] + col_dists,
                                  colLast))

    row_borders = np.round(row_borders).astype(np.int)
    col_borders = np.round(col_borders).astype(np.int)

    row_pairs = zip(row_borders[:-1], row_borders[1:])
    col_pairs = zip(col_borders[:-1], col_borders[1:])

    rc_pairs = product(row_pairs, col_pairs)
    bboxes = [(p[0][0], p[1][0], p[0][1], p[1][1]) for p in rc_pairs]
    return bboxes
