from itertools import product

import numpy as np
import scipy as sp
from scipy import ndimage

import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt

import skimage
from skimage import (morphology, segmentation, exposure, feature, filters,
                     measure, transform, util, io)

from toolz.curried import *



read_image = io.imread

@curry
def rescale(scale, img):
    return transform.rescale(img, scale,
                             mode = "constant",
                             preserve_range = True).astype(img.dtype)

invert = util.invert

equalize_adaptive = exposure.equalize_adapthist
equalize_hist = exposure.equalize_hist

clear_border = segmentation.clear_border

disk_selem = morphology.disk
binary_opening = morphology.binary_opening
binary_erosion = morphology.binary_erosion


def threshold_otsu(img):
    return img > filters.threshold_otsu(img)

def threshold_li(img):
    return img > filters.threshold_li(img)

def threshold_isodata(img):
    return img > filters.threshold_isodata(img)

@curry
def threshold_gaussian(block_size, sigma, img):
    return img > filters.threshold_local(img, block_size,
                                         method = "gaussian",
                                         param = sigma)

def remove_small_objects(**args):
    return curry(morphology.remove_small_objects, **args)

def remove_small_holes(**args):
    return curry(morphology.remove_small_holes, **args)

@curry
def pad_image(nrows, ncols, img):
    padwidth = int(min(np.array(img.shape)/np.array((nrows,ncols))))
    return util.pad(img, padwidth, mode="constant")


def disk_opening(radius):
    return curry(morphology.binary_opening, selem = morphology.disk(radius))

def disk_erosion(radius):
    return curry(morphology.binary_erosion, selem = morphology.disk(radius))


def adaptive_binarize_image(img, block_size=None, sigma=None, nrows=8, ncols=12, elemsize=None,
                            min_size = 50):
    if block_size is None:
        block_size = (2 * (max(img.shape)//200)) + 1
    if sigma is None:
        sigma = 2 * block_size 

    if elemsize is None:
        elemsize = min(1, int(min(img.shape) * 0.005))
    disk_opening = curry(binary_opening, selem = morphology.disk(elemsize))
    border_clearing = curry(clear_border, buffer_size = elemsize)
    

    return pipe(img,
                threshold_gaussian(block_size, sigma),
                disk_opening,
                border_clearing,
                remove_small_objects(min_size),
                remove_small_holes(min_size),
                pad_image(nrows, ncols))

    bimg = img > filters.threshold_local(img, block_size, param = sigma) 
    bimg = morphology.binary_opening(bimg)
    bimg = segmentation.clear_border(bimg)
    padwidth = int(min(np.array(img.shape)/np.array((nrows,ncols))))
    padimg = util.pad(bimg, padwidth, mode="constant")
    bimg = filter_by_peakfinding(padimg, nrows=nrows, ncols=ncols)
    bimg = util.crop(bimg, padwidth, copy=True)
    bimg = ndimage.binary_fill_holes(bimg)
    return bimg, block_size

def filter_by_grid_peaks(binaryimg, nrows=8, ncols=12):
    """Filter objects in binary image using coarsened histogram."""
    limg = morphology.label(binaryimg)
    regions = measure.regionprops(limg)
    peaks = find_grid_peaks(limg, nrows, ncols)
    regions = [r for r in regions if pts_in_bbox(r.bbox, peaks).size]
    fimg = np.in1d(limg.ravel(), [r.label for r in regions])
    fimg.shape = limg.shape
    return fimg

def regions_in_bboxes(labeled_img, bboxes):
    regions = measure.regionprops(labeled_img)
    centroids = [r.centroid for r in regions]
    regions = [r for r in regions if pts_in_bbox]
    

def find_grid_peaks(labeledimg, nrows=8, ncols=12,
                    mult = 7, min_dist = 3, return_hist=False):
    regions = measure.regionprops(labeledimg)
    pts = np.vstack([i.coords for i in regions])
    nr, nc = labeledimg.shape
    binr, binc = mult * nrows, mult * ncols
    hist2d, xedges, yedges = np.histogram2d(pts[:, 0], pts[:, 1],
                                            bins=(binr, binc),
                                            range=((0, nr), (0, nc)))
    peaks = feature.peak_local_max(hist2d, min_distance = min_dist, indices=True)
    xmidpts = (xedges[:-1] + xedges[1:]) / 2.0
    ymidpts = (yedges[:-1] + yedges[1:]) / 2.0
    rowcolpeaks = np.array([(xmidpts[i[0]], ymidpts[i[1]]) for i in peaks])
    if return_hist:
        return rowcolpeaks, hist2d
    return rowcolpeaks

def bbox_to_poly(bbox):
    minr, minc, maxr, maxc = bbox
    pt1 = (minr, minc)
    pt2 = (minr, maxc)
    pt3 = (maxr, maxc)
    pt4 = (maxr, minc)
    return np.array([pt1, pt2, pt3, pt4])


def pts_in_bbox(bbox, pts):
    poly = bbox_to_poly(bbox)
    return pts[measure.points_in_poly(pts, poly)]

def points_within_bboxes(points, bboxes):
    in_a_box = np.zeros(len(points), dtype = np.bool)
    for bbox in bboxes:
        poly = bbox_to_poly(bbox)
        in_this_box = measure.points_in_poly(points, poly)
        in_a_box = np.logical_or(in_a_box, in_this_box)
    return in_a_box
    

def region_encloses_grid_center(region, grid_centers):
    poly = bbox_to_poly(region.bbox)
    hits = measure.points_in_poly(grid_centers, poly)
    if np.sometrue(hits):
        return True, np.argwhere(hits).flatten()
    else:
        return False, None

#  _____     _   _                 _                    _     _  
# | ____|___| |_(_)_ __ ___   __ _| |_ ___    __ _ _ __(_) __| | 
# |  _| / __| __| | '_ ` _ \ / _` | __/ _ \  / _` | '__| |/ _` | 
# | |___\__ \ |_| | | | | | | (_| | ||  __/ | (_| | |  | | (_| | 
# |_____|___/\__|_|_| |_| |_|\__,_|\__\___|  \__, |_|  |_|\__,_| 
#                                            |___/               


def find_grid_bboxes(binary_img, min_gap, min_n):
    labeled_img = morphology.label(binary_img)
    regions = measure.regionprops(labeled_img)
    centroids = np.vstack([r.centroid for r in regions])
    row_centers, col_centers = estimate_grid_centers(centroids, min_gap, min_n)
    grid_centers = list(product(row_centers, col_centers))
    grid_bboxes = grid_bboxes_from_centers(row_centers, col_centers, labeled_img.shape)
    return grid_bboxes, grid_centers, labeled_img, regions, centroids
    

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


# def sub_imshow(img, cmap='gray'):
#     fig = plt.figure()
#     #naxes = len(fig.get_axes())
#     ax = fig.add_subplot(111)#naxes+1,naxes+1)
#     ax.imshow(img, cmap=cmap, aspect='equal')
#     #fig.subplots_adjust(wspace=0.1)
#     return img
    

def multi_imshow(images, cmap = "gray"):
    fig, axes = plt.subplots(1, len(images))
    for i, image in enumerate(images):
        axes[i].imshow(image, cmap = cmap)
    return fig, axes
