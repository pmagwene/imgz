from __future__ import print_function

from toolz import *

import numpy as np
import scipy as sp

import matplotlib
from matplotlib.widgets import RectangleSelector, Button
from matplotlib import pyplot as plt

from skimage import (io, util, exposure, transform, equalize, morphology, segmentation)


class PersistentRectangleSelector(RectangleSelector):
    def release(self, event):
        super(PersistentRectangleSelector, self).release(event)
        self.to_draw.set_visible(True)
        self.canvas.draw()   

class SelectorContainer(object):
    def __init__(self, selector):
        self.selector = selector

    def extents(self, event):
        return self.selector.extents

    def quit(self, event):
        plt.close("all")



def select_ROI(img, cmap='gray'):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom = 0.2)  
    ax.imshow(img, cmap=cmap)

    selector = PersistentRectangleSelector(ax,
                                           lambda e1,e2: None,
                                           drawtype = "box",
                                           useblit = False,
                                           button = [1],
                                           spancoords = "data",
                                           interactive = True)

    ax_done = plt.axes([0.45, 0.05, 0.2, 0.075])
    btn_done = Button(ax_done, "Done")
    btn_done.on_clicked(lambda e: plt.close("all"))

    plt.show(block=True)
    xmin, xmax, ymin, ymax =  selector.extents
    return (int(xmin), int(xmax), int(ymin), int(ymax))
    

def equalize_from_ROI(img, roi):
    xmin, xmax, ymin, ymax = roi
    mask = np.zeros(img.shape)
    mask[ymin:ymax, xmin:xmax] = 1
    return exposure.equalize_hist(img, mask = mask)



def rescale(img, value):
    return transform.rescale(img, value,
                             mode="constant",
                             preserve_range=True).astype(img.dtype)


invert = util.invert

equalize_adaptive = equalize.equalize_adapthist

equalize_hist = equalize.equalize_hist
