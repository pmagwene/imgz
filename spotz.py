import os.path

import click




# @curry
# def pad_image(nrows, ncols, img):
#     padwidth = int(min(np.array(img.shape)/np.array((nrows,ncols))))
#     return util.pad(img, padwidth, mode="constant")




# def adaptive_binarize_image(img, block_size=None, sigma=None, nrows=8, ncols=12, elemsize=None,
#                             min_size = 50):
#     if block_size is None:
#         block_size = (2 * (max(img.shape)//200)) + 1
#     if sigma is None:
#         sigma = 2 * block_size 

#     if elemsize is None:
#         elemsize = min(1, int(min(img.shape) * 0.005))
#     disk_opening = curry(binary_opening, selem = morphology.disk(elemsize))
#     border_clearing = curry(clear_border, buffer_size = elemsize)
    

#     return pipe(img,
#                 threshold_gaussian(block_size, sigma),
#                 disk_opening,
#                 border_clearing,
#                 remove_small_objects(min_size),
#                 remove_small_holes(min_size),
#                 pad_image(nrows, ncols))

#     bimg = img > filters.threshold_local(img, block_size, param = sigma) 
#     bimg = morphology.binary_opening(bimg)
#     bimg = segmentation.clear_border(bimg)
#     padwidth = int(min(np.array(img.shape)/np.array((nrows,ncols))))
#     padimg = util.pad(bimg, padwidth, mode="constant")
#     bimg = filter_by_peakfinding(padimg, nrows=nrows, ncols=ncols)
#     bimg = util.crop(bimg, padwidth, copy=True)
#     bimg = ndimage.binary_fill_holes(bimg)
#     return bimg, block_size


import gridder
import segmenter


@click.group()
def cli():
    pass

cli.add_command(gridder.main, "gridder")
cli.add_command(segmenter.main, "segmenter")


if __name__ == "__main__":
    cli()
