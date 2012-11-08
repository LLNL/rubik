"""
This file defines functions that operate on numpy arrays.  Most are the
fundamental rubik transforamtions that permute array elements.
"""
import numpy as np

def data(arr):
    """This gets the address of a numpy array's data buffer."""
    return arr.__array_interface__["data"][0]

def view_to_base(view, index=None):
    """Given a numpy view and an index into it, this will convert the index in
    the view to the corresponding index in the base.  Example::

        base = np.empty([4,4])
        view = base[1::2, 1::2]
        print view_to_base(view, (0,0))
        (1,1)

    If you call this without an index, it returns a function that can
    be used to efficently convert a lot of indices for the same view::

        vtob = view_to_base(view)
        print vtob((0,0))
        (1,1)
    """
    offset = (data(view) - data(view.base)) / view.dtype.itemsize
    corner = np.unravel_index(offset, view.base.shape)
    scale = [v / b for v, b in zip(view.strides, view.base.strides)]

    def vtob(index):
        return tuple([c + i * s for c,i,s in zip(corner, index, scale)])

    if index == None:
        return vtob
    else:
        return vtob(index)

def base_to_view(view, index=None):
    """Given a numpy view and an index into it, this will convert the index in
    the base to the corresponding index in the view.  Example::

        base = np.empty([4,4])
        view = base[1::2, 1::2]
        print view_to_base(view, (0,0))
        (1,1)

    If you call this without an index, it returns a function that can
    be used to efficently convert a lot of indices for the same view::

        btov = base_to_view(view)
        print btov((1,1))
        (0,0)
    """
    offset = (data(view) - data(view.base)) / view.dtype.itemsize
    corner = np.unravel_index(offset, view.base.shape)
    scale = [v / b for v, b in zip(view.strides, view.base.strides)]

    def btov(index):
        return tuple([(i - c) / s for c,i,s in zip(corner, index, scale)])
    if index == None:
        return btov
    else:
        return btov(index)

def hyperplane(arr, axis, index):
    """ This generates a slice list that will select one hyperplane out of a
    numpy ndarray by fixing one axis to a particular coordinate.
    """
    selector = [slice(None)] * arr.ndim
    selector[axis] = index
    return arr[selector]


def mod(arr, dim, chunk, nchunks):
    """ Given an array, the dimension (axis) it's being sliced on, the chunk
    and the number of chunks, returns a slice that divides that dimension
    into modulo sets.
    """
    return slice(chunk, None, nchunks)


def div(arr, dim, chunk, nchunks):
    """ Given an array, the dimension (axis) it's being sliced on, the chunk
    and the number of chunks, returns a slice that divides that dimension
    into contiguous pieces. If nchunks doesn't evenly divide arr.shape[dim],
    the last slice will include the remainder.
    """
    chunksize = arr.shape[dim] / nchunks
    start = chunk * chunksize
    end = start + chunksize

    # last slice takes any leftovers for uneven divides
    if chunk == nchunks - 1: end = None

    return slice(start, end)


def cut(arr, divisors, slicers = div):
    """ Given an array and a list of divisors, up to one per dimension, cuts
    the array using the slice generator functions in 'slicers'. If slicers
    is a function, use that for all axes. If slicers is an array, use one
    slicer per axis. If no slicers are provided, use div for all axes.
    """
    # If you just pass a slicer, it uses that for everything
    if hasattr(slicers, '__call__'):
        slicers = [slicers] * len(divisors)

    # Create a new numpy array to hold each cut of the original array
    parts = np.ndarray(divisors, dtype=object)

    # assign a view to each partition
    for index in np.ndindex(parts.shape):
        slice = [slicers[dim](arr, dim, i, parts.shape[dim]) for dim, i in enumerate(index)]
        parts[index] = arr[slice]

    # return the numpy array containing each subview
    return parts


def shear(arr, axis, direction, slope = 1):
    """ Shear the set of hyperplanes in arr defined by axis.
    direction determines the dimension along which we shear.
    slope specifies how steep the shear should be.

    Here are some examples in 2d. In 2d, each 'hyperplane' is a line, but
    the routine is general for the nd case::

	Start with a 2d array:

	0
	^
	|  6 7 8
        |  3 4 5
        |  0 1 2
        ----------> 1

	shear(0, 1, 1) - shear hyperplanes defined by axis 0 in 1 direction
	with a slope of 1:

        7 8 6
        5 3 4
        0 1 2

	shear(1, 0, 2) - shear hyperplanes defined by axis 1 in 0 direction
	with a slope of 2:

        6 1 5
        3 7 2
        0 4 8
    """
    # Can't shear a hyperplane in a perpendicular direction.
    if axis == direction:
        raise Exception("Error: axis cannot be same as shear direction.")

    # compensate for subtracted dimension
    if direction > axis:
        direction -= 1

    for i in xrange(1, arr.shape[axis]):
        plane = hyperplane(arr, axis, i)
        plane.flat = np.roll(plane, i * slope, axis=direction).flat


def tilt(arr, axis, direction, slope = 1):
    """ Tilt the set of hyperplanes defined by axis perpendicular to the
    hyperplanes. direction defines the dimension in which the tilt is
    performed. slope specifies how steep the tilt should be.

    Intuitively, in 3d, tilting a set of 2d planes (say XY) in the
    direction of its perpendicular (Z) along one of its dimensions (X or Y)
    is the same as shearing a set of perpendicular [hyper]planes (YZ or XZ
    respectively) along the perpendicular (Z). In other words,

    tile(0, 2, slope) = shear(2, 0, slope)
    tile(0, 1, slope) = shear(1, 0, slope)
    """
    # 'axis' is the subtracted dimension and hence cannot tilt in that dimension
    if axis == direction:
        raise Exception("Error: axis cannot be the same as the tilt dimension.")

    # compensate for subtracted dimension
    if axis > direction:
        axis -= 1

    for i in xrange(1, arr.shape[direction]):
        plane = hyperplane(arr, direction, i)
        plane.flat = np.roll(plane, i * slope, axis=axis).flat


def zigzag(arr, axis, direction, depth = 1, stride=1):
    """ Zigzag shifts hyperplanes against each other in alternating directions
    arr, axis, and direction have the same meaning as for shear and tilt.
    This command causes hyperplanes to be shifted in the indicated
    direction. The shift grows linearly up to the depth specified in the
    parameter depth over stride hyperplanes.
    """
    # 'axis' is the subtracted dimension and hence cannot zigzag in that
    # dimension
    if axis == direction:
        raise Exception("Error: axis cannot be the zigzag dimension.")

    # compensate for subtracted dimension
    if axis > direction:
        axis -= 1

    for i in xrange(1, arr.shape[direction]):
        base = (i/(stride*2))*(stride*2)+stride
        shift = depth-(abs(i-base)*depth)/stride
        plane = hyperplane(arr, direction, i)
        plane.flat = np.roll(plane, shift, axis=axis).flat
