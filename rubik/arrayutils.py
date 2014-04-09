################################################################################
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Todd Gamblin et al. <tgamblin@llnl.gov>
# LLNL-CODE-599252
# All rights reserved.
#
# This file is part of Rubik. For details, see http://scalability.llnl.gov.
# Please read the LICENSE file for further information.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the disclaimer below.
#
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the disclaimer (as noted below) in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the LLNS/LLNL nor the names of its contributors may be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE
# U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################
"""
This file defines functions that operate on numpy arrays.  Most are the
fundamental rubik transforamtions that permute array elements.
"""
import numpy as np

def data(arr):
    """This gets the address of a numpy array's data buffer."""
    return arr.__array_interface__["data"][0]

def parent_corner(parent, view):
    """Finds index of the (0, 0, ...) element of view within its parent."""
    if view.base is not parent.base and view.base is not parent:
        raise ValueError("View and parent must be related.")

    offset = data(view) - data(parent)
    corner = []
    for s in parent.strides:
        corner.append(offset / s)
        offset %= s
    return tuple(corner)

class IndexConverter(object):
    """Given a numpy view and an index into it, this will convert the index in
    the view to the corresponding index in the parent.  Example::

        parent = np.empty([4,4])
        view = parent[1::2, 1::2]
        ic = IndexConverter(view)

        print ic.view_to_parent((0,0))
        (1,1)

        print ic.parent_to_view((1,1))
        (0,0)
    """
    def __init__(self, parent, view):
        self.corner = parent_corner(parent, view)
        self.scale = [v / b for v, b in zip(view.strides, parent.strides)]

    def view_to_parent(self, index):
        return tuple([c + i * s for c,i,s in zip(self.corner, index, self.scale)])

    def parent_to_view(self, index):
        return tuple([(i - c) / s for c,i,s in zip(self.corner, index, self.scale)])


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

    tilt(0, 2, slope) = shear(2, 0, slope)
    tilt(0, 1, slope) = shear(1, 0, slope)
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
