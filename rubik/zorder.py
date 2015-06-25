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
This file provides routines to transform the elements of an ndarray from
dimension-major order to Z-order.
"""

import numpy as np
import itertools
import math

def le_power_of_2(num):
    """ Gets a power of two less than or equal to num. Works for up to 64-bit
    numbers.
    """
    num |= (num >> 1)
    num |= (num >> 2)
    num |= (num >> 4)
    num |= (num >> 8)
    num |= (num >> 16)
    num |= (num >> 32)
    return num - (num >> 1)


def b(num, bytes=8):
    bytefields = ["{0:08b}".format(num >> 8*i & 0xFF) for i in reversed(range(bytes))]
    return " ".join(bytefields).replace('0', '-')


def h(num, bytes=8):
    quads = bytes / 2
    quadfields = ["{0:04x}".format(num >> 8*i & 0xFFFF) for i in reversed(range(quads))]
    return " ".join(quadfields).replace('0', '-')


class ZEncoder(object):
    """ Class representing a set of bitmasks for encoding/decoding
    n-dimensional morton numbers.

    Parameters:
	ndim	number of dimensions to encode/decode for.
        bits	number of bits in the generated codes.

    Note: Codes are internally generated with as many bits as are necessary,
    then they are returned as either numpy.uint32 or numpy.uint64, depending
    on how many bits are needed to represent the codes.
    """
    filters = {}

    def __init__(self, ndim, bits = 64):
        self.ndim = ndim
        self.bits = bits
        self.filter = self.get_filter(ndim, bits)

    @classmethod
    def for_shape(cls, shape):
        # Default to 32-bit codes, but use 64-bit codes if needed for a shape.
        maxdim = max(shape)
        if math.ceil(math.log(maxdim, 2)) >= (32 / len(shape)):
            bits = 64
        else:
            bits = 32
        return ZEncoder(len(shape), bits)

    def get_filter(self, ndim, bits):
        """ Get a possibly memoized filter for the dimensions bit length
	specified.
	"""
        key = (ndim, bits)
        filters = self.__class__.filters
        if not key in filters:
            filters[key] = ZEncoder.create_filter(ndim, bits)
        return filters[key]

    @classmethod
    def create_filter(cls, ndim, bits):
	""" This creates a filter (a set of bitmasks) that can be used to
	quickly generate n-dimensional Z codes (Morton codes). The filter is
	based on the method described here:
	http://graphics.stanford.edu/~seander/bithacks.html#InterleaveBMN

        Parameters:

	ndim
	  number of dimensions in the z curve
        bits
	  bit width of morton codes to be generated. Each coordinate gets
	  bits/ndim bits.
        """
	# Construct the initial mask: this selects just the lower (bits/ndim)
	# bits of a number
        width = bits / ndim
        mask = (1 << width) - 1

        # Each bit in a coordinate needs to move (ndim * width) positions.
	# We'll move them using iterative power-of-2 shifts for O(log(bits))
	# complexity.
        # First get the max shift we need to do:
        max_shift = le_power_of_2(ndim * (width-1))
        filter = [[mask, 0, max_shift]]

	# Now figure out which bits need to be moved by each shift, and build
	# masks.
        shift = max_shift
        while shift > 0:
            mask = 0
            shifted = 0
            shift_mask = ~(shift-1)
            for bit in range(width):
                distance = ndim * bit - bit
                shifted |= (shift & distance)
                mask |= 1 << bit << (distance & shift_mask)
            if shifted:
                filter.append([mask, shift, shift >> 1])
            shift >>= 1

        # set last rshift to zero for compact operation.
        filter[-1][2] = 0
        return filter

    def spread(self, x):
        """ Applies filter to spread the bits of x apart by ndim-1 zeros. """
        for mask, shift, rshift in self.filter:
            x = (x | (x << shift)) & mask
        return x

    def compact(self, x):
        """ Applies filter in reverse to push spread bits back together. """
        for mask, shift, rshift in self.filter[-1::-1]:
            x = (x | (x >> rshift)) & mask
        return x

    def encode(self, point):
        """ Takes a point and returns a morton code for that point. """
        if len(point) != self.ndim:
            raise Exception("Error: Can't encode %d-dimensional point with %d-dimensional ZEncoder." % (len(point), self.ndim))
        code = 0
        for i,x in enumerate(point):
            code |= self.spread(x) << i

        # Python will do arithmetic with the largest ints it can.
        # This ensures we return an int32 when it was asked for.
        return code

    def decode(self, value):
	""" Given an ndim-dimensional morton code, returns the corresponding
	point as a tuple.
	"""
        return tuple([self.compact(value >> i) for i in range(self.ndim)])

    def __bytes(self):
        """ Number of bytes needed to represent the masks in this filter. """
        if self.bits > 32:
            return 8
        else:
            return 4

    def __str(self, format):
	""" Prints out each mask in the filter line along with its
	corresponding left and right shifts. format paramter determines how to
	format binary numbers. Options are b or h.
        """
        fields = ["%s %2d %2d" % (format(mask, self.__bytes()), shift, rshift) for mask, shift, rshift in self.filter]
        return "\n".join(fields)

    def hex_str(self):
	""" formatted string with masks in the filter in hexadecimal, along
	with their left and right shifts.
	"""
        return self.__str(h)

    def bin_str(self):
	""" formatted string with masks in the filter in binary, along with
	their left and right shifts.
	"""
        return self.__str(b)

    def __str__(self):
        """ Equivalent to bin_str() """
        return self.bin_str()


def zenumerate(shape, proc):
    """ Enumerates points in the shape in Z order. Currently dumps the morton
    codes into an array and sorts them, then regenerates points in that
    order. This is O(nlogn) time. We could do better for matrices with more
    even aspect ratios by enuerating all morton codes from 0 on and
    converting to points, but that is O(n^2) for irregular shapes.
    """
    # Build a buffer of encoded z values and sort them
    zencoder = ZEncoder.for_shape(shape)
    buffer = []
    for point in np.ndindex(*shape):
        if proc == 1:
            if Check_if_available(point) == -1:
                continue
        buffer.append(zencoder.encode(point))
    buffer.sort()

    # Decode z values in order and yield each.
    for code in buffer:
        yield zencoder.decode(code)


def zorder(arr, proc):
    """ Transform the elements of an ndarray from dimension-major order to z
    order. This modifies the array.
    """
    buffer = arr.copy()
    i=0
    for index in zenumerate(arr.shape, proc):
        arr[index] = buffer.flat[i]
        i += 1


def Check_if_available(point):
    """Checks if the processor is available for mapping. Returns -1 if the processor is not available for mapping.
    Assumes that every processor with y-coordinate divisible by 2 is not availabe for mapping.
    """
    if point[1] % 2 == 0:
        return -1
    return 1

