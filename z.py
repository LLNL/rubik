#!/usr/bin/env python2.6

ze2d = [0x0000ffff, 0x00ff00ff, 0x0f0f0f0f, 0x33333333, 0x55555555]
ze3d = [0x000003ff, 0xff0000ff, 0x0300f00f, 0x030c30c3, 0x09249249]

zd2d = [0x55555555, 0x33333333, 0x0f0f0f0f, 0x00ff00ff, 0x0000ffff]
zd3d = [0x09249249, 0x030c30c3, 0x0300f00f, 0xff0000ff, 0x000003ff]

import numpy as np

def b(x):
    return "{0:064b}".format(x & 0xFFFFFFFFFFFFFFFF)

def b32(x):
    return "{0:032b}".format(x & 0xFFFFFFFF)

def print_filter(filter, printer = b):
    for elt in filter:
        print printer(elt)

#print_filter(ze2d, b32)
#print
#print_filter(ze3d, b32)
#print

def le_power_of_2(num):
    """Gets a power of two less than or equal to num.  Works for up to 64-bit numbers."""
    num |= (num >> 1)
    num |= (num >> 2)
    num |= (num >> 4)
    num |= (num >> 8)
    num |= (num >> 16)
    num |= (num >> 32)
    return num - (num >> 1)

class ZFilter(object):
    """Class representing a set of bitmasks for encoding/decoding n-dimensional morton numbers.
       Parameters:
          ndim     number of dimensions to encode/decode for.
          dtype    datatype of filter.  Determined by the width of the encoded morton numbers.
       Note: numpy.uint64 doesn't seem to support left shift, so this will not work
       with it.  Use int64 instead.
    """
    filters = {}

    def __init__(self, ndim, dtype):
        self.ndim = ndim
        self.dtype = dtype
        self.filter = self.get_filter(ndim, dtype)

    def get_filter(self, ndim, dtype):
        """Get a possibly memoized filter for the dimensions and dtype specified"""
        key = (ndim, dtype)
        filters = self.__class__.filters
        if not key in filters:
            filters[key] = self.create_filter(ndim, dtype)
        return filters[key]

    def create_filter(self, ndim, dtype):
        # Construct the initial mask: this masks out the lower width / ndim bits of a number
        max_width = np.dtype(dtype).itemsize * 8
        one = dtype(1)
        width = max_width / ndim
        mask = (one << width) - 1
        filter = [mask]

        # This loop creates progressively smaller masks
        width = le_power_of_2(width-1)
        while width > 0:
            mask = (one << width) - 1
            shift = width * ndim
            while (shift < max_width):
                mask |= dtype(mask << shift)
                shift <<= 1
            filter.append(mask)
            width >>= 1

        return filter

    def spread(self, x):
        x &= self.filter[0]
        i = 1 << (len(self.filter) - 4 + self.ndim)
        for mask in self.filter[1:]:
            x = (x ^ (x << i)) & mask
            i >>= 1
        return x

    def compact(self, x):
        x &= self.filter[-1]
        i = (1 << self.ndim-2)
        for mask in self.filter[-2::-1]:
            x = (x ^ (x >> i)) & mask
            i <<= 1
        return x

    def encode(self, point):
        code = 0x0
        for i,x in enumerate(point):
            code |= self.spread(x) << i
        return code

    def decode(self, value):
        return [self.compact(x[i] >> i) for i in self.ndim]
        self.compact(x)
        pass

    def __str__(self):
        return "\n".join([b32(elt) for elt in self.filter])

zfilter = ZFilter(3, np.int32)
for x in range(0,4):
    for y in range(0,4):
        for z in range(0,4):
            print "(%d,%d,%d) %d " % (x,y,z,zfilter.encode((x,y,z))),
        print
    print
    print

zfilter = ZFilter(2, np.int32)
for x in range(0,4):
    for y in range(0,4):
        print "(%d,%d) %d " % (x,y,zfilter.encode((x,y))),
    print




#for dim in range(2,8):
#    print "%s:" % dim
#    filter = zfilter(dim, np.uint32)
#    for elt in filter:
#        print b(elt & 0xFFFFFFFFFFFFFFFF)

