#!/usr/bin/env python

import numpy as np
import itertools

def mod(arr, dim, chunk, nchunks):
    return slice(chunk, None, nchunks)

def div(arr, dim, chunk, nchunks):
    chunksize = arr.shape[dim] / nchunks
    start = chunk * chunksize
    end = start + chunksize

    # last slice takes any leftovers for uneven divides
    if chunk == nchunks - 1: end = None

    return slice(start, end)


def cut(arr, divisors, modes = None):
    # By default cut just divides everything
    if not modes:
        modes = [div] * len(divisors)

    # Make an iterator over the cartesian product of the ranges of each divisor value.  This gives us a
    # set of unique identifiers for each subdivision of the array.
    slice_ids = itertools.product(*[xrange(d) for d in divisors])

    # Map the slice generator to each dimensional index in the slice id to get a slice.
    def get_slice_for_dim(dim, chunk):
        slicer = modes[dim]
        return slicer(arr, dim, i, divisors[dim])
    slices = [[get_slice_for_dim(d, i) for d, i in enumerate(id)] for id in slice_ids]

    # Slice the array up and return views for each of the mod set slices.
    return [arr[s] for s in slices]


dims = np.array([4, 4])
foo = np.zeros(dims, dtype=np.int32)
foo.flat = xrange(dims.prod())

#x = range(dims[0:-1].prod())
#foo.flat = np.array(zip(x,x)).flat

print foo
print
print
for arr in cut(foo, [2, 3], [div, mod]):
    print arr
    print "==="

