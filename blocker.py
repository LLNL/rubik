#!/usr/bin/env python
description = """\
Blocker generates mapping files for torus and mesh networks according to
structured transformations of blocks within the ranks.

Todd Gamblin tgamblin@llnl.gov
"""
import numpy as np
import zorder
import optparse, itertools, sys

class Process(object):
    """The process class represents a single task in a parallel application with a unique id
       in [0,ntasks).  Processes exist in a doubly linked list that runs through the full id
       space.  This allows us to quickly run through the ranks in order, even after a network
       topology has been permuted extensively.

       See Process.make_list() to create lists of processes from ranges of identifiers.
    """
    def __init__(self, id, next=None, prev=None):
        """Constructs a process with a particular id, optionally as part of a list.
           Parameters:
             id      arbitrary process identifier.
             next    next Process in a list.
             prev    previous Process in a list.
        """
        self.id      = id
        self.coord   = None

        self.next = next
        if next: next.prev = self
        self.prev = prev
        if prev: prev.next = self

    @classmethod
    def make_list(cls, iterable):
        """Takes an iterable over a list of identifiers and builds a linked list of
           Processes out of it.  Can be used with ranges to create lists of ranks, but allows
           arbitrary process identifiers.
        """
        head = None
        tail = None
        for element in iter(iterable):
            tail = Process(element, None, tail)
            if not head: head = tail
        return head

    def __iter__(self):
        """Iterate over this list forward, starting from this node."""
        node = self
        while node:
            yield node
            node = node.next

    def __reversed__(self):
        """Iterate over this list backwards, starting from this node."""
        node = self
        while node:
            yield node
            node = node.prev

    def __str__(self):
        """String representation for printing is just the identifier."""
        return "%d" % self.id

    def __array__(self, type):
        """Allows a Process list to be assigned directly into a numpy ndarray."""
        return np.array([x for x in self], dtype=type)


def hyperplane(arr, axis, index):
    """This generates a slice list that will select one hyperplane out of a numpy ndarray by
       fixing one axis to a particular coordinate.
    """
    selector = [slice(None)] * arr.ndim
    selector[axis] = index
    return arr[selector]


def mod(arr, dim, chunk, nchunks):
    """Given an array, the dimension (axis) it's being sliced on, the chunk and the number of chunks,
       returns a slice that divides that dimension into modulo sets.
    """
    return slice(chunk, None, nchunks)


def div(arr, dim, chunk, nchunks):
    """Given an array, the dimension (axis) it's being sliced on, the chunk and the number of chunks,
       returns a slice that divides that dimension into contiguous pieces.  If nchunks doesn't evenly
       divide arr.shape[dim], the last slice will include the remainder.
    """
    chunksize = arr.shape[dim] / nchunks
    start = chunk * chunksize
    end = start + chunksize

    # last slice takes any leftovers for uneven divides
    if chunk == nchunks - 1: end = None

    return slice(start, end)


def cut(arr, divisors, slicers = div):
    """Given an array and a list of divisors, up to one per dimension, cuts the array using the
       slice generator functions in 'slicers'.  If slicers is a function, use that for all axes.
       If slicers is an array, use one slicer per axis.  If no slicers are provided, use div for
       all axes.
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


def tilt(arr, axis, direction, slope = 1):
    """Tilts the elements in arr along the specified axis.
       direction determines the axis whose direction we should tilt in.
       slope specifies how steep the tilt should be. Here are some examples in 2d.
       In 2d, each 'hyperplane' is a line, but the routine is general for the nd case.

       Start with 2d array:                                    0 1 2
                                                               3 4 5
                                                               6 7 8

       Tilt array along 0 axis in 1 direction w/slope 1:       0 1 2
                                                               5 3 4
                                                               7 8 6

       Tilt array along 1 axis in 0 direction w/slope 2:       0 4 8
                                                               3 7 2
                                                               6 1 5
    """
    if axis == direction:
        raise Exception("Error: axis cannot be same as tilt direction.")

    for i in xrange(arr.shape[axis]):
        plane = hyperplane(arr, axis, i)
        if direction > axis: direction -= 1   # compensate for subtracted dimension
        plane.flat = np.roll(plane, i * slope, axis=direction).flat

class Partition(object):
    """Tree of views of an initial Box.  Each successive level is a set of views of the top-level box."""
    def __init__(self, box, parent):
        """Constructs a child Partition.  Children have a view of the top-level array rather than a direct
           copy, and they do not have the Process list that the top-level Partition has.
        """
        self.box       = box
        self.procs     = None
        self.parent    = parent
        self.children  = np.array([], dtype=object)

    @classmethod
    def create(cls, shape):
        """Constructs the top-level partition, with the original numpy array and a process list
           running through it.
        """
        box      = np.ndarray(shape, dtype=object)
        procs    = Process.make_list(xrange(0, box.size))
        box.flat = procs
        p = Partition(box, None)
        p.procs = procs
        return p

    # === Partitioning routines =================================================================
    def div(self, divisors):
        self.cut(divisors, div)

    def tile(self, tiles):
	divisors = [0]*len(tiles)
	for i in range(len(tiles)):
	    divisors[i] = self.box.shape[i] / tiles[i]
        self.cut(divisors, div)

    def mod(self, divisors):
        self.cut(divisors, mod)

    def cut(self, divisors, slicers):
        """Cuts this partition into a set of views, and make children out of them. See cut()."""
        views = cut(self.box, divisors, slicers)   # Get an array of all the subpartitions (views)

        def create_child_partition(view):
            return Partition(view, self)
        create_children = np.vectorize(create_child_partition)
        self.children = create_children(views)     # Make a Partition object out of each view

    # === Reordering Routines ===================================================================
    def transpose(self, axes):
        """Transpose this partition by permuting its axes according to the axes array.
           See numpy.transpose().
        """
        self.box.flat = self.box.transpose(axes).flat

    def tilt(self, axis, direction, slope):
        """Tilts the box in this partition along one axis in the direction of another.  See tilt()."""
        tilt(self.box, axis, direction, slope)

    def zorder(self):
        """Reorder the processes in this box in z order."""
        zorder.zorder(self.box)

    # === Other Operations =====================================================================
    def map(self, other):
        """Map the other partition onto this one.  First checks if partition sizes are compatible."""
        # We will assign every element of other to self.  We need to swap in other's process list so
        # that things are consistent.
        if self.procs:
            self.procs = other.procs

        if self.children.size:
            # We only want to do assignment at the leaves, so follow the views until we get there.
            for child, otherchild in zip(self.children.flat, other.children.flat):
                child.map(otherchild)
        else:
            # At leaves, assign procs through the view.
            self.box.flat = other.box.flat

    def compatible(self, other):
        """True if and only if other can be mapped to self.  This implies that at each level of the
           partition tree, self and other have the same number of children, and that the ith child
           of a node in self has the same size as the ith child of the corresponding node in other.
        """
        return (self.box.size == other.box.size and
                self.children.size != other.children.size and
                all(x.compatible(y) for x,y in zip(self.children, other.children)))

    def assign_coordinates(self):
        """Assigns coordinates to all processes based on their position in the array."""
        for index, i in np.ndenumerate(self.box):
            self.box[index].coord = index

    def write_map_file(self, stream = sys.stdout):
        """Write a map file to the specified stream.  By default this writes to sys.stdout."""
        if not self.procs:
            raise Exception("Error: Must call write_map_file on root Partition")
        self.assign_coordinates()  # Put coords into all procs
        for proc in self.procs:
            format = " ".join(["%s"] * len(proc.coord)) + "\n"
            stream.write(format % proc.coord)

    def __getitem__(self, coords):
        """Returns the child at a pearticular index"""
        return self.children[coords]

    def __iter__(self):
        for child in self.children.flat:
            yield child

    def plot(self, graph=None):
        import pygraphviz as pgv  # only import this if we call plot function
        if not graph:
            graph = pgv.AGraph(bgcolor="transparent")
            graph.node_attr['shape'] = "circle"

            for proc in self.box.flat:
                graph.add_node(proc.id,f="foo")

            for child in self.children.flat:
                child.plot(graph)

            return graph
        else:
            name = "-".join([str(x.id) for x in self.box.flat])
            subgraph = graph.subgraph([x.id for x in self.box.flat], name="cluster%s" % name)
            for child in self.children.flat:
                child.plot(subgraph)

    def __invert_helper(self):
        for path in self.box.flat:
            path.append(self)
        for child in self.children.flat:
            child.__invert_helper()

    def invert(self, coordinate=None):
        """Returns an array of the same shape as this Partition's box.  Each cell of this array
           will contain a list of partitions that contain that cell, ordered from top to bottom.
        """
        temp = self.box.copy()
        for i in xrange(self.box.size):
            self.box.flat[i] = []
        self.__invert_helper()

        flat_box = self.box.copy()
        self.box.flat = temp.flat
        return flat_box

