"""
This file defines the basic Partition class in Rubik.
"""

import zorder
import sys
import numpy as np
from arrayutils import *

class Partition(object):
    """ Tree of views of an initial Box. Each successive level is a set of
    views of the top-level box.
    """

    class PathElement(object):
	""" This class describes a partition in a hierarchy. It contains the
	partition and its index within its parent partition.
        """
        def __init__(self, partition, index):
            self.partition = partition
            self.index = index

        def get_flat_index(self):
            """ Uses the partition and the index to create a multi-index. """
            if not self.partition.parent:
                return 0
            else:
                return np.ravel_multi_index(self.index, self.partition.parent.children.shape)
        flat_index = property(get_flat_index)


    def __init__(self, box, parent, index, flat_index, level):
	""" Constructs a child Partition. Children have a view of the top-level
	array rather than a direct copy, and they do not have the Process
	list that the top-level Partition has.
        """
        self.box        = box
        self.procs      = None
        self.parent     = parent
        self.index      = index
        self.flat_index = flat_index
        self.level      = level
        self.children   = np.array([], dtype=object)

    # === Convenience attributes -- to be more numpy-like ==============
    shape = property(lambda self: self.box.shape)
    size  = property(lambda self: self.box.size)
    ndim  = property(lambda self: self.box.ndim)

    # === Partitioning routines ========================================
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
	""" Cuts this partition into a set of views, and make children out of
	them. See cut().
	"""
        views = cut(self.box, divisors, slicers)   # Get an array of all the subpartitions (views)

        # Create partitions so that they know their index and flat index
        # within the parent's child array.
        flat_index = 0
        for index in np.ndindex(views.shape):
            views[index] = Partition(views[index], self, index, flat_index, self.level + 1)
            flat_index += 1

        # Finally assign the numpy array to children
        self.children = views

    # === Reordering Routines ==========================================
    def transpose(self, axes):
        """ Transpose this partition by permuting its axes according to the axes
	array. See numpy.transpose().
        """
        self.box.flat = self.box.transpose(axes).flat

    def shear(self, axis, direction, slope):
	""" Shears the hyperplanes in this partition defined by one axis in the
	direction of another. See shear().
	"""
        shear(self.box, axis, direction, slope)

    def tilt(self, axis, direction, slope):
	""" Tilts the hyperplanes in this partition defined by one axis in one
	of the other directions. See tilt().
	"""
        tilt(self.box, axis, direction, slope)

    def zigzag(self, axis, direction, depth, stride):
	""" Zigzags the hyperplanes in this partition defined by one axis in one
	of the other directions. See zigzag().
	"""
        zigzag(self.box, axis, direction, depth, stride)

    def zorder(self):
        """ Reorder the processes in this box in z order. """
        zorder.zorder(self.box)

    # === Other Operations =============================================
    def leaves(self):
      """ Return all leaves for a partition. """
      if self.children.size:
	for child in self.children.flat:
	  for leaf in child.leaves():
	    yield leaf
      else:
	yield self

    def map(self, other):
	""" Map the other partition onto this one. First checks if partition
	sizes are compatible.
        """
        if self.procs:
            self.procs = other.procs

	myleaves = [x for x in self.leaves()]
	otherleaves = [x for x in other.leaves()]
	if len(myleaves) != len(otherleaves):
	    raise Exception("Error: Partitions are not compatible")

        for leaf, otherleaf in zip(self.leaves(), other.leaves()):
	    leaf.box.flat = otherleaf.box.flat

    def compatible(self, other):
	""" True if and only if other can be mapped to self.  This implies that
	at each level of the partition tree, self and other have the same
	number of children, and that the ith child of a node in self has the
	same size as the ith child of the corresponding node in other.
        """
        return (self.box.size == other.box.size and
                self.children.size != other.children.size and
                all(x.compatible(y) for x,y in zip(self.children, other.children)))

    def assign_coordinates(self):
        """ Assigns coordinates to all processes based on their position in the
	array.
	"""
        for index, i in np.ndenumerate(self.box):
            self.box[index].coord = index

    def write_map_file(self, stream = sys.stdout):
        """ Write a map file to the specified stream. By default this writes to
	sys.stdout.
	"""
        if not self.procs:
            raise Exception("Error: Must call write_map_file on root Partition")
        self.assign_coordinates()  # Put coords into all procs
        for proc in self.procs:
            format = " ".join(["%s"] * len(proc.coord)) + "\n"
            stream.write(format % proc.coord)

    def __getitem__(self, coords):
        """ Returns the child at a particular index. """
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

    def traverse_cells(self, visitor, path=None):
	""" Call a visitor function on each cell in the Partition. The visitor
	should look like this::

	  def visitor(global_index, path, element, index):
	      pass

	  Parameters:
		global_index
		  This is the index in the top-level partition i.e. the one you
		  called traverse_cells on.

		path
		  This is a list of PathElements describing the nesting of the cell
		  within partitions. For a PathElement p there are two properties of
		  interest:

		p[l].partition
		  the Partition at nesting level l

		p[l].index
		  the index of p[l] in its parent partition

		element
		  The element at self.box[global_index]

		index
		  The local index of the element within its parent partition
        """
        if not path:
	    # TODO: we probably shouldn't modify the contents if we want the
	    # Partition to be an abstract container.  Consider wrapping the
	    # elements in our own class
            path = []
            self.assign_coordinates()
            path.append(Partition.PathElement(self, (0,)*self.box.ndim))

        if self.children.size:
            for index, child in np.ndenumerate(self.children):
                path.append(Partition.PathElement(child, index))
                child.traverse_cells(visitor, path)
                path.pop()
        else:
            for index, elt in np.ndenumerate(self.box):
                visitor(elt.coord, path, elt, index)

    def invert(self):
	""" Returns an array of the same shape as this Partition's box. Each
	cell of this array will contain a list of partitions that contain
	that cell, ordered from top to bottom.
        """
        flat_box = self.box.copy()
        def path_assigner(global_index, path, elt, index):
            flat_box[global_index] = path[:]

        self.traverse_cells(path_assigner)
        return flat_box
