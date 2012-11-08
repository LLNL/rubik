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

    def __init__(self, box, parent, index, flat_index, level):
	""" Constructs a child Partition. Children have a view of the top-level
	array rather than a direct copy, and they do not have the Process
	list that the top-level Partition has.
        """
        self.box        = box
        self.vtob       = None
        self.btov       = None
        self.procs      = None
        self.parent     = parent
        self.index      = index
        self.flat_index = flat_index
        self.level      = level
        self.children   = np.array([], dtype=object)

    # === Convenience attributes -- to be more numpy-like ==============
    @property
    def shape(self):
        return self.box.shape

    @property
    def size(self):
        return self.box.size

    @property
    def ndim(self):
        return self.box.ndim

    # === Index conversion routines ====================================
    def parent_to_self(self, index):
        if not self.btov:
            self.btov = base_to_view(self.box)
        return self.btov(index)

    def self_to_parent(self, index):
        if not self.vtob:
            self.vtob = view_to_base(self.box)
        return self.vtob(index)

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
            p = Partition(views[index], self, index, flat_index, self.level+1)
            views[index] = p
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
    def depth(self):
        return 1 + max([child.depth() for child in self.children.flat] + [0])

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
        for index in np.ndindex(self.box.shape):
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

    @property
    def xancestors(self):
        """Yields this partition's ancestors, starting at the root and ending
        with this partition.
        """
        if self.parent:
            for a in self.parent.xancestors:
                yield a
        yield self

    @property
    def ancestors(self):
        """Returns a list of this partition's ancestors, starting at the root
        and ending with this partition.
        """
        return list(self.xancestors)

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


    class PathNode(object):
	"""This contains metadata for a particular data element in a partition.
        It contains the partition and the element's index within the partition.
        It also contains methods for getting the flat index of the element
        within that partition.
        """
        def __init__(self, partition, index = None):
            self.partition = partition
            self.index = index

        @property
        def flat_index(self):
            return np.ravel_multi_index(self.index, self.partition.shape)

        @property
        def element(self):
            """Returns the element at this index in the partition."""
            return self.partition.box[self.index]


    def traverse_cells(self, visitor):
	""" Call a visitor function on each cell in the Partition. The visitor
	should look like this::

	  def visitor(path):
	      pass

        The path passed in is a list of PathNodes describing the nesting of
        the cell within partitions.  From the path, you can get all the
        containing partitions of the element it points to, the element,
        and both n-dimensional and flat indices of the element within each
        partition.  See PathNode for more details.
        """
        if self.children.size:
            for index, child in np.ndenumerate(self.children):
                child.traverse_cells(visitor)
        else:
            for index, elt in np.ndenumerate(self.box):
                # Build a list of PathNodes containing ancestor partitions
                path = [Partition.PathNode(p) for p in self.ancestors]
                path[-1].index = index
                # assign index of elt within each partition to each PathNode
                i = -2
                while i >= -len(path):
                    child = path[i+1]
                    path[i].index = child.partition.self_to_parent(child.index)
                    i -= 1
                # Now visit the element with its path.
                visitor(path)

    def invert(self):
	""" Returns an array of the same shape as this Partition's box. Each
	cell of this array will contain a list of partitions that contain
	that cell, ordered from top to bottom.
        """
        flat_box = self.box.copy()
        def path_assigner(path):
            flat_box[path[0].index] = path

        self.traverse_cells(path_assigner)
        return flat_box

