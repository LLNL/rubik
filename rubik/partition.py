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
This file defines the hierarchical Partition class in Rubik.
"""
import zorder
import sys
import pickle
import numpy as np

from contextlib import closing
from arrayutils import *
from itertools import ifilter
from collections import deque
from operator import itemgetter
from pyprimes import awful
from pyprimes import factors as fact
from pyprimes import strategic
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
        self.iconv      = None
        self.parent     = parent
        self.index      = index
        self.flat_index = flat_index
        self.level      = level
        self.cutargs    = None
        self.children   = np.array([], dtype=object)
        self.elements   = []

    @classmethod
    def empty(clz, shape):
        box = np.empty(shape, dtype=object)
        index = (0,) * len(box.shape)
        return Partition(box, None, index, 0, 0)

    @classmethod
    def fromlist(clz, shape, elements):
        p = Partition.empty(shape)
        p.flat = elements
        return p

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

    def __iter__(self):
        for child in self.children.flat:
            yield child

    @property
    def flat(self):
        return FlatIterator(self)

    @flat.setter
    def flat(self, elts):
        self.elements = [Meta(e) for e in elts]
        self.box.flat = self.elements

    # === Index conversion routines ====================================
    def parent_to_self(self, index):
        if not self.iconv:
            self.iconv = IndexConverter(self.parent.box, self.box)
        return self.iconv.parent_to_view(index)

    def self_to_parent(self, index):
        if not self.iconv:
            self.iconv = IndexConverter(self.parent.box, self.box)
        return self.iconv.view_to_parent(index)

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
        # Get an array of all the subpartitions (views)
        views = cut(self.box, divisors, slicers)

        assert(all(view.base != None for view in views))

        # Create partitions so that they know their index and flat index
        # within the parent's child array.
        flat_index = 0
        for index in np.ndindex(views.shape):
            p = Partition(views[index], self, index, flat_index, self.level+1)
            views[index] = p
            flat_index += 1

        # Assign the numpy array to children
        self.children = views

        # Record the cut so we can copy this object later.
        self.cutargs = (divisors, slicers)

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


    def tilt_cont(self, slope):
    	#    """ Tilts the hyperplanes defined by axis and in all directions in cyclic order."""
    	tilt_cont(self.box, slope)

    def tilt_combi(self, slope):
    	#    """ Tilts the hyperplanes for every combination of axis and directions."""
    	tilt_combi(self.box, slope)


    def zigzag(self, axis, direction, depth, stride):
	""" Zigzags the hyperplanes in this partition defined by one axis in one
	of the other directions. See zigzag().
	"""
        zigzag(self.box, axis, direction, depth, stride)

    def zorder(self):
        """ Reorder the processes in this box in z order. """
        zorder.zorder(self.box)
    
    def zorder_cray(self):
        return zorder.zenumerate(self.box.shape)
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
	myleaves = [x for x in self.leaves()]
	otherleaves = [x for x in other.leaves()]
	if len(myleaves) != len(otherleaves):
	    raise Exception("Error: Partitions are not compatible")

        # Need to copy Meta objects between partitions (metadata is separate).
        copies = dict((e, e.copy()) for e in other.box.flat)
        self.elements = [copies[e] for e in other.elements]
        for leaf, otherleaf in zip(myleaves, otherleaves):
	    leaf.box.flat = [copies[o] for o in otherleaf.box.flat]

    def compatible(self, other):
        """ True if and only if other can be mapped to self.  This is true
            if the leaves of self have the same order, number, and size as
            the leaves of self."""
        myleaves = [x for x in self.leaves()]
        otherleaves = [x for x in other.leaves()]
        if len(myleaves) != len(otherleaves):
            return false
            return all(m.size == o.size for m, o in zip(myleaves, otherleaves))


    def assign_coordinates(self):
        """Assigns each element its coordinate in the box."""
        for index in np.ndindex(self.box.shape):
            self.box[index].coord = index

    def write_map_file(self, stream=sys.stdout):
        """ Write a map file to the specified stream. By default this writes to
	sys.stdout.
	"""
        close = False
        # make it easy for folks and open a file for them.
        if type(stream) == str:
            stream = open(stream, "w")
            close = True

        if self.parent == None:
            elements = self.elements
        else:
            # Here we generate a map file from the elements in the order they
            # were provided to the root of the tree.  This is about the best we
            # can do for mapping a subpartition, since the semantics aren't
            # obvious.
            my_elts = set(self.box.flat)
            elements = ifilter(my_elts.__contains__, self.root.elements)

        self.assign_coordinates()
        for elt in elements:
            format = " ".join(["%s"] * len(elt.coord)) + "\n"
            stream.write(format % elt.coord)

        if close:
            stream.close()

    def recursive_partitioning(self, big_box, num_pes, factors, dirVec, dirIdx): # recursive partiton by factors of the given numpes 
        "sort big_box in 'direction'"
        dirIdx = (dirIdx +1) % 3
        big_box = sorted(big_box, key=itemgetter(dirVec[dirIdx]))
        temp = []
        big_box = [big_box]
#        print 'big_box'
#        print big_box
        j=0
        finalResult = []
        for fact in factors:
#        fact = factors[0]
            for partition in big_box:
                for i in range(0, num_pes, num_pes/fact):
                    if num_pes/fact == 1:
                        finalResult.append(partition[i:i+num_pes/fact][0])
                    else:
                        temp.append(partition[i:(i+num_pes/fact)]) # splitting
                print("i : %d, fact : %d\n") % (i, fact)
#               print temp
#               print '\n'
#            print temp
            if len(finalResult) == 0:
                dirIdx = (dirIdx + 1) % 3
#            print("i : %d, fact : %d, direction : %d\n") % (i, fact, direction)
                big_box = [ sorted(i, key=itemgetter(dirVec[dirIdx])) for i  in temp ]
                temp = []
#            print big_box
#            print '\n'
                num_pes = num_pes/fact
            else: 
                break
        big_box = finalResult
#        print big_box
        return big_box

    def assign_coordinates_cray(self, big_box, big_torus, type1 = 'zorder', dimVector=None):
        """ Assigns the elements their coordinates as per actual cray grid in the user specified order. -> In this function, the coordinates are ordered in the speicifed method by the user such as z-order, row order and grid-order. 
        """
        buffer = []
        if type1 == "zorder":
            zorderedIndex = big_torus.zorder_cray()
            for i in zorderedIndex: #np.ndindex(big_torus.box.shape):
                if big_box[i] != -1:
                    temp = i+(int(big_box[i][0]),int(big_box[i][1]))
                    buffer.append(temp)
        elif type1 == "row_order":
            for i_big in np.ndindex(big_box.shape):                
                if big_box[i_big] != -1:
                    temp = i_big+(int(big_box[i_big][0]), int(big_box[i_big][1]))
                    buffer.append(temp)
        elif type1 == "rcb_order":
            num_pes = len(self.elements)
            directions = [0,1,2]
            if dimVector != None:
               dimVector = sorted(list(enumerate(dimVector)), key=itemgetter(1),reverse=True)
               directions = list(map(int, zip(*dimVector)[0]))
            print directions
            if strategic.is_prime(awful.isprime, num_pes):
                print 'deal with prime number with normal bisection' # I'm apply the recursive graph bisection or recursive spectral bisection for the next step
            else:
                factors = fact.factorise(num_pes)
                j=0
                temp =[]
                print ("factors[0] : %d, number of pes in the first partiton : %d\n") % (factors[0], num_pes/factors[0])
                for i_big in sorted(np.ndindex(big_box.shape), key=itemgetter(3)):#directions[0])):
                    if big_box[i_big] != -1:
                        print i_big
                        temp.append(i_big+(int(big_box[i_big][0]), int(big_box[i_big][1]))) #first splitting here. The numpy array is split into python lists so that further partition can be done so easily
                        j=j+1
                        if j >= (num_pes/factors[0]):
#                            print temp
                            for i in self.recursive_partitioning(temp, num_pes/factors[0], factors[1:len(factors)], directions, 0):
                                buffer.append(i)
                            temp = []
                            j=0
        
        elif type1 == "grid_order": # this mapping is not complete
            for i_big in np.ndindex(big_box.shape):
              if big_box[i_big] != -1:
                 temp = i_big+(int(big_box[i_big][0]), int(big_box[i_big][1]))
                 print i_big
            for j in buffer:
                print "grid_order"

        i = 0

        for elt in self.elements: #np.ndindex(self.box.shape):
            temp = elt.coord
            elt.coord = elt.coord + (i,) # buffer[i] #
            self.box[temp] = elt
            i += 1
        print buffer
        return buffer


    def write_map_cray(self, big_box, big_torus, type1 = 'zorder', stream=sys.stdout, dimVector=None):
        """ Writes a map file for cray machines to a specified stream.
        """
        close = False
        """ Write mpi ranks into this custom rank order file so that a mpi application can be executed using this file"""       

        rankReorderFile = open("MPICH_RANK_ORDER", "w")
        rankReorderBuffer = []
  
        if self.parent != None:
            print "this partition is not root box, write_map_cray only can be applied to a root partition"
            sys.exit()

        if type1 == 'grid_order':
           checkResult = gridOrder.grid_vector_check(dimVector, self.box.shape)
           if checkResult[0] == False:
              print checkResult[1]
              sys.exit()
 
        if type(stream) == str:
            stream = open(stream, "w")
            close = True

        if self.parent == None:
            elements = self.elements
        else:
            my_elts = set(self.box.flat)
            elements = ifilter(my_elts.__contains__, self.root.elements)

        self.assign_coordinates() #assign coordinates in a logical torus box
        buffer = self.assign_coordinates_cray(big_box, big_torus, type1, dimVector) #buffer the physical coordinates in the real torus network in the specified order such as z-order, row-order and grid-order
        temp_coords = []

        #the followings are for mapping the logical coordinates into the physical coordinates in the real torus network with holes such as BlueWaters. 

        for elt in elements: ##first collects logical orders in a temporary list. These coordinates are determined by the user specified partitioning and transformations. 
            format = " ".join(["%s"] * len(elt.coord)) # + "\n"
            temp_coords.append(elt.coord)
        i=0
        finalResult = []
        for coord in sorted(temp_coords):# sort the coordinates in a logical box because they are ordered by the user specified partitioning and transformations. 
            finalResult.append(coord + buffer[i]) ## each coordinates in the sorted logical box starting from (0, 0, 0) to (n, n, n) are mapped to each coordintates in the real torus network in the order specified by the user with the type1 variable. 
            i+=1

        for finalCoord in sorted(finalResult, key=itemgetter(len(self.box.shape))):
            format = " ".join(["%s"] * len(finalCoord))  + "\n"
            stream.write(format % finalCoord)
            rankReorderBuffer.append((finalCoord[len(self.box.shape)], finalCoord[len(finalCoord)-2]))#write the final order of mpi ranks in to map_file and MPICH_ORDER_FILE so that mpi applications could run with this mapping. 
        rankReorderBuffer = zip(*sorted(rankReorderBuffer, key=itemgetter(1)))
        rankReorderFile.write(",".join(map(str,rankReorderBuffer[0])))
        rankReorderFile.close()
        if close:
            stream.close()
        return rankReorderBuffer[0]

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

    @property
    def root(self):
        p = self
        while p.parent != None:
            p = p.parent
        return p

    def plot(self, graph=None):
        import pygraphviz as pgv  # only import this if we call plot function
        if not graph:
            graph = pgv.AGraph(bgcolor="transparent")
            graph.node_attr['shape'] = "circle"

            for elt in self.flat:
                graph.add_node(elt.id, f="foo")

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
            return self.partition.box[self.index].value

        @property
        def meta(self):
            """Returns the Meta object at this index in the Partition.
            Rubik uses this to put metadata on items in the Partition.
            """
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

    def get_elements_in_root_order(self):
        """This gets a list of the elements currently in *this* partition, but
        they are in the order they appear in the root elements list."""

    def copy(self, clone=None):
        """Return a copy of this partition with the same cuts applied to it, and
        with the same elements contained in this one."""
        if not clone:
            clone = Partition.empty(self.box.shape)

            # Create copies of only elements in this partition's box.
            copies = dict((e, e.copy()) for e in self.box.flat)

            # Create a list of copied elements in the order they appear in the
            # root partition's element list.  This allows us to copy a subtree
            # of a partition tree into its own partition tree.
            for elt in ifilter(copies.__contains__, self.root.elements):
                clone.elements.append(copies[elt])

            clone.box.flat = [copies[elt] for elt in self.box.flat]

        if self.children.size:
            clone.cut(*self.cutargs)
            for child, cp_child in zip(self.children.flat, clone.children.flat):
                child.copy(cp_child)

        return clone

    def __copy__(self):
        """Make sure that if someone uses the copy module that it works"""
        return self.copy()

    def __repr__(self):
        """Print a nice string representation of this array."""
        box = np.empty(self.box.shape, dtype=object)
        box.flat = [elt.value for elt in self.box.flat]
        return box.__repr__()

    def get_cuts(self, cuts=None):
        """Returns a list of cuts in this partition in depth-first order."""
        if not cuts:
            cuts = deque()

        cuts.append(self.cutargs)
        for child in self.children.flat:
            child.get_cuts(cuts)
        return cuts

    def __getstate__(self):
        """This encodes the partition as its root element list, its root
        box, and all of the cuts applied to it in depth-first order."""
        self.assign_coordinates()
        return (self.shape, self.elements, self.get_cuts())

    def do_cuts(self, cuts):
        """Replays a deque of cuts from get_cuts()"""
        if not cuts:
            raise ValueError("cuts list was empty!")

        cutargs = cuts.popleft()
        if cutargs:
            self.cut(*cutargs)

        for child in self.children.flat:
            child.do_cuts(cuts)

    def __setstate__(self, state):
        """Decodes output of __getstate__."""
        shape, elements, cuts = state

        self.box = np.empty(shape, dtype=object)
        self.iconv      = None
        self.parent     = None
        self.index      = (0,) * len(shape)
        self.flat_index = 0
        self.level      = 0
        self.cutargs    = None
        self.children   = np.array([], dtype=object)
        self.elements   = elements

        for elt in elements:
            self.box[elt.coord] = elt

        self.do_cuts(cuts)

class Meta(object):
    """Everything in the partition is wrapped in a Meta object that
    stores attributes like color and coords.  This is so that we don't
    destructively modify data contained in a partition. """
    def __init__(self, value, color=None, coord=None):
        self.value = value
        self.color = color
        self.coord = coord

    def copy(self):
        return Meta(self.value, self.color, self.coord)

class FlatIterator(object):
    """This class implements the ndarray.flat-like semantics. """
    def __init__(self, partition):
        self.partition = partition

    def __iter__(self):
        for elt in self.partition.flat:
            yield elt.value

    def __getitem__(self, index):
        return self.partition.flat[index].value
