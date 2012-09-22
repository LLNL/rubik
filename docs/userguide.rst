User Guide
==========

Rubik is a tool that simplifies the process of creating task mappings for
structured applications.  Rubik allows an application developer to specify
communicating groups of processes in a virtual application topology succinctly
and map them onto groups of processors in a physical network topology.  Both
the application topology and the network topology must be Cartesian, but the
dimensionality of either is arbitrary.  This allows users to easily map
low-dimensional structures such as planes to higher-dimensional structures like
cubes to increase the number of links used for routing.

Rubik also provides embedding operations that adjust the way tasks are laid out
within groups.  These operations are intended to optimize particular types of
communication among ranks in a group, either by shifting them to increase the
number of available links for communication between processor pairs or by
moving communicating ranks closer together on the Cartesian topology to reduce
latency.  In conjunction with Rubik's mapping semantics, these operations allow
users to create a wide variety of task layouts for structured codes by
composing a few fundamental operations, which we describe in the following
sections.

Partition Trees
---------------

The fundamental data structure in Rubik is the *partition tree*, a hierarchy of
*n*-D Cartesian spaces. We use partition trees to specify groups of tasks (or
processes) in the parallel application and groups of processors (or nodes) on
the network.  Nodes of a partition tree represent boxes, where a *box* is an
*n*-D Cartesian space.  Each element in a box is an object that could be a task
or a processor.  New boxes are filled by default with objects numbered by rank
(much like MPI communicators).

Every partition tree starts with a single root *box* representing the full
*n*-D Cartesian space to be partitioned.  We construct a box from a list of its
dimensions, e.g., a 4 x 4 x 4 3D application domain.  From the root, the tree
is subdivided into smaller child boxes representing communication groups (MPI
sub-communicators) in the application.  Child boxes in a partition tree are
disjoint, and the union of any node's child boxes is its own box.  Unlike other
tools, which are are restricted to two or three dimensions, Rubik's syntax
works for any number of dimensions. An arbitrary number of dimensions can be
specified when a box is constructed.

Partitioning Operations
-----------------------

Mapping
-------

Permuting operations
--------------------


Dimensionality-independent operations
-------------------------------------
