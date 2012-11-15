Basic Tutorial
==============

This tutorial outlines the basic steps in generating a mapping file using
Rubik. For more detailed documentation, refer to the :doc:`userguide`.

The Basics
----------
Rubik is based on the idea of creating hierarchical groups within
recursively partitioned *n*-dimensional cartesian spaces.

Creating a Mapfile
------------------

Create a python file::

    #!/usr/bin/env python2.6

    from rubik import *

Specify the application topology::

    app = box([16, 8, 16])
    app.tile([1, 8, 16])

Specify the processor topology::

    torus = box([8, 8, 32])
    torus.tile([8, 8, 2])

Call map to embed the application in the processor torus::

    torus.map(app)

Specify operations (such as tilt, zigzag etc) to change the ordering of MPI ranks to processors::

    torus.tile(2, 0, 1)

Write out a mapfile::

    f = open('mapfile', 'w')
    torus.write_map_file(f)
    f.close()

Visualizing 3D Mappings
-----------------------
Rubik has visualization support for three-dimensional Cartesian spaces. A
particular partition (app or torus) can be visualized using::

    import rubik.view as rv

    # Create an application
    app = box([16, 8, 16])
    app.tile([1, 8, 16])

    # This will color the application by its partitions.
    rv.color(app)

    # If you want to *see* the effect of a mapping you need to map after
    # you color the source partition.
    torus.map(app)

    # Now view the app and the torus mappings.  You can see the colors
    # from the application mapped into thte torus.
    rv.viewbox(app)
    rv.viewbox(torus)

See the documentation for :mod:`rubik.view.color` for more information.

Creating Hierarchical Mappings
------------------------------
Instead of apply operations such as tile to the entire torus partition, they
can be applied to sub-domains as follows::

    for child in torus:
	child.tile(2, 0, 1)
