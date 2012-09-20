Quick Tutorial
==============

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

    torus.map(pf3d)

Specify operations (such as tilt, zigzag etc) to change the ordering of MPI ranks to processors::

    torus.tile(2, 0, 1)
    
