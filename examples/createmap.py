#!/usr/bin/env python

from rubik import *

# application topology
app = box([8, 8])
app.tile([4, 4])
print app
print ""

# processor topology
torus = box([4, 4, 4])
torus.tile([1, 4, 4])

torus.map(app)
print torus
print ""

f = open('mapfile', 'w')
torus.write_map_file(f)
f.close()
