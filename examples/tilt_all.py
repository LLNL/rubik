#!/usr/bin/env python

from rubik import *
Z, Y, X = 0, 1, 2
# application topology
app = box([2, 8, 4])
app.tile([1, 4, 4])
print app
print ""

# manually using tilt
app1 = app.copy()
app1.tilt(Z, Y, 1)
app2 = app1.copy()
app2.tilt(Z, X ,1)
app3 = app2.copy()
app3.tilt(Y, X, 1)
print app3
print ""

# processor topology
torus = box([4, 4, 4])
torus.tile([1, 4, 4])
torus1 = torus.copy()

torus.map(app3)
print torus
print ""

f = open('mapfile', 'w')
torus.write_map_file(f)
f.close()

# using tilt_combi
app4 = app.copy()
app4.tilt_combi(1)

torus1.map(app4)
print torus1
print ""

f1 = open('mapfile1', 'w')
torus1.write_map_file(f1)
f1.close()
