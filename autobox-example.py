#!/usr/bin/env python

import rubik

print "Autmatically obtaining Geometry and tasks per node:"
box = rubik.autobox()
print box.shape
print box.size
print box.ndim

print
print "Autmatically obtaining Geometry, binding tasks per node to 64:"
box = rubik.autobox(64)
print box.shape
print box.size
print box.ndim
