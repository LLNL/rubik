#!/usr/bin/env python

from rubik import *
import rubik.view as rv

Z, Y, X = 0, 1, 2  # Assign names to dimensions
t0 = box([4, 4, 4]) # Create a box
t0.tile([4, 4, 1])
for child in t0:
    child.tile([4, 1, 1])
rv.color(t0)

t1 = t0.copy()
t1.tilt(X, Z, 1)  # Tilt X (YZ) planes along Z

t2 = t1.copy()
t2.tilt(Y, Z, 1)  # Tilt Y (XZ) planes along Z

rv.viewbox(t0, t1, t2)
