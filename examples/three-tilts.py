#!/usr/bin/env python

from rubik import *
import rubik.view as rv

Z, Y, X = 0, 1, 2  # Assign names to dimensions
t0 = box([4,4,4]) # Create a box
rv.color(t0)

t1 = t0.copy()
t1.tilt(Z, X, 1)  # Tilt Z (XY) planes along X

t2 = t1.copy()
t2.tilt(X, Y, 1)  # Tilt X (YZ) planes along Y

rv.viewbox(t0, t1, t2)
