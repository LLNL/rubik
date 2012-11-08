#!/usr/bin/env python
#
# This file uses the transparency option on the Rubik visualizer
# to demonstrate the hierarchy of Rubik's partition trees.
# We show a series of progressively deeper div operations and
# how the Partitions are structured after each operation.
#
from rubik import *
import rubik.rubikview as rv

# Basic 4x4x4 box
p1 = box([4,4,4])

# Same box divided in two along z axis.
p2 = box([4,4,4])
p2.div([1,1,2])

# p2, but with each child divided into 3 parts.
p3 = box([4,4,4])
p3.div([1,1,2])
for child in p3:
    child.div([2,1,2])

# Now divide each of THOSE children into 4 parts.
p4 = box([4,4,4])
p4.div([1,1,2])
for child in p4:
    child.div([2,1,2])
    for grandchild in child:
        grandchild.div([1,2,1])

# Color each partitioned box
for p in (p1, p2, p3, p4):
    rv.color(p)

rv.viewbox(p1, p2, p3, p4, transparent=True, rotation=(45, 3, 3, 1))
