#!/usr/bin/env python

from rubik import *

# application topology
app = box_cray([4, 2])
app.tilt(0, 1, 2)

big_torus, big_box = autobox_cray(numpes="8")

torus = box_cray([2, 2, 2])
type1 = 'zorder'

torus.map(app)

f = open('mapfile_new', 'w')
torus.write_map_cray(big_box, big_torus, type1, f)
f.close()
