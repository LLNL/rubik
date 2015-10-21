#!/usr/bin/env python

#import rubik.view as rv
from rubik import *

# application topology
app = box_cray([8,8])

app.tilt(0, 1, 2)
#rv.color(app)
#rv.viewbox(app)
big_torus, big_box = autobox_cray(numpes="64")

torus = box_cray([4, 4, 4])
type1 = 'zorder'

torus.map(app)

f = open('mapfile_new', 'w')
torus.write_map_cray(big_box, big_torus, type1, f)
#rv.viewbox(torus)

f.close()
