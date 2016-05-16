#!/usr/bin/env python

#import rubik.view as rv
from rubik import *

# application topology
app = box_cray([8,8])
app.tile([1,8])
#rv.color(app)
#rv.viewbox(app)
big_torus, big_box, dimVector = autobox_cray(numpes="64")

torus = box_cray([4, 4, 4])
type1 = 'rcb_order'
torus.tile([2,2,2])
torus.map(app)
#torus.tilt(0,1,2)
f = open('mapfile_new', 'w')
#gridVector = (2,2) #
torus.write_map_cray(big_box, big_torus, type1, f, dimVector)# the last parameter is for grid mapping but it is not fully implemented so ignore it. 
#torus.write_map
#rv.viewbox(torus)

f.close()
