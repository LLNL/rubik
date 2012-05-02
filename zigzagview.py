#!/usr/bin/env python

from rubik import *
import rubik.rubikview as rv

def main():
    p = box([8, 8, 8])
    p.div([2,2,2])
    p.traverse_cells(rv.assign_flat_index_gradient_color)
#    p.tilt(1,0,1)
    p.zigzag(1,0,1,1)
    p.zigzag(2,1,1,1)
    p.zigzag(0,2,1,1)

#    q = Partition.create([8,8,8])
#    q.zigzag(1,0,2,2)
#    q.tilt(1,0,1)
#    q.tile([8,2,8])
#    q.map(p)

#    for child in q:
#        child.tilt(0,1,1)
#        child.tilt(0,2,1)


    rv.view_in_app(p, rv.ColoredFaceRenderer())

if __name__ == "__main__":
    main()
    sys.exit()
