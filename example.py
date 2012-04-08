#!/usr/bin/env python2.6

from rubik import *

if __name__ == "__main__":
    # application topology
    app = box([8,8])
    # app.div([2,2])
    app.tile([4,4])
    print app.box
    print ""

    # processor topology
    torus = box([4,4,4])
    # torus.div([4,1,1])
    torus.tile([1,4,4])

    torus.map(app)
    print torus.box
    print ""

    f = open('mapfile', 'w')
    torus.write_map_file(f)
    f.close()

    sys.exit(0)


    p = box([32,16,16,6,2])
    print p.box
    p.zorder()
    p.tilt(1, 0, 2)

    print p.box

    p.transpose([1,0])
    p.div([2,2])

    print p.box

    q = box([4,4])
    q.mod([2,2])
    q.div([4,1])
    q.map(p)

    print q.box

    q.write_map_file()
