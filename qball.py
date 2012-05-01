#!/usr/bin/env python2.6

import sys
from rubik import *

if __name__ == "__main__":
    #   npcol  nrowmax 
    blm = [16, 128] # Si512.i
    tblm = []
    tblm.append(int(sys.argv[1]))
    tblm.append(int(sys.argv[2]))

    #      X  Y   Z  T
    bgp = [8, 8, 8, 4] # 1024 node BG/P torus
    tbgp = []
    tbgp.append(int(sys.argv[3]))
    tbgp.append(int(sys.argv[4]))
    tbgp.append(int(sys.argv[5]))
    tbgp.append(int(sys.argv[6]))

    # application topology
    app = box(blm)
    app.tile(tblm)
    # print app.box
    # print ""

    # processor topology
    torus = box(bgp)
    torus.tile(tbgp)

    torus.map(app)
    # print torus.box
    # print ""

    numpes = blm[0] * blm[1]

    f = open('mapfile-%s-%s-%s-%s-%s-%s-%s' % (numpes,tblm[0],tblm[1],tbgp[0],tbgp[1],tbgp[2],tbgp[3]), 'w')
    torus.write_map_file(f)
    f.close()
    print "output written to mapfile"

    sys.exit(0)

