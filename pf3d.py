#!/usr/bin/env python2.6

import sys
from blocker import *

if __name__ == "__main__":
    #  mp_r mp_q mp_p
    blm = [16,8,16] # bench_ltr_mem
    tblm = []
    tblm.append(int(sys.argv[1]))
    tblm.append(int(sys.argv[2]))
    tblm.append(int(sys.argv[3]))

    #      Z Y X T
    bgp = [8,8,8,4] # 512 node BG/P torus
    tbgp = []
    tbgp.append(int(sys.argv[4]))
    tbgp.append(int(sys.argv[5]))
    tbgp.append(int(sys.argv[6]))
    tbgp.append(int(sys.argv[7]))

    
    # application topology
    app = Partition.create(blm)
    app.tile(tblm)
    # print app.box
    # print ""

    # processor topology
    torus = Partition.create(bgp)
    torus.tile(tbgp)

    torus.map(app)
    torus.tilt(0, 1, 1)
    torus.tilt(0, 2, 1)
    # print torus.box
    # print ""

    f = open('mapfile-%s-%s-%s-%s-%s-%s-%s' % (tblm[0],tblm[1],tblm[2],tbgp[0],tbgp[1],tbgp[2],tbgp[3]), 'w')
    torus.write_map_file(f)
    f.close()
    print "output written to mapfile"

    sys.exit(0)

