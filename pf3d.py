#!/usr/bin/env python2.6

from blocker import *

if __name__ == "__main__":
    #  mp_r mp_q mp_p
    blm = [8,32,8] # bench_ltr_mem
    tblm = [1,32,8]
    #      Z Y X T
    bgp = [8,8,8,4] # 512 node BG/P torus
    tbgp = [4,8,8,1]
    
    # application topology
    app = Partition.create(blm)
    app.tile(tblm)
    # print app.box
    print ""

    # processor topology
    torus = Partition.create(bgp)
    torus.tile(tbgp)

    torus.map(app)
    # print torus.box
    print ""

    f = open('mapfile-%s-%s-%s-%s-%s-%s-%s' % (tblm[0],tblm[1],tblm[2],tbgp[0],tbgp[1],tbgp[2],tbgp[3]), 'w')
    torus.write_map_file(f)
    f.close()

    sys.exit(0)

