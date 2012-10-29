#!/usr/bin/env python

from rubik import *
import rubik.rubikview as rv

if __name__ == "__main__":
    p = box([8, 8, 8])
    p.div([2, 2, 2])
    p.traverse_cells(rv.assign_flat_index_gradient_color)

    p.zigzag(1, 0, 1, 1)
    p.zigzag(2, 1, 1, 1)
    p.zigzag(0, 2, 1, 1)

    rv.viewbox(p)
