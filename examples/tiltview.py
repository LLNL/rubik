#!/usr/bin/env python

import sys
import rubik.rubikview as rv
from rubik import *

if __name__ == "__main__":
    p = box([8, 8, 8])
    p.tile([1])
    p.traverse_cells(rv.assign_flat_index_gradient_color)

    q = box([8, 8, 8])
    q.tile([4, 4, 4])
    q.map(p)

    for child in q:
        child.tilt(0, 1, 1)
        child.tilt(0, 2, 1)

    rv.view_in_app(q, rv.ColoredFaceRenderer())
