#!/usr/bin/env python

from rubik import *
import rubik.rubikview as rv

app = box([32, 8, 16])
app.tile([1])
app.traverse_cells(rv.assign_flat_index_gradient_color)

app.tilt(0, 2, 1)	# tilt XY planes in X
app.tilt(0, 1, 1)	# tilt XY planes in Y

rv.view_in_app(app, rv.ColoredFaceRenderer(0))
