#!/usr/bin/env python

import rubik

print "Automatically obtaining geometry:"
box = rubik.autobox()
print box.shape
print "#Dimensions: %d #Tasks: %d" % (box.ndim, box.size)

print
print "Automatically obtaining geometry, binding tasks per node to 4:"
box = rubik.autobox(tasks_per_node=4)
print box.shape
print "#Dimensions: %d #Tasks: %d" % (box.ndim, box.size)

print
print "Automatically obtaining geometry, binding tasks per node to 4 and total number of tasks to 256:"
box = rubik.autobox(tasks_per_node=4, num_tasks=256)
print box.shape
print "#Dimensions: %d #Tasks: %d" % (box.ndim, box.size)
