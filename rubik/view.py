################################################################################
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Todd Gamblin et al. <tgamblin@llnl.gov>
# LLNL-CODE-599252
# All rights reserved.
#
# This file is part of Rubik. For details, see http://scalability.llnl.gov.
# Please read the LICENSE file for further information.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the disclaimer below.
#
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the disclaimer (as noted below) in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the LLNS/LLNL nor the names of its contributors may be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE
# U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################
"""
This is an interface to RubikView.  RubikView will launch viewers for
Rubik partitions in a separate process.

Note that this module takes care NOT to import Qt, and only runs code
that would import Qt on the remote viewer processes.  Qt won't work
properly if you import it on the main process before spawninig viewers.
"""
import sys, math
from color import *
import numpy as np
import multiprocessing as mp
from face import *

rubik_colors = [
    Color(0.000, 0.000, 1.000), # blueberry
    Color(1.000, 0.000, 0.000), # maraschino
    Color(0.250, 0.500, 0.000), # fern
    Color(1.000, 1.000, 0.000), # lemon
    Color(0.000, 0.500, 1.000), # agua
    Color(1.000, 0.000, 1.000), # magenta
    Color(0.200, 0.200, 0.200), # tungsten
    Color(1.000, 0.500, 0.000), # tangerine
    Color(0.700, 0.700, 0.700), # magnesium
    Color(0.000, 1.000, 0.500), #"sea foam
    Color(0.500, 0.250, 0.000), # mocha
    Color(0.500, 0.000, 1.000), # grape
    Color(0.000, 1.000, 1.000), # turquoise
    Color(0.000, 1.000, 0.000), # spring
    Color(1.000, 0.400, 0.400), # salmon
    Color(0.500, 0.500, 0.000)] # asparagus


def colored_face_renderer(**kwargs):
    """This will render faces with the color assigned to each element
    in the partition.  If an element has no color attribute, it will
    be rendered gray.

    Options:
      transparent      Default False.  If true renders transparent hierarchy
                       around leaf partitions.
      alpha            Transparency of non-leaf levels.
      default_color    Color to use if there is no color attribute.  Default
                       is gray.
      margin           Margin of empty space around separate partitions.
    """
    transparent = kwargs.get("transparent", False)
    alpha = kwargs.get("alpha", 0.25)
    default_color = Color(*kwargs.get("default_color", (0.2, 0.2, 0.2)))

    default_margin = 0.1
    if transparent: default_margin = 0.15
    margin = kwargs.get("margin", default_margin)

    def render(path, level, connections):
        leaf = (level == len(path) - 1)

        for face in all_faces:
            # don't draw internal faces.
            if connections[face]: continue

            color = path[level].meta.color
            if not color:
                color = shade_by_index(path, level, default_color)

            my_margin = margin
            if transparent:
                my_margin *= level
                if not leaf:
                    color = default_color.with_alpha(alpha)
            elif not leaf:
                return

            index = path[0].index
            yield Face(face, index, 1, my_margin, connections, color)

    return render

def shade_by_index(path, level, color):
    """Shades the color depending on its index in the partition at <level>
    in <path>."""
    partition  = path[level].partition
    shade = 1 - (path[level].flat_index / float(partition.size))
    return color.mix(Color(shade, shade, shade), .66)

def level_gradient_colorer(level=-1, **kwargs):
    """This returns a colorer that assigns unique colors to each partition
    at <level> in the encountered path.  The default is to assign colors by
    leaf partitions, which is equivalent to supplying -1. If you only want
    to color by the root partition, supply level 0.

    The level refers to the position in the path. Within colored partitions,
    elements are colored from light to dark by their flat index within the
    partition."""
    part_colors = ColorMapper(rubik_colors, **kwargs)

    def getcolor(path):
        partition  = path[level].partition
        return shade_by_index(path, level, part_colors[partition])

    return getcolor


def color(partition, **kwargs):
    """This function traverses a partition with the default coloring
    function.  This is intended to make Rubik easier to script by
    allowing people not to have to worry about how things are colored.

    Optional parameters:
      colorer  Optionally pass a colorer to this routine.  The colorer
               should be some function that takes a path and returns
               a color.

    By default, this uses a colorer built with
    :meth:`level_gradient_colorer`. If you do not supply a custom
    colorer, you can pass level_gradient_colorer's keyword args
    directly to the color function, e.g.::

      color(app, level=-2)
    """
    colorer = kwargs.get("colorer", level_gradient_colorer(**kwargs))

    def assign_color(path):
        path[0].meta.color = colorer(path)
    partition.traverse_cells(assign_color)


def _view_server(kwargs, queue):
    from PySide.QtGui import QApplication, QMainWindow, QFrame, QGridLayout
    from RubikView import RubikView

    # Grab the partition to view from the queue.
    app = QApplication(sys.argv)

    frame = QFrame()
    layout = QGridLayout()
    frame.setLayout(layout)

    mainwindow = QMainWindow()
    mainwindow.setCentralWidget(frame)
    mainwindow.resize(1024, 768)
    mainwindow.move(30, 30)

    # Find next highest square number and fill within that shape
    partitions = queue.get()
    side = math.ceil(math.sqrt(len(partitions)))
    aspect = (side, side)

    views = []
    for i, partition in enumerate(partitions):
        rview = RubikView(mainwindow)
        if "renderer" in kwargs:
            rview.set_face_renderer(kwargs["renderer"])
        else:
            rview.set_face_renderer(colored_face_renderer(**kwargs))

        rview.set_partition(partition)

        r, c = np.unravel_index(i, aspect)
        layout.addWidget(rview, r, c)
        views.append(rview)

    mainwindow.show()
    mainwindow.raise_()

    rotation = kwargs.get("rotation", (45, 3, 3, 1))
    for rview in views:
        rview.set_rotation_quaternion(*rotation)

    app.exec_()


def viewbox(*partitions, **kwargs):
    queue = mp.Queue()
    queue.put(partitions)
    p = mp.Process(target=_view_server, args=(kwargs, queue))
    p.start()

