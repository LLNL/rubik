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
This is the viewer window for Rubik.  It is in a separate module because it
actually imports Qt, which we try to avoid in the main rubik process.

See the view module for more documentation.
"""
from itertools import ifilter, ifilterfalse
from glutils import *
import numpy as np

from PySide.QtGui import *
from OpenGL.GL import *
from face import *

from PySide.QtCore import *
from GLWidget import GLWidget

def crop(image, crop_criteria):
    """ Crops the transparent background pixels out of a QImage and returns the
    result.
    """
    def min_x(image):
        for x in range(image.width()):
            for y in range(image.height()):
                if not crop_criteria(image.pixel(x,y)): return x

    def max_x(image):
        for x in range(image.width()-1, -1, -1):
            for y in range(image.height()):
                if not crop_criteria(image.pixel(x,y)): return x+1

    def min_y(image):
        for y in range(image.height()):
            for x in range(image.width()):
                if not crop_criteria(image.pixel(x,y)): return y

    def max_y(image):
        for y in range(image.height()-1, -1, -1):
            for x in range(image.width()-1, -1, -1):
                if not crop_criteria(image.pixel(x,y)): return y+1

    mx, Mx = min_x(image), max_x(image)
    my, My = min_y(image), max_y(image)
    return image.copy(mx, my, Mx - mx, My - my)


def translate(index, dim, amount):
    """ Translate an n-dimensional index in the specified dimension by amount.
    """
    l = list(index)
    l[dim] += amount
    return tuple(l)


def pad(index, dims):
    """ Pads an index out to the specified number of dimensions. Index is padded
    using zeros.
    """
    return tuple(list(index) + ([0] * (dims - len(index))))


class RubikView(GLWidget):
    def __init__(self, face_renderer, parent=None, **kwargs):
        """ Creates a view of the specified partition using the supplied face
        renderer. face_renderer should be a cell handler suitable for
        passing to the iterate_cells routine. It is used to create the faces
        this RubikView will render.
        """
        GLWidget.__init__(self, parent)

        self.partition = None
        self.face_renderer = None

        self.solid_faces = []
        self.solid_face_list = DisplayList(self.draw_solid_faces)

        self.show_axis = False

    def set_partition(self, partition):
        self.partition = partition
        if partition.box.ndim != 3:
            raise Exception("Can only view 3-dimensional partitions.")
        self.paths = partition.invert()

        # Compute maxdepth here to save cycles later
        self.maxdepth = max(len(l) for l in self.paths.flat)

        # Initial translation is dependent on the size of the shape we're using
        shape = self.paths.shape

        depth = 3 * max(shape[0:2])
        self.translation = np.array([0.0, 0.0, -depth])
        self.update()

    def set_face_renderer(self, face_renderer):
        self.face_renderer = face_renderer
        self.update()

    def initializeGL(self):
        glClearColor(1.0, 1.0, 1.0, 0.0)

        glClearDepth(1.0)
        glDepthFunc(GL_LESS)

        glLightfv(GL_LIGHT0, GL_AMBIENT,  [0.2, 0.2, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE,  [1.0, 1.0, 1.0, 1.0])
        glLightfv(GL_LIGHT0, GL_POSITION, [-1.0, -1.0, 2.0, 1.0])

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE)
        glShadeModel(GL_SMOOTH)

        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT,         GL_NICEST)
        glHint(GL_LINE_SMOOTH_HINT,            GL_NICEST)
        glHint(GL_POINT_SMOOTH_HINT,           GL_NICEST)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_LINE_SMOOTH)
        glEnable(GL_POLYGON_SMOOTH)


    def iterate_cells(self, renderer, results = None):
	""" Iterates over all cells in the array, and calls the provided
	renderer function for each cell. The function should look
	something like this::

	  def renderer(path, level, connections):
	      pass

        The handler should be a generator function that yields faces to
        be rendered by this RubikView.

        Parameters of the handler:
          path          This is a path to the element being rendered.
                        You can get at the element and its containing
                        partitions using the path.  See
                        Partition.PathElement for more on this.

          level         This is the level the renderer should generate
                        faces for.  If you only want to render leaf
                        partitions, you can just return when level
                        is not equal to len(path)-1.  If you want to
                        handle nesting of some sort, you should handle
                        the other levels as well.

          connections	Connections that this cell has with its
			neighbors at the specified level. This array will
			have 6 boolean-valued elements, one for each face of
			the cell being iterated. You can use the values of
			the global all_faces (i.e. left, right, down, up,
			far, near) to iterate over the connections array.
			This could be used, e.g., to tell you whether you
			need to draw a face between your cell and a
			particular neighbor.
        """
        shape = self.paths.shape
        for index in np.ndindex(shape):
            # Path is our local list in the inverted partition structure.
            path = self.paths[index]

	    # This loop iterates over levels of the partition structure and
	    # computes connections between cells
            for l in range(len(path)):
                # List to hold connections in each direction
                connect = [False] * 6
                for dim in range(self.paths.ndim):
		    # Determine whether we're connected to our neighbor in the
		    # negative direction along dim
                    low = translate(index, dim, -1)
                    if (low[dim] >= 0 and len(self.paths[low]) > l
                        and self.paths[low][l].partition == path[l].partition):
                        connect[2*dim] = True
		    # Determine whether we're connected to our neighbor in the
		    # positive direction along dim
                    high = translate(index, dim, 1)
                    if (high[dim] <= shape[dim]-1 and len(self.paths[high]) > l
                        and self.paths[high][l].partition == path[l].partition):
                        connect[2*dim+1] = True

                # Pass a 3d index to the cell handler and let it do its job
                center = pad(index, 3)

                for face in renderer(path, l, connect):
                    results.append(face)

    def ready(self):
        return (self.partition != None and self.face_renderer != None)

    def update(self):
        if not self.ready():
            return

        self.solid_face_list.update()
        self.faces = []
        self.iterate_cells(self.face_renderer, self.faces)

    def draw_solid_faces(self):
        solid_faces = ifilterfalse(Face.transparent, self.faces)
        with glSection(GL_QUADS):
            for face in solid_faces:
                face.draw()

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        if not self.ready():
            return

        # This does translation and rotation for us; see GLWindow for docs
        self.orient_scene()
        glTranslatef(*((s-1) / -2.0 for s in self.paths.shape))

        # Must sort faces in far-to-near z order for transparency
        mdl = np.array(glGetDoublev(GL_MODELVIEW_MATRIX)).flat
        camera = np.array([-(mdl[0] * mdl[12] + mdl[1] * mdl[13] + mdl[2] * mdl[14]),
                            -(mdl[4] * mdl[12] + mdl[5] * mdl[13] + mdl[6] * mdl[14]),
                            -(mdl[8] * mdl[12] + mdl[9] * mdl[13] + mdl[10] * mdl[14])])

        # render solid faces first
        glDepthMask(GL_TRUE)
        self.solid_face_list()

        # render transparent faces afterwards, without depth writing.
        if any(f.transparent for f in self.faces):
            def depth(face):
                return np.linalg.norm(camera - face.center)
            transparent_faces = list(ifilter(Face.transparent, self.faces))
            transparent_faces.sort(key=depth)

            glEnable(GL_BLEND)
            glDepthMask(GL_FALSE)
            glDisable(GL_CULL_FACE)
            with glSection(GL_QUADS):
                for face in transparent_faces:
                    face.draw()
            glEnable(GL_CULL_FACE)
            glDepthMask(GL_TRUE)
            glDisable(GL_BLEND)

        if self.show_axis:
            self.draw_axis()

    def saveImage(self, transparent):
        name, selectedFilter = QFileDialog.getSaveFileName(
            self, "Save Image", "rubik-image.png", filter="*.png")
        if name:
            image = self.grabFrameBuffer(withAlpha=transparent)
            if transparent:
                image = crop(image, lambda p: not qAlpha(p))
            else:
                image = crop(image, lambda p: qGray(p) == 255)
            image.save(name)

    def keyReleaseEvent(self, event):
        super(RubikView, self).keyReleaseEvent(event)
	# This adds the ability to save an image file if you hit 'p' while the
	# viewer is running.
        if event.key() == Qt.Key_P:
	    self.saveImage(False)
        elif event.key() == Qt.Key_T:
            self.saveImage(True)
        elif event.key() == Qt.Key_R:
            print self.rotation
        elif event.key() == Qt.Key_A:
            self.show_axis = not self.show_axis
            self.updateGL()
