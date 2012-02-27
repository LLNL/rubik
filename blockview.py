#!/usr/bin/env python

import sys, math, itertools
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *
from OpenGL.GL import *

from blocker import *
import glwindow
import numpy as np

# True for perspective projection, false for ortho
perspective = True

black, white = ((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0))
clear_color = white

# Really basic color list.  Smart coloring could use some work.
# Note that this color list has no alpha values.  Use add_alpha to add this.
class color:
    def __init__(self, name, value):
        self.name = name
        self.value = value

colors = [
    color("red",           (1.000, 0.000, 0.000)),
    color("green",         (0.000, 1.000, 0.000)),
    color("blue",          (0.000, 0.000, 1.000)),
    color("yellow",        (1.000, 1.000, 0.000)),
    color("cyan",          (0.000, 1.000, 1.000)),
    color("magenta",       (1.000, 0.000, 1.000)),
    color("white",         (1.000, 1.000, 1.000)),
    color("black",         (0.000, 0.000, 0.000)),
    color("grey",          (0.745, 0.745, 0.745)),
    color("light grey",    (0.827, 0.827, 0.827)),
    color("orange red",    (1.000, 0.271, 0.000)),
    color("dark orange",   (1.000, 0.549, 0.000)),
    color("orange",        (1.000, 0.647, 0.000)),
    color("gold",          (1.000, 0.843, 0.000)),
    color("light yellow",  (1.000, 1.000, 0.878)),
    color("dark green",    (0.000, 0.392, 0.000)),
    color("pale green",    (0.596, 0.984, 0.596)),
    color("navy",          (0.000, 0.000, 0.502)),
    color("royal blue",    (0.255, 0.412, 0.882)),
    color("sky blue",      (0.529, 0.808, 0.922)),
    color("light blue",    (0.678, 0.847, 0.902)),
    color("turquoise",     (0.251, 0.878, 0.816)),
    color("aquamarine",    (0.498, 1.000, 0.831)),
    color("dark violet",   (0.580, 0.000, 0.827)),
    color("purple",        (0.627, 0.125, 0.941)),
    color("violet",        (0.933, 0.510, 0.933)),
    color("deep pink",     (1.000, 0.078, 0.576)),
    color("hot pink",      (1.000, 0.412, 0.706)),
    color("pink",          (1.000, 0.753, 0.796)),
    color("brown",         (0.647, 0.165, 0.165)),
    color("coral",         (1.000, 0.498, 0.314)),
    color("khaki",         (0.941, 0.902, 0.549)),
    color("beige",         (0.961, 0.961, 0.863)),
    color("light coral",   (0.941, 0.502, 0.502)),
    color("ivory",         (1.000, 1.000, 0.941))]

def add_alpha(color, alpha):
    with_alpha = list(color)
    with_alpha.append(alpha)
    return tuple(with_alpha)

# Indices for stencil arrays.
all_faces = range(6)
left, right, down, up, far, near = all_faces

def set_perspective(fovY, aspect, zNear, zFar):
    """NeHe replacement for gluPerspective"""
    fH = math.tan(fovY / 360.0 * math.pi) * zNear
    fW = fH * aspect
    glFrustum(-fW, fW, -fH, fH, zNear, zFar)

def set_ortho(maxdim, aspect):
    halfheight = maxdim
    halfwidth = aspect * halfheight
    glOrtho(-halfwidth, halfwidth, -halfheight, halfheight, -10, 100)


def translate(index, dim, amount):
    """Translate an n-dimensional index in the specified dimension by amount."""
    l = list(index)
    l[dim] += amount
    return tuple(l)

def pad(index, dims):
    """Pads an index out to the specified number of dimensions.
       Index is padded using zeros.
    """
    return tuple(list(index) + ([0] * (dims - len(index))))


class Face(object):
    def __init__(self, face, cell_center, width, margin, connect, color=None):
        self.color = color
        self.solid = not (color[3] < 1.0)

        # compute margins for connections by trimming ones where a
        # connection doesn't exist.
        margins = [width / 2.0] * 6
        for f in all_faces:
            if not connect[f]: margins[f] -= margin
        lm, rm, dm, um, fm, nm = margins

        c = cell_center
        self.corners = np.empty([4], dtype=object)
        if face == left:
            self.normal     = (-1.0, 0.0, 0.0)
            self.center     = (c[0] - lm, c[1],      c[2])
            self.corners[0] = (c[0] - lm, c[1] + um, c[2] + nm)
            self.corners[1] = (c[0] - lm, c[1] + um, c[2] - fm)
            self.corners[2] = (c[0] - lm, c[1] - dm, c[2] - fm)
            self.corners[3] = (c[0] - lm, c[1] - dm, c[2] + nm)

        elif face == right:
            self.normal     = (1.0, 0.0, 0.0)
            self.center     = (c[0] + rm, c[1],      c[2])
            self.corners[0] = (c[0] + rm, c[1] + um, c[2] - fm)
            self.corners[1] = (c[0] + rm, c[1] + um, c[2] + nm)
            self.corners[2] = (c[0] + rm, c[1] - dm, c[2] + nm)
            self.corners[3] = (c[0] + rm, c[1] - dm, c[2] - fm)

        elif face == down:
            self.normal     = (0.0, -1.0, 0.0)
            self.center     = (c[0],      c[1] - dm, c[2])
            self.corners[0] = (c[0] - lm, c[1] - dm, c[2] - fm)
            self.corners[1] = (c[0] + rm, c[1] - dm, c[2] - fm)
            self.corners[2] = (c[0] + rm, c[1] - dm, c[2] + nm)
            self.corners[3] = (c[0] - lm, c[1] - dm, c[2] + nm)

        elif face == up:
            self.normal     = (0.0, 1.0, 0.0)
            self.center     = (c[0],      c[1] + um, c[2])
            self.corners[0] = (c[0] + rm, c[1] + um, c[2] - fm)
            self.corners[1] = (c[0] - lm, c[1] + um, c[2] - fm)
            self.corners[2] = (c[0] - lm, c[1] + um, c[2] + nm)
            self.corners[3] = (c[0] + rm, c[1] + um, c[2] + nm)

        elif face == far:
            self.normal     = (0.0, 0.0, -1.0)
            self.center     = (c[0],      c[1],      c[2] - fm)
            self.corners[0] = (c[0] - lm, c[1] + um, c[2] - fm)
            self.corners[1] = (c[0] + rm, c[1] + um, c[2] - fm)
            self.corners[2] = (c[0] + rm, c[1] - dm, c[2] - fm)
            self.corners[3] = (c[0] - lm, c[1] - dm, c[2] - fm)

        elif face == near:
            self.normal     = (0.0, 0.0, 1.0)
            self.center     = (c[0],      c[1],      c[2] + nm)
            self.corners[0] = (c[0] + rm, c[1] + um, c[2] + nm)
            self.corners[1] = (c[0] - lm, c[1] + um, c[2] + nm)
            self.corners[2] = (c[0] - lm, c[1] - dm, c[2] + nm)
            self.corners[3] = (c[0] + rm, c[1] - dm, c[2] + nm)

    def draw(self):
        if self.color:
            glColor4f(*self.color)

        tl, tr, bl, br = self.corners
        glNormal3f(*self.normal)
        glVertex3f(*self.corners[0])
        glVertex3f(*self.corners[1])
        glVertex3f(*self.corners[2])
        glVertex3f(*self.corners[3])

    def draw_normal(self):
        glColor4f(0,0,0,1.0)
        scale = 0.1
        glVertex3f(*self.center)
        glVertex3f((scale * self.normal[0]) + self.center[0],
                   (scale * self.normal[1]) + self.center[1],
                   (scale * self.normal[2]) + self.center[2])


class BlockerView(glwindow.GLWindow):
    def __init__(self, partition, face_renderer, parent=None):
        """Creates a view of the specified partition using the supplied face renderer.
           face_renderer should be a cell handler suitable for passing to the iterate_cells routine.
           It is used to create the faces this BlockerView will render.
        """
        glwindow.GLWindow.__init__(self, parent)

        if partition.box.ndim > 3:
            raise Exception("Can only view up to 3-dimensional partitions.")
        self.paths = partition.invert()

        self.face_renderer = face_renderer
        self.faces = None

        # Compute maxdepth here to save cycles later
        self.maxdepth = max(len(l) for l in self.paths.flat)

        # Initial tranlation is dependent on the size of the shape we're using
        shape = self.paths.shape
        depth = 3 * shape[2]
        self.translation = np.array(
            [-0.5 * shape[0], -0.5 * shape[1], -depth])

    def initializeGL(self):
        glShadeModel(GL_SMOOTH)
        glClearColor(*clear_color)

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


    def resizeGL(self, width, height):
        if (height == 0):
            height = 1
        glViewport(0, 0, width, height)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()

        aspect = float(width)/height
        if perspective:
            set_perspective(45.0, aspect, 0.1, 100.0)
        else:
            maxdim = max(self.paths.shape)
            set_ortho(maxdim, aspect)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()


    def iterate_cells(self, cell_handler, results = None):
        """Iterates over all cells in the array, and calls the provided cell_handler function for each cell.
           The function should look something like this:

               def cell_handler(index, level, connections, results):
                   pass

           Parameters of the handler:
           index         This is the index within the top-level partition of the cell that's being iterated.
                         Note that this will always be a 3d index, with y and z set to zero if either of those
                         dimensions is not needed.  You don't need to pad this yourself.

           level         The level within the partition hierarchy that we're calling this handler for.  i.e.
                         if a cell is contained within 3 nested partitions, handler will be called 3 times
                         with 0,1, and 2 as values for level.

           connections   Connections that this cell has with its neighbors at the specified level.  This array
                         will have 6 boolean-valued elements, one for each face of the cell being iterated.
                         You can use the values of the global all_faces (i.e. left, right, down, up, far, near)
                         to iterate over the connections array.  This could be used, e.g., to tell you whether
                         you need to draw a face between your cell and a particular neighbor.

           results       This is the results list passed to iterate_cells.  Your handler function can append
                         to this list as it creates Faces (or anything else)
        """
        shape = self.paths.shape
        for index in np.ndindex(shape):
            # Path is our local list in the inverted partition structure.
            path = self.paths[index]

            # This loop iterates over levels of the partition structure and computes connections between cells
            for l in range(len(path)):
                # List to hold connections in each direction
                connect = [False] * 6
                for dim in range(self.paths.ndim):
                    # Determine whether we're connected to our neighbor in the negative direction along dim
                    low = translate(index, dim, -1)
                    if low[dim] >= 0 and len(self.paths[low]) > l and self.paths[low][l] == path[l]:
                        connect[2*dim] = True
                    # Determine whether we're connected to our neighbor in the positive direction along dim
                    high = translate(index, dim, 1)
                    if high[dim] <= shape[dim]-1 and len(self.paths[high]) > l and self.paths[high][l] == path[l]:
                        connect[2*dim+1] = True

                # Pass a 3d index to the cell handler and let it do its job
                center = pad(index, 3)
                cell_handler(self, center, l, connect, results)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # This does translation and rotation for us; see GLWindow for docs
        self.orient_scene()

        if not self.faces:
            self.faces = []
            self.iterate_cells(self.face_renderer, self.faces)

        # Must sort faces in far-to-near z order for transparency
        mdl = np.array(glGetDoublev(GL_MODELVIEW_MATRIX)).flat
        camera = np.array([-(mdl[0] * mdl[12] + mdl[1] * mdl[13] + mdl[2] * mdl[14]),
                            -(mdl[4] * mdl[12] + mdl[5] * mdl[13] + mdl[6] * mdl[14]),
                            -(mdl[8] * mdl[12] + mdl[9] * mdl[13] + mdl[10] * mdl[14])])
        for face in self.faces:
            face.depth = np.linalg.norm(camera - face.center)
        self.faces.sort(lambda f1, f2: cmp(f1.depth, f2.depth))

        # transparency: test for whether alpha is less than 1.0
        def transparent(face):
            return face.color[3] < 1.0
        transparent_faces = itertools.ifilter(transparent, self.faces)
        solid_faces = itertools.ifilterfalse(transparent, self.faces)

        # render solid faces first
        glDepthMask(GL_TRUE)
        glBegin(GL_QUADS)
        for face in solid_faces:
            face.draw()
        glEnd()

        # render transparent faces afterwards, without depth writing.
        glEnable(GL_BLEND)
        glDepthMask(GL_FALSE)
        glDisable(GL_CULL_FACE)
        glBegin(GL_QUADS)
        for face in transparent_faces:
            face.draw()
        glEnd()
        glEnable(GL_CULL_FACE)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)



def make_nested_faces(blockerview, index, level, connections, faces):
    """Hierarchical renderer that shows tree decomposition with transparent boxes.  Deeper partition
       levels are drawn as progressively smaller boxes within their encosing partitions' boxes,
       and outer boxes are made transparent so that inner boxes can be seen.
    """
    def get_color(level):
        color = colors[level].value
        if level < blockerview.maxdepth-1:
            return add_alpha(color, 0.25)
        else:
            return add_alpha(color, 1.0)

    # Create faces only when there is NOT a connection between our cell and the neighbor
    # Cells at deeper levels have increasingly large margins -- 0.1 units per level
    for face in all_faces:
        if not connections[face]:
            faces.append(Face(face, index, 1, 0.1*level, connections, get_color(level)))


def make_leaf_faces(blockerview, index, level, connections, faces):
    """Really basic leaf coloring scheme.  Colors each leaf by its position *within* its parent.
       By default, this leaves no space between the leaves.
    """
    # Get the path to the cell at index
    path = blockerview.paths[index]
    leaf_level = len(path)-1

    # This forces us to only render leaf cells
    if level != leaf_level: return

    partition = path[level]
    for face in all_faces:
        if not connections[face]:
            color_index = partition.flat_index % len(colors)
            color = add_alpha(colors[color_index].value, 1.0)
            faces.append(Face(face, index, 1, 0.1, connections, color))


def main():
    app = QApplication(sys.argv)    # Create a Qt application

    p = Partition.create([4, 4, 4])
    p.div([2, 1, 1])
    for child in p:
        child.div([2,2,2])

    mainwindow = QMainWindow()
    glview = BlockerView(p, make_leaf_faces, mainwindow)
#    glview = BlockerView(p, make_nested_faces, mainwindow)

    mainwindow.setCentralWidget(glview)
    mainwindow.resize(800, 600)
    mainwindow.move(30, 30)

    mainwindow.show()
    app.exec_()    # Enter Qt application main loop


if __name__ == "__main__":
    main()
    sys.exit()
