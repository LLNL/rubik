#!/usr/bin/env python

import sys, math, itertools
from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *
from OpenGL.GL import *

from blocker import *
import numpy as np

# True for perspective projection, false for ortho
perspective = True
solid_interior = True

# If the interior blocks are solid, use more opacity to reveal them
if solid_interior: alpha = 0.25
else:              alpha = 0.4

black, white = ((0.0, 0.0, 0.0, 1.0), (1.0, 1.0, 1.0, 1.0))
clear_color = white

colors = [(1.0, 0.3, 0.3, alpha),
          (0.3, 1.0, 0.3, alpha),
          (0.3, 0.3, 1.0, alpha),
          (0.3, 1.0, 1.0, alpha),
          (1.0, 0.3, 1.0, alpha),
          (1.0, 1.0, 0.3, alpha)]

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


class BlockerView(QGLWidget):
    def __init__(self, partition, parent=None):
        QGLWidget.__init__(self, parent)
        if partition.box.ndim > 3:
            raise Exception("Can only view up to 3-dimensional partitions.")
        self.paths = partition.invert()
        self.faces = None

        self.last_pos = [0,0,0]
        self.dragging = False

        # Initial tranlation is dependent on the size of the shape we're using
        shape = self.paths.shape
        depth = 3 * max(shape)
        self.trans = [-0.5 * max(shape), -0.5 * max(shape), -depth]

        # Initial rotation is just the identity matrix.
        self.total_rotate = np.identity(4)

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


    def map_to_sphere(self, x, y):
        width, height = self.width(), self.height()
        v = [0,0,0]

        v[0] = (2.0 * x - width) / width
        v[1] = (height - 2.0 * y) / height

        d = math.sqrt(v[0]*v[0] + v[1]*v[1])
        if d >= 1.0: d = 1.0

        v[2] = math.cos((math.pi/2.0) * d)

        a = v[0]*v[0]
        a += v[1]*v[1]
        a += v[2]*v[2]
        a = 1 / math.sqrt(a)

        v[0] *= a
        v[1] *= a
        v[2] *= a

        return v

    def mousePressEvent(self, event):
        x, y = event.x(), event.y()
        self.last_pos = self.map_to_sphere(x, y)
        self.dragging = True

    def mouseReleaseEvent(self, event):
        self.dragging = False

    def mouseMoveEvent(self, event):
        if not self.dragging: return

        x, y = event.x(), event.y()
        cur_pos = self.map_to_sphere(x, y)

        dx = cur_pos[0] - self.last_pos[0]
        dy = cur_pos[1] - self.last_pos[1]
        dz = cur_pos[2] - self.last_pos[2]

        tb_angle = 0
        tb_axis = [0,0,0]

        if dx != 0 or dy != 0 or dz != 0 :
            # compute theta and cross product
            tb_angle = 90.0 * math.sqrt(dx*dx + dy*dy + dz*dz)
            tb_axis[0] = self.last_pos[1]*cur_pos[2] - self.last_pos[2]*cur_pos[1]
            tb_axis[1] = self.last_pos[2]*cur_pos[0] - self.last_pos[0]*cur_pos[2]
            tb_axis[2] = self.last_pos[0]*cur_pos[1] - self.last_pos[1]*cur_pos[0]

            # update position
            self.last_pos = cur_pos

        # Once rotation has been computed, use OpenGL to add our rotation to the
        # current modelview matrix.  Then fetch the result and keep it around.
        glLoadIdentity()
        glRotatef(0.5*tb_angle, tb_axis[0] , tb_axis[1], tb_axis[2])
        glMultMatrixd(self.total_rotate)
        self.total_rotate = glGetDouble(GL_MODELVIEW_MATRIX)

        self.updateGL()


    def get_color(self, level):
        color = colors[level]
        maxdepth = max(len(l) for l in self.paths.flat)
        if solid_interior and level == maxdepth-1:
            color=(color[0], color[1], color[2], 1.0)
        return color


    def make_faces(self):
        faces = []
        shape = self.paths.shape
        for index in np.ndindex(shape):
            path = self.paths[index]

            for l in range(len(path)):
                connect = [False] * 6
                for dim in range(self.paths.ndim):
                    low = translate(index, dim, -1)
                    if low[dim] >= 0 and len(self.paths[low]) > l and self.paths[low][l] == path[l]:
                        connect[2*dim] = True
                    high = translate(index, dim, 1)
                    if high[dim] <= shape[dim]-1 and len(self.paths[high]) > l and self.paths[high][l] == path[l]:
                        connect[2*dim+1] = True

                center = pad(index, 3)
                for face in all_faces:
                    if not connect[face]:
                        faces.append(Face(face, center, 1, 0.1*l, connect, self.get_color(l)))
        return faces

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glLoadIdentity()
        shape = self.paths.shape
        depth = 3 * max(shape)

        glTranslatef(*self.trans)
        glMultMatrixd(self.total_rotate)

        if not self.faces:
            self.faces = self.make_faces()

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


def main():
    app = QApplication(sys.argv)    # Create a Qt application

    p = Partition.create([4,4,8])
    p.tile([4,4,1])

    p = Partition.create([4, 4, 4])
    p.div([2, 1, 1])
    for child in p:
        child.div([2,2,2])
        for c in child:
            c.div([2,1,1])

    mainwindow = QMainWindow()
    glview = BlockerView(p, mainwindow)
    mainwindow.setCentralWidget(glview)

    mainwindow.resize(800, 600)
    mainwindow.move(30, 30)

    mainwindow.show()
    app.exec_()    # Enter Qt application main loop


if __name__ == "__main__":
    main()
    sys.exit()
