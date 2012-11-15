"""
This class represents a face in a RubikView window.
"""

from OpenGL.GL import *
import numpy as np

# Indices for stencil arrays, given names for convenience.
all_faces = range(6)
left, right, down, up, far, near = all_faces

class Face(object):
    def __init__(self, face, cell_center, width, margin, connect, color=None):
        self.color = color
        self.solid = color and not (color[3] < 1.0)

        # compute margins for connections by trimming ones where a connection
        # doesn't exist.
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

    def transparent(self):
        return self.color[3] < 1.0
