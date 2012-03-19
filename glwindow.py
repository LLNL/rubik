"""\
Basic support for interactive OpenGL widgets in Qt.  See GLWindow class docs for details.

Authors:
    Todd Gamblin, tgamblin@llnl.gov

Original rotation code was borrowed liberally from boxfish by Kate Isaacs and Josh Levine.
"""

from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *
from OpenGL.GL import *

import math
import numpy as np

class GLWindow(QGLWidget):
    """ This class implements basic support for interactive OpenGL application.  This includes support
        for rotation and translation using the mouse.  Other than handling mouse events for basic interactive
        features, this is just a regular QGLWidget, so the user still needs to implement resizeGL(), initializeGL(),
        and paintGL() to get their scene drawn.
    """

    def __init__(self, parent=None):
        """Sets up initial values for dragging variables, translation, and rotation matrices."""
        QGLWidget.__init__(self, parent)

        # Initialize last position and dragging flag to support mouse interaction
        self.last_pos = [0,0,0]
        self.dragging = False

        # Initialize key tracking
        self.pressed_keys = set()
        self.setFocusPolicy(Qt.StrongFocus)

        self.translation = np.zeros([3])      # Initial translation is zero in all directions.
        self.rotation = np.identity(4)        # Initial rotation is just the identity matrix.


    def map_to_sphere(self, x, y):
        """This takes local x and y window coordinates and maps them to an arcball sphere
           based on the width and height of the window.  This is used for quaternion rotation
           later in mouseMoveEvent().
        """
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
        """Maps the click location to the sphere and records this in lastPos.  Also records that
           dragging has begun.
        """
        x, y = event.x(), event.y()
        self.last_pos = self.map_to_sphere(x, y)
        self.dragging = True

    def mouseReleaseEvent(self, event):
        """Ends dragging so that mouseMoveEvent() will know not to adjust things. """
        self.dragging = False

    def mouseMoveEvent(self, event):
        """This method rotates the scene around as the mouse moves, and it calls updateGL()
           to notify the UI that the system needs updating.  Rotation is quaternion (axis/angle)
           based.
        """
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
        glMultMatrixd(self.rotation)
        self.rotation = glGetDouble(GL_MODELVIEW_MATRIX)

        self.updateGL()


    def wheelEvent(self, event):
        """Does translation in response to wheel events.  Within paintGL(), you will
           need to either call self.orient_scene() or do your own glTranslate() and
           glRotate() based on self.translation and self.rotation.
           """
        if event.orientation() == Qt.Orientation.Vertical:
            if int(Qt.Key_Shift) in self.pressed_keys:
                self.translation[1] += .01 * event.delta()
            else:
                self.translation[2] += .01 * event.delta()

        elif event.orientation() == Qt.Orientation.Horizontal:
            self.translation[0] -= .01 * event.delta()


        self.updateGL()

    def keyPressEvent(self, event):
        self.pressed_keys.add(event.key())

    def keyReleaseEvent(self, event):
        if event.key() in self.pressed_keys:
            self.pressed_keys.remove(event.key())


    def orient_scene(self):
        """You should call this from paintGL() to orient the scene before rendering.
           This will do translation and rotation so that rendering happens at the right
           location and orientation.
        """
        glLoadIdentity()
        glTranslatef(*self.translation)
        glMultMatrixd(self.rotation)
