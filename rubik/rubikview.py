"""
This is a basic viewer for Rubik in the form of a Qt Widget. You can plug
this into a PySide GUI to view Rubik boxes using various types of renderers.
"""

import sys, math
from itertools import ifilter, ifilterfalse

from PySide.QtCore import *
from PySide.QtGui import *
from PySide.QtOpenGL import *
from OpenGL.GL import *

from rubik import *
from rubik.color import *

from GLWidget import GLWidget
from glutils import *
import numpy as np

# True for perspective projection, false for ortho
perspective = True

black, white, transparent = ((0.0, 0.0, 0.0, 1.0),
                             (1.0, 1.0, 1.0, 1.0),
                             (1.0, 1.0, 1.0, 0.0))
clear_color = transparent

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

# Indices for stencil arrays.
all_faces = range(6)
left, right, down, up, far, near = all_faces

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


class Face(object):
    def __init__(self, face, cell_center, width, margin, connect, color=None):
        self.color = color
        self.solid = not (color[3] < 1.0)

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


class RubikView(GLWidget):
    def __init__(self, face_renderer, parent=None):
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

            color = getattr(path[level].element, "color", default_color)
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


def level_gradient_colorer(level=-1):
    """This returns a colorer that assigns unique colors to each partition
    at <level> in the encountered path.  The default is to assign colors by
    leaf partitions, which is equivalent to supplying -1. If you only want
    to color by the root partition, supply level 0.

    The level refers to the position in the path. Within colored partitions,
    elements are colored from light to dark by their flat index within the
    partition."""
    part_colors = ColorMapper(rubik_colors)

    def getcolor(path):
        partition  = path[level].partition
        shade = 1 - (path[level].flat_index / float(partition.size))
        return part_colors[partition].mix(Color(shade, shade, shade), .66)

    return getcolor


def color(partition, **kwargs):
    """This function traverses a partition with the default coloring
    function.  This is intended to make Rubik easier to script by
    allowing people not to have to worry about how things are colored.

    Optional parameters:
      colorer  Optionally pass a colorer to this routine.  The colorer
               should be some function that takes a path and returns
               a color.

    By default, this uses a level_gradient_colorer.  If you do not
    supply a custom colorer, you can pass level_gradient_colorer's
    keyword args directly to the color function, e.g.::

      color(app, level=-2)
    """
    colorer = kwargs.get("colorer", level_gradient_colorer(**kwargs))

    def assign_color(path):
        path[0].element.color = colorer(path)
    partition.traverse_cells(assign_color)


def viewbox(*partitions, **kwargs):
    """This is a convenience function for making a viewer app out of a
    RubikView. This handles the basics of making a Qt application and
    displaying a main window, so that you can write simple scripts to bring
    up a RubikView.

    This simply builds an app, brings it to the front, and returns the
    result of Qt's exec_() function after it executes the app.

    Optional parameters:
      If you use the default renderer, you can pass colored_face_renderer's
      keyword args directly to viewbox.  e.g.::

          viewbox(app, transparent=True)

      See colored_face_renderer for details on allowed arguments.

      renderer  Optionally pass a custom renderer to draw the faces with.
                By default uses colored_face_renderer.

      rotation  Useful if you want to set a particular starting rotation.
		For example, if you want to generate a set of images with
		the same viewpoint.  Supply a tuple as follows:

                  (angle, x, y, z)

                where (x,y,z) define a vector about which the rotation
		should be performed, and angle is the number of degrees to
		rotate. See GLWidget.set_rotation() for the implentation;
		this just calls that function.
    """
    app = QApplication(sys.argv)
    mainwindow = QMainWindow()

    frame = QFrame()
    layout = QGridLayout()
    frame.setLayout(layout)

    mainwindow.setCentralWidget(frame)
    mainwindow.resize(1024, 768)
    mainwindow.move(30, 30)

    # Find next highest square number and fill within that shape
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

    for rview in views:
        if "rotation" in kwargs:
            rview.set_rotation_quaternion(*kwargs["rotation"])

    # Enter Qt application main loop
    return app.exec_()
