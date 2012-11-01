Download and Install
====================

Download
--------
Rubik is available on the `PAVE <https://scalability.llnl.gov/performance-analysis-through-visualization/software.php>`_ website.

Install
--------
Rubik requires a Python installation (2.5+) and numpy. If you have these two
on your machine, you should be able to generate map files for your jobs.

Optionally, if you want to visualize Rubik mappings in 3D, you can install Qt,
OpenGL for Python, and `PySide <http://qt-project.org/wiki/PySide>`_, the free
Python bindings for Qt. To build the documentation, you need `Sphinx
<http://sphinx-doc.org>`_.

On a mac, you can get all this fairly easily through macports::

    port install py27-numpy py27-pyside py27-sphinx py27-opengl

That should automatically handle dependencies, installing numpy, Qt, PySide,
and Sphinx for Python 2.7 all at once.
