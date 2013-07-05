#!/usr/bin/env python

from distutils.core import setup

# Check dependencies
try:
    import numpy
except:
    print "WARNING: Rubik requires numpy.  You will need to install numpy for anything to work."

try:
    import PySide
    import OpenGL
except:
    print "WARNING: To use the rubik.view package for visualizations, you need to install PySide and OpenGL."

setup(name='rubik',
      # Distribution and version information
      version='1.0.1',
      packages=['rubik', 'rubik.tests'],

      # Other metadata
      license='LLNL BSD',
      description='Rubik Task Mapping Tool',
      author='Todd Gamblin',
      author_email='tgamblin@llnl.gov',
      url='https://github.com/tgamblin/rubik',
      long_description='Rubik provides a simple and intuitive interface to create a wide variety of mappings for structured communication patterns. Rubik supports a number of elementary operations such as splits, tilts, or shifts, that can be combined into a large number of unique patterns.',
      )
