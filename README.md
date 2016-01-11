Rubik v1.0.1
============

Rubik generates mapping files for torus and mesh networks according to
structured transformations of blocks within the ranks.

Author:
  Todd Gamblin tgamblin@llnl.gov

Contributors:
  Abhinav Bhatele bhatele@llnl.gov
  Martin Schulz schulzm@llnl.gov

Source Code for Rubik can be found [on GitHub](https://github.com/llnl/rubik).

Documentation:
    https://scalability.llnl.gov/performance-analysis-through-visualization/software/rubik/docs

More information on rubik at LLNL:
    https://scalability.llnl.gov/performance-analysis-through-visualization/software.php

Running Rubik Scripts:
To use rubik, either add the <distribution>/rubik directory to your PYTHONPATH,
or just be sure to run scripts in the root directory.  A proper setup.py and
installation process is forthcoming.

## Pre-requisites

To generate map files with Rubik, you will need an installation of numpy.  To
visualize Rubik partitions, you will need PySide (python Qt bindings) and
OpenGL for Python.  To build the documentation you will need sphinx.

You can install all this relatively easily on a mac through MacPorts:

    port install py27-numpy py27-pyside py27-sphinx py27-opengl

For more on installation, see [the installation docs](https://scalability.llnl.gov/performance-analysis-through-visualization/software/rubik/docs/intro.html#install).
