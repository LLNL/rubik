"""
Rubik outputs mapping files for changing task layout on torus and mesh
networks. The input to Rubik is the application topology (including process
groups) and the processor topology. Various operations are supported on the
groups as well as the entire processor partition.

:Author:
  `Todd Gamblin <mailto:tgamblin@llnl.gov>`_

:Contributors:
  `Abhinav Bhatele <mailto:bhatele@llnl.gov>`_,
  `Martin Schulz <mailto:schulzm@llnl.gov>`_

:Version: 1.0
"""

import sys
sys.path.append('/soft/apps/python/python-2.6.4-fen-gcc/numpy-1.3.0/lib/python2.6/site-packages/numpy')	# Intrepid at Argonne
sys.path.append('/soft/apps/python/python-2.6.6-fen-gcc/usr/lib64/python2.6/site-packages')		# Vesta at Argonne
import zorder

from partition import *
from process import *
from box import *
