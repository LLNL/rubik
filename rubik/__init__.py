"""
Rubik generates mapping files for torus and mesh networks according to
structured transformations of blocks within the ranks.

Author:        Todd Gamblin tgamblin@llnl.gov

Contributors:  Abhinav Bhatele bhatele@llnl.gov
               Martin Schulz schulzm@llnl.gov
"""
import sys
sys.path.append('/soft/apps/python/python-2.6.4-fen-gcc/python/lib/python2.6/site-packages') # Intrepid at Argonne
sys.path.append('/soft/apps/python/python-2.6.6-fen-gcc/usr/lib64/python2.6/site-packages') # Vesta at Argonne

import zorder
from partition import *
from process import *
from box import *
