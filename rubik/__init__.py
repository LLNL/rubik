"""
Rubik generates mapping files for torus and mesh networks according to
structured transformations of blocks within the ranks.

Author:        Todd Gamblin tgamblin@llnl.gov

Contributors:  Abhinav Bhatele bhatele@llnl.gov
               Martin Schulz schulzm@llnl.gov
"""

import zorder
from partition import *
from process import *
from box import *
