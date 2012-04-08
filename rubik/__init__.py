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

def box(shape):
    """Constructs the top-level partition, with the original numpy array and a
       process list running through it.
    """
    box = np.ndarray(shape, dtype=object)
    index = (0,) * len(box.shape)

    p = Partition(box, None, index, 0, 0)
    p.procs = [Process(i) for i in xrange(0, box.size)]
    p.box.flat = p.procs
    return p

