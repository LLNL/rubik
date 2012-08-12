"""
This file defines routines for creating boxes in Rubik.
"""

from partition import *
from process import *
import numpy as np
import subprocess
import os
import re

# Constant for environment variables
SLURM_JOBID           = "SLURM_JOBID"
SLURM_TASKS_PER_NODE = "SLURM_TASKS_PER_NODE"

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


def autobox(tasks_per_node=None):
    """This routine tries its best to figure out dimensions of the partition automatically
       from the environment. Typically this involves asking the scheduler for the partition
       geometry.

       This is designed to be run within a run script, after the partition is allocated but
       before the job is launched.
    """
    if SLURM_JOBID in os.environ:
        # If SLURM_JOBID is present, then we can just open a subprocess to ask scontrol about the job.
        jobid = os.environ[SLURM_JOBID]
        scontrol_proc = subprocess.Popen(["scontrol", "-o", "show", "job", jobid], stdout=subprocess.PIPE)
        out_data, in_data = scontrol_proc.communicate()

        # Once we read the job description, dump key-value pairs into a dict so we can get them by name.
        job_info = dict([kvp.split("=") for kvp in out_data.split(" ")])

        # Now split the dimensions out of the network Geometry
        dims = [int(dim) for dim in job_info["Geometry"].split("x")]

        # And finally either take the user-specified number of tasks per node, or grab it from SLURM,
        # and use it as the last dimension.
        if tasks_per_node:
            dims.append(tasks_per_node)
        else:
            if not SLURM_TASKS_PER_NODE in os.environ:
                raise Exception("SLURM doesn't tell us the number of tasks per node")
            else:
                ntasks = os.environ[SLURM_TASKS_PER_NODE]
                ntasks = re.match(r'(\d+)', ntasks).group(0)
                dims.append(int(ntasks))

        return box(dims)
    else:
        raise Exception("Unable to automatically determine partition shape.  Did you run Rubik outside of a valid allocation?")



