"""
This file defines routines for creating boxes automatically in Rubik by
querying the system for dimensions/shape of the allocated job partition.
"""

from partition import *
from process import *

import numpy as np
import subprocess
import os
import re

# Constant for environment variables
SLURM_JOBID	= "SLURM_JOBID"
COBALT_JOBID	= "COBALT_JOBID"
COBALT_PARTNAME	= "COBALT_PARTNAME"
COBALT_JOBSIZE	= "COBALT_JOBSIZE"
PLATFORM	= "PLATFORM"

def box(shape):
    """ Constructs the top-level partition, with the original numpy array and a
    process list running through it.
    """
    box = np.ndarray(shape, dtype=object)
    index = (0,) * len(box.shape)

    p = Partition(box, None, index, 0, 0)
    p.procs = [Process(i) for i in xrange(0, box.size)]
    p.box.flat = p.procs
    return p

def create_bg_shape_executable(exe_name):
    """ Creates an executable that obtains the torus dimensions from the IBM
    MPIX routines.
    """
    bg_shape_source = "%s.C" % exe_name
    source_file = open(bg_shape_source, "w")
    source_file.write("""#include <iostream>
#include <mpi.h>
#if defined(__bgp__)
  #include <dcmf.h>
#elif defined(__bgq__)
  #include <mpix.h>
#endif
using namespace std;

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
    #if defined(__bgp__)
        DCMF_Hardware_t hw;
        DCMF_Hardware(&hw);
        cout << hw.xSize << "x" << hw.ySize << "x" << hw.zSize << "x" << hw.tSize;
        cout << endl;
    #elif defined(__bgq__)
        MPIX_Hardware_t hw;
        MPIX_Hardware(&hw);
        cout << hw.Size[0];
        for (int i=1; i < hw.torus_dimension; i++) {
            cout << "x" << hw.Size[i];
        }
        cout << endl;
    #endif
    }
    MPI_Finalize();
    return(0);
}
    """)
    source_file.close()
    if subprocess.call(["mpicxx", "-o", exe_name, bg_shape_source]) != 0:
        raise Exception("Unable to compile %s executable!" % exe_name)


def autobox(tasks_per_node=1):
    """ This routine tries its obtain the dimensions of the partition
    automatically. On Blue Gene/Q, we compile an executable (if it's not already
    built) and run it to query the system. This is designed to be run within a
    run script, after the partition is allocated but before the job is launched.
    """
    prefs_directory = os.path.expanduser(".")
    if not os.path.isdir(prefs_directory):
        os.makedirs(prefs_directory)
    bg_shape = os.path.join(prefs_directory, "bg-shape")
    if not os.path.exists(bg_shape):
	create_bg_shape_executable(bg_shape)

    if SLURM_JOBID in os.environ:
	# LLNL Blue Gene/P or Q
        run_command = ["srun", bg_shape]

    elif COBALT_JOBID in os.environ:
	# ANL Blue Gene/P or /Q
	if PLATFORM in os.environ:
	    platform = os.environ[PLATFORM]
	    num_nodes = os.environ[COBALT_JOBSIZE]
	    part_name = os.environ[COBALT_PARTNAME]

	    if re.match('linux-rhel_6-ppc64', platform):
		# ANL Blue Gene/Q
		run_command = ["runjob",
			       "-p", "1",
			       "-n", num_nodes,
			       "--block", part_name,
			       "--verbose=INFO",
			       "--envs", "BG_SHAREDMEMSIZE=32MB",
			       ":", bg_shape]
	    elif re.match('linux-sles10-ppc64', platform):
		# ANL Blue Gene/Q
		run_command = ["cobalt-mpirun",
			       "-nofree",
			       "-np", num_nodes,
			       "-mode", "smp",
			       bg_shape]
	    else:
		raise Exception("Unsupported platform")
	else:
	    raise Exception("PLATFORM should be set as an env variable")
    else:
        raise Exception("Unsupported scheduler environment!")

    def get_dims():
	""" Runs the job and communicates the output back to Rubik. """
        srun_proc = subprocess.Popen(run_command, stdout=subprocess.PIPE)
        out_data, err_data = srun_proc.communicate()
        if srun_proc.wait() != 0:
            return None
        try:
            return [int(dim) for dim in out_data.split("x")]
        except ValueError, e:
            return None

    # Try to run the executable and rebuild and retry once if it fails the
    # first time. This is to handle things like driver changes, that a
    # rebuild will fix
    dims = get_dims()
    if not dims:
        create_bg_shape_executable(bg_shape)
        dims = get_dims()
        if not dims:
            raise Exception("Error invoking bg-shape: %s" % err_data)

    dims.append(tasks_per_node)
    return box(dims)
