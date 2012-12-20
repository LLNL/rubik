################################################################################
# Copyright (c) 2012, Lawrence Livermore National Security, LLC.
# Produced at the Lawrence Livermore National Laboratory
# Written by Todd Gamblin et al. <tgamblin@llnl.gov>
# LLNL-CODE-599252
# All rights reserved.
#
# This file is part of Rubik. For details, see http://scalability.llnl.gov.
# Please read the LICENSE file for further information.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the disclaimer below.
#
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the disclaimer (as noted below) in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of the LLNS/LLNL nor the names of its contributors may be
#       used to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY, LLC, THE
# U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
################################################################################
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
SLURM_NNODES	= "SLURM_NNODES"
COBALT_JOBID	= "COBALT_JOBID"
COBALT_PARTNAME	= "COBALT_PARTNAME"
COBALT_JOBSIZE	= "COBALT_JOBSIZE"
PLATFORM	= "PLATFORM"

def box(shape):
    """ Constructs the top-level partition, with the original numpy array and a
    process list running through it.
    """
    size = np.product(shape)
    return Partition.fromlist(shape, [Process(i) for i in xrange(size)])

def create_bg_shape_executable(exe_name):
    """ Creates an executable that obtains the torus dimensions from the IBM
    MPIX routines.  The executable is compiled and placed in the current
    working directory.
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
        cout << hw.xSize << "x" << hw.ySize << "x" << hw.zSize;
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


def autobox(**kwargs):
    """ This routine tries its obtain the dimensions of the partition
    automatically. On Blue Gene/Q, we compile an executable (if it's not already
    built) and run it to query the system. This is designed to be run within a
    run script, after the partition is allocated but before the job is launched.
    """
    num_nodes = None
    if kwargs:
	# the user must specify the number of tasks_per_node if he provides any
	# arguments to autobox
	if 'tasks_per_node' not in kwargs:
	  raise ValueError("autobox requires tasks_per_node")
	else:
	  tasks_per_node = kwargs['tasks_per_node']
	if 'num_tasks' in kwargs:
	  num_tasks = kwargs['num_tasks']
	  num_nodes = str(num_tasks / tasks_per_node)
    else:
	# the default is assumed to be SMP mode
	tasks_per_node = 1

    prefs_directory = os.path.expanduser(".")
    if not os.path.isdir(prefs_directory):
        os.makedirs(prefs_directory)
    bg_shape = os.path.join(prefs_directory, "bg-shape")
    if not os.path.exists(bg_shape):
	create_bg_shape_executable(bg_shape)

    if SLURM_JOBID in os.environ:
	# LLNL Blue Gene/P or Q
	if num_nodes is None:
	  num_nodes = os.environ[SLURM_NNODES]
	run_command = ["srun",
		       "-N", num_nodes,
		       bg_shape]

    elif COBALT_JOBID in os.environ:
	# ANL Blue Gene/P or /Q
	if PLATFORM in os.environ:
	    platform = os.environ[PLATFORM]
	    part_name = os.environ[COBALT_PARTNAME]
	    if num_nodes is None:
		num_nodes = os.environ[COBALT_JOBSIZE]

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
