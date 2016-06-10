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
from pyprimes import factors as fact
from sets import Set

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

def box_cray(shape):
    """ Constructs the top-level partition, with the original numpy array and a
    process list running through it. Same working as box.
    """
    size = np.product(shape)
    return Partition.fromlist(shape, [Process(i) for i in xrange(size)])

def create_executable():
    pythonPath = os.environ["PYTHONPATH"]
    pythonPath = pythonPath.split(':')
    pythonPath = [i for i in pythonPath if 'rubik' in i][0]
    print pythonPath
    if subprocess.call(["cc", pythonPath+'/rubik/'+"Topology.c", "-o", "topology"]) != 0:
       raise Exception("Unable to compile Topology executable!")

def decide_torus_shape(dimVector, numpes):
    """ This funciton is used to decide the shape of logical torus approximated to the shape of real allocation. 
    The reason why this functionn is needed is that user needs some logical rubik box to map their application grid onto. 
    Depending on how this logical torus is shaped, the benefit of rubik script can be varied significantly. 
    """
    ppn = dimVector[len(dimVector)-1]
    dimVector = dimVector[0:len(dimVector)-1]

    dimVector = sorted(list(enumerate(dimVector)), key=itemgetter(1),reverse=True)
#    print dimVector
    factors = fact.factorise(numpes/ppn)
    tempShape = [1]*len(dimVector)
    finalShape = []
    numLeft=len(tempShape)
    idx=0
#    print factors
    for factor in factors:
        print idx
        print factor
        print tempShape
        print finalShape
        print dimVector
        tempShape[idx] = tempShape[idx] * factor
        if tempShape[idx] >= dimVector[idx][1]:
            finalShape.append((dimVector.pop(idx)[0],tempShape.pop(idx)))
            numLeft-=1
        else:
            idx = idx + 1
        if numLeft != 0:
            idx = idx % numLeft
#    print idx
    print finalShape
    if numLeft >0:
        for idx in range(numLeft):
            finalShape.append((dimVector[idx][0],tempShape[idx]))
    finalShape.append((len(finalShape),ppn))
    return list(zip(*sorted(finalShape, key=itemgetter(0)))[1])

def autobox_cray(**kwargs):
    """ This obtains the dimensions of the partition and available coordinates are discovered.
    """
    numpes = kwargs['numpes']
    topo = kwargs['queryTopo']
    if (int)(topo) == 1:
      if os.path.isfile("./topology") != True:
          create_executable()
      subprocess.call(["aprun", "-n", numpes, "./topology", numpes])
#        cuboidShape = '9x4x8@9.14.0'
#        ""This code is to obtain the shape of the assigned cuboid, this will be used for further partitoning""
#        cuboidShape = subprocess.Popen("checkjob $PBS_JOBID | grep 'Placement' | awk '{print $NF;}'", stdout=subprocess.PIPE, shell=True).stdout.read()         
#        cuboidShape = cuboidShape.split('@')[0]
#        cuboidShape = map(int, cuboidShape.split('x')) 
#       print "Aprun is called" 
#    ppn = (int)(kwargs['ppn'])
#        cuboidShape.append(ppn*2)
    f = open((str)(numpes)+'_'+"Topology.txt", "r")
    dims = f.readline().rstrip('\n').split("x")
    dims = [int(i) for i in dims]

    setList =[] #a list of Sets for each dimension to figure out the shape of the allocation received from BlueWaters
    for i in range(len(dims)):
        setList.append(Set())

    check_coord = np.ones((int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])),dtype=object)
    check_coord *= -1
    for line in f:
        row =  [(int)(i) for i in line.split()] # p, n, x, y, z, t = line.split()
        k=0
        for j in map(int,row[2:len(row)]):
            setList[k].add(j)
            k+=1
        check_coord[int(row[2])][int(row[3])][int(row[4])][int(row[5])] = [int(row[0]),int(row[1])];

#           check_coord[int(x)][int(y)][int(z)][int(t)] = [p,n];
#           check_coord[int(x)][int(y)][int(z)][int(t)][1] = n;
    cuboidShape = []
    for eachSet in setList:
        cuboidShape.append(len(eachSet))

    f.close()
    print cuboidShape
    return box_cray(dims), check_coord, cuboidShape, decide_torus_shape(cuboidShape, (int)(numpes)) 

def autobox_sim(**kwargs):
    """ This obtains the dimensions of the partition and available coordinates are discovered.
    """
    filename = kwargs['filename'] 
    f = open(filename,"r")
    next(f)
#    ppn = (int)(kwargs['ppn'])
    dims = [24,24,24,64] #BlueWaters
    check_coord = np.ones((int(dims[0]), int(dims[1]), int(dims[2]), int(dims[3])),dtype=object)
    check_coord *= -1
    p=0
    setList =[]
    for i in range(len(dims)):
        setList.append(Set())
    for line in f:
        row =  [(int)(i) for i in line.split()]
        k=0
        for j in map(int,row):
            setList[k].add(j)
            k+=1
        check_coord[int(row[0])][int(row[1])][int(row[2])][int(row[3])] = [p,p];
        p+=1
    cuboidShape=[]
    for eachSet in setList:
        cuboidShape.append(len(eachSet))
#        cuboidShape.append(ppn*2) 
    numpes = kwargs['numpes']
    factors = fact.factorise(numpes)

    f.close()
    print cuboidShape
    return box_cray(dims), check_coord,  cuboidShape, decide_torus_shape(cuboidShape, (int)(numpes)) 
