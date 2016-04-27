#include <stdlib.h>
#include <pmi.h>
#include <rca_lib.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <mpi.h>
#define HAS_RCA_MAX_DIMENSION 1

int *pid2nid = NULL;           
int maxX = -1;
int maxY = -1;
int maxZ = -1;
int maxNID = -1;

void getDimension(int *maxnid, int *xdim, int *ydim, int *zdim);

int getMeshCoord(int nid, int *x, int *y, int *z) {

    rca_mesh_coord_t xyz;
    int ret = -1;
    ret = rca_get_meshcoord(nid, &xyz);
    if (ret == -1) return -1;
    *x = xyz.mesh_x;
    *y = xyz.mesh_y;
    *z = xyz.mesh_z;
    return ret;
}

void pidtonid(int numpes, int *pid2nid) {
  int maxNID, maxX, maxY, maxZ;

  int i, nid, ret;
  for (i=0; i<numpes; i++) {
    PMI_Get_nid(i, &nid);
    pid2nid[i] = nid;
  }
}

void getDimension(int *maxnid, int *xdim, int *ydim, int *zdim)
{
  int i = 0, ret;
  rca_mesh_coord_t dimsize;

  if(maxNID != -1) {
    *xdim = maxX;
    *ydim = maxY;
    *zdim = maxZ;
    *maxnid = maxNID;
    return;
  }

  *xdim = *ydim = *zdim = 0;
#if HAS_RCA_MAX_DIMENSION
  rca_get_max_dimension(&dimsize);
  maxX = *xdim = dimsize.mesh_x+1;
  maxY = *ydim = dimsize.mesh_y+1;
  maxZ = *zdim = dimsize.mesh_z+1;
  maxNID = *maxnid = *xdim * *ydim * *zdim * 2;
#endif
}

typedef struct coord
{
  int x, y, z, t, nid1, pid1;
}coord;

void main(int argc, char *argv[])
{
  FILE *fp;
  MPI_Init(&argc, &argv);
  int myRank;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);

  if(!myRank) { // Only master node do the following mapping work.
        char *fname=(char*)malloc(sizeof(char)*(strlen("_Topology.txt")+strlen(argv[1])));
        strcpy(fname, argv[1]);
//        strcat(fname, argv[1]);
        strcat(fname, "_Topology.txt");
        fp = fopen(fname,"w");//numpe_Topology.txt
        int nid,i,j,k,l,lx=0,ly=0,lz=0,xdim,ydim,zdim,maxnid;
        int numpes = atoi(argv[1]);
        coord *details;
        int ****coords2pid;
        int *pid2nid;
        int numCores = 2*sysconf(_SC_NPROCESSORS_ONLN); // 2 times of _SC_NPROCESSORS_ONLN because each physical position can have 2 nodes in it.

        pid2nid = (int *)malloc(sizeof(int) * numpes);
        pidtonid(numpes, pid2nid);
        getDimension(&maxnid, &xdim, &ydim, &zdim);

        fprintf(fp, "%d x %d x %d x %d \n", xdim, ydim, zdim, numCores); //maximum dimensions of Bluewaters 

        details = (coord *)malloc(numpes*sizeof(coord)); //matrix containing 

        coords2pid = (int ****)malloc(xdim*sizeof(int***));
        for(i=0; i<xdim; i++) {
                coords2pid[i] = (int ***)malloc(ydim*sizeof(int**));
                for(j=0; j<ydim; j++) {
                        coords2pid[i][j] = (int **)malloc(zdim*sizeof(int*));
                        for(k=0; k<zdim; k++) {
                                coords2pid[i][j][k] = (int *)malloc(numCores*sizeof(int));
                    }
                }
        }

        for(i=0; i<xdim; i++)
                for(j=0; j<ydim; j++)
                        for(k=0; k<zdim; k++)
                                for(l=0; l<numCores; l++)
                                        coords2pid[i][j][k][l] = -1;

        for(i=0; i<numpes; i++)
        {
                nid = pid2nid[i];
                getMeshCoord(nid, &lx, &ly, &lz);

                details[i].x = lx;
                details[i].y = ly;
                details[i].z = lz;

                l = 0;
                while(coords2pid[lx][ly][lz][l] != -1)
                        l++;
                coords2pid[lx][ly][lz][l] = i;
                details[i].t = l;
                details[i].nid1 = nid;
                details[i].pid1 = i;
        }
    for(i=0; i<numpes; i++)
        {
fprintf(fp, "%d \t %d \t %d \t %d \t %d \t %d \n", details[i].pid1, details[i].nid1, details[i].x, details[i].y, details[i].z, details[i].t);
        }
        fclose(fp);
        free(fname);

        free(details);
        for(i=0; i<xdim; i++) {
                for(j=0; j<ydim; j++) {
                        for(k=0; k<zdim; k++) {
                                free(coords2pid[i][j][k]);
                        }
                        free(coords2pid[i][j]);
                }
                free(coords2pid[i]);
        }
        free(coords2pid);
  }
}

