/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpi.h"
#include "kernel.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define BUFLEN 512

//#define DEBUG

void showResult(int m, int k, int *out)
{
	int i,j;
	for(i=0; i<m; i++)
	{
		for(j=0; j<k; j++)
		{
			printf("%d ", out[i*k+j]);
			if(j == k-1)
			{
				printf("\n");
			}	
		}    	
	}        	
}            	

int main(int argc, char *argv[]){
	int myid, numprocs, next, namelen;

	int i;
	int rc;
	struct timespec start, end;
	double comtime;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

	int m,n,k;
	int *V, *out;				// host copies
	int *D;						
	FILE *fp;
	if(argc != 2)
	{
		printf("Usage: knn <inputfile>\n");
		exit(1);
	}
	if((fp = fopen(argv[1], "r")) == NULL)
	{
		printf("Error open input file!\n");
		exit(1);
	}
	fscanf(fp, "%d %d %d", &m, &n, &k);

	V = (int *) malloc(m*n*sizeof(int));
	out = (int *) malloc(m*k*sizeof(int));

	for(i=0; i<m*n; i++)
	{
		fscanf(fp, "%d", &V[i]);
	}

	fclose(fp);

	MPI_Init(&argc,&argv);
	MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	MPI_Comm_rank(MPI_COMM_WORLD,&myid);
	MPI_Get_processor_name(processor_name,&namelen);
	if(numprocs != 2) goto end;

	if(myid == 0) beforeStart(processor_name);
	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 1) beforeStart(processor_name);
	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME,&start);
	MPI_Barrier(MPI_COMM_WORLD);

	launch(myid, m, n, k, V, out);

	MPI_Barrier(MPI_COMM_WORLD);
	if(myid == 1)
	{
		rc = MPI_Send(out+(m/2)*k, (m/2)*k, MPI_INT, 0, 1, MPI_COMM_WORLD);
	}
	if(myid == 0)
	{
		rc = MPI_Recv(out+(m/2)*k, (m/2)*k, MPI_INT, 1, 1, MPI_COMM_WORLD, &status);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	clock_gettime(CLOCK_REALTIME,&end);
	if(myid == 0){
		comtime = (double)(end.tv_sec-start.tv_sec)+(double)(end.tv_nsec-start.tv_nsec)/(double)1000000000L;
		showResult(m, k, out);
		printf("%f\n", comtime);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	free(V);
	free(out);
	MPI_Barrier(MPI_COMM_WORLD);
end:
	MPI_Finalize();

//	free(V);
//	free(out);

	return (0);
}
