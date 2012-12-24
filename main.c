/* -*- Mode: C; c-basic-offset:4 ; -*- */
/*
 *  (C) 2001 by Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#include "mpi.h"
#include "knn.h"
#include <stdio.h>
#include <string.h>

#define BUFLEN 512

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
	int m,n,k;
	int i;
	int *V, *out;				// host copies
	FILE *fp;
	float time;

	int myid, numprocs, next, namelen;
	int A[32];
	int i;
	char buffer[BUFLEN], processor_name[MPI_MAX_PROCESSOR_NAME];
	MPI_Status status;

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
/*
	while(fscanf(fp, "%d %d %d", &m, &n, &k) != EOF)
	{
		V = (int *) malloc(m*n*sizeof(int));
		out = (int *) malloc(m*k*sizeof(int));

		for(i=0; i<m*n; i++)
		{
			fscanf(fp, "%d", &V[i]);
		}

		launch(V, out);

		showResult(m, k, out);
		if(m == 1024) {
			printf("SMALL:");
		} else if(m == 4096) {
			printf("MIDDLE:");
		} else if(m == 16384) {
			printf("LARGE:");
		}
		printf("%f\n", time);

		free(V);
		free(out);
	}
	fclose(fp);
*/
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
	launch(V, out, time);
//	printf("%s \n",processor_name);
	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	showResult(m, k, out);
	if(m == 1024) {
		printf("SMALL:");
	} else if(m == 4096) {
		printf("MIDDLE:");
	} else if(m == 16384) {
		printf("LARGE:");
	}
	printf("%f\n", time);

	free(V);
	free(out);

	return (0);
}
