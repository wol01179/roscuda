#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>


/*
Topic :: Blocks

GPU computing is about massive parallelism add<<<N,1>>> - execute N times in parallel

Terminology: each parallel invocation of add()is referred to as a block
- The set of blocks is referred to as a grid.
- Each invocation can refer to its block index using blockIdx.x

By using blockIdx.xto index into the array, each block handles a different element of the array

*/

#define N 512

__global__ void add1(int *a,int *b,int *c) {
	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];	
}

void cudamain1() 
{	
	
	int *a, *b, *c;		// host copies of a,b,c
	int *d_a, *d_b,*d_c;   // device copies of a,b,c
	int size = N * sizeof(int);

	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Allocspace for host copies of a, b, c and setup input values
	a = (int *)malloc(size); //random_ints(a, N);
	b = (int *)malloc(size); //random_ints(b, N);
	c = (int *)malloc(size);

	// Copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add1<<<N,1>>>(d_a,d_b,d_c);

	// Copy result back to host
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	//printf("c : %d\n",c);

	// Cleanup
	free(a); free(b); free(c);
	cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
	return;
}
