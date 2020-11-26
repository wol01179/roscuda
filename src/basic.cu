#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

/* What is __global__ ? 
 1. It indicates a function that runs on the device.
 2. 
*/

/*
 FYI,
 device function processed by NVIDIA compiler.
		will execute on the device. / will be called from the host. 
<<<>>> is called from host code to device code.-"kernel launch"

Simple CUDA API for handling device memory
- cudaMalloc(), cudaFree(), cudaMemcpy()
- Similar to the C equivalents malloc(), free(), memcpy()	

void *malloc(size_t size)
필요한 크기를 동적으로 할당하여 사용합니다.
데이터 크기에 맞춰서 할당해줘야 하므로 "(데이터타입*)malloc(sizeof(데이터타입)*할당크기);"형식으로 할당합니다.

void free(void *ptr)
할당 메모리는 반드시 free함수를 통해 메모리 해제를 해야합니다.

*/
__global__ void add(int *a,int *b,int *c) {
	*c = *a+*b;
}

void cudamain() 
{	
	
	int a, b, c;		// host copies of a,b,c
	int *d_a, *d_b,*d_c;   // device copies of a,b,c
	int size = sizeof(int);

	// Allocate space for device copies of a,b,c
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Setup input values
	a = 2; b= 7;

	// Copy inputs to device
	cudaMemcpy(d_a, &a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, &b, size, cudaMemcpyHostToDevice);

	// Launch add() kernel on GPU
	add<<<1,1>>>(d_a,d_b,d_c);

	// Copy result back to host
	cudaMemcpy(&c, d_c, size, cudaMemcpyDeviceToHost);

	// Cleanup
	cudaFree(d_a);cudaFree(d_b);cudaFree(d_c);
	printf("c : %d\n",c);
	return;
}
