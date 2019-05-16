#include <cuda.h>
#include <cuda_runtime.h>
#include <iostream>
#include <device_launch_parameters.h>

using namespace std;

__global__ void init(float *a, float *b, int n, int m)
{
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;

	float sum = 0;
	if (row == 0 || row == n - 1 || col == 0 || col == n - 1)
		return;
	for (int m = row - 1; m <= row + 1; m++)
		for (int n = col - 1; n <= col + 1; n++) {
			sum += a[m*n + n];

		}
	sum /= 9;
	b[row*n + col] = sum;
}


int main()
{
	int N = 512;
	int M = 512;

	float *a, *b;
	a = new float[N*M];
	b = new float[N*M];

	//initializare matrice
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			a[i*N + j] = (i + j) % 2;
		}
	}

	float *a_d, *b_d;

	//alocare device
	cudaMalloc((void**)&a_d, N*M * sizeof(float));
	cudaMalloc((void**)&b_d, N*M * sizeof(float));

	dim3 nBlock(N / 32, N / 32, 1);
	dim3 nThreadsBlock(32, 32, 1);

	cudaMemcpy(a_d, a, N * M * sizeof(float), cudaMemcpyHostToDevice);
	init << <nThreadsBlock, nBlock >> > (a_d, b_d, N, M);
	cudaMemcpy(b, b_d, N*M * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < M; j++)
		{
			cout << b[i + j * N] << " ";
		}
		cout << "\n";
	}

	return 0;
}