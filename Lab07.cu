#include <iostream>
#include <math.h>
using namespace std;
__global__ void max(float* data, float* data_output, int size) {
	__shared__ float max[128];
	unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int tid = threadIdx.x;
	if (index >= size) {
		return;
	}
	max[tid] = data[index];
	__syncthreads();
	if (tid == 0) {
		for (unsigned int i = 1; i < blockDim.x; i++) {

			if (max[tid] < max[i]) {
				max[tid] = max[i];
			}
		}
		data_output[blockIdx.x] = max[tid];
	}

}
int main() {
	int N = 128;
	dim3 nThreadsPerBlock(128, 1, 1);
	dim3 nBlocks(N / 128, N / 128, 1);


	//float vector_intrare[N];
	float *vector_intrare, *vector_iesire;
	vector_intrare = new float[N];
	vector_iesire = new float[N];

	// Initializare vector
	for (int i = 0; i < N; i++) {
		vector_intrare[i] = rand() % 99 + 1;
	}
	for (int i = 0; i < N; i++)
	{
		cout << vector_intrare[i] << " ";
	}
	cout << endl;
	float *vector_intrare_d, *vector_iesire_d;
	//      a_d = new float[N*M];
	//      b_d = new float[N*M];

	cudaMalloc((void**)&vector_intrare_d, N * sizeof(float));
	cudaMalloc((void**)&vector_iesire_d, N * sizeof(float));

	cudaMemcpy(vector_intrare_d, vector_intrare, N * sizeof(float), cudaMemcpyHostToDevice);
	max << <nBlocks, nThreadsPerBlock >> > (vector_intrare_d, vector_iesire_d, N);
	cudaMemcpy(vector_iesire, vector_iesire_d, N * sizeof(float), cudaMemcpyDeviceToHost);

	/*for (int i = 0; i < N; i++) {

			cout << vector_iesire[N ] << " ";
		}
		cout << "\n";*/
	cout << *vector_iesire;

	return 0;
}