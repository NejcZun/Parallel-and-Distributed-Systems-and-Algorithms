#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1000

int compare(const void* first, const void* second){
	return *(int*)first - *(int*)second;
}
void merge(int ar1[], int ar2[], int m, int n){
	for (int i = n - 1; i >= 0; i--){
		int j, last = ar1[m - 1];
		for (j = m - 2; j >= 0 && ar1[j] > ar2[i]; j--)
			ar1[j + 1] = ar1[j];
		if (j != m - 2 || last > ar2[i])
		{
			ar1[j + 1] = ar2[i];
			ar2[i] = last;
		}
	}
}

int main(int argc, char** argv) {
	double t1, t2, result;
	int* polje = (int*) malloc(N * sizeof(int));
	int* semi_sorted = NULL;
	int* sorted = NULL;
	srand(time(NULL));
	for (int i = 0; i < N; i++) polje[i] = rand() % N;
	int world_rank;
	int world_size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	int* distributedCount = (int*)malloc(world_size * sizeof(int));
	int* displacement = (int*)malloc(world_size * sizeof(int));
	int* temp_disp = (int*)malloc(world_size * sizeof(int));

	if (world_rank == 0) t1 = MPI_Wtime();
	for (int i = 0; i < world_size; i++) {
		distributedCount[i] = round(N / (world_size * 1.0f));
		displacement[i] = i * distributedCount[i];
	}
	if ((N % world_size) != 0) {
		distributedCount[world_size-1] = N % distributedCount[0];
	}

	int* pod_polje = (int*)malloc(distributedCount[world_rank] * sizeof(int));
	
	MPI_Scatter(polje, distributedCount[world_rank], MPI_INT, pod_polje, distributedCount[world_rank], MPI_INT, 0, MPI_COMM_WORLD);
	
	qsort(pod_polje, distributedCount[world_rank], sizeof(int), compare);

	if (world_rank == 0)semi_sorted = (int*)malloc(N * sizeof(int));

	MPI_Gather(pod_polje, distributedCount[world_rank], MPI_INT, semi_sorted, distributedCount[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

	if (world_rank == 0) {

		temp_disp = displacement;
		sorted = (int*)malloc(N * sizeof(int));

		for (int i = 0; i < N; i++) {
			int min = RAND_MAX;
			int izbran;
			for (int j = 0; j < world_size; j++) {
				if (temp_disp[j] < temp_disp[j] + distributedCount[j]) {
					if (min > semi_sorted[temp_disp[j]]) {
						min = semi_sorted[temp_disp[j]];
						izbran = j;
					}
				}
			}
			sorted[i] = min;
			temp_disp[izbran]++;
		}
			
		//qsort(sorted, world_size, sizeof(int), compare);
		printf("Sortiran array:\n");
		t2 = MPI_Wtime();
		result = t2 - t1;
		for (int i = 0; i < N; i++) printf("%d ", sorted[i]);
		free(sorted);
	}
	free(polje);
	free(pod_polje);
	if (world_rank == 0) printf("\nPorabljen cas: %f s\n", result);
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Finalize(); 
}