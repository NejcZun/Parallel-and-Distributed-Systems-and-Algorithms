#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 1000000

int compare(const void* first, const void* second) {
	return *(int*)first - *(int*)second;
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
		distributedCount[i] = ceil(N / (world_size * 1.0f));
		displacement[i] = i * distributedCount[i];
		temp_disp[i] = displacement[i];
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

		sorted = (int*)malloc(N * sizeof(int));
		
		for (int i = 0; i < N; i++) {
			int min = RAND_MAX;
			int izbran=0;
			for (int j = 0; j < world_size; j++) {
				if (temp_disp[j] < displacement[j] + distributedCount[j]) {
					//printf("Disp: %d < %d\n", temp_disp[j], displacement[j] + distributedCount[j]);
					if (min > semi_sorted[temp_disp[j]]) {
						min = semi_sorted[temp_disp[j]];
						izbran = j;
					}
				}
			}
			sorted[i] = min;
			temp_disp[izbran]++;
		}
		
			
		printf("Sortiran array:\n");
		t2 = MPI_Wtime();
		//qsort(sorted, N, sizeof(int), compare);
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