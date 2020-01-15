#define _CRT_SECURE_NO_WARNINGS
#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
int main(int argc, char** argv) {

	int my_rank;
	int size, i, t;
	MPI_Status stat;

	MPI_Init(&argc, &argv); 
	MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); 
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	char msg[2048];
	char* temp;

	if (my_rank == 0) {
		sprintf(msg, "%d - ", my_rank);
		MPI_Send(msg, strlen(msg)+1, MPI_BYTE, 1, 1, MPI_COMM_WORLD);
		MPI_Recv(msg, sizeof(msg), MPI_BYTE, size-1, 1, MPI_COMM_WORLD, &stat);
	}else if (my_rank > 0) {
		MPI_Recv(msg, sizeof(msg), MPI_BYTE, my_rank - 1, 1, MPI_COMM_WORLD, &stat);

		if (!(my_rank+1 < size)) {
			MPI_Send(msg, strlen(msg)+1, MPI_BYTE, 0, 1, MPI_COMM_WORLD);
			temp = (char*)malloc(sizeof(char) * 3);
			snprintf(temp, sizeof(temp), "0");
			strcat(msg, temp);
			printf("%s", msg);
		}else {
			temp = (char*)malloc(sizeof(char) * 3);
			snprintf(temp, sizeof(temp), "%d - ", my_rank);
			strcat(msg, temp);
			MPI_Send(msg, strlen(msg)+1, MPI_BYTE, my_rank + 1, 1, MPI_COMM_WORLD);
		}
	}
	MPI_Finalize();

}