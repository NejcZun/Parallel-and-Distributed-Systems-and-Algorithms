#include <pthread.h>
#include <stdlib.h>
#include <stdio.h>
#include "timing.h"
#include <math.h>

#define NTHR 8
#define N 100000

pthread_t threads[NTHR];

typedef struct _thread_data_t {
	int threadID;
} thread_data_t;

thread_data_t thr_struct[NTHR];

//int arr[] = { 2, 1, 4, 9, 5, 3, 6, 10};
int arr[N];
pthread_barrier_t barrier;

//izpis kaj je v arrayju
void printStep() {
	for (int i = 0; i < N; i++)printf("%d ", arr[i]);
	printf("\n");
}
//zamenja 2 elementa
void swap(i, j) {
	int temp = arr[i];
	arr[i] = arr[j];
	arr[j] = temp;
}
//generiramo array glede na podan N
void generateNumbers() {
	for (int i = 0; i < N; i++) {
		arr[i] = rand() % N + 1;
	}
}

void* threadFunction(void* args) {
	thread_data_t* threadData = (thread_data_t*)args;;
	int id = threadData->threadID;

	for (int j = 0; j < N; j++) {  //za vsako vrstico
		int i = j % 2;
		while (i < N) {  //even odd sort po vrstici
			int even = i + id * 2; //i
			int odd = i + id * 2 + 1; //i+1
			if (odd >= N) break; //ce je vecji od N smo na konc - \o/ - pojdimo v novo vrstico
			if (arr[even] > arr[odd]) {
				swap(even, odd);
			}
			i += NTHR * 2;
		}
		i = j % 2;
		if (i == 1) pthread_barrier_wait(&barrier); //ko pride do lihe spet zato k delamo 2 na enkrat
	}
	pthread_exit(NULL);
}


void main() {
	/*Generate a bigger array*/
	srand(time(NULL));
	generateNumbers();
	printf("Array generated.\n\n");
	//printStep();
	printf("Started sorting.\n");
	pthread_barrier_init(&barrier, NULL, NTHR);
	for (int i = 0; i < NTHR; i++) {
		thr_struct[i].threadID = i;
		pthread_create(&threads[i], NULL, threadFunction, &thr_struct[i]);
	}
	BEGCLOCK(sinhrono);
	for (int i = 0; i < NTHR; i++) {
		pthread_join(threads[i], NULL);
	}
	ENDCLOCK(sinhrono);
	printf("Sorting ended.\n");
	//printStep();

	//glede na to da je array velik ga glih nebi preverju na uc - ce je le ze mozno
	for (int i = 0; i < N - 1; i++) {
		if (arr[i] > arr[i + 1]) {
			break; printf("Ne dela");
		}
	}
	printf("Array is sorted!");
}