#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include "timing.h"

#define NTHR 8
#define N 10000
#define BLOCKSIZE 16


typedef struct _thread_data_t {
	int threadID;
} thread_data_t;

pthread_t threads[NTHR];
thread_data_t thr_struct[NTHR];
long vsotaStatic = 0;
long vsotaDynamic = 0;
long long nextBlockIndex = NTHR * BLOCKSIZE;
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t blockLock = PTHREAD_MUTEX_INITIALIZER;

int vsotaDeljiteljev(int st) {
	if (st == 2) return 0;
	int vsota = 0;
	for (int i = 2; i < sqrt(st); i++) 
		if (st % i == 0)vsota += (i + st / i);
	if ((int)sqrt(st) * (int)sqrt(st) == st) vsota += sqrt(st);
	return vsota + 1;
}

void* threadFunctionStatic(void* args) {
	thread_data_t threadData = *((thread_data_t*)args);
	printf("[Static] Started thread %d\n", threadData.threadID);
	for (int i = threadData.threadID; i < N; i+=NTHR) {
		int si = vsotaDeljiteljev(i);
		int sj = vsotaDeljiteljev(si);
		if (sj == i && si != i) {
			pthread_mutex_lock(&lock);
			vsotaStatic += i;
			pthread_mutex_unlock(&lock);
		}
	}
	pthread_exit(NULL);
}

void* threadFunctionDynamic(void* args) {
	thread_data_t threadData = *((thread_data_t*)args);
	int startIndex = threadData.threadID * BLOCKSIZE;
	printf("[Dynamic] Started thread %d\n", threadData.threadID);
	
	while (startIndex < N) {
		for (int i = startIndex; i < startIndex + BLOCKSIZE && i < N; i++) {
			int si = vsotaDeljiteljev(i);
			int sj = vsotaDeljiteljev(si);
			if (sj == i && si != i) {
				pthread_mutex_lock(&lock);
				vsotaDynamic += i;
				pthread_mutex_unlock(&lock);
			}
		}
		pthread_mutex_lock(&blockLock);
		startIndex = nextBlockIndex;
		nextBlockIndex += BLOCKSIZE;
		pthread_mutex_unlock(&blockLock);
	}

	pthread_exit(NULL);
}


void main(int argc, char args[]) {

	//Staticno

	// Create threads:
	for (int i = 0; i < NTHR; i++) {
		thr_struct[i].threadID = i;
		pthread_create(&threads[i],NULL, threadFunctionStatic, &thr_struct[i]);
	}
	BEGCLOCK(staticno);
	for (int i = 0; i < NTHR; i++) {
		pthread_join(threads[i], NULL);
	}
	ENDCLOCK(staticno);
	printf("[Static]: Vsota prijateljskh stevil do %d je: %ld\n", N, vsotaStatic);

	//Dinamicno 
	for (int i = 0; i < NTHR; i++) {
		thr_struct[i].threadID = i;
		pthread_create(&threads[i], NULL, threadFunctionDynamic, &thr_struct[i]);
	}
	BEGCLOCK(dinamicno);
	for (int i = 0; i < NTHR; i++) {
		pthread_join(threads[i], NULL);
	}
	ENDCLOCK(dinamicno);
	printf("[Dynamic]: Vsota prijateljskh stevil do %d je: %ld\n", N, vsotaDynamic);

}


/*
Ugotovitve:
N = 1000000, NTHR = 8, BLOCKSIZE 500
Staticno: 0.887000, Dinamicno: 0.772000
Pohitritev: 1.148x

 N 10000, NTHR 8, BLOCKSIZE 16
Staticno: 0.004000, Dinamicno: 0.002000
Pohitritev: 2x
*/