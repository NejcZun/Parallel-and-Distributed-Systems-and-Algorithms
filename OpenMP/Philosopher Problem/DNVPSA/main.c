#include <omp.h>      
#include <stdio.h>    
#include <stdlib.h>  
#include <windows.h> 

#define N 5
#define EAT 1

static omp_lock_t fork[N];

void philosopher() {
#pragma omp barrier //pocaka na vse
	int id = omp_get_thread_num();
	//printf("%d", id);
	int right_fork;
	int left_fork;

	if (id < N - 1) {  //da gre na desno
		right_fork = id;
		left_fork = (id + 1) % 5;
	}
	else { //da gre na levo
		right_fork = 0;
		left_fork = id;
	}

	int i;
	for (i = 0; i < EAT; i++) {
		printf("philosopher %d is thinking\n", id);
		omp_set_lock(&fork[left_fork]);
		omp_set_lock(&fork[right_fork]);

		printf("philosopher %d is eating | L fork: %d , R fork: %d\n", id, left_fork, right_fork);
		Sleep(100);

		omp_unset_lock(&fork[left_fork]);
		omp_unset_lock(&fork[right_fork]);
	}
}

void main(int argc, char* argv) {
	omp_set_num_threads(N);
	int i;
	for (i = 0; i < N; i++)omp_init_lock(&fork[i]);
#pragma omp parallel num_threads(N)
	{
		philosopher();
	}
	for (i = 0; i < N; i++) omp_destroy_lock(&fork[i]);
}