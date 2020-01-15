#define _DEFAULT_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/sem.h>
#include <sys/msg.h>
#include <sys/types.h>
#include <err.h>
#include <unistd.h>

#define N 5
#define MEMSIZE 512

union semun {
	int val;
	struct semid_ds *buf;
	unsigned short *array;
} sem_args;

void generiraj_sporocilo(int id, void *buffer){
	if(id==N) sprintf(buffer,"%s", "\0");
	else sprintf(buffer,"ID %d: Sporocilo!\n", id);
}

int main(int argc, char *argv[]){
	int shm, sem, semcontrol;
	key_t key;
	void *pointer;
	int exit;
	sem_args.val = 1;
	
	struct sembuf sestevanje;
	struct sembuf odstevanje;
	
	/* Pristevanje */
	sestevanje.sem_num = 0;
	sestevanje.sem_op = 1;
	sestevanje.sem_flg = 0;
	
	/* Odstevanje */
	odstevanje.sem_num = 0;
	odstevanje.sem_op = -1;
	odstevanje.sem_flg = 0;
	
	if((key = ftok("/tmp", 'a')) < 0) perror("Napaka pri generiranju kljuca");
	if((shm = shmget(key, MEMSIZE, 0664 | IPC_CREAT)) < 0) perror("Napaka pri deljenem polnilniku");
	if((sem = semget(key, 1, 0664 | IPC_CREAT)) < 0) perror("Napaka pri semaforju");
	if((semcontrol = semctl(sem, 1, SETVAL, sem_args)) < 0) perror("Napaka pri vrednosti semaforja.");
	if((pointer = shmat(shm,NULL, 0)) < 0) perror("Napaka pri povezovanju na deljen pomnilnik"); 
	
	for(int i=1; i<=N; i++){
		if( semop(sem, &odstevanje, 1) < 0) perror("Napaka pri odstevanje vrednosti semaforja");
		generiraj_sporocilo(i, pointer);
		sleep(2);
		if( semop(sem, &sestevanje, 1) < 0) perror("Napaka pri sestevanje vrednosti semaforja");
		sleep(2);
	}
	if((exit = shmdt(pointer)) < 0) perror("Napaka pri zapiranju deljenega pomnilnika");
	
	return 0;
}