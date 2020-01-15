#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "timing.h"

#define NTHR 8
#define N 1000000
#define BLOCKSIZE 64

long vsotaStatic = 0;
long vsotaDynamic = 0;
long vsotaGuided = 0;
long long nextBlockIndex = NTHR * BLOCKSIZE;

int vsotaDeljiteljev(int st) {
	if (st == 2) return 0;
	int vsota = 0;
	for (int i = 2; i < sqrt(st); i++)
		if (st % i == 0)vsota += (i + st / i);
	if ((int)sqrt(st) * (int)sqrt(st) == st) vsota += sqrt(st);
	return vsota + 1;
}

void ompFunctionStatic() {
	int i;
#pragma omp parallel for schedule(static) reduction(+:vsotaStatic)
	for (i = 0; i < N; i ++) {
		int si = vsotaDeljiteljev(i);
		int sj = vsotaDeljiteljev(si);
		if (sj == i && si != i) {
			vsotaStatic += i;
		}
	}
}

void ompFunctionDynamic() {
	int i;
#pragma omp parallel for schedule(dynamic, BLOCKSIZE) reduction(+:vsotaDynamic)
	for (i = 0; i < N; i++) {
		int si = vsotaDeljiteljev(i);
		int sj = vsotaDeljiteljev(si);
		if (sj == i && si != i) {
			vsotaDynamic += i;
		}
	}
}
void ompFunctionGuided() {
	int i;
#pragma omp parallel for schedule(guided) reduction(+:vsotaGuided)
	for (i = 0; i < N; i++) {
		int si = vsotaDeljiteljev(i);
		int sj = vsotaDeljiteljev(si);
		if (sj == i && si != i) {
			vsotaGuided += i;
		}
	}
}
void main(int argc, char args[]) {


	omp_set_num_threads(NTHR);

	//Staticno
	BEGCLOCK(staticno);
	ompFunctionStatic();
	ENDCLOCK(staticno);
	printf("[Static]: Vsota prijateljskh stevil do %d je: %ld\n", N, vsotaStatic);


	//Dinamicno 
	BEGCLOCK(dynamic);
	ompFunctionDynamic();
	ENDCLOCK(dynamic);
	printf("[Dynamic]: Vsota prijateljskh stevil do %d je: %ld\n", N, vsotaDynamic);

	//Guided
	BEGCLOCK(guided);
	ompFunctionGuided();
	ENDCLOCK(guided);
	printf("[Guided]: Vsota prijateljskh stevil do %d je: %ld\n", N, vsotaGuided);
}


/*
Ugotovitve:
	N: 1000000, NTHR: 8, BLOCKSIZE: 16
	Staticno:    2.417000
	Dinamicno:   1.905000
	Guided:      1.975000
	Pohitritev:  1.26x

	N: 100000, NTHR: 12, BLOCKSIZE: 16
	Staticno:    0.094000
	Dinamicno:   0.065000
	Guided:      0.066000
	Pohitritev:  1.45x
	
	N: 1000000, NTHR: 8, BLOCKSIZE: 64
	Staticno:    2.353000
	Dinamicno:   1.878000
	Guided:      1.885000
	Pohitritev:  1.25x

	N: 1000000, NTHR: 16, BLOCKSIZE: 512
	Staticno:    2.384000
	Dinamicno:   1.905000
	Guided:      1.884000
	Pohitritev:  1.15x
*/