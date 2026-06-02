#include <stdio.h>
#include <time.h>
#include "tact.h"

#define N 10


double m1[N*N];
double m2[N*N];
double m3[N*N];

int main(void){
    int i;

    srand(time(NULL));
    randmat(m1, N, N);
    randmat(m2, N, N);
    
    struct timespec t1={0, 0}, t2={0, 0};
    clock_gettime(CLOCK_MONOTONIC, &t1);

    
    for (i = 0; i < 1000000; i++) {
	matmul(m3, m1, m2, N, N, N);
	//matmul0(m1, N, N, m2, N, N, m3);
    }
    
    clock_gettime(CLOCK_MONOTONIC, &t2);

    matprint(m1, N, N);
    matprint(m2, N, N);
    matprint(m3, N, N);
    
    //printf("time: %lf\n", (double)(t2-t1));
    double sec = ((double)t2.tv_sec+1.0e-9*t2.tv_nsec) - ((double)t1.tv_sec+1.0e-9*t1.tv_nsec);
    printf("time: %.5f sec\n", sec);

    return 0;
}
