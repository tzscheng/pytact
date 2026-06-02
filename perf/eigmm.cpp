#include <iostream>
#include <Eigen/Dense>
#include <ctime>

using Eigen::MatrixXd;
int N = 10;

int main(){
    //MatrixXd m1(N, N);
    //MatrixXd m2(N, N);
    //MatrixXd m3(N, N);

    MatrixXd m1 = MatrixXd::Random(N, N);
    MatrixXd m2 = MatrixXd::Random(N, N);
    MatrixXd m3(N, N); // = MatrixXd::Random(N, N);

    struct timespec t1={0, 0}, t2={0, 0};
    clock_gettime(CLOCK_MONOTONIC, &t1);

    
    for(int i=0; i < 1000000; i++){
	m3 = m1 * m2;
    }


    clock_gettime(CLOCK_MONOTONIC, &t2);


    std::cout << m3 << std::endl;

    
    //printf("time: %lf\n", (double)(t2-t1));
    double sec = ((double)t2.tv_sec+1.0e-9*t2.tv_nsec) - ((double)t1.tv_sec+1.0e-9*t1.tv_nsec);
    printf("time: %.5f sec\n", sec);

    return 0;
}
