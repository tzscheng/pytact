#!/bin/bash

#gcc -W -Wall -o cmm cmm.c -I../native -lm
gcc -O3 -W -Wall -march=native -ffast-math -funroll-loops -o cmm cmm.c -I../native -lm


#g++ -W -Wall -o eigmm eigmm.cpp -I/usr/include/eigen3
g++ -O3 -march=native -ffast-math -funroll-loops -DEIGEN_NO_DEBUG -W -Wall -o eigmm eigmm.cpp -I/usr/include/eigen3

