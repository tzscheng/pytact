#!/bin/bash

#basic interface
gcc -W -Wall -o basic basic.c

#rb10 chrono
g++ -march=native -O3 -shared -I../ -I/usr/include/eigen3 -I/usr/include/irrlicht -I/usr/local/include/chrono/collision/bullet -I../ -o chenv.so -fPIC chenv.cpp -lChrono_core -lChrono_irrlicht -lm

