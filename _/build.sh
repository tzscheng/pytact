#!/bin/bash

#gcc -W -Wall -shared -fPIC -o libtact.so rbd.c ccd.c penalty.c lcp.c tact.c render.c -lm -lEGL -lGL -lGLEW -lGLU -lglfw -lturbojpeg -lzstd
gcc -W -Wall -O3 -march=native -ffast-math -funroll-loops -shared -fPIC -o libtact.so rbd.c ccd.c penalty.c lcp.c tact.c render.c -lm -lEGL -lGL -lGLEW -lGLU -lglfw -lturbojpeg -lzstd
#sudo cp libtact.so /opt/fg

#standard io
#gcc -W -Wall -shared -o io-std.so io-std.c -fPIC -include ../can/sockcan.h -lm -D ENABLE_CANFD


g++ -shared -I/usr/local/include/mujoco -o mujoco.io -fPIC mjcore.cpp -lmujoco -lGL -lglfw -lturbojpeg
#sudo cp mujoco.io /opt/fg
