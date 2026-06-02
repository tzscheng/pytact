#!/bin/bash

#3dof
gcc -W -Wall -I/usr/local/include/mujoco 3dof.c -lmujoco -lGL -lglfw -lm -o 3dof
