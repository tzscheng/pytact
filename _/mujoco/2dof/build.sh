#!/bin/bash

gcc -W -Wall -I/usr/local/include/mujoco 2dof.c -lmujoco -lGL -lglfw -lm -o 2dof

