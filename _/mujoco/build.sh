#!/bin/bash

g++ -shared -I/usr/local/include/mujoco -o mujoco.io -fPIC mjcore.cpp -lmujoco -lGL -lglfw
sudo cp mujoco.io /opt/fg
