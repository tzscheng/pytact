#!/usr/bin/env -S uv run python
import sys, os
HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, PROJECT_ROOT)
import socket, numpy as np, tact

def scene_path(name):
    if os.path.isabs(name) or os.path.exists(name) or os.path.exists(name + '.yaml'): return name
    if os.path.sep in name or (os.path.altsep and os.path.altsep in name): return name
    return os.path.join(HERE, name)

def usage():
    print("Usage: ./runner.py <scene>")
    print("Examples:")
    print("  ./runner.py test-convex")
    print("  ./runner.py test-concave")
    print("  ./runner.py box-sphere")

if len(sys.argv) != 2:
    usage()
    sys.exit(0)

scene = scene_path(sys.argv[1])
env = tact.Env(scene, render=True, redraw=4)

#pull = zmq.Context().socket(zmq.PULL)
#pull.bind('ipc:///dev/shm/zmq-pp0')
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind(('0.0.0.0', 6600))
sock.setblocking(False)


mode = [0]*len(env.q) #set all joint torque mode
cnt = 0

def msgproc(msg):
    global cnt
    w = msg.split()
    if not w: return
    if w[0] == 'quit': env.finish(); exit(0)
    elif w[0] == 'reset': env.reset(); cnt = 0

    
while True:
    #try: msg = pull.recv(flags=zmq.NOBLOCK)
    #except zmq.ZMQError: pass
    #else: msgproc(msg); continue
    try: msg, clnt_addr = sock.recvfrom(1024)
    except BlockingIOError: pass
    else: msgproc(msg.decode()); continue
    
    u = np.zeros(env.dof)
    y = env.step(u)
    
    #if len(y) > 0: print(cnt, y)
    #else: print(cnt)
    
    cnt += 1

    #if cnt == 100: break
