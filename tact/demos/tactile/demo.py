#!/usr/bin/env -S uv run python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pyzmq>=25",
#   "pyyaml>=6.0",
# ]
# ///
# -*- mode: python -*-
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import tact
import zmq


def main():
    ap = argparse.ArgumentParser(description="Run tactile demo and publish tactile frames over /dev/shm.")
    ap.add_argument("scene", nargs="?", default="demo",
                    help="scene name/path, default: demo")
    ap.add_argument("-l", "--headless", action="store_true", help="disable render window")
    ap.add_argument("-t", "--redraw", type=int, default=4, help="render redraw interval")
    ap.add_argument("--amp", type=float, default=18.0, help="pitch/roll amplitude in degrees")
    ap.add_argument("--freq", type=float, default=0.35, help="pitch/roll frequency in Hz")
    ap.add_argument("--phase", type=float, default=90.0, help="roll phase lead in degrees")
    ap.add_argument("--kp", type=float, default=1000.0, help="joint position gain")
    ap.add_argument("--kd", type=float, default=40.0, help="joint velocity gain")
    args = ap.parse_args()

    scene = args.scene
    if not os.path.isabs(scene) and not os.path.exists(scene) and not os.path.exists(scene + ".yml"):
        scene = os.path.join(HERE, scene)

    env = tact.Env(scene, render=not args.headless, redraw=args.redraw)

    ctx = zmq.Context()
    tactile_socks = {}
    for spec in getattr(env, "tactiles", []):
        sock = ctx.socket(zmq.PUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.bind("ipc:///dev/shm/%s" % spec["name"])
        tactile_socks[spec["name"]] = sock

    print("publishing tactiles:", ", ".join(tactile_socks) if tactile_socks else "(none)")
    print(f"driving pad joints: amp={args.amp:g} deg  freq={args.freq:g} Hz")
    print("Ctrl-C or ESC in render window to exit")

    cnt = 0
    amp = np.deg2rad(args.amp)
    phase = np.deg2rad(args.phase)
    omega = 2.0 * np.pi * args.freq
    kp = np.full(env.dof, args.kp)
    kd = np.full(env.dof, args.kd)
    try:
        while True:
            t = cnt * env.dt
            pitch = amp * np.sin(omega * t)
            roll = amp * np.sin(omega * t + phase)
            pitch_d = amp * omega * np.cos(omega * t)
            roll_d = amp * omega * np.cos(omega * t + phase)
            q_ref = np.array([pitch, roll])
            qd_ref = np.array([pitch_d, roll_d])
            env.step(q_ref=q_ref, qd_ref=qd_ref, kp=kp, kd=kd)

            for name, buf in env.tactile_frames():
                tactile_socks[name].send(buf)

            cnt += 1
    except KeyboardInterrupt:
        pass
    finally:
        env.finish()


if __name__ == "__main__":
    main()
