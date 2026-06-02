#!/usr/bin/env -S uv run python
"""raymap visualization demo.

Loads raymap_demo.yml (a small scene with one of each shape type), shoots two
depth maps from two named camera frames, and saves the result as side-by-side
images. Pass --show to also open the PNG in the system image viewer.

Usage:
    cd <repo>/fg/tact/examples
    ./raymap_demo.py                  # angular projection, range output (LiDAR-like)
    ./raymap_demo.py --pinhole        # RGB-D camera style (lines stay straight)
    ./raymap_demo.py --perpendicular  # cos-corrected: flat surfaces stay flat
    ./raymap_demo.py --show           # opens the PNG in your image viewer
"""
import os, sys, argparse, time, shutil, subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')   # headless: write PNG, no GUI dependency
import matplotlib.pyplot as plt

# Resolve tact package: examples/ → tact/ → fg/ on sys.path
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(_here)))
import tact


def render(env, frame, width, height, dth, pinhole, perpendicular):
    t0 = time.perf_counter()
    D = env.raymap(frame, width=width, height=height, dth=dth,
                   pinhole=pinhole, perpendicular=perpendicular)
    dt_ms = (time.perf_counter() - t0) * 1000
    n_hit = int((D >= 0).sum())
    return D, dt_ms, n_hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene',  default='raymap_demo', help='YAML name under examples/')
    ap.add_argument('--frame',  default=None, help='frame name (default: render both cam_top and cam_tilt)')
    ap.add_argument('--width',  type=int,   default=120)
    ap.add_argument('--height', type=int,   default=90)
    ap.add_argument('--dth',    type=float, default=0.8, help='degrees per pixel (horizontal)')
    ap.add_argument('--pinhole', action='store_true',
                    help='pinhole / rectilinear projection (RGB-D camera). Default: angular (LiDAR-like).')
    ap.add_argument('--perpendicular', action='store_true',
                    help='cos-correction → camera-Z depth (flat surfaces stay flat). Default: range.')
    ap.add_argument('--show',   action='store_true', help='open the saved PNG in the system image viewer')
    ap.add_argument('--out',    default='raymap_demo.png')
    args = ap.parse_args()

    os.chdir(_here)
    env = tact.Env(args.scene)
    print(f'loaded {args.scene}: {len(env.m.ctype)} shapes, frames={list(env.m.fdict.keys())}')

    frames = [args.frame] if args.frame else ['cam_top', 'cam_tilt']

    proj = 'pinhole' if args.pinhole else 'angular'
    mode = 'perpendicular depth' if args.perpendicular else 'range'
    fig, axes = plt.subplots(1, len(frames), figsize=(6*len(frames), 5), squeeze=False)
    for ax, fname in zip(axes[0], frames):
        D, dt_ms, n_hit = render(env, fname, args.width, args.height, args.dth,
                                 args.pinhole, args.perpendicular)
        Dm = np.where(D >= 0, D, np.nan)
        vmin = float(np.nanmin(Dm)) if n_hit > 0 else 0.0
        vmax = float(np.nanmax(Dm)) if n_hit > 0 else 1.0
        im = ax.imshow(Dm, cmap='viridis_r', vmin=vmin, vmax=vmax,
                       interpolation='nearest', origin='upper')
        ax.set_title(f'{fname} · {proj} · {mode}\n'
                     f'{args.width}×{args.height}, dth={args.dth}° · {dt_ms:.1f} ms · '
                     f'hits {n_hit}/{D.size} · [{vmin:.2f}, {vmax:.2f}] m')
        ax.set_xticks([]); ax.set_yticks([])
        plt.colorbar(im, ax=ax, fraction=0.046, label=f'{mode} (m) — white = miss')

    fig.tight_layout()
    fig.savefig(args.out, dpi=120, bbox_inches='tight')
    print(f'saved {args.out}')
    if args.show:
        opener = next((c for c in ('xdg-open', 'open', 'eog', 'feh') if shutil.which(c)), None)
        if opener: subprocess.Popen([opener, os.path.abspath(args.out)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else: print('--show: no image opener found (xdg-open/open/eog/feh); open the PNG manually.')


if __name__ == '__main__':
    main()
