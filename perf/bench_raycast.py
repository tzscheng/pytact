#!/usr/bin/env -S uv run python
# Single-thread raycast benchmark for the LiDAR depth path (tact_raycast_batch).
#
# Measures the pure raycast cost (cached _ray_grid + _raycast_batch), NOT the wire encoder — zstd is
# a separate, unaffected stage. Two scenes:
#   syn : synthetic scene, N raycast shapes scattered around the sensor (dial N to stress
#         the per-ray shape loop; the real dog scene is low-N).
#   dog : the actual `./start dog -e steps1` scene (dog model + steps1 terrain). Only the
#         terrain boxes are raycast-on (the robot's own shapes opt out), so N is small.
#
# Pins to core 0 for stable numbers. Reports median ms/frame over `reps` blocks of `iters`.
# --save-ref / --check-ref give a bitwise correctness gate: every optimization stage must
# reproduce the baseline output exactly (a correct frustum cull / reject changes nothing).
#
# Run from a dir where `import tact` works (e.g. fg/dog or fg). Examples:
#   uv run python ../tact/perf/bench_raycast.py --scene dog --save-ref /tmp/ref_dog.npy
#   uv run python ../tact/perf/bench_raycast.py --scene syn --n 100 --save-ref /tmp/ref_syn100.npy
import os, sys, time, argparse, tempfile
# Run as a script, sys.path[0] is this file's dir (tact/perf) so `import tact` self-imports
# and fails (same trap start.py handles). Prepend cwd so a sibling `tact` (e.g. fg/dog/tact
# symlink, or fg/) resolves as the package.
sys.path.insert(0, os.getcwd())
import numpy as np

MESH_OBJ = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'examples/objs/10.obj')

def gen_syn_yml(n, seed, path, kind='mix'):
    """N raycast shapes on a shell around the sensor. Shell radius [1,6] m, random sizes —
    a forward-looking 128deg FoV sees ~a third (gives the frustum cull something to cull)
    at varied depth. Fixed seed → reproducible placement.
      kind='mix'  : box/sphere/cylinder mix (cheap primitives).
      kind='mesh' : all 10.obj (5120-face) meshes — the ray-sphere reject's payoff case,
                    since a non-rejected mesh runs a 5120-triangle loop per ray."""
    rng = np.random.default_rng(seed)
    lines = []
    lines.append("materials:")
    lines.append("    mat1: {normal: [20000, 200], tangent: [20000, 200, 1.0], spin: [200, 2, 0.05], roll: [200, 2, 0.02], restitution: 0.0}")
    lines.append("bodies:")
    lines.append("  - name: root")
    lines.append("    shapes:")
    for i in range(n):
        v = rng.normal(size=3); v /= np.linalg.norm(v)   # uniform direction on sphere
        p = v * rng.uniform(1.0, 6.0)
        p[2] += 0.5  # sensor is at z=0.5
        pos = f"[{p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}]"
        if kind == 'mesh':
            shp = f"{{type: mesh, pos: {pos}, file: {MESH_OBJ}, rgba: [0.7,0.7,0.7,1.0]}}"
        else:
            k = i % 3
            if k == 0:
                s = rng.uniform(0.15, 0.5, size=3)
                shp = f"{{type: box, pos: {pos}, param: [{s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f}], rgba: [0.7,0.7,0.7,1.0]}}"
            elif k == 1:
                shp = f"{{type: sphere, pos: {pos}, param: [{rng.uniform(0.15,0.5):.3f}, 0, 0], rgba: [0.7,0.7,0.7,1.0]}}"
            else:
                shp = f"{{type: cylinder, pos: {pos}, param: [{rng.uniform(0.1,0.3):.3f}, {rng.uniform(0.2,0.6):.3f}, 0], rgba: [0.7,0.7,0.7,1.0]}}"
        lines.append(f"      - {shp}")
    lines.append("lidars:")
    lines.append("  - {name: synlidar, type: 2d, body: root, pos: [0, 0, 0.5], euler: [0, -90, 0], eulerseq: xyz, res: [320, 240], dth: 0.4, fps: 30, pinhole: true}")
    open(path, 'w').write("\n".join(lines) + "\n")

def build_env(args):
    import tact
    if args.scene == 'dog':
        env = tact.Env('dog', render=False)
        env.add(tact.pkg_dir + '/examples/steps1')
        frame = 'lidar1'
    else:
        base = os.path.join(tempfile.gettempdir(), f'syn_{args.shapes}_{args.n}_{args.seed}')
        gen_syn_yml(args.n, args.seed, base + '.yml', kind=args.shapes)  # Env(name) opens name + '.yml'
        env = tact.Env(base, render=False)
        frame = 'synlidar'
    env.reset()
    return env, frame

def scene_info(env):
    cast = np.asarray(env.m.craycast)
    on = int(cast.sum())
    from collections import Counter
    names = {100:'mesh',101:'box',102:'sphere',103:'cyl',104:'cap'}
    hist = Counter(names.get(env.m.ctype[i], env.m.ctype[i]) for i in range(len(env.m.ctype)) if cast[i])
    return len(cast), on, dict(hist)

def raymap(env, frame, w, h, dth, pinhole):
    """Depth map from the primitives (raymap() was inlined into lidar_frames
    2026-06-06; this measures the same pre-encoding pipeline: cached _ray_grid
    + one tact_raycast_batch call)."""
    return env._raycast_batch(frame, env._ray_grid(w, h, dth, pinhole)).reshape(h, w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--scene', choices=['syn','dog'], default='syn')
    ap.add_argument('--shapes', choices=['mix','mesh'], default='mix', help='synthetic shape kind')
    ap.add_argument('--n', type=int, default=100, help='synthetic raycast shape count')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--w', type=int, default=320); ap.add_argument('--h', type=int, default=240)
    ap.add_argument('--dth', type=float, default=0.4)
    ap.add_argument('--pinhole', type=int, default=1)
    ap.add_argument('--iters', type=int, default=300)
    ap.add_argument('--warmup', type=int, default=30)
    ap.add_argument('--reps', type=int, default=5)
    ap.add_argument('--save-ref', default=None)
    ap.add_argument('--check-ref', default=None)
    args = ap.parse_args()

    try: os.sched_setaffinity(0, {0})
    except Exception: pass

    env, frame = build_env(args)
    nsh, on, hist = scene_info(env)
    ph = bool(args.pinhole)

    # warmup (also triggers any mesh lazy-load before timing)
    for _ in range(args.warmup):
        D = raymap(env, frame, args.w, args.h, args.dth, ph)

    if args.save_ref:
        np.save(args.save_ref, D); print(f"saved ref -> {args.save_ref}")
    if args.check_ref:
        ref = np.load(args.check_ref)
        diff = float(np.max(np.abs(D - ref)))
        nan_mismatch = int(np.sum(np.isnan(D) != np.isnan(ref)))
        ok = (diff == 0.0 and nan_mismatch == 0)
        print(f"correctness vs {args.check_ref}: max|diff|={diff:g} nan_mismatch={nan_mismatch} -> {'IDENTICAL' if ok else 'DIFFERS'}")

    # timed: reps blocks of iters; per-frame ms = block_time/iters
    per_frame_ms = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        for _ in range(args.iters):
            raymap(env, frame, args.w, args.h, args.dth, ph)
        dt = time.perf_counter() - t0
        per_frame_ms.append(dt / args.iters * 1e3)
    per_frame_ms = np.array(per_frame_ms)
    med = float(np.median(per_frame_ms))

    scene_str = args.scene + (f"-{args.shapes}(N={on})" if args.scene=='syn' else "(steps1)")
    print(f"scene={scene_str} res={args.w}x{args.h} dth={args.dth} pinhole={ph} "
          f"| n_shape={nsh} raycast_on={on} types={hist}")
    print(f"  per-frame ms: median={med:.3f}  min={per_frame_ms.min():.3f}  max={per_frame_ms.max():.3f}  "
          f"({1e3/med:.1f} fps)  [reps={args.reps} iters={args.iters}]")

if __name__ == '__main__':
    main()
