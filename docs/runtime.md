# Runtime: render/frameskip + IPC contract (`start` script)

Operational detail for the `start` runner. `AGENTS.md` keeps the short command
table; ZMQ camera/lidar dispatch details live here.

## Frameskip / render cadence

Current `start` is a fixed-step loop without wall-clock sleep pacing. There is
no `-s` speed flag.

- `-f N`: physics ticks per controller update. The last full command tuple
  `(tau, q_ref, qd_ref, kp, kd)` is held between controller updates (ZOH).
- `-t N`: GUI redraw interval in physics ticks for tact `Env`; render interval
  passed into `mjenv` for `-m`.
- `-l`: headless/no render window.

Real hardware pacing lives in the eio/backend loop. The generic sim path
advances as fast as Python/render allows.

## IPC contract

Bound by `start` over ZMQ IPC on `/dev/shm`:
- PULL `ipc:///dev/shm/default` — incoming commands (whitespace-split string). Words `quit` and `reset` are handled in `start`; everything else is forwarded to `controller.msgproc(w)`.
- PUB `ipc:///dev/shm/proprio` (CONFLATE) — float32 packed proprioceptive vector `y`. Generic `start` publishes every 33 sim ticks and every 16 real-HW ticks.
- PUB `ipc:///dev/shm/<camera-name>` for each camera in `env.cameras`. The runner binds one PUB socket per camera and sends frames from `env.camera_frames()`. `camera_frames()` owns rate-gating and type-to-encoder dispatch: `rgb` → JPEG bytes, `depth` → zstd-compressed float32 metric depth via EGL render. The sim core has no zmq dependency; sockets/send stay in the runner. `kida.run`/`single.run` also bind LAN `tcp://0.0.0.0:<port>` for cameras declaring a `port`.
- PUB `ipc:///dev/shm/<lidar-name>` for each lidar in `env.lidars`. `env.lidar_frames()` owns fps-vs-step gating and ray dispatch. Lidar wire is raw float32, no compression: `2d` → range-along-ray depth map with `-1` no-hit pixels; `3d` → sensor-frame point cloud `(N, 3)`, `N` varying per frame; decode with `np.frombuffer(buf, '<f4').reshape(-1, 3)`.
- PUB `ipc:///dev/shm/<tactile-name>` for each surface-taxel tactile sensor in `env.tactiles`. `env.tactile_frames()` maps the previous tact LCP contact reports to fixed body-local taxel samples; tactile specs with `cell: [du, dv]` integrate 2-point manifolds as line segments and 3+-point manifolds as contact patches over taxel surface cells. `radius` is the receptive margin: it inflates cells for segment/patch overlap and acts as the point-contact splat radius. Wire is raw little-endian float32 `(N, C)`, with `N` = sample count and `C=len(channels)` from YAML; decode with `np.frombuffer(buf, '<f4').reshape(N, C)`.
- Sensor publishing is capability-based. tact `Env` exposes cameras/lidars/tactiles. MuJoCo `CEnv` can publish lidar when `start -m` resolves lidar specs from the project YAML. Real hardware has no sim sensors; external drivers publish compatible topics.
- Without `-d`, every received command is logged to `/dev/shm/out.txt` with the step counter.

The cartpole subdir has its own (identical) `start`; new projects typically copy this script.
