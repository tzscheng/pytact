#!/usr/bin/env -S uv run python
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pyzmq>=25",
# ]
# ///
"""Live terminal heatmap for tactile demo frames.

Run demo.py first so it publishes ipc:///dev/shm/top_taxels, then run this
viewer:

    cd <repo>/fg/tact/tact/demos/tactile
    uv run plot.py
"""
import argparse
import sys
import time

import numpy as np
import zmq

RESET = "\033[0m"
CLEAR_SCREEN = "\033[2J\033[H"
HOME = "\033[H"
CLEAR_LINE = "\033[K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"

PALETTE = [
    (20, 20, 30),
    (20, 70, 130),
    (25, 120, 190),
    (35, 170, 150),
    (130, 190, 70),
    (220, 190, 45),
    (235, 120, 35),
    (220, 55, 45),
    (250, 245, 220),
]

CHANNELS = ["normal", "shear_u", "shear_v", "pressure"]


def bg(rgb):
    r, g, b = rgb
    return f"\033[48;2;{r};{g};{b}m"


def color_for(v, vmax):
    if vmax <= 0:
        t = 0.0
    else:
        t = max(0.0, min(1.0, float(v) / vmax))
    idx = int(round(t * (len(PALETTE) - 1)))
    return PALETTE[idx]


def parse_shape(s):
    try:
        rows_s, cols_s = s.lower().split("x", 1)
        rows, cols = int(rows_s), int(cols_s)
    except Exception:
        raise SystemExit("--shape must look like 20x20")
    if rows <= 0 or cols <= 0:
        raise SystemExit("--shape dimensions must be positive")
    return rows, cols


def channel_index(value):
    if value in CHANNELS:
        return CHANNELS.index(value)
    try:
        idx = int(value)
    except ValueError:
        raise SystemExit(f"--channel must be one of {CHANNELS} or a numeric index")
    if not 0 <= idx < len(CHANNELS):
        raise SystemExit(f"--channel index must be in [0, {len(CHANNELS) - 1}]")
    return idx


def connect(names):
    ctx = zmq.Context.instance()
    poller = zmq.Poller()
    socks = {}
    for name in names:
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.connect(f"ipc:///dev/shm/{name}")
        poller.register(sock, zmq.POLLIN)
        socks[sock] = name
    return poller, socks


def decode_payload(buf, ntaxel, nchan, chan):
    arr = np.frombuffer(buf, dtype="<f4")
    if arr.size == ntaxel:
        return arr
    if arr.size == ntaxel * nchan:
        return arr.reshape(ntaxel, nchan)[:, chan]
    return None, arr.size


def grid_lines(name, values, rows, cols, vmax, channel_name, numbers):
    grid = values.reshape(rows, cols)
    lines = [
        f"{name}",
        f"  channel={channel_name}  rows: y -to+  cols: x -to+",
        f"  sum={grid.sum():8.3f}  max={grid.max():8.3f}  nonzero={(grid > 1e-6).sum():3d}",
    ]
    for row in grid[::-1]:
        blocks = [bg(color_for(v, vmax)) + "  " + RESET for v in row]
        if numbers:
            nums = " ".join(f"{v:6.1f}" for v in row)
            lines.append("  " + "".join(blocks) + "   " + nums)
        else:
            lines.append("  " + "".join(blocks))
    return lines


def render_lines(lines, previous_count):
    out = [HOME]
    for line in lines:
        out.append(line)
        out.append(CLEAR_LINE)
        out.append("\n")
    for _ in range(max(0, previous_count - len(lines))):
        out.append(CLEAR_LINE)
        out.append("\n")
    sys.stdout.write("".join(out))
    sys.stdout.flush()
    return len(lines)


def main():
    ap = argparse.ArgumentParser(description="Live terminal heatmap for tactile sensor frames.")
    ap.add_argument("--names", nargs="+", default=["top_taxels"],
                    help="tactile IPC names to subscribe")
    ap.add_argument("--shape", default="20x20", help="taxel grid shape, e.g. 20x20")
    ap.add_argument("--channels", type=int, default=4,
                    help="payload channel count; demo.yml uses 4")
    ap.add_argument("--channel", default="normal",
                    help=f"channel to display: {', '.join(CHANNELS)} or numeric index")
    ap.add_argument("--hz", type=float, default=20.0, help="terminal refresh rate")
    ap.add_argument("--max", dest="vmax", type=float, default=0.0,
                    help="fixed color max; 0 means auto-scale from current frame")
    ap.add_argument("--timeout", type=float, default=2.0, help="seconds before showing stale data")
    ap.add_argument("--numbers", action="store_true",
                    help="append per-row numeric values; wide for 20x20")
    args = ap.parse_args()

    rows, cols = parse_shape(args.shape)
    ntaxel = rows * cols
    if args.channels <= 0:
        raise SystemExit("--channels must be positive")
    chan = channel_index(args.channel)
    if chan >= args.channels:
        raise SystemExit("--channel index must be smaller than --channels")
    channel_name = CHANNELS[chan] if chan < len(CHANNELS) else str(chan)

    poller, socks = connect(args.names)
    data = {name: np.zeros(ntaxel, dtype=np.float32) for name in args.names}
    stamp = {name: 0.0 for name in args.names}
    period = 1.0 / args.hz if args.hz > 0 else 0.05
    next_draw = 0.0
    drawn_lines = 0

    sys.stdout.write(HIDE_CURSOR + CLEAR_SCREEN)
    sys.stdout.flush()
    try:
        while True:
            events = dict(poller.poll(max(1, int(period * 1000))))
            now = time.time()
            for sock in events:
                name = socks[sock]
                decoded = decode_payload(sock.recv(), ntaxel, args.channels, chan)
                if isinstance(decoded, tuple):
                    _, got = decoded
                    drawn_lines = render_lines([
                        f"{name}: bad payload: {got} floats, expected "
                        f"{ntaxel} or {ntaxel * args.channels}"
                    ], drawn_lines)
                    continue
                data[name] = decoded.copy()
                stamp[name] = now

            if now < next_draw:
                continue
            next_draw = now + period

            active = [name for name in args.names if now - stamp[name] <= args.timeout]
            shown = active if active else args.names
            frame_max = max(float(data[name].max()) for name in shown)
            vmax = args.vmax if args.vmax > 0 else max(frame_max, 1e-6)

            lines = [
                "tactile terminal heatmap  (Ctrl-C to exit)",
                f"sockets: {', '.join(args.names)}",
                f"shape: {rows}x{cols}   payload: float32 ({ntaxel},{args.channels})   "
                f"color max: {vmax:.3f}",
                "",
            ]
            for name in shown:
                if now - stamp[name] > args.timeout:
                    age = "never" if stamp[name] == 0.0 else f"{now - stamp[name]:.1f}s"
                    lines.append(f"{name}  waiting/stale: no frame for {age}")
                    lines.append("")
                else:
                    lines.extend(grid_lines(name, data[name], rows, cols, vmax,
                                            channel_name, args.numbers))
                    lines.append("")
            drawn_lines = render_lines(lines, drawn_lines)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write(SHOW_CURSOR + RESET + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
