#!/bin/bash
# walking_perf.sh — zen walking sim에 perf stat 붙여서 하드웨어 카운터 잡기
# 데스크탑 vs 랩탑 IPC/cache/branch 비교용. 양쪽에서 그대로 실행.
#
# 사용:
#   bash walking_perf.sh           # 기본 (vsync 그대로)
#   bash walking_perf.sh --no-vsync # vsync 끄고 (순수 compute 비교)
#
# 결과는 stdout + /tmp/walking_perf_<host>_<vsync>.txt 에 저장됨.

set -u

NO_VSYNC=0
[ "${1:-}" = "--no-vsync" ] && NO_VSYNC=1

HOST=$(hostname)
TAG="$([ $NO_VSYNC -eq 1 ] && echo novsync || echo vsync)"
OUT="/tmp/walking_perf_${HOST}_${TAG}.txt"

# 종료 시 정리 (Ctrl-C 포함)
cleanup() {
    pkill -9 -f "venv.*python.*start" 2>/dev/null || true
    rm -f /dev/shm/default /dev/shm/proprio /dev/shm/out.txt 2>/dev/null || true
    if [ -n "${PARANOID_ORIG:-}" ]; then
        sudo sysctl -w kernel.perf_event_paranoid="$PARANOID_ORIG" >/dev/null 2>&1 || true
    fi
}
trap cleanup EXIT INT TERM

echo "=== walking_perf @ $HOST ($TAG) ==="
echo "출력 파일: $OUT"
echo

# 1) perf 권한
PARANOID_ORIG=$(cat /proc/sys/kernel/perf_event_paranoid)
echo "[1/5] perf_event_paranoid: $PARANOID_ORIG → 0 (sudo 필요)"
sudo sysctl -w kernel.perf_event_paranoid=0 >/dev/null

# 2) 환경 정보
echo "[2/5] 환경 정보 수집"
{
    echo "=== HOST: $HOST ==="
    echo "=== date: $(date) ==="
    echo "=== mode: $TAG ==="
    echo
    echo "--- CPU ---"
    grep -m1 "model name" /proc/cpuinfo
    echo "max boost: $(cat /sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq) kHz"
    echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"
    echo "EPP: $(cat /sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference 2>/dev/null || echo n/a)"
    echo "amd_pstate: $(cat /sys/devices/system/cpu/amd_pstate/status 2>/dev/null || echo n/a)"
    echo
} > "$OUT"

# 3) 클린 + sim 띄우기
echo "[3/5] zen sim 시작 + walking 활성화"
pkill -9 -f "venv.*python.*start" 2>/dev/null || true
sleep 2
rm -f /dev/shm/default /dev/shm/proprio /dev/shm/out.txt 2>/dev/null

VSYNC_ENV=""
if [ $NO_VSYNC -eq 1 ]; then
    # NVIDIA와 mesa 양쪽 env var 다 세팅 (해당 안 되는 건 무시됨)
    VSYNC_ENV="__GL_SYNC_TO_VBLANK=0 vblank_mode=0"
fi

cd ~/uv/fg/zen
env $VSYNC_ENV ./start zen -e5 -C > /tmp/zen_perf.log 2>&1 &
SIM_BG=$!
disown $SIM_BG 2>/dev/null || true

sleep 6
~/uv/fg/zmqmsg r
sleep 2
~/uv/fg/zmqmsg 1
sleep 3

# 4) PID 찾고 perf 실행
PY=$(ps -eo pid,comm,args --no-headers | awk '$2~/^python/ && /\.\/start zen/ {print $1; exit}')
if [ -z "$PY" ]; then
    echo "ERROR: sim 프로세스 못 찾음. /tmp/zen_perf.log 확인."
    tail -10 /tmp/zen_perf.log >> "$OUT"
    exit 1
fi
echo "[4/5] sim PID = $PY, perf stat 20초 측정"
echo "--- perf stat output ---" >> "$OUT"

perf stat -p "$PY" \
    -e task-clock,cycles,instructions,branches,branch-misses,L1-dcache-load-misses,LLC-load-misses \
    sleep 20 2>> "$OUT"

# 코어 0 클럭 한 번 더 캡쳐 (측정 직후)
echo "" >> "$OUT"
echo "--- post-measure cpu0 freq ---" >> "$OUT"
echo "scaling_cur_freq: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq) kHz" >> "$OUT"

# 5) 마무리 (cleanup trap이 sim kill + paranoid 복원)
echo "[5/5] 측정 완료"
echo
echo "=========== 결과 ==========="
cat "$OUT"
echo "============================"
echo
echo "이 파일을 공유: $OUT"
