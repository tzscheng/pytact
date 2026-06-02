# tact 성능 비교: 데스크탑 vs 랩탑 (핸드오프)

작성: 2026-04-30, 데스크탑 세션에서.
**v2** — 데스크탑에서 40% 재현 + vsync OFF 측정 후 가설 갱신.

## 배경 / 의문

`./start zen -e5 -C` + 보행 활성화 시:
- 데스크탑 CPU% ≈ 40%
- 랩탑 CPU% ≈ 38%

랩탑이 미세하게(~5%) 더 효율적으로 보이는 이유.

## 핵심 발견 (데스크탑 측정 후 갱신)

1. **`start` 루프엔 dt 스로틀이 없음** (`zen/start:110-138`). 헤드리스(`-l`)로 돌리면 100% flat-out.
2. **유일한 throttle은 NVIDIA driver 단의 강제 vsync (60Hz)** — `render.c`에 `glfwSwapInterval()` 호출이 **없는데도** 동작. `__GL_SYNC_TO_VBLANK=0`을 주면 풀려서 99%까지 올라감.
3. `_win_render` 경로엔 `glReadPixels`가 없음. PCIe readback은 `-z` 카메라 publish할 때 `egl_render`에서만 발생.
4. **CPU% = (vblank 한 주기당 compute 시간) / 16.67ms**.
5. **랩탑 38% vs 데스크탑 40% = vblank당 compute가 ~0.3ms 짧다** = 약 5% 빠른 단스레드 성능.

### 어제 v1 가설 ("NVIDIA discrete GPU readback") — **폐기**

- vsync OFF에서 데스크탑이 99%로 정상 saturated. driver가 spin/poll하는 게 아님.
- `_win_render`엔 readback 자체가 없음.
- readback 가설은 **`-z` 옵션 시나리오로만 격리됨** (그땐 진짜로 dGPU vs iGPU 차이가 클 가능성).

### 갱신된 가설

랩탑이 5% 빠른 이유는 **단스레드 compute 우위**:
- **Zen 5 (Ryzen 7 PRO 350, 모바일)** vs **Zen 4 (Ryzen 9 7900, 데스크탑)**:
  - Zen 5는 **full-width AVX-512** (Zen 4는 double-pumped 256-bit). `rbd.c`의 `mm66`/`mv66`/`mTm66`/`mTv66` 6×6 inline 커널이 `-march=native -funroll-loops`로 SIMD화 → 핫패스 직접 영향.
  - Zen 5 IPC가 ~10–15% 높음.
  - 부스트 클럭은 데스크탑(5.485GHz)이 더 높지만, IPC + AVX-512 이득이 클럭 격차를 상쇄하고도 약간 남음.
- (보조 요인) GPU driver per-frame submission cost 차이(0.3ms 수준이면 충분).

## 머신 사양

| | 데스크탑 (이미 측정) | 랩탑 (측정 필요) |
|---|---|---|
| CPU | Ryzen 9 7900 (Zen 4, 12C/24T, boost 5.485GHz) | Ryzen 7 PRO 350 (Zen 5, 모바일, ~5.0GHz) |
| GPU | NVIDIA RTX 5060 Ti (discrete) + AMD Raphael iGPU | Radeon 860 (iGPU, RDNA 3.5) |
| RAM | 30GB | 8GB |
| AVX-512 | 있음 (double-pumped) | 있음 (full-width) |

## 데스크탑 측정 결과 (단일 코어 0 pin)

`./start`는 시작 시 `os.sched_setaffinity(0, {0})`으로 코어 0에 핀 → pidstat %CPU는 **단일 코어 기준 100% = 1코어 풀로**.

### 보행 활성화 시 (사용자 시나리오)

```
cd ~/uv/fg/zen
./start zen -e5 -C &
sleep ~6
~/uv/fg/zmqmsg r        # 'r'로 ready 상태로 shift
sleep 2
~/uv/fg/zmqmsg 1        # '1'로 wk1 walking 컨트롤러 활성화
sleep 3                 # 보행 안정화
# 측정
```

| 조건 | CPU% |
|---|---|
| vsync ON (기본) | **40.4%** ← 사용자 관찰값 일치 |
| vsync OFF (`__GL_SYNC_TO_VBLANK=0`) | **99.4%** |

→ vblank(16.67ms) 한 주기당 compute ≈ **6.7ms**, idle wait ≈ 10ms.

### 기타 구성 (참고용)

| 명령어 | CPU% | 비고 |
|---|---|---|
| `./start -l dog` | 100% | 헤드리스, throttle 없음 |
| `./start -l -C dog` | 100% | 동일 |
| `./start dog` (no -C, no walk) | 100% | compute > vblank, vsync 미스 |
| `./start -C dog` (-t 16, 아이들) | 7% | 보행 미활성화. compute 가벼움 |
| `./start -t 1 -C dog` (아이들) | 3% | 매 step 렌더, vsync 매번 |
| `./start -t 100 -C dog` (아이들) | 25% | 100step씩 묶어 렌더 |
| `./start -C zen` (-t 16, 아이들) | 7% | dog와 동일 패턴 |

### 순수 compute 벤치 (단일 코어 핀, 3회 중앙값)

```bash
cd ~/uv/fg/tact/perf
gcc -O3 -W -Wall -march=native -ffast-math -funroll-loops -o cmm cmm.c ../rbd.c -I../ -lm
g++ -O3 -march=native -ffast-math -funroll-loops -DEIGEN_NO_DEBUG -o eigmm eigmm.cpp -I/usr/include/eigen3
for i in 1 2 3; do taskset -c 0 ./cmm; done
for i in 1 2 3; do taskset -c 0 ./eigmm; done
cd ~/uv/fg && taskset -c 0 uv run python ~/uv/fg/tact/perf/npmm.py
```

데스크탑:
- `cmm` (10×10 matmul × 1M iter, rbd.c의 generic matmul): **0.25s**
- `eigmm`: **0.11s**
- `npmm`: **0.78s**

## 랩탑에서 측정해야 할 것

### 1. 보행 시나리오 (핵심)

vsync ON / OFF 둘 다.

```bash
# vsync ON
cd ~/uv/fg/zen && ./start zen -e5 -C &
sleep 6
~/uv/fg/zmqmsg r; sleep 2; ~/uv/fg/zmqmsg 1; sleep 3
PY=$(ps -eo pid,comm,args --no-headers | awk '$2~/^python/ && /\.\/start zen/ {print $1; exit}')
pidstat -p $PY 1 10
pkill -9 -f "uv run python ./start"
rm -f /dev/shm/default /dev/shm/proprio /dev/shm/out.txt

# vsync OFF (랩탑은 mesa이므로 vblank_mode=0; AMD 드라이버는 MESA_GLTHREAD/AMD_DEBUG 등도 있음)
vblank_mode=0 ./start zen -e5 -C &
sleep 6
~/uv/fg/zmqmsg r; sleep 2; ~/uv/fg/zmqmsg 1; sleep 3
PY=$(ps -eo pid,comm,args --no-headers | awk '$2~/^python/ && /\.\/start zen/ {print $1; exit}')
pidstat -p $PY 1 10
pkill -9 -f "uv run python ./start"
```

기대치:
- vsync ON: **38%** 근처 (사용자가 보던 값)
- vsync OFF: **99~100%** (compute saturated)

→ 랩탑 (1) = 38%, (2) = 99%면 가설 확정. compute가 ~5% 빠른 게 핵심 원인.

→ 만약 vsync OFF에서도 차이가 있다면, 그건 step rate를 따로 잡아야 보임 (둘 다 100% near). 그땐 sim cnt를 시간당 출력하는 짧은 instrumentation 필요.

### 2. 순수 compute 벤치

```bash
cd ~/uv/fg/tact/perf
gcc -O3 -W -Wall -march=native -ffast-math -funroll-loops -o cmm cmm.c ../rbd.c -I../ -lm
g++ -O3 -march=native -ffast-math -funroll-loops -DEIGEN_NO_DEBUG -o eigmm eigmm.cpp -I/usr/include/eigen3
for i in 1 2 3; do taskset -c 0 ./cmm; done
for i in 1 2 3; do taskset -c 0 ./eigmm; done
cd ~/uv/fg && taskset -c 0 uv run python ~/uv/fg/tact/perf/npmm.py
```

랩탑 결과 채우기:
- cmm: ___s (데스크탑 0.25s)
- eigmm: ___s (데스크탑 0.11s)
- npmm: ___s (데스크탑 0.78s)

→ Zen 5의 full-width AVX-512가 `eigmm`(SIMD 의존도 높음)에서 더 큰 이득을 보일 가능성. `cmm`(rbd.c의 scalar inner loop)은 비슷할 수도.

## 만약 가설이 확정되면

이건 "버그/병목"이 아니라 **단순한 IPC/SIMD 우위**라서 별 액션 없음. 의미 있는 후속 작업이 있다면:

- (검증) `-z headcam:rgb …` 시나리오는 별도 — 거기선 dGPU vs iGPU readback 차이가 큰 폭으로 나올 수 있음 (어제 v1 가설). 별 측정.
- (튜닝) `rbd.c`의 6×6 inline 커널을 명시적 AVX intrinsics로 작성하면 Zen 4에서도 full-width 효과를 더 끌어낼 여지가 있음. 그러나 `-march=native -funroll-loops`로 이미 vectorize되니까 효과는 marginal.
- (옵션) `render.c`에 `glfwSwapInterval(0)` 명시 + 자체 dt-throttle 도입 → vsync에 안 묶이고 정확한 sim rate 보장. 별개 개선.

## 관련 파일/위치

- 렌더 호출 경로: `tact/render.c:850 win_render`, `glfwSwapBuffers` at line 1034
- 헤드리스 토글: `dog/start:73-74`, `zen/start:73-74` (`-l` 플래그)
- redraw 간격: `tact/tact.py:2089` (`if self.render and self.cnt % self.redraw == 0`)
- C dynamics 핫패스: `tact/rbd.c` (`matmul`, `mm66`/`mv66`/`mTm66`/`mTv66` 6×6 inline 커널)
- ZMQ 명령어 도구: `~/uv/fg/zmqmsg`
- 보행 컨트롤러: `zen/zen.py:40-49 msgproc` (`r` → shift to ready, `1` → wk1 활성화)

## 데스크탑에서 이미 확인된 것

- `lscpu`, `nvidia-smi`, `lspci`, `free -h` 결과 (위 표 참조)
- `ldconfig -p | grep EGL` → `libEGL_nvidia.so.0`, `libEGL_mesa.so.0` 둘 다 설치됨
- 다양한 `./start` 구성 CPU% 측정 (위 표 참조)
- 보행 시나리오 vsync ON/OFF 측정 (40.4% / 99.4%)
- perf 벤치 cmm/eigmm/npmm 단일 코어 핀 결과

## 랩탑에서 이어서 할 일

1. 위 [측정해야 할 것 1, 2] 실행해서 표 채우기.
2. (1)에서 vsync ON ≈ 38%, vsync OFF ≈ 99%면 → "Zen 5 compute 우위" 가설 확정.
3. 차이가 다른 곳(예: vsync OFF에서도 큰 격차)에서 나면 → 추가 instrumentation으로 step rate 직접 측정.
4. 결과 따라 후속 작업 결정 (대개는 단순 결론으로 마무리).
