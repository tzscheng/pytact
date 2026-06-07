# backend-interface.md — backend core contract + capability ledger

> **Status**: 2026-06-06 제정. core interface N=6 확정, conformance test =
> `tests/test_backend_contract.py` (이 문서의 CORE 마커와 코드를 대조).

## 1. 설계 원칙

Env(tact) 와 CEnv(mujoco/chrono/real) 의 인터페이스 괴리가 자라는 것을 막기 위한
분류 체계. 괴리의 척도는 Env 클래스의 메서드 개수가 아니라 **호출자(start,
kida.run 등 runner / controller)가 backend 에 기대하는 표면**이다.

1. 모든 backend 는 **core interface** 와 **capability(부가 interface)** 두 층을 가진다.
2. core interface (아래 N=6) 는 모든 backend 가 반드시 구현한다.
3. capability 는 backend 마다 기능적으로 구현 가능한 것만 제공한다
   (예: mjenv 는 `mj_ray` 로 `height_scan` 을 구현할 수 있다 — parity 후보).
4. 어떤 capability 를 호출해도 되는지 구분하는 것은 **호출자의 책임**이다
   (관례는 §4). 부재는 정당하며, 빈 stub 으로 메꾸지 않는다.
5. core/capability 분류 없이 표면이 늘지 않도록 conformance test 가 이 문서와
   코드의 일치를 강제한다. 불필요해진 항목은 지속적으로 줄인다.

## 2. Core interface (N=6)

모든 backend (`Env`, `CEnv`×{mujoco, chrono, real}) 가 구현하는 계약.
2026-06-06 기준 4개 backend 전부 이미 만족 — 선언 비용 0 인 절단면.

<!-- CORE: step reset finish backend has_pd dt -->

| 멤버 | 시그니처 / 타입 | 계약 |
|---|---|---|
| `step` | `step(tau=None, q_ref=None, qd_ref=None) → y` | 세 입력 채널은 동등 우선순위·독립 optional. `tau=None` → zero feedforward; `q_ref/qd_ref=None` → 내부 PD 비활성. PD 미지원 backend 는 조용히 무시 (판별은 `has_pd`). 반환 y = 고정 길이 proprio vector (float64 ndarray) |
| `reset` | `reset() → y` | 초기 상태 복원 후, integrator 를 **한 step 도 전진시키지 않은** 진짜 post-reset 관측 반환 (zero-step 보장) |
| `finish` | `finish() → None` | 정리. idempotent. tact 는 no-op |
| `backend` | `str` | free-form 라벨 (`'tact'/'mujoco'/'chrono'/'real'/...`) — controller 가 게인·타이밍 분기에 사용 |
| `has_pd` | `bool` | `q_ref/qd_ref` 를 내부에서 소화하는지. controller 가 자기 책임 하에 override 가능 |
| `dt` | `float \| None` | physics timestep (sec). **None 허용이 계약의 일부** (chrono/real). runner 는 None 이면 rate 도출을 포기하고 controller 자체 pacing 에 맡김 |

관측 채널 구분 (계약의 배경):
- **y (`step()` 반환)** = proprioception — 실기에서 control bus 에 동기로 실려오는
  것 (encoder, IMU, 그리고 f/t 센서·contact switch 처럼 버스에 실리는 신호 포함).
- **sensor topic** = exteroception — 별도 디바이스 드라이버가 비동기 publish 하는
  것 (camera/lidar). sim 에서는 tact 전용 capability (§3) 가 그 드라이버의 대역.
- **oracle 쿼리** (`height_scan` 등) = sim-only GT — 런타임 센서
  아키텍처가 아니라 학습/디버그용 치트 경로. 분류 기준은 proprio/extero 의미가
  아니라 **"실기에서 어느 배선으로 오는가"** (control bus 동기 → y, 드라이버
  비동기 → topic).

## 3. Capability ledger

| capability | 계약 (요지) | tact | mujoco | chrono | real | parity |
|---|---|:-:|:-:|:-:|:-:|---|
| ~~`get_z(x, y)`~~ | **2026-06-06 제거** (sim-trick 최소화): 절대 world-z 는 실기에 존재하지 않는 양. 후속은 `height_scan` + "stance-foot FK z anchor + 상대 Δh" 레시피 (`StepGenerator2/4` 참조). mjenv/chenv 의 C export 는 legacy 잔존 — CEnv 는 이름 차단으로 포워딩 사고 방지 | ✗ | ✗ | ✗ | ✗ | — |
| ~~`get_rgb_image`~~ (+ `get_depth_image`/`get_lidar_image`/`get_lidar_points`) | **2026-06-06 전부 제거** (원칙 (5)): CEnv wrapper 는 live 호출자 0 의 streaming 유물이었고, Env 쪽 4종은 `camera_frames`/`lidar_frames` 로 inline (소비자는 frames() generator 만 읽음; ad-hoc 접근: lidar 는 `_ray_grid`+`clib.tact_raycast_frame` 조합 (recipe 는 `sim.py`; `_raycast_frame` 도 2026-06-06 lidar_frames 로 inline), camera 는 due tick 의 `camera_frames()` (raymap/raycloud/_render_frame 도 2026-06-06 frames() 로 inline)). C export 는 `mjenv.cpp` 에 존치 | ✗ | ✗ | ✗ | ✗ | — |
| `height_scan` | **유일한 terrain 쿼리** — base-relative (G,2) offsets → 상대고도 (G,), `MiniElevationMap.sample` 계약의 GT twin. 실 elevation map 이 주는 양과 동일 형태라 trick-free | ✓ | ✓ (CEnv parity wrapper — mjenv `get_z` C export 를 init-probe 해 instance 에 bind; miss → NaN → validity. probe 실패 backend 는 `hasattr` False) | ✗ | ✗ (소비자는 blind fallback) | **달성 (tact+mujoco)** |
| `cameras` / `lidars` / `camera_frames()` / `lidar_frames()` | 센서 publish 명세 + (name, bytes) frame 공급 | ✓ | ✗ | ✗ | ✗ (실 드라이버가 동일 topic 직접 publish) | **비목표** — sim 전용 드라이버 대역 |
| `add` / `delete` / `groups` / `edit` | 동적 토폴로지 편집 | ✓ | ✗ | ✗ | ✗ | 비목표 — CEnv 에서 부재 (`hasattr` False) + `__getattr__` 차단 리스트로 dlsym 충돌 방어 (§4) |
| `height_scan` / `env.m.*` (fk, jacob, …) | sim-native 도구상자 / oracle (`env.raycast` 는 2026-06-06 height_scan 으로 inline — 유일 소비자; ad-hoc ray 는 `clib.tact_raycast_world` 직접, recipe 는 `sim.py`) | ✓ | (height_scan 은 parity ✓ — 위 row) | ✗ | ✗ | 비목표 |
| `step(kp=, kd=)` per-step PD gains | implicit joint-PD 게인의 유일 채널 (YAML `k:` 2026-06-07 제거 — 게인은 plant 가 아니라 control-policy 입력; `start` 가 controller 의 `kp`/`kd` attr 를 매 tick 읽어 전달; reference 만 있고 게인 없으면 Model.step/CEnv.step 양쪽에서 ValueError). **mujoco 도 동일 인터페이스** (2026-06-07): mjenv `step(tau, q_ref, qd_ref, kp, kd, y)` 가 매 step position actuator 의 gainprm/biasprm 을 stateless 하게 caller 게인으로 씀 — XML 의 kp/kv 는 구조 placeholder 로 격하, nominal save/restore toggle 제거. qd_ref 는 caller kd 로 motor 채널 folding. 단 **수치 동치는 아님** — tact 는 implicit(분모, 무조건 안정 + 유효감쇠 ≈kd+kp·dt), mujoco 는 적분기 의존 explicit. real 은 driver/firmware (kida eio: hardcode) 소관 — eio/chenv 의 step 시그니처는 다음 사용 시 +2 인자 재컴파일 | ✓ | ✓ | ✗ | ✗ (firmware) | **인터페이스 달성 (tact+mujoco)** — 게인 값의 1:1 이전은 비목표 (적분 처리 상이) |
| `get_dt` / `set_redraw` | runner 전용 plumbing — CEnv init 에서 probe, core `dt` 와 render cadence 의 구현 수단 | — | ✓ | ✗ | ✗ | (사용자 호출 대상 아님) |

parity 열: **지향** = 여러 backend 가 같은 계약으로 구현해야 하는 항목 (이름이 같으면
의미·단위·실패 거동까지 같아야 함 — capability 당 한 장의 계약이 원칙). **비목표** =
단일 backend 전용임을 명시 (없는 backend 에서 흉내내지 않는다).

## 4. 호출자 관례

- **probe 는 init 시점에, fail-fast.** capability 가 필요한 controller/runner 는
  생성 시점에 확인하고 없으면 즉시 명확한 에러로 죽는다. 보행 중간의
  `AttributeError` 금지.
- **ledger 에 선언된 capability 는 선언된 메서드로 호출한다.** CEnv 에서 C 심볼 기반
  capability 는 `CEnv.__init__` 의 probe + `argtypes`/`restype` 선언을 마친 메서드를
  통한다 (`get_dt` probe 가 모범). raw `__getattr__` 포워딩은 **per-robot eio 고유 명령의
  정규 통로로 존치한다** (실측: `set_abf` — dog/module; 로봇별 심볼이라 tact 쪽
  allowlist 로 대체 불가). 단 두 가지 제약:
  ① 포워딩으로 부르는 심볼의 시그니처 책임은 그 프로젝트(eio 작성자)에 있다 —
  ctypes 는 argtypes 미선언 `_FuncPtr` 를 그대로 내주므로 double↔int 조용한 marshal
  오답이 가능. ② **tact 전용 이름 (`add`/`delete`/`groups` + sensor publishing 4종)
  은 포워딩에서 차단된다** (`CEnv._TACT_ONLY`) — mutation API 의 짧고 흔한 C 이름이
  미래 backend .so 의 무관한 심볼에 dlsym 으로 붙는 사고 방지.
- 단일 backend 전용 capability (sensor publishing, add/delete) 는 hasattr probe 보다
  **`env.backend` 라벨 분기**가 명시적이다 — `start` 의 sensor socket guard 가 예.

## 5. 검증

`tests/test_backend_contract.py` (self-validating, baseline 없음):
- 이 문서의 `<!-- CORE: ... -->` 마커와 테스트의 CORE 목록 일치
- tact `Env` + fake-cdll `CEnv` 가 core 6 멤버를 올바른 종류(callable/속성·타입)로 제공
- `step`/`reset` 반환 계약 (shape·dtype), `reset` zero-step 보장, `dt=None` 경로
- ledger 분류 일치: tact 전용 7종 (`add`/`delete`/`groups` + sensor 4종) 이 CEnv 에서
  부재 (`hasattr` False) 이면서 `__getattr__` 차단으로 fail-fast, 비차단 cdll 심볼은
  여전히 포워딩 (eio 통로 보존)
