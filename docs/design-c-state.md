# C-side state object 도입 설계 (`tact_t`)

> **Status**: 이 문서는 `tact_t` 핸들 **도입 당시(2026-05 이전)의 planning 기록**이다. C 경로는 이제 항상 활성이고 `use_c` 외부 옵션은 제거됐다(`model.use_c = True` 하드코딩, parity용 내부 토글로만 유지). **살아있는 계약은 §3.5 Lifecycle invariants** — 코드(`_clib.py`, `native/tact.{c,h}`, `sim.py`)가 `§3`/`§3.5`를 직접 인용한다. 나머지 본문은 historical이라 아래 옛 식별자를 쓴다 — 현행 매핑:
>
> | 본문 (planning 당시) | 현행 |
> |---|---|
> | `ccd.c`, `common.h` | `native/tact.{c,h}` (+ collision 측은 `narrow.c`/`mpr.c`/`ray.c`/`shape.c`로 split) |
> | `contact_ccd` | `collision_check` (narrow-phase dispatch) |
> | `tact_step` (thin wrapper) | `tact_step_lcp` (LCP 1-step, λ caller-threaded) |
> | `_cache_c_pointers` | `_create_c_handle` |
> | `step(q, qd, _tau, cff=...)` + `use_c` 분기 | `step(q, qd, tau, q_ref, qd_ref, kp, kd, ctx)` — 순수 함수, `design-pure-step.md` 참조 |
> | `mu` 0.8 hardcoded | per-material YAML `materials:` (`design-lcp-perf.md`) |

## 1. Motivation

현재의 C 포팅 패턴 — "Python이 모든 상태를 들고, C 함수는 매 호출마다 포인터/배열을 인자로 받는 stateless 라이브러리" — 이 측정상 천장에 부딪혔다.

오늘까지 측정 (µs/call, 워밍업 후):

| 항목 | nb=2 | nb=18 | 관찰 |
|---|---:|---:|---|
| `gravity()` Python | 35 | 295 | nb에 선형 |
| `gravity()` C-path | 3.5 | 4.8 | **nb와 거의 무관, 평탄** |
| `fk()` 3d C-path | 5.9 | 7.0 | 마찬가지로 평탄 |
| `step()` C-path | (미측정) | (미측정) | `_fk` + `contact_ccd` + `euler_step` 3번 ctypes 왕복 |

C 경로의 시간이 **nb에 거의 평탄**하다는 것은 비용의 대부분이 C 본체가 아니라 **호출 표면**에 있다는 뜻이다. 구체적으로:

- 매 호출 `np.ascontiguousarray` + `data_as` (q, qd, tau, g 등).
- ctypes argument marshaling — `step()`은 17개 포인터/스칼라를 매번 push.
- step() 안에서 `_fk` → `contact_ccd` → `euler_step`이 **각각 따로** ctypes 경계를 넘음.
- `feedback()`의 14개 case 루프가 Python에 남아 nb에 선형 비용을 추가.

`_cache_c_pointers()`에 30개 넘는 attribute가 쌓이는 것 자체가 "C 측에 진짜 state object가 있어야 한다"는 신호.

## 2. Goals / Non-goals

**Goals**

- C 측에 인스턴스 상태를 담는 `tact_t` 구조체를 두고, hot-path 함수가 그 핸들 하나만 받게 만든다.
- `step()` 전체를 single C call로 묶어 ctypes 왕복을 1회로 줄인다 (현재 4회).
- `feedback()`의 14 case 루프를 C로 내려, nb에 선형인 Python 비용을 제거한다.
- 측정 가능한 목표: `step()` C-path가 **nb에 평탄에 가까운** 비용 곡선을 보일 것 (예: nb=18에서 현재 추정치의 1/3 이하).

**Non-goals**

- MuJoCo의 `mjModel`/`mjData` 풀 카피 — immutable build, JIT recompile 같은 깊은 구조는 도입하지 않는다.
- YAML 로더의 C 이식 — 모델 build phase는 Python에 남긴다 (yaml + frame 합성 + prefix renaming의 유연성을 깨고 싶지 않다).
- `gravity()` / `fk()` 같은 single-shot query의 추가 최적화 — step() ROI가 압도적이라 우선 step()에 집중.
- 기존 use_c=False 경로 제거 — Python reference path는 parity 검증용으로 유지.

**Guiding principle: Python-first by default**

C 코어는 어디까지나 **step 가속 전용**이다. 새 기능을 추가할 때 기본 위치는 Python이고, C로 내릴지는 다음 두 조건을 모두 만족할 때만 결정한다:

1. **Hot path임이 측정으로 입증** — 매 step 호출되거나 inner loop에 포함된다.
2. **Python 구현 시 nb에 비례하는 지속 비용** — 단발 setup 비용은 해당 없음.

Hot path가 아니면 Python에 남긴다. C struct가 비대해지기 시작하는 순간(예: YAML schema 필드 ↔ struct 필드 1:1 미러링 유혹) 정체성 침범의 신호로 간주하고 멈춘다. tact의 차별점 — YAML 스키마, controllers (pid/jtc/ctc/hyc), sim/real 통일, swappable backend (MuJoCo/Chrono/real), 컨트롤러 라이브러리 — 는 모두 Python에 남는 것이 원칙.

이 원칙에 따라 Phase 3에서 추가 함수의 C 이식 여부는 자동 적용이 아니라 **개별 측정 후 판단**한다.

## 3. Proposed design

### 3.1 Build vs runtime split

- **Build phase (Python)**: 기존 `model.__init__` + `add()` + `edit()` 그대로. YAML을 Python 자료구조로 펼치고, frame registry/contact pair 등을 채운다. `use_c=True`이면 마지막에 `tact_create(self, ...)` 호출로 C 측 struct에 일괄 복사.
- **Runtime phase (C)**: 빌드 끝난 시점부터는 `tact_step` 등 hot-path 함수가 **핸들만** 받아 내부에서 모든 일을 처리. q/qd/tau만 in/out으로 marshal.

이 분기는 MuJoCo의 `mjModel` (build 시 만들고 이후 immutable) + `mjData` (runtime mutable) 분리와 같은 정신.

### 3.2 `tact_t` 구조체 (C)

기존 `_cache_c_pointers()`가 들고 있던 것을 그대로 옮긴다:

```c
typedef struct {
    /* fixed model (build phase에서 채워지고 step 동안 read-only) */
    int     nb;
    int    *parent;          /* nb */
    int    *jtype;            /* nb */
    double *X;                /* 36*nb (spatial transforms) */
    double *I6;               /* 36*nb (spatial inertias) */
    double *Ti;               /* 16*nb (per-body initial transforms) */
    double  g[3];
    double  dt;
    int     solver;           /* 1=euler, 2=rk4 */

    /* contact (build phase에 결정) */
    int     n_shape;
    int     n_pair;
    int    *ctype;            /* n_shape */
    int    *cbody;            /* n_shape */
    double *cshape;           /* 3*n_shape */
    double *ctran;            /* 16*n_shape */
    double *cparam;           /* 4*n_shape */
    int    *cpair;            /* 2*n_pair */
    double  mu;               /* 현재 0.8 hardcoded; YAML에서 읽도록 추후 확장 */

    /* dynamic state (step마다 갱신) */
    double *T;                /* 16*nb (FK 결과) */
    double *f_ext;            /* 6*nb (contact_ccd 결과) */
    double *f, *a, *v;        /* 6*nb (rne 결과) */
    double *qdd;              /* nb */
    double *cfs_buf;          /* 8*max(n_pair,1) */
    int     cfs_count;

    /* feedback descriptor (build phase) */
    int     n_feeds;
    int    *feeds;            /* flat encoding of (kind, frame_idx[]) */

    /* frame registry (build phase) — feedback이 ftran을 참조 */
    int     n_frames;
    int    *fbody;            /* n_frames */
    double *ftran;            /* 16*n_frames */
    double *ftran_inv;        /* 16*n_frames (case 14에서만 쓰임 — 미리 계산) */

    /* scratch */
    double *workspace;        /* max(120,12)*nb */
} tact_t;
```

소유: `tact_create`가 모든 buffer를 단일 arena에서 할당하고, `tact_destroy`가 한 번에 해제. Python 측은 `ctypes.c_void_p` 핸들만 잡고 있는다 — 30개의 numpy attribute가 1개의 핸들로 압축된다.

### 3.3 Public C API (1차 마일스톤)

```c
tact_t *tact_create(/* 모든 build-phase 데이터 */);
void    tact_destroy(tact_t *h);

/* hot path: step 전체를 single call. q,qd,tau는 caller 소유 */
void    tact_step(tact_t *h,
                  const double *q, const double *qd, const double *tau,
                  double *q_next, double *qd_next,    /* out: nb each */
                  double *y, int *n_y,                 /* out: feedback vector */
                  double *cfs, int *n_cfs);            /* out: 8*n_cfs doubles */

/* read accessors — render/디버깅용 (hot path 아님) */
const double *tact_get_T(tact_t *h);     /* 16*nb */
const double *tact_get_v(tact_t *h);     /* 6*nb */
const double *tact_get_f_ext(tact_t *h); /* 6*nb */
```

Python 측 `model.step`은 이 single call로 줄어든다:

```python
def step(self, q, qd, _tau, cff=False):
    tau = _tau - self.ff*qd - self.sk*q
    if self.is_locked: ...
    if self.use_c:
        tact.tact_step(self._h, q.ctypes..., qd.ctypes..., tau.ctypes...,
                       None, None, None, None,
                       self._q_next.ctypes..., self._qd_next.ctypes...,
                       self._y_buf.ctypes..., self._n_y,
                       self._cfs_buf.ctypes..., self._n_cfs)
        # cfs, y는 미리 잡아둔 buffer에서 slice
    else:
        ... (기존 Python 경로 그대로)
```

ctypes 왕복: 4회 → 1회. feedback 루프: Python 14-case → C 14-case (작성 1회).

### 3.4 feedback의 C 이식

`feedback`은 14개 case가 있고 각 case가 frame_idx별로 다른 양을 뽑아 y에 append한다. Python에선 list append + np.cross / matmul이 case마다 산재. C에선:

- build phase에서 `feeds`를 `(kind, n, frame_idx[n])` 시퀀스로 평탄화해 `tact_t.feeds`에 저장.
- `tact_step` 안에서 step 끝에 단일 루프로 y 출력 버퍼에 직접 기록.
- y의 길이는 build phase에서 결정 가능 → Python은 미리 `np.empty(n_y)` 잡아두고 그 포인터만 넘김.

이 부분이 사실 ROI가 가장 큰 곳일 수 있다 (case 6/8/12/14가 cross product 여러 번 + matmul을 매 step 호출).

### 3.5 Lifecycle invariants

핸들 기반으로 가면 외부에서 raw pointer/view를 들고 있는 코드가 dangling 위험을 만난다. 다음 invariant를 명문화한다:

1. **`add()` / `edit()`은 핸들 무효화 사건이다.** 호출 직후 내부 동작은 `tact_destroy(self._h)` + `tact_create(self._h, ...)`로 동일 메모리 위치에 다시 만든다 — 그러나 그 시점에 외부에서 들고 있던 `tact_get_T(h)` 등의 view는 모두 dangling으로 간주.

2. **외부 view는 매 변형 후 재취득**. 렌더러/컨트롤러가 model에서 raw view를 받아 캐시했다면, `add()`/`edit()` 직후 다시 받아야 한다. 호출자 책임. 현 코드베이스 기준 영향 받는 곳은 `sim` 클래스의 렌더 경로 1~2곳 — convention으로 충분하고 자동 invalidation 콜백은 도입하지 않는다.

3. **호출자 API는 그대로**. `m.add(...)` / `m.edit(...)` 자체는 Python에서 보면 변화 없음. 핸들 destroy/recreate는 그 메서드 끝에서 자동 처리 (`_create_c_handle()` 호출 — 옛 `_cache_c_pointers()` 자리).

4. **edit()가 hot loop에 들어오는 경우**는 별도. 현재 호출자(grep 결과)는 setup-time 사용이 대부분이라 destroy/recreate 비용을 감수할 만함. 만약 미래에 동적 페이로드 변경처럼 매 step `edit()`를 부르는 시나리오가 생기면 `tact_update_inertia(h, body_idx, ...)` 같은 핀포인트 setter를 별도 추가 — 그때 가서 측정 후 결정.

## 4. Migration plan

세 단계로 점진. 매 단계 끝에 parity test 통과가 게이트.

**Phase 1 — `tact_create`/`tact_destroy` + `tact_step`**

- C: `tact_t` 구조체 정의, `tact_create` (Python에서 raw 배열들을 받아 arena에 복사), `tact_destroy`.
- C: `tact_step`을 기존 `_fk` + `contact_ccd` + `euler_step`/`rk4_step`을 차례로 호출하는 thin wrapper로 작성. feedback은 일단 Python에 남기고 `tact_get_T`로 읽어옴.
- Python: `model._cache_c_pointers` → `_create_c_handle`로 개명, 핸들만 들고 있도록 단순화. 기존 numpy buffer attribute는 점진적으로 제거.
- 측정: 1차 win은 ctypes 왕복 4→1.

**Phase 2 — feedback의 C 이식**

- build phase에서 feeds/frames/ftran_inv 전부 C struct에 평탄화.
- C `tact_feedback(tact_t *h, double *y_out)` 작성, `tact_step`에서 호출.
- Python `feedback()`은 use_c=False 경로용으로만 남김.

**Phase 3 — 정리 & 추가 함수 이식 검토**

- `gravity()` / `fk()` / `jacob()` / `error()`도 같은 패턴으로 핸들 기반으로 전환할지 판단. 단발 호출이라 Phase 1/2만큼 ROI는 크지 않을 수 있음 — 측정 후 결정.
- `add()` / `edit()`이 호출되면 핸들을 destroy + recreate (Phase 1에서는 immutable로 가정).

## 5. Open questions / risks

- **contact pair 가변 길이 cfs**. n_pair는 build phase에 결정되지만, 실제 충돌하는 페어 수(cfs_count)는 매 step 다르다. `cfs_buf`를 max 크기로 잡고 count만 in/out으로 가져오는 패턴 (현재와 동일) 으로 유지.

- **`y` 출력 버퍼 크기**. feedback 결과 y의 길이는 build phase에서 결정 — feeds를 훑어 미리 계산 가능. Python은 `self._y_buf = np.empty(n_y)` 한 번 잡아두고 매 step에 동일 buffer 재사용.

- **ctypes 한계**. ctypes로 struct를 다루기 시작하면 Python측 mirror struct 정의가 또 boilerplate가 된다. 단순한 `void*` 핸들로만 다루고, 모든 접근은 함수 호출로 가는 것이 깔끔. 이미 그렇게 설계됨.

- **렌더 측 T 접근의 view 캐싱 빈도**. 현재 `sim`은 매 redraw 시 fresh로 T를 받는 패턴이라 view 캐싱이 없으면 invariant (3.5) 만족 — 그러나 향후 controller가 T view를 캐시하기 시작하면 그 시점에 invalidation을 명시적으로 해야 함. 코드 리뷰 게이트로 둠.

(라이프사이클 / 정체성 / Python-first 원칙은 §2와 §3.5에서 결정됨)

## 6. 첫 PR 범위 제안

위 Phase 1만:

1. `common.h`/`ccd.c`에 `tact_t` 구조체 + `tact_create`/`tact_destroy`/`tact_step` 추가.
2. `tact.py`의 `_cache_c_pointers`를 `_create_c_handle`로 교체, `step()`의 use_c 블록을 single `tact_step` 호출로 축약.
3. parity test (이번 세션에서 쓴 것) 통과.
4. `step()` µs/call 측정 — nb={2, 6, 18}에서 현재 대비 개선폭 보고.

분량 추정: tact.py ~80 lines 변경, common.h/ccd.c ~150 lines 추가. 1~2일 작업.
