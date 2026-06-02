# design-pure-step.md — `Model.step()` 의 referential transparency (`ctx` / `SolverState`)

> **Status**: Phase 1+2 shipped 2026-05-25. `Model.step` 은 referentially transparent
> (single-thread). thread-safe / batched / autodiff 까지 가려면 workspace 외출(§7,
> Option C+)이 추가로 필요하며 아직 미구현.

## 1. 목표

`Model.step()` 을 **pure function** (referentially transparent) 으로 만든다 — 같은 입력
`(q, qd, tau, q_ref, qd_ref, ctx)` 이면 항상 같은 출력 `(q_next, qd_next, y, ctx_next)`,
숨은 상태 의존 없음. `Env.step()` 은 편의를 위해 **stateful wrapper** 로 그대로 둔다
(내부에 carry 하나 보관 → Env 사용자는 변화 없음).

**왜:**
- **재현성 / golden-value 테스트** — 같은 입력 → 같은 출력. `tests/regression/test_traj.py`
  가 의존하는 결정성.
- **state checkpoint / branch** — `ctx` 를 fork 해 여러 미래를 안전하게 탐색 (MPC 롤아웃,
  MPPI, time-travel 디버그). `examples/cartpole/cartpole.py` 의 MPC 가 실제 사례.
- **미래의 differentiable physics / batched RL rollout** — side effect 없는 step 이 전제.
  지금 단계가 그 stepping stone (전체 비교는 메모리 `tact-pure-function-design` 참고).

## 2. 계약 (contract)

```python
Model.step(q, qd, tau=None, q_ref=None, qd_ref=None, ctx=None)
    → (q_next, qd_next, y, ctx_next)
```

- `ctx` (`SolverState`) 는 스텝 간 넘어가는 **지속 솔버 상태** (현재 LCP warm-start λ 하나).
- `ctx=None` → **cold start** (λ 전부 0).
- `ctx` 는 **변형되지 않는다** — `ctx_next` 는 새 객체. 그래서 같은 `ctx` 로 두 번 step 하면
  두 독립 분기가 나온다 (fork-safe).
- `Model.zero_state()` → 현재 토폴로지에 맞는 cold `SolverState` (외부 pure 호출자가 초기
  carry 를 만들 때). 길이 `6 · MAX_PTS_PER_PAIR · n_pair`.

### `SolverState`
```python
class SolverState(NamedTuple):
    lam: np.ndarray          # LCP warm-start λ
    # 미래 hidden state(예: stick-slip anchor)는 여기 필드만 추가 → 시그니처 불변
```
- **NamedTuple** 인 이유: ① 필드 추가로 일반화 (미래 숨은 상태 대비 — 시그니처 churn 없음),
  ② JAX-pytree 친화 (나중 vmap/scan/AD), ③ self-documenting (`ctx.lam`).
- **인자명 `ctx`**: `q/qd`(동역학 상태)와 혼동을 피하려 `state` 대신, Drake `Context` 선례를
  따라 `ctx` 채택. (`carry`(JAX scan) 도 후보였음.)

## 3. 왜 lam_prev 하나만 막고 있었나

`step` 을 stateful 하게 만들던 숨은 상태:

| 항목 | 상태? | 비고 |
|---|---|---|
| LCP `lam_prev` (warm-start λ) | **YES** — 유일 블로커 | 이 문서가 외출시킨 것 |
| penalty `s_*_world` (brush) | ~~YES~~ | penalty solver **2026-05-24 제거** → 더 이상 없음 |
| `h->T`, workspace, `h->f/a/v` | NO ("externally pure") | 매 호출 덮어씀 → 입력에만 의존, 다음 호출에 영향 X |
| mesh storage | NO | build-time immutable, model identity 의 일부 |
| 전역 솔버 노브 (`erp/slop/cfm_scale/v_rest_thresh/iters/tol`, `dt`, `g`) | NO | build-time immutable config (`set()` 제거로 런타임 변경 경로 없음) |

→ penalty 제거 후 **`lam_prev` 하나만 인자로 외출**하면 Model 전체 runtime API 가 externally
pure 가 된다.

## 4. 구현

### Python (`sim.py`)
- `Model.step(..., ctx=None)`:
  - **C 경로** (`use_c`): `ctx.lam`(없으면 zeros)을 `lam_in`, fresh `lam_out` 을 받아
    `tact_step_lcp` 에 전달 → `ctx_next = SolverState(lam=lam_out)`.
  - **Python 경로** (`use_c=False`): `contact_lcp(lam_prev = ctx.lam or None)` → 
    `ctx_next = SolverState(lam=lcp_info['lam_full'])`.
  - **minimal 솔버**: warm-start 없음 → `ctx_next = ctx` (passthrough).
- `self.lam_prev` (옛 Model 상태) **제거** — carry 로 일원화.
- `Env`: `self._ctx` 보관, `step` 에서 thread, `__init__`/`add`/`delete`/`reset` 에서 `None`
  리셋 (토폴로지 변경 시 cpair 크기가 바뀌어 carry 무효).

### C (`tact.c`/`tact.h`/`_clib.py`)
- `tact_step_lcp(..., double *lam_in, double *lam_out)`:
  - `lam_in==NULL` → `h->lam_prev` fallback (legacy stateful). 비-NULL → 그 버퍼 사용.
  - `contact_lcp` 가 이미 `lam_prev`(in)/`lam_full_out`(out) 을 분리 지원 — `lam_in != lam_out`
    이면 out 을 in 으로 **seed-copy** 후 active slot 만 갱신 → **`lam_in` 불변**.
- `h->lam_prev` 필드는 NULL-fallback 버퍼로 유지 (제거하지 않음).

## 5. 핵심 성질: bit-identical → baseline 재캡처 불필요

Env 가 같은 warm-start λ 를 그대로 흘리면 **산술이 동일** → 변경 전 trajectory 와 **비트동일**.
실제로 `tests/regression/test_traj.py` 10/10 (atol 1e-12) 가 **재캡처 없이** 통과한다.

이는 `design-lcp-perf.md` 의 win(b)(재결합으로 1e-12 차이 → 재캡처 필요) 와 **정반대** 의 위치다.
warm-start 를 "숨김"에서 "명시적 인자"로 바꿨을 뿐 같은 항을 같은 순서로 더하므로 reassociation
이 없다.

## 6. cold vs warm

`ctx=None` 으로 **매 스텝 cold** 면 결정적이고 순수하지만, **warm(Env 기본)과 수치가 다르다** —
PGS 가 `iters`(기본 20)에서 잘리기 때문:

| | 시작점 | iters=20 결과 | iters↑ |
|---|---|---|---|
| warm | 직전 λ (≈해) | 거의 수렴 | 수렴 |
| cold | 0 | **under-converged** | 수렴 |

- contact-rich(box_wall, 138 접촉) 에서 cold-매스텝 vs warm 은 iters=20 에서 ~7e-4 차이.
- iters 를 충분히 올리면 **둘 다 같은 해로 수렴** (단일 접촉 sphere_test 에서 Δ=0 확인).
- 즉 warm-start 는 수치 꼼수가 아니라 같은 fixpoint 로의 가속. 큰 결합 더미는 PGS 수렴이 느려
  cold 가 iters=400 으로도 미수렴 (= warm-start 가 중요한 이유, `design-lcp-perf.md` 참고).
- **권장**: 재현성/테스트엔 cold(혹은 carry thread), 일반 시뮬엔 Env 가 warm-thread.

## 7. 범위 경계 (이 단계가 *주지 않는* 것)

| 원하는 것 | 필요한 것 | 상태 |
|---|---|---|
| 단일 스레드 재현성 / fork | `ctx` 외출 (이 문서) | ✅ shipped |
| **thread-safety / batched parallel** | workspace 까지 외출 (handle 공유 race 제거) | ❌ Option C+ (미구현) |
| **autodiff / diff-physics** | 미분 backend + functional 배열 | ❌ 미래 |

- `h->workspace`/`h->T` 등은 여전히 handle 안에 있다 → **같은 handle 을 두 스레드가 step 하면
  race**. 진짜 thread-safe pure 는 workspace 도 인자/per-call 로 빼야 한다 (Option C+).
- `add`/`delete`/`edit` 은 **build-phase mutation** (MuJoCo `mjModel` 과 동일 성격) — runtime
  query 가 아니므로 pure 대상 아님. 토폴로지 변경 시 `ctx` 는 무효(Env 가 리셋, 외부 호출자는
  `zero_state()` 재호출).
- A 행렬은 **dense-stored 유지** (`design-lcp-perf.md` I5) — autodiff/IFT-adjoint 친화적
  선택이라 diff-physics 방향과 정합.

## 8. 테스트

`tests/test_pure_step.py` — 구조적 불변량 assert (golden baseline 불필요, self-validating):
zero_state 크기, cold 결정성, **Model(ctx thread)==Env 비트동일**, ctx 불변/fork 독립,
warm/cold 동일 해(@high iters), py↔C thread tracking. 7/7 pass.

테스트 입력은 `tests/scenes/` 의 frozen fixture 사용 (`examples/` 변동과 분리,
`tests/scenes/README.md`).
