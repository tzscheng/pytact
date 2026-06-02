# design-autodiff.md — differentiable `tact` (DRAFT / planning)

> **Status**: **계획 초안. 미구현.** autodiff는 아직 동기 + "문을 열어둔 설계 결정"으로만
> 존재한다(`design-pure-step.md`, `design-lcp-perf.md §5` / I5). 이 문서는 *어떻게 구현할지*
> 의 결정 fork를 정리한다 — 합의 전까지 어떤 것도 확정 아님.

## 1. 목표 & use cases

`step` 을 통과하는 gradient 를 얻어, 학습/최적화가 동역학을 관통하게 한다.

| use case | 필요한 gradient |
|---|---|
| System identification | ∂(궤적/관측)/∂(mass, inertia, μ, restitution, geometry) |
| Gradient trajectory opt / MPC | ∂cost/∂tau (롤아웃 전체) |
| Model-based RL / policy gradient | ∂(q_next,qd_next)/∂(q,qd,tau) |
| Design optimization | ∂성능/∂(설계 파라미터) |

MVP 후보는 **state/control 미분** `∂(q_next,qd_next)/∂(q,qd,tau)` (가장 흔하고 모델 파라미터
미분의 부분집합).

## 2. step 의 어디가 미분가능한가

`step` = FK/Jacobian → ABA(forward dyn) → CRB(M) → **LCP 접촉** → semi-implicit Euler.

| 단계 | 미분 | 비고 |
|---|---|---|
| FK, Jacobian, CRB(M), ABA | **매끄러움** | 해석적 도함수 or 표준 AD |
| 적분기 | 매끄러움 | 단순 |
| **LCP 접촉** | **어려움** | 비매끄러움(active-set 변화). 단 **정칙화 볼록 LCP**(CFM)는 IFT로 미분 가능 |

핵심: tact 의 접촉은 **정칙화된 볼록 문제**(CFM + Stewart-Trinkle)라 — penalty 제거가 여기서
중요 — 해에서 KKT 조건을 음함수정리(IFT)로 미분하면 매끄러운 gradient 경로가 열린다.
penalty(스프링-댐퍼 brush)였다면 비매끄러움이 더 고약했을 것 (`design-lcp-perf.md §5`).

## 3. 접근법 (Fork A — 구현 전략)

| | A1. IFT-adjoint custom op | A2. JAX/PyTorch 재구현 | A3. black-box (finite-diff / Enzyme) |
|---|---|---|---|
| 방식 | C forward 재사용 + backward는 KKT adjoint 풀이. PyTorch `autograd.Function` / JAX `custom_vjp` | 동역학+접촉을 JAX로 재작성, XLA가 AD (MJX/Brax 방식) | C 소스 자동미분(Enzyme) 또는 유한차분 |
| gradient | **정확(해석적)** | 정확 | finite-diff: 근사·noisy / Enzyme: PGS 펼침 위험 |
| batching/GPU | 별도(Option C+ 필요) | **vmap/jit/GPU 같이 옴** | ✗ |
| 구현량 | 중 (adjoint 유도·구현) | **대 (이중 코드베이스 동기)** | 소(finite-diff) / 중(Enzyme) |
| 성능(비-AD) | C 그대로 유지 | C 성능 상실 | C 유지 |
| 기존 설계와 정합 | **높음** (dense A + block M factor 재사용, I5) | 중 (재작성) | 낮음 |

**추천: A1 (IFT-adjoint custom op).** 빠른 C forward 유지, 정확한 gradient, 그리고 dense A·
재사용 가능한 block M factor·정칙화 볼록 LCP 가 **이미 A1 을 겨냥해** 결정돼 있었다
(`design-lcp-perf.md §5`). A2(JAX)는 GPU 대규모 배칭이 진짜 병목이 될 때만 재고.

## 4. IFT-adjoint 경로 수학 개요 (A1 채택 시)

매 스텝 접촉은 정칙화 볼록 문제로 λ(접촉 임펄스)를 푼다 (cone 제약, A=J M⁻¹ Jᵀ, c=J qd_free−b).
해 λ\* 에서 KKT 가 성립 → loss 의 upstream gradient 는 **해에서 KKT 시스템의 transpose(adjoint)
를 한 번 풀어** 얻는다 (OptNet / diffcp 결과). 핵심:

- **LCP 해 자체의 미분** `∂λ/∂A, ∂λ/∂c` — IFT/diffcp. **forward 의 dense A factorization 을 그대로
  재사용** (backward 가 같은 행렬의 전치 풀이) → S1 의 block M factor 도 재활용.
- **A(q), c(q,qd,tau) 의 미분** — `A=J M⁻¹ Jᵀ`, `c=J qd_free − b` 는 q(J,M 경유)·qd·tau 에 의존.
  이 매끄러운 부분은 **해석적(spatial-algebra 도함수) 또는 표준 AD**.
- **하이브리드 권장**: 비매끄러운 LCP fixpoint 만 IFT 로, 나머지 매끄러운 동역학(A·c 구성,
  적분기)은 AD/해석. 어려운 부분만 손으로, 쉬운 부분은 자동.

## 5. 결정해야 할 fork (미정)

- **B. backend/API**: PyTorch `autograd.Function`(FFI 단순, 로보틱스 학습 흔함) vs JAX `custom_vjp`
  (vmap/jit 유리, C callback 필요) vs backend-중립 C adjoint + 얇은 바인딩. → 1차 PyTorch 유력.
- **C. 미분 대상 범위**: state/control(MVP) → 이후 model params(μ, mass, geometry). 어디까지/언제.
- **D. 비매끄러움 처리**: 접촉 make/break 에서 gradient 불연속. CFM 정칙화가 완화하지만 제거 못함.
  randomized smoothing / 서브gradient 수용 / 기대치 설정 중 택. (§7 한계 참고)
- **E. Option C+ 와 의존순서**: 단일 샘플 reverse-mode adjoint(A1, 한 번의 backward)는
  **Option C+ 불요** — 현재 pure step 위에서 가능. **배칭/vmap 은 Option C+(workspace 외출)
  필요**. → A1 단일궤적 먼저, 배칭은 C+ 이후.

## 6. 전제조건 & 단계 제안

**이미 갖춰진 것**: pure step(referential transparency, Phase 1–3) · 정칙화 볼록 LCP substrate ·
dense A + reusable block M factor(I5) · ctx 외출. → A1 단일궤적은 **추가 인프라 없이 착수 가능**.

**제안 단계** (각 단계 끝 검증·보고):
- **Phase A**: 단일 샘플 IFT-adjoint, PyTorch custom op, `∂(q_next,qd_next)/∂(q,qd,tau)`.
  finite-diff(gradcheck)로 검증. (Option C+ 불요)
- **Phase B**: 미분 대상을 model params(μ/mass/restitution/geometry)로 확장.
- **Phase C**: 배칭/vmap → **Option C+(workspace 외출) 선행 필요**. 필요 시 A2(JAX) 재고 지점.
- **Phase D(선택)**: GPU 대규모 RL 이 병목이면 JAX 경로.

## 7. 리스크 / 정직한 한계

- **접촉 gradient 신뢰성**: 충돌/강성 접촉에서 미분물리 gradient 는 폭발·편향되기 쉽다(문헌 공통:
  Brax, "differentiable simulation" 연구들). CFM 정칙화가 완화하나 근본 해결 아님 — **gradient 가
  항상 유용하리란 보장 없음.** 기대치를 미리 합의.
- **유지비**: A2(JAX 재구현)는 C 와 **이중 코드베이스 동기** 부담이 큼 → GPU 배칭이 강제하기 전엔
  A1 유지.
- **adjoint 정확성**: A1 backward 의 정확성은 KKT 구조(해의 active set)에 의존 → gradcheck 필수.

## 8. 관련 문서 / 선례

- `design-pure-step.md` (전제인 pure step), `design-lcp-perf.md` I5·§5 (dense A·convex LCP =
  IFT-adjoint 친화, S3 drift 경고), 메모리 `tact-pure-function-design`(엔진 비교: MJX/Brax pure AD,
  Drake AutoDiff Context).
- 외부: OptNet(Amos & Kolter), diffcp(Agrawal et al.) — convex 해의 미분; MJX/Brax — functional AD 사례.
