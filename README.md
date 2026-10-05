# Unified ReAct vs Plan Routing Experiment

## Experiment

이전 두 실험의 결과를 하나의 routing hypothesis로 통합했다.

핵심 가설은 다음과 같다.

```text
0 = ReAct
    다음 action/query가 intermediate observation에 의존하는 경우

1 = Plan
    필요한 action/search/tool set을 실행 전에 미리 결정할 수 있는 경우
```

Dataset은 총 200개로 구성했다.

| Group | N | Dependency Label | 가설 |
|---|---:|---:|---|
| SQuAD | 50 | Plan | 검색 query를 upfront에서 결정 가능 |
| HotpotQA | 50 | ReAct | 이전 retrieval 결과가 다음 query에 필요 |
| BFCL Single-tool | 50 | Plan | 필요한 tool을 upfront에서 결정 가능 |
| BFCL Multi-tool | 50 | Plan | 필요한 tool set을 upfront에서 결정 가능 |

비교 전략:

```text
Always ReAct
Always Plan
Router → ReAct / Plan
Static Dependency Rule
```

공통 평가는 다음과 같이 정의했다.

```text
Retrieval:
task_success = full supporting-document retrieval
task_score   = support recall

Tool:
task_success = exact tool-set match
task_score   = tool F1
```

---

## Overall Results

| Method | Route Acc. | Task Success | Task Score | Avg. Steps | LLM Calls |
|---|---:|---:|---:|---:|---:|
| Always Plan | 0.75 | 0.865 | 0.9208 | 1.595 | 1.00 |
| Always ReAct | 0.25 | 0.880 | 0.9292 | 1.575 | 1.89 |
| **Router** | **0.84** | **0.900** | **0.9383** | 1.585 | 1.00* |
| **Static Dependency** | 1.00 | **0.905** | **0.9408** | 1.590 | cached |

가장 중요한 결과는 **Router가 두 fixed strategy를 모두 이겼다는 점**이다.

```text
Task Success

Always Plan   0.865
Always ReAct  0.880
Router        0.900
Static Rule   0.905
Oracle Best   0.915
```

Router는:

```text
vs Always Plan  : +3.5%p
vs Always ReAct : +2.0%p
```

의 task success 개선을 보였다.

또한 theoretical per-query best strategy인 Oracle 0.915와의 차이도 **1.5%p**에 불과했다.

따라서 이번 실험에서는 처음으로:

> **Adaptive execution routing이 best fixed strategy를 실제로 넘어섰다.**

---

# Results by Task Family

## Retrieval

| Method | Task Success | Task Score |
|---|---:|---:|
| Always Plan | 0.75 | 0.855 |
| **Always ReAct** | **0.82** | 0.885 |
| **Router** | **0.82** | **0.890** |
| Static Dependency | **0.83** | **0.895** |

Retrieval에서는 ReAct가 여전히 강했다.

```text
Plan   0.75
ReAct  0.82
Router 0.82
```

Router는 ReAct와 동일한 task success를 유지하면서 support-recall 기반 task score는 소폭 높였다.

즉 routing을 추가해도 retrieval quality가 크게 손상되지 않았다.

---

## Tool Selection

| Method | Task Success | Task Score |
|---|---:|---:|
| Always Plan | **0.98** | **0.9867** |
| Always ReAct | 0.94 | 0.9733 |
| **Router** | **0.98** | **0.9867** |
| Static Dependency | **0.98** | **0.9867** |

Tool task에서는 이전 실험과 동일하게 Plan이 강했다.

```text
Plan   0.98
Router 0.98
ReAct  0.94
```

Router는 거의 모든 BFCL query를 Plan으로 보내면서 Always Plan의 성능을 그대로 유지했다.

즉 전체적인 Router 성능 향상은 주로 다음 조합에서 발생했다.

```text
Retrieval
→ ReAct 성향

Tool orchestration
→ Plan 성향
```

---

# Dataset-Level Analysis

## BFCL Multi-tool

| Method | Success | Score |
|---|---:|---:|
| Always Plan | **1.00** | **1.0000** |
| Always ReAct | 0.92 | 0.9733 |
| Router | **1.00** | **1.0000** |

Router:

```text
49 / 50 → Plan
1 / 50  → ReAct
```

로 98% routing accuracy를 기록했다.

필요한 tool set이 original request에서 명확하게 드러나는 경우에는 upfront planning이 가장 적합했다.

---

## BFCL Single-tool

| Method | Success | Score |
|---|---:|---:|
| Plan | 0.96 | 0.9733 |
| ReAct | 0.96 | 0.9733 |
| Router | 0.96 | 0.9733 |

세 전략의 quality가 완전히 동일했다.

Router는:

```text
50 / 50 → Plan
```

을 선택했다.

따라서 이 환경에서는 single-tool에서 ReAct loop를 수행할 이유가 거의 없었다.

```text
Question
→ required tool known upfront
→ direct / plan selection
```

이면 충분했다.

---

## SQuAD

| Method | Success | Score |
|---|---:|---:|
| **Plan** | **0.94** | **0.94** |
| ReAct | 0.92 | 0.92 |
| **Router** | **0.94** | **0.94** |

Router:

```text
46 / 50 → Plan
4 / 50  → ReAct
```

로 dependency rule과 92% 일치했다.

SQuAD처럼 standalone query에서 필요한 retrieval query 자체가 처음부터 명확한 경우에는 upfront query generation이 충분히 강했다.

---

## HotpotQA

| Method | Success | Score |
|---|---:|---:|
| Plan | 0.56 | 0.77 |
| **ReAct** | **0.72** | **0.85** |
| Router | 0.70 | 0.84 |
| Static Dependency | **0.72** | **0.85** |

HotpotQA가 이번 실험에서 가장 중요한 dataset이다.

```text
ReAct success = 0.72
Plan success  = 0.56
```

로 **16%p 차이**가 발생했다.

이는 multi-hop retrieval에서:

```text
Search
  ↓
Intermediate entity 발견
  ↓
Follow-up query 생성
  ↓
Search
```

처럼 다음 action이 observation에 의존하기 때문이다.

그러나 Router는 HotpotQA에서:

```text
ReAct : 23 / 50 = 46%
Plan  : 27 / 50 = 54%
```

만 선택했다.

즉 Router가 HotpotQA의 observation dependency를 충분히 잘 판단하지 못했다.

그 결과:

```text
Static Dependency 0.72
Router            0.70
```

으로 약 **2%p의 손실**이 발생했다.

---

# Router Analysis

Router confusion matrix:

```text
             Predicted
             ReAct   Plan

Gold ReAct     23      27
Gold Plan       5     145
```

Overall route accuracy:

```text
168 / 200 = 84%
```

하지만 dataset은:

```text
Gold Plan  = 150
Gold ReAct = 50
```

으로 불균형하기 때문에 84%만 보면 안 된다.

Class별 recall은:

```text
PLAN detection
145 / 150 = 96.7%

REACT detection
23 / 50 = 46.0%
```

이다.

즉 현재 Router는 **Plan을 매우 잘 찾지만 ReAct가 필요한 observation-dependent task를 놓치는 경향**이 있다.

Balanced accuracy는 약:

```text
(0.967 + 0.460) / 2
≈ 0.713
```

이다.

따라서 Router의 다음 개선 포인트는 명확하다.

> **“Can the next action really be known before seeing the observation?”을 더 보수적으로 판단하도록 만드는 것.**

---

# Router Routing Pattern

실제 routing 결과:

| Group | ReAct | Plan |
|---|---:|---:|
| BFCL Multi | 2% | **98%** |
| BFCL Single | 0% | **100%** |
| SQuAD | 8% | **92%** |
| HotpotQA | **46%** | 54% |

세 dataset에서는 거의 예상대로 동작했다.

문제는 HotpotQA였다.

Router가 HotpotQA의 절반 이상을 Plan으로 보냈는데, 실제 결과에서는 ReAct가 명백하게 더 강했다.

따라서 단순히:

```text
"This question has several information needs"
→ Plan
```

으로 판단하면 부족하다.

더 중요한 질문은:

```text
"Can I name the later search query NOW,
before observing the result of the first search?"
```

이다.

답이 `No`라면 ReAct가 더 자연스럽다.

---

# Paired Comparison

동일한 200개 query에 대한 paired comparison:

```text
Router vs Always ReAct
win  = 7
loss = 3
tie  = 190

Router vs Always Plan
win  = 7
loss = 0
tie  = 193
```

이 결과는 매우 중요하다.

Router는 Always Plan과 비교해서:

```text
7 wins
0 losses
193 ties
```

를 기록했다.

즉 **Plan을 기본 전략으로 사용하면서 필요한 일부 observation-dependent query에서만 ReAct로 전환하는 구조**가 효과적이었다.

반면 ReAct와 비교하면:

```text
7 wins
3 losses
```

로 여전히 positive net gain이었다.

---

# How Often Does Strategy Choice Actually Matter?

실제 ReAct / Plan 결과를 query별로 직접 비교하면 다음과 같다.

| Group | ReAct Wins | Plan Wins | Tie |
|---|---:|---:|---:|
| BFCL Multi | 0% | 8% | **92%** |
| BFCL Single | 0% | 0% | **100%** |
| HotpotQA | **22%** | 6% | **72%** |
| SQuAD | 0% | 2% | **98%** |

가장 중요한 관찰은 **대부분의 query에서는 strategy 선택이 결과를 바꾸지 않는다는 점**이다.

전체 200개 중 대부분은 tie였다.

즉 Router의 목적은 모든 query를 정교하게 분류하는 것이 아니다.

실제 가치가 있는 것은:

```text
small but important subset
```

에서 잘못된 fixed strategy를 피하는 것이다.

특히 HotpotQA에서는:

```text
ReAct wins = 22%
Plan wins  = 6%
```

으로 ReAct가 실제로 필요한 subset이 존재했다.

---

# Why the Unified Router Works

이전 실험에서는 각각 하나의 fixed strategy가 강했다.

### Retrieval

```text
HotpotQA
ReAct > Plan
```

### Tool Selection

```text
BFCL
Plan > ReAct
```

이번 실험에서는 두 환경을 합쳤기 때문에 각각의 강점을 선택할 수 있었다.

```text
                        User Request
                             ↓
             Can actions be known upfront?
                    /                 \
                  YES                  NO
                   ↓                   ↓
                 PLAN                ReAct
                                    ↓
                               Observation
                                    ↓
                               Next Action
```

이 routing principle은 `single vs multi`, `simple vs complex`, `single-hop vs multi-hop`보다 일반적이다.

---

# Main Finding

이번 실험에서 가장 중요한 결과는 다음이다.

```text
Always Plan   = 86.5%
Always ReAct  = 88.0%
Router        = 90.0%
Static Rule   = 90.5%
Oracle Best   = 91.5%
```

즉:

> **Neither ReAct nor Plan is globally optimal.**

그리고:

> **Routing based on information dependency can outperform both fixed strategies.**

Router는 best fixed strategy인 ReAct보다 **+2.0%p** 높았고, Oracle best와의 차이는 **1.5%p**였다.

---

# Revised Execution Policy

이번 결과까지 반영하면 production policy는 다음과 같이 정리할 수 있다.

```text
1. Is the procedure deterministic?
   → Fixed Workflow

2. Can the action graph be inferred upfront?
   → Plan / Direct Execute

3. Does the next action depend on intermediate observations?
   → ReAct

4. Does routing actually beat the best fixed strategy?
   → Only then keep the Router
```

이번 unified experiment에서는 4번 조건까지 만족했다.

```text
Router 0.900
>
Best Fixed 0.880
```

따라서 앞선 개별 실험과 달리 **heterogeneous task environment에서는 Router를 둘 근거가 생겼다.**

---

# Important Caveat: Route Labels Are Heuristics

`route_accuracy`에서 사용한 label은 절대적인 ground truth가 아니다.

현재 label은:

```text
SQuAD       → Plan
HotpotQA    → ReAct
BFCL Single → Plan
BFCL Multi  → Plan
```

이라는 information-dependency hypothesis에서 만든 것이다.

실제 query별 winner를 보면:

```text
Hotpot:
ReAct 11
Plan   3
Tie   36
```

처럼 동일 dataset 안에서도 최적 strategy가 다르다.

따라서 최종 production router는 dataset-level label보다:

```text
Query
 ↓
Run ReAct offline
Run Plan offline
 ↓
Compare utility
 ↓
Winner label
```

로 학습하는 것이 더 적절하다.

예:

```text
utility =
    task_success
  + α * task_score
  - β * latency
  - γ * LLM calls
```

이렇게 생성한 outcome-based label이 Select-then-Solve 방식과도 더 가깝다.

---

# Latency Caveat

현재 실험은 cross-method cache를 사용했다.

따라서 다음 latency:

```text
Plan   1.62 s
ReAct  2.85 s
Router 1.43 s
```

를 그대로 비교하면 안 된다.

Router의:

```text
llm_calls      = 1.00
llm_cache_hits = 1.075
```

는 execution 단계가 이전 fixed baseline 실행 결과를 재사용했음을 의미한다.

따라서:

- Task quality 비교: **유효**
- Router accuracy: **유효**
- Router standalone latency: **비교 불가**

실제 latency/cost 평가는 다음처럼 cache 없이 다시 수행해야 한다.

```bash
python3 unified_dependency_router_eval.py \
  --n-squad 50 \
  --n-hotpot 50 \
  --n-bfcl-single 50 \
  --n-bfcl-multi 50 \
  --corpus-size 1000 \
  --top-k 3 \
  --max-steps 3 \
  --no-cache
```

실제 production에서는 Router가 추가 호출이므로 latency overhead가 존재한다.

---

# Conclusion

이번 실험에서는 처음으로 **Router가 모든 fixed execution strategy를 넘어섰다.**

```text
Task Success

Always Plan        0.865
Always ReAct       0.880
Router             0.900
Static Dependency  0.905
Oracle Best        0.915
```

이전 실험과 함께 보면 다음과 같은 일관된 패턴을 확인할 수 있다.

```text
BFCL Tool Selection
Required actions known upfront
→ Plan

HotpotQA Retrieval
Next query depends on retrieved evidence
→ ReAct

Mixed Environment
→ Adaptive Router
```

따라서 최종 설계 원칙은 다음과 같다.

> **Use Plan when the action graph can be inferred before execution. Use ReAct when the action graph must be discovered through intermediate observations. Use a Router only when these different execution regimes coexist and routing beats the best fixed policy.**

현재 unified experiment에서는 바로 그 조건이 성립했다.

```text
Router > Always ReAct > Always Plan
```

그리고 Router의 남은 가장 큰 개선점은 **observation-dependent ReAct task의 recall**, 특히 HotpotQA에서 현재 46%에 불과한 ReAct routing recall을 높이는 것이다.
