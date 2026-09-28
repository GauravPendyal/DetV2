# Detector V2 — Stage Log

Running record of each stage: what was built, tests, and measured results on the
**held-out local eval sets** (`Datasets/*.jsonl`, 769 rows, never trained on).
Metrics reported per-dataset (never hidden behind one combined number).

Gate view = the routing decision HalluciGuard actually sees (`next_action == Verify` → positive).
Thresholds: LOW ≤ 0.30 < MEDIUM < 0.50 ≤ HIGH; LOW→Accept, else→Verify.

---

## Stage 0 — Claim decomposition + structural features + heuristic baseline

**Built:** rule-based claim segmenter; structural feature extractor (numbers,
dates, proper nouns, superlatives, absolutes, over-assertion, hedging,
false-premise cues, specificity); hand-set logistic `StructuralSignal`;
`HeuristicScorer` (mean of available signals); response aggregators
(max/mean/top-k/noisy-OR); canonical adapter + `DetectorAgent`; local eval
loader; full metrics harness (P/R/F1, AUROC, AUPRC, ECE, Brier, confusion,
latency, recall-vs-routing curve).

**Tests:** 51 passed (claims, structural, degraded-mode, scoring, aggregation,
pipeline, contract, agent, adversarial, long-context, metrics).

**Results (gate view):**

| dataset | n | recall | prec | f1 | auroc |
|---|---|---|---|---|---|
| halluciguard_dataset | 187 | 0.583 | 0.444 | 0.505 | 0.352 |
| ragtruth | 84 | 0.111 | 0.143 | 0.125 | 0.554 |
| truthfulqa | 498 | 0.131 | 0.897 | 0.228 | 0.432 |
| COMBINED | 769 | 0.215 | 0.556 | 0.310 | — |

Macro-avg gate F1 = 0.286, recall = 0.275. Cold start ≈ negligible (no models).
Warm p95 latency < 0.5 ms/response.

**Reading (honest):** near-random — AUROC ≤ 0.55 everywhere and *below* 0.5 on
halluciguard_dataset, i.e. the hand-set weights are partly mis-signed. This is
the intended floor: it proves the harness end-to-end and gives Stage 1 something
to beat by *learning* the weights instead of guessing them. Not a shippable
detector.

**Artifact:** `reports/stage0_heuristic_eval.json`.

---

## Stage 1 — Logistic-regression fusion on Stage-0 structural features

**Built:** `LogRegResponseModel` (StandardScaler → LogisticRegression, C=1.0,
class_weight="balanced") + `LogRegResponseScorer` injected into the pipeline as a
response-level scorer; dual featurizers (`featurize_text` for training,
`featurize_claims` for inference) proven to emit identical vectors; training
script with **dataset-wise held-out group** protocol; eval wiring via
`--response-model`.

**Data / leakage discipline:** trained on HaluEval **qa + dialogue** only.
HaluEval **summarization** held out entirely as a cross-group test. Local
`Datasets/*.jsonl` (769 rows) never seen in training — pure held-out eval.
Context / judge.evidence excluded from features.

**Tests:** 58 passed (+7: Stage-1 featurizer agreement, empty→zero vector,
train/predict/proba range, separable-signal learning, save/load roundtrip,
pipeline integration).

**Training (HaluEval, holdout = summarization):**
train f1=0.696 auroc=0.794 · val f1=0.666 auroc=0.789 ·
holdout-summarization f1=0.667 recall=1.000 prec=0.500 **auroc=0.831 ece=0.463 brier=0.444**.
Top weights: `total_tokens` +1.55, `d_absolute_density_mean` −0.96,
`n_claims` −0.83, `num_count_per_tok` −0.70, `d_specificity_mean` +0.64.

**Results on held-out local sets (gate view + AUROC + routing):**

| dataset | n | pos_rate | recall | prec | f1 | **auroc** | auprc | **frac_routed** |
|---|---|---|---|---|---|---|---|---|
| halluciguard_dataset | 187 | 0.51 | 0.948 | 0.850 | 0.897 | **0.966** | 0.977 | 0.57 |
| ragtruth | 84 | 0.21 | 1.000 | 0.214 | 0.353 | **0.407** | 0.184 | 1.00 |
| truthfulqa | 498 | 0.80 | 0.957 | 0.811 | 0.878 | **0.529** | 0.821 | 0.94 |
| COMBINED | 769 | — | 0.957 | 0.741 | 0.835 | — | — | — |

Macro-avg gate F1 = 0.709, recall = 0.968. Warm p95 ≤ 1.1 ms/response.

**V1 comparison (threshold@0.50, same local sets):** V1 combined recall 0.264 /
f1 0.391 (per-set recall: halluciguard 0.938, ragtruth 0.278, truthfulqa 0.101).
Stage 1 combined recall 0.957 — a large recall gain.

**Reading (honest — the gate F1 is partly a mirage):** AUROC, not gate-F1, tells
the real story. Genuine discrimination appears on **halluciguard_dataset only**
(AUROC 0.966, frac_routed 0.57 ≈ pos_rate 0.51). On **ragtruth** AUROC is 0.407
(*below random*) and on **truthfulqa** 0.529 (≈ random) — the strong-looking
recall/F1 there come from `frac_routed` = 1.00 / 0.94: the model routes almost
everything to Verify, so on positive-skewed sets (truthfulqa pos_rate 0.80) recall
and F1 look high for free. So Stage 1 is a real improvement in absolute recall over
V1, but its *ranking* ability generalizes to only one of three held-out
distributions. The top weight being `total_tokens` confirms the features are
surface-correlated (length proxies), not semantic. The holdout-group ECE 0.463
(predicts ~everything positive) further shows the scores are **not** calibrated —
motivating Stage 8. This is exactly the case for Stage 2: a semantic claim-level
encoder that can rank by meaning rather than surface form.

**Artifacts:** `reports/stage1_train.json`, `reports/stage1_logreg_eval.json`,
`models/stage1_logreg.joblib` (+ `.meta.json`).

---

## Stage 2 — DeBERTa-v3 claim-level encoder (always-on learned signal)

**Built:** `DebertaClaimEncoder` (original 2-class head fine-tuned on a plain
`microsoft/deberta-v3-xsmall` LM backbone — **not** the HaluEval DistilBERT, not a
wrapped detector); `EncoderSignal` (per-claim `Signal` with a batched `prepare()`
precompute hook, dependency-injectable, honest degradation when the model/torch is
missing); a `prepare(claims, ctx)` hook added to the `Signal` base + pipeline so a
whole response is scored in one forward pass; manual AdamW fine-tune loop
(`train.py`), `tools/train_stage2.py`, and `--encoder-model` eval wiring.

**Data / leakage discipline:** fine-tuned on HaluEval **qa + dialogue** only;
**summarization** held out as a cross-group test; local `Datasets/*.jsonl` never
seen. **Weak claim labels** (each segmented claim inherits its response's label) —
noisy by construction, reported honestly. Input is `(query, claim)`; response risk
= top-k-mean of per-claim probs.

**Config:** backbone deberta-v3-xsmall, 3432 training claims (pos 0.53), 2 epochs,
batch 16, lr 2e-5, max_len 128, CPU. Train time ≈ 891 s.

**Training (HaluEval):** val response AUROC **0.933** F1 0.853 ECE **0.105** (good,
near-calibrated in-family); holdout-summarization response AUROC **0.719** F1 0.668
ECE 0.418 (transfers, but miscalibrated on the unseen group).

**Results on held-out local sets:**

| dataset | n | pos | recall | prec | f1 | **auroc** | auprc | **ece** | frac_routed | p95 |
|---|---|---|---|---|---|---|---|---|---|---|
| halluciguard_dataset | 187 | 0.51 | 0.979 | 0.959 | 0.969 | **0.993** | 0.995 | **0.034** | 0.52 | 58 ms |
| ragtruth | 84 | 0.21 | 1.000 | 0.214 | 0.353 | **0.411** | 0.195 | 0.709 | 1.00 | 140 ms |
| truthfulqa | 498 | 0.80 | 0.756 | 0.798 | 0.777 | **0.482** | 0.781 | 0.263 | 0.76 | 49 ms |

Macro gate F1 = 0.700, recall = 0.912. Cold start ≈ 3.5 s (model load + first
inference). Warm p95 49–140 ms/response (CPU transformer — ~100× Stage 1).

**Reading (honest):** the semantic encoder is a decisive win on the HaluEval-family
distribution — `halluciguard_dataset` AUROC 0.966 → **0.993**, and now genuinely
*selective* (frac_routed 0.52 ≈ base rate 0.51) and, unexpectedly, well-calibrated
there (ECE 0.034) without any calibration step. But it does **not** solve
cross-dataset generalization: `ragtruth` (AUROC 0.41, routes everything) and
`truthfulqa` (AUROC 0.48 — random) are no better than Stage 1. This is diagnostic,
not a bug: **ragtruth** hallucination is defined against a *source document* we
deliberately withhold (leakage guard), so no context-free signal can rank it — that
is exactly Stage 3's job (grounded entailment, context-aware). **truthfulqa** turns
on world-knowledge truth of common misconceptions, which a small encoder trained on
HaluEval qa/dialogue has no way to know. Net: macro gate F1 is flat vs Stage 1
(0.709 → 0.700) and combined recall drops (0.957 → 0.807) because the encoder stops
over-routing — a *more honest* operating point, not a worse detector. The encoder
earns its place as the always-on signal on in-family data; the remaining two
distributions need context (Stage 3) and, ultimately, learned fusion + calibration
(Stages 6/8).

**Artifacts:** `reports/stage2_train.json`, `reports/stage2_encoder_eval.json`,
`models/stage2_encoder/`.

---

## Stage 3 — MiniCheck-style grounded entailment (context-aware, optional signal)

**Built:** `EntailmentSignal` — a context-aware per-claim `Signal` that chunks the
source document (`chunk_context`, sentence-grouped ~900 chars, 1-sent overlap, ≤12
chunks) and scores every `(chunk, claim)` pair, reducing to `risk = min over chunks
of P(unsupported)` (the best-supporting chunk wins — MiniCheck's *method*, not its
model). Honest abstention is the defining behavior: no context → `available=False`,
reason `context_absent`, nothing fabricated. Model = **our own** `(context, claim) →
P(unsupported)` head fine-tuned on a plain deberta-v3-xsmall LM (reuses the Stage-2
`DebertaClaimEncoder` class; different slots + different training data). NOT a
downloaded MiniCheck / detector.

**Data / leakage discipline:** fine-tuned on HaluEval **qa + dialogue** *with their
context field* (`knowledge`); HaluEval **summarization** (context = `document`) held
out entirely as a grounded cross-group test. Public **RAGTruth**
(`wandb/RAGTruth-processed` test, real source `context`) used as a **fully-external
grounded test — never trained on**. Local 769 rows stay held-out eval. Weak claim
labels (claim inherits response label), reported as noisy.

**Config:** deberta-v3-xsmall, 2965 `(context,claim)` pairs (pos 0.535), 2 epochs,
batch 16, lr 2e-5, max_len 256, CPU. Train time ≈ 842 s.

**Training (HaluEval):** val response AUROC **0.889** F1 0.797 ECE 0.128;
holdout-summarization response AUROC **0.561** F1 0.547 ECE 0.022 — i.e. it learns
grounding in-family but barely generalizes to the unseen document-context group.

**Tests:** 68 passed (+5: chunk splits long doc, context-absent abstains
(`context_absent`, value None), grounded claim scores lower risk than unsupported,
best-supporting-chunk wins, pipeline completes with context / degrades without).

**Head-to-head — Stage 2 vs Stage 3 on public RAGTruth (n=600, pos_rate 0.35, WITH
source context — the whole motivation for this stage):**

| signal | auroc | auprc | ece | brier | gate f1 | @0.50 f1 | routed | p95 ms |
|---|---|---|---|---|---|---|---|---|
| stage2_encoder (context-free) | **0.317** | 0.260 | 0.556 | 0.544 | 0.519 | 0.519 | 1.00 | 265 |
| stage3_entailment (grounded) | **0.331** | 0.264 | **0.246** | **0.331** | 0.519 | 0.417 | 1.00 | **3200** |

**Honest abstention on context-free local sets (must not fabricate):**
halluciguard_dataset 187/187, ragtruth 84/84, truthfulqa 498/498 → **100 % degraded,
routed-all=yes, zero scored rows.** The grounded signal invents nothing when there is
no document to check against (incl. the local ragtruth set, whose source was stripped).

**Reading (honest — this is a mixed/negative result, not a win):** grounded
entailment did **not** deliver the expected ranking win on RAGTruth. **Both** signals
sit *below random* on AUROC (0.317 / 0.331); Stage 3 edges Stage 2 by only +0.014 —
inside noise. So the headline claim "grounding fixes RAGTruth" is **not** supported by
this run. What Stage 3 *does* buy: (1) **calibration** — ECE 0.556→0.246 and Brier
0.544→0.331 (both ~halved); (2) a **usable routing curve** — Stage 2 is degenerate
(routes 100 % until threshold 0.9, only 1 TN), while Stage 3's scores actually spread
(threshold@0.50 separates 92 TN, and you can pick a budget: ≤50 % routed → recall
0.214, ≤30 % → 0.133); (3) **correct honest abstention** everywhere context is absent.
Why sub-random ranking despite val AUROC 0.889? Domain shift (trained HaluEval
qa/dialogue `knowledge`; RAGTruth is Summary/QA/Data2txt — the summarization holdout
already fell to 0.561), max_len-256 truncation of long RAGTruth documents (the true
supporting span can be cut), noisy weak labels, and a tiny xsmall backbone. Cost is
real: **12× slower** (p95 3200 ms vs 265, max 7 s) for no ranking gain. **Verdict:**
keep Stage 3 as an *optional, calibration-improving* grounded signal with honest
abstention — but on current evidence it is **not** a standalone RAGTruth fix and must
not be presented as one. Its value will be decided by learned fusion (Stage 6), not
in isolation.

**Artifacts:** `reports/stage3_train.json`, `reports/stage3_grounded_eval.json`,
`models/stage3_entailment/`.

<!-- STAGE6_APPEND_ANCHOR -->

---

## Stage 6 — Learned fusion (logistic regression) over Stage 1/2/3 signals

**Question this stage answers:** does a *learned* fusion of the three base signals
improve **cross-dataset ranking** over the best single signal, or not? Per the brief:
be conservative — if fusion does not help generalization, say so plainly. Stage 1/2/3
artifacts are untouched; nothing is wired into live orchestration.

### 1. Fusion training data (provenance — leakage-clean, out-of-sample for every base signal)
- **HaluEval `summarization`** — 1200 rows. The one HaluEval group held out from
  Stages 1, 2 **and** 3, so all three base-signal features on it are genuinely
  out-of-sample (avoids stacking leakage).
- **`tals/vitaminc`** (external grounded, fallback `pietrolesci/nli_fever`) — 1200 rows.
  `evidence`→context, `claim`→response, SUPPORTS→0 / REFUTES→1, NOT-ENOUGH-INFO dropped.
  Never seen by any stage.
- Pool = **2400 rows**, pos_rate 0.458, entailment available on **2400/2400** (both
  sources carry context). Split **train 1920 / val 480**, stratified by source×label.
- **NEVER trained on:** public RAGTruth and the local 769 (`Datasets/*.jsonl`) — both
  are held-out **test** only.

### 2. Exact feature inputs (train == inference, canonical block order)
- `structural` = `[s1_prob]` — Stage-1 LR probability (Stage-1 artifact reused, unmodified).
- `encoder` = `[s2_mean, s2_max, s2_topk]` — Stage-2 per-claim probs, aggregated (k=3).
- `entailment` = `[s3_mean, s3_max, s3_topk, s3_avail]` — Stage-3 per-claim probs +
  availability flag. Missing Stage 3 → mean-imputed (train constants) with `s3_avail=0`;
  entailment-present train rows are **duplicated with entailment masked** so the LR can
  learn the Stage-3-absent regime. Impute constants (train-only): s1 0.751, s2
  mean/max/topk 0.876/0.886/0.879, s3 mean/max/topk 0.531/0.543/0.534.
- Model: `StandardScaler → LogisticRegression(C=1.0, class_weight="balanced")`, one per
  ablation subset. Degrades honestly (score None → route to Verify) iff *no* subset block
  is available.

### 3. Learned weights (standardized coefficients)
| cfg | intercept | s1 | s2_mean | s2_max | s2_topk | s3_mean | s3_max | s3_topk | s3_avail |
|---|---|---|---|---|---|---|---|---|---|
| A | -0.00 | 0.22 | | | | | | | |
| B | -0.02 | | 1.79 | -0.58 | -1.06 | | | | |
| D | -0.03 | 0.27 | 2.08 | -0.80 | -1.22 | | | | |
| G | -0.04 | 0.30 | 2.63 | -0.76 | -1.81 | 0.87 | -0.42 | -0.44 | -0.00 |

The fusion is **encoder-dominated** (G: s2_mean +2.63 vs s1 +0.30, s3_mean +0.87), and
`s3_avail`≈0 (the masked-duplicate augmentation kept the flag non-informative). This is
the mechanism behind the negative result below: fusion largely tracks the encoder, and
so inherits the encoder's weaknesses on the context-present set.

### 4. All 7 ablations × 4 held-out datasets (AUROC — ranking is the fair cross-model metric)
| cfg | subset | halluciguard_dataset | ragtruth (local) | truthfulqa | ragtruth_public |
|---|---|---|---|---|---|
| A | s1 | 0.966 | 0.402 | **0.529** | **0.578** |
| B | s2 | **0.993** | 0.592 | 0.481 | 0.282 |
| C | s3 | *degraded* | *degraded* | *degraded* | 0.289 |
| D | s1+s2 | 0.986 | 0.544 | 0.525 | 0.286 |
| E | s2+s3 | **0.993** | **0.596** | 0.481 | 0.275 |
| F | s1+s3 | 0.966 | 0.402 | **0.529** | 0.290 |
| G | s1+s2+s3 | 0.984 | 0.549 | 0.524 | 0.273 |

Best single-signal vs **full fusion G**, per dataset: halluciguard B 0.993 > G 0.984;
ragtruth B/E ~0.59 > G 0.549; truthfulqa A 0.529 ≳ G 0.524; **ragtruth_public A 0.578
≫ G 0.273**. **G never wins on any dataset** and is *strictly dominated* by the best
single signal everywhere — catastrophically so on the one grounded set.

Supporting metrics (thr@0.50 P/R/F1 · ECE · Brier · degraded · p50/p95 ms):
| cfg·dataset | P | R | F1 | ECE | Brier | degraded | p50/p95 ms |
|---|---|---|---|---|---|---|---|
| B · halluciguard | 0.97 | 0.98 | **0.97** | 0.37 | 0.165 | 0 | 56/68 |
| G · halluciguard | 1.00 | 0.31 | 0.48 | 0.38 | 0.208 | 0 | 57/69 |
| B · ragtruth_public | 0.03 | 0.01 | 0.01 | 0.30 | 0.331 | 0 | 388/1149 |
| A · ragtruth_public | 0.35 | 1.00 | 0.52 | **0.21** | 0.269 | 0 | 2/6 |
| G · ragtruth_public | 0.05 | 0.01 | 0.02 | 0.33 | 0.352 | 0 | 1528/**6842** |

thr@0.50 F1 is confounded by miscalibration under domain shift (fusion was calibrated
on the summarization+VitaminC pool; test-set ECE is 0.30–0.40 everywhere vs val 0.04–0.06),
so AUROC is the honest cross-model comparison.

### 5. Honest degradation & routing (the availability contract holds)
- **Config C (entailment-only) degrades on 100% of local rows** — 187/187, 84/84,
  498/498, **zero scored rows**: with no context it abstains and fabricates nothing.
  On context-present ragtruth_public it scores all 600.
- Configs with structural/encoder (A,B,D,E,F,G) never degrade on local (those blocks are
  always available) — the pipeline still routes correctly.
- Routing curve, ragtruth_public: encoder B gives a usable curve (≤70% routed → recall
  0.481); structural A, despite the best AUROC, compresses all scores high and routes
  ~100% (recall 0.0 at every ≤70% budget) — a degenerate operating point. G ≈ B (≤50%
  routed → recall 0.252) but at ~1.5s median / 6.8s p95 latency.

### 6. Comparison vs Stage 1/2/3 (single signals) and vs V1
**vs the base signals** — fusion is not a ranking improvement:
- HaluEval-family (halluciguard_dataset): Stage 2 alone **0.993**; fusion G 0.984, D
  0.986. Adding structural/entailment to the encoder *slightly dilutes* it.
- RAGTruth-public (context present — Stage 3's motivation): Stage 3 alone 0.331 (Stage-3
  eval, n=600) / here entailment-block 0.289; **structural alone is the only above-random
  config at 0.578**; every encoder/entailment-containing fusion is **sub-random**
  (0.27–0.29). Fusion made grounded RAGTruth *worse*, not better.
- What entailment still buys (unchanged from Stage 3): best calibration on the grounded
  set (C: ECE 0.228, lowest of any config there) — but not ranking.

**vs V1** (`../Detector_agent/eval_report.json`, local 769, threshold@0.50; V1 not re-run):
| dataset | V1 P/R/F1 | best V2-fusion cfg (AUROC) |
|---|---|---|
| halluciguard_dataset | 0.80 / **0.94** / **0.865** | B/E AUROC 0.993 (ranking far better; @0.50 F1 miscalibrated) |
| ragtruth (local) | 0.28 / 0.28 / 0.278 | B/E AUROC ~0.59 |
| truthfulqa | 0.82 / 0.10 / 0.179 | A AUROC 0.529 |
| combined@0.50 | 0.75 / 0.26 / 0.391 | — |
V1 has no RAGTruth-public. V1's calibrated @0.50 recall on halluciguard_dataset (0.94)
exceeds fusion's @0.50 recall because fusion is miscalibrated on these test
distributions; on **ranking** (AUROC) V2's encoder/fusion is far stronger in-family
(0.99 vs a threshold V1 report gives no AUROC for). No honest single-number win either way.

### 7. Verdict (conservative)
**Full fusion does NOT improve cross-dataset ranking.** On all four held-out sets the
best *single* signal matches or beats config G, and G is strictly dominated everywhere —
most starkly on the grounded RAGTruth-public set, where structural-alone ranks at 0.578
while every encoder/entailment fusion collapses to sub-random 0.27–0.29. The learned LR
is encoder-dominated (s2_mean +2.63) and therefore inherits the encoder's anti-correlation
on context-present RAGTruth; the entailment block and its availability flag add no ranking
signal (`s3_avail`≈0). Fusion also did not fix calibration on the test distributions
(ECE 0.30–0.40). Entailment's only retained value is calibration on grounded data, at a
real cost (~1.5s median / 6.8s p95 for config G).

The honest reading: **these three signals do not combine into a better cross-dataset
ranker via a linear stack.** No signal generalizes across the family boundary (encoder
wins in-family, structural edges the grounded set, neither transfers), so a single LR
cannot do better than pick one — and it picks the encoder, which hurts the grounded set.
This is a **negative result reported as such**, not tuned away. It does not justify
promoting fusion into orchestration, an MLP, or Stage-3 latency work on current evidence;
if pursued, the next lever is *per-domain routing / context-gated signal selection*
(explicitly out of scope here), not a bigger fusion head.

**Tests:** 79 passed (11 new fusion tests + existing 68) — feature-name/canonical order,
`_agg3`, entailment impute+flag, missingness augmentation (both regimes), degradation when
all blocks absent, model save/load roundtrip, scorer parity (ClaimSignals == components).

**Artifacts:** `models/stage6_fusion/{A..G}.joblib` + `manifest.json`,
`reports/stage6_fusion_train.json`, `reports/stage6_fusion_eval.json`,
`detector_v2/datasets/external_grounded.py`, `detector_v2/features/fusion.py`,
`detector_v2/scoring/fusion.py`, `tools/train_stage6_fusion.py`, `tools/eval_stage6_fusion.py`.

---

## Stage 7 — Final calibration + routing + detector selection

**Goal (fast track):** pick a FINAL core signal by evidence (no new large model),
calibrate it on a leakage-safe held-out split (NOT the 769), tune the routing
threshold for hallucination recall, emit the full 8-field contract with a
calibrated probability, and compare vs V1. No new transformer/MLP was trained.

**1) Error analysis → detector selection (held-out local sets, from the stage evals):**

| signal | halluciguard AUROC | ragtruth AUROC | truthfulqa AUROC | combined AUROC |
|---|---|---|---|---|
| Stage 1 structural LR | 0.966 | 0.407 | 0.529 | 0.568 |
| **Stage 2 encoder** | **0.993** | 0.411 | 0.482 | **0.643** |

Both collapse cross-dataset (ragtruth/truthfulqa near-random) — the known ceiling
from Stage 6, where fusion did **not** beat the best single signal. The encoder
has the best combined and in-family AUROC and is the always-on, context-free
learned signal, so **FINAL CORE = Stage 2 encoder**. Stage 3 entailment stays
**optional** (abstains without context; never wired into the default path).

**2) Calibration (leakage-safe):** the encoder's uncalibrated response-risk is
mapped to P(hallucinated) by **isotonic regression**, fit on **HaluEval
`summarization`** (the group held out from Stage 2 → out-of-sample) — the 769-row
local set and RAGTruth are **never** touched. The pool (800 rows, balanced) is
split 50/50 (stratified): one half fits the calibrator, the other tunes the
threshold and measures post-fit calibration (never fit on). Isotonic beat Platt:
**ECE 0.4232 → 0.0483**, Brier 0.4261 → 0.2167 on the held-out selection half.
`calibrated=True` is now emitted on the calibrated path.

**3) Threshold:** swept 0.05–0.95 on the calibrated selection half; chose the
**largest threshold still meeting a 0.90 hallucination-recall floor** (highest
recall at least routing cost) → **threshold = 0.35** (recall 0.905, routing 0.770
on the balanced calibration set). Full recall/routing tradeoff table is in
`reports/stage7_calibration.json`.

**4) confidence_score is NOT a copy of the probability:** it is the
threshold-relative decision distance — `(p−t)/(1−t)` above the boundary,
`(t−p)/t` below — so it is 0 at the gate (a coin-flip) and →1 at the extremes.
It measures decisiveness, an axis orthogonal to risk magnitude (smoke test: p=0.789,
confidence=0.676 — clearly different values).

**5) Final eval on the held-out local 769 (calibrated gate @0.35) vs V1 (risk_gate @0.30):**

| dataset | DetV2 recall | DetV2 routing | V1 recall | V1 routing |
|---|---|---|---|---|
| halluciguard | **0.958** | **0.497** | 0.938 | 0.615 |
| ragtruth | **0.556** | 0.655 | 0.333 | 0.250 |
| truthfulqa | **0.148** | 0.161 | 0.113 | 0.108 |
| **COMBINED** | **0.314** | 0.297 | 0.275 | 0.247 |

DetV2 catches **more** hallucinations than V1 on every dataset; combined recall
0.314 vs 0.275 (gate F1 0.435 vs 0.402). On the in-family set it is strictly
better (higher recall at lower routing). Routing cost is modestly higher combined
(0.297 vs 0.247). **Honest caveat:** truthfulqa recall stays low for both — a
context-free encoder cannot catch adversarial common-misconception answers — and
combined ranking AUROC 0.607 is the same cross-dataset ceiling reported in Stage 6.

**6) Honest degradation preserved:** empty/degenerate → degraded + Verify + null
probability (`calibrated=False`); all signals unavailable → `no_signal_available`
+ Verify; missing context → entailment abstains but the encoder core still scores
(never fabricated). Verified by tests.

**7) Smoke test** — "Who painted the Mona Lisa?" / "…painted by Pablo Picasso in
1750." → prob **0.789**, confidence 0.676, risk **HIGH**, action **Verify**,
`calibrated=True`. Correctly flagged.

**Tests:** 90 passed (11 new Stage-7 tests + existing 79) — calibrator
fit/monotonicity/range/save-load, decision-confidence (0 at threshold, 1 at
extremes, not a copy of p), and calibrated-pipeline PASS/VERIFY/threshold-flip/
missing-context/degraded-model/calibrated-range/canonical-contract paths.

**Verdict:** DetV2 (calibrated encoder core) is a modest but real improvement over
V1 on the held-out local set — higher hallucination recall everywhere and far
better calibration — while preserving honest degradation and the canonical
contract. Cross-dataset ranking outside the HaluEval family remains weak (known
limitation). Recommended: **ready for orchestration integration behind the
existing Accept/Verify gate**, with the caveat that truthfulqa-style adversarial
prompts should lean on the Verify path.

**Artifacts:** `detector_v2/calibration/calibrator.py`,
`detector_v2/calibration/build.py`, `models/stage7_calibration/calibrator.joblib`,
`reports/stage7_calibration.json`, `reports/stage7_eval.json`,
`tools/train_stage7_calibration.py`, `tools/eval_stage7.py`,
`tests/test_stage7_calibration.py`; pipeline gains an optional `calibrator`
(None → unchanged behavior). Stage 1/2/3/6 artifacts and V1 untouched.




