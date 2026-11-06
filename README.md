# FinFlow-Eval: A 90-Day Cash-Flow & Liquidity Evaluation Benchmark for LLMs

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Grader: Deterministic](https://img.shields.io/badge/Grader-Deterministic%20(90%2F90%20Verified)-success.svg)](graders/grade_task.py)
[![Zero-Cost Eval](https://img.shields.io/badge/Local%20Run-Standard%20Library%20($0)-brightgreen.svg)](run_eval.py)

## 1. Short Description
**FinFlow-Eval** is a deterministic evaluation benchmark designed to measure how effectively frontier Large Language Models (LLMs) solve forward-looking personal cash-flow, liquidity underwriting, and multi-constraint financial planning problems over a 90-day horizon. Across ten realistic multi-currency scenarios, OpenAI Codex (GPT-6 Luna) achieved a **10% pass rate** (1/10) with a mean rubric score of **0.471**, frequently failing at temporal conflict resolution, pending transaction accounting, and budget constraint enforcement.

## 2. Video Walkthrough Link
* **Video Walkthrough URL**: [Insert your unlisted Loom / YouTube / Google Drive share link here]
*(See the recording guide and word-for-word timed script in `.backup/VIDEO_WALKTHROUGH_SCRIPT.md`).*

---

## 3. Overview & Motivation

## Key Highlights

- **Real-World Financial Invariants:** Enforces that a user's liquid cash balance must never violate a strict minimum liquidity floor across any calendar day over a 91-day window ($t_0$ through $t_{90}$).
- **Heterogeneous Multi-Source Reconciliation:** Combines historical bank transaction ledgers, card authorization lifecycle states (`settled`, `pending`, `scheduled`, `cancelled`), foreign exchange tables, multi-option payment structures (lump-sum, installments, partial splits), and unstructured employer messages.
- **Deterministic 6-Tier Grader:** Zero reliance on stochastic LLM-as-a-judge. Scored purely via deterministic code evaluating schema, numeric safe headroom within currency tolerances, affordability classification, payment method selection, exact temporal tranche schedules, and grounded explanation verification.
- **90/90 Adversarial Test Suite:** Includes an automated test suite verifying that every reference answer achieves a perfect 1.0000 score while 70 adversarial traps (naive static balances, boundary violations, cancelled hold conflation, and output hedging) are strictly rejected.
- **Zero-Dependency Reproducibility:** The reference simulator, baseline models, and grader validation run entirely on the Python standard library ($0 API cost, zero external framework overhead).

---

## Benchmark Tasks Overview

FinFlow-Eval includes 10 comprehensive evaluation scenarios spanning 5 currencies (`USD`, `EUR`, `ZAR`, `INR`, `IDR`):

| Task | Request & Currency | Scenario & Core Challenge | Safe Headroom | Affordability Status / Method | Payment Plan | Discretionary Adjustments |
| :--- | :--- | :--- | ---: | :--- | :--- | :--- |
| **`task_01`** | `request_01` (ZAR) | Distinguish cancelled card authorization from pending hold; anchor on confirmed scheduled salary. | `25,256.00` | `affordable_now` / `full_payment` | `2024-03-03:25256` | `none` |
| **`task_02`** | `request_02` (IDR) | Parse Indonesian HR payroll notice; enforce strict user installment duration limit ($\le$ 7 months). | `18,336,765.49` | `affordable_with_plan` / `installments` | 3 monthly tranches (`15,952,906.67` each) | `none` |
| **`task_03`** | `request_04` (IDR) | Exclude unapproved performance bonus; account for scheduled school fee; defer purchase to future payday. | `10,609,286.07` | `affordable_later` / `wait` | `2024-06-15:12693000` | `none` |
| **`task_04`** | `request_06` (EUR) | Factor in temporary pay reduction; optimize permitted subscription cancellation (`stop:event_476`). | `603.82` | `affordable_with_plan` / `full_payment` | `2026-01-03:620.40` | `stop:event_476` |
| **`task_05`** | `request_07` (INR) | Employer shifts payday (15th $\rightarrow$ 23rd); detect mid-month liquidity trough and select installment smoothing. | `86,237.22` | `affordable_with_plan` / `installments` | 3 monthly tranches (`68,432.00` each) | `none` |
| **`task_06`** | `request_11` (IDR) | Exclude unconfirmed sales commissions; reduce flexible dining expense to its minimum allowed floor. | `12,429,866.62` | `affordable_with_plan` / `full_payment` | `2025-05-03:13110000` | `reduce_to:event_989:665950` |
| **`task_07`** | `request_12` (ZAR) | Account for seasonal employment contract termination; respect user installment-only preference. | `65,164.00` | `affordable_with_plan` / `installments` | 3 monthly tranches (`22,590.19` each) | `none` |
| **`task_08`** | `request_19` (INR) | Compare zero-fee two-step partial payment split against fee-bearing financing options. | `30,022.77` | `affordable_with_plan` / `partial_payment` | `2024-09-04:30022.77\|2024-09-15:9637.23` | `none` |
| **`task_09`** | `request_21` (USD) | Exclude unrealized non-cash portfolio valuation; combine compound multi-event flexible budget reductions. | `1,543.35` | `affordable_with_plan` / `full_payment` | `2026-04-03:1574.40` | `stop:event_1815\|reduce_to:event_1816:23.50` |
| **`task_10`** | `request_25` (IDR) | Reconcile cross-currency foreign income (USD $\rightarrow$ IDR FX table); verify complete mathematical infeasibility. | `376,083.40` | `not_affordable` / `not_recommended` | `none` | `none` |

---

## Grading Methodology & Verification

The evaluation engine (`graders/grade_task.py`) scores model outputs strictly via code against verified reference solutions across 6 weighted gates:

1. **Schema & Anti-Gaming Validity (10%, Hard Gate):** Enforces exact JSON keys, valid enum states, bounded numbers, and rejects multi-answer "hedging".
2. **Immediate Safe Headroom Accuracy (25%, Hard Gate):** Compares `amount_safe_to_pay` against simulated ground-truth within currency-specific tolerances ($\max(\text{currency\_floor}, 0.2\% \times \text{requested\_amount})$).
3. **Affordability Status Classification (20%, Hard Gate):** Exact match on `affordable_now`, `affordable_with_plan`, `affordable_later`, or `not_affordable`.
4. **Recommended Payment Method Selection (15%, Hard Gate):** Exact match on `full_payment`, `installments`, `partial_payment`, `wait`, or `not_recommended`.
5. **Temporal Schedule & Date Accuracy (15%, Hard Gate):** Parses and checks each tranche date and amount in `payment_plan` and verifies `earliest_date_for_full_payment`.
6. **Budget Optimization & Explanation Grounding (15%, Hard Gate):** Verifies that flexible spending reduction tokens (`stop:<id>`, `reduce_to:<id>:<amt>`) match minimal mathematical requirements and that explanations cite actual case figures without contradiction.

### Grader Unit & Adversarial Test Suite

The grader is validated via `graders/test_graders.py`, running **90 automated test cases**:
- **20 Positive Checks:** Verifies exact reference answers and benign formatting variants (markdown code fences, reversed ordering of independent cuts).
- **70 Adversarial Checks:** Confirms rejection of naive balance heuristics, out-of-bounds safe amounts, ungrounded explanations, extra JSON keys, and multi-object payloads.

---

## Measured Benchmark Results

| System | Description | Evaluations | Pass Rate | Mean Score | Exact Safe Headroom | Runtime | API Cost |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`oracle_reference`** | Deterministic 90-day simulator self-consistency check | 10 / 10 | **100.0%** | **1.0000** | 10 / 10 | 0.07s | $0 |
| **`rolling_mean_no_override`** | Ablation baseline (ignores message overrides & flexible cuts) | 15 / 30 | **50.0%** | **0.7480** | 27 / 30 | 0.30s | $0 |
| **`naive_ledger_baseline`** | Static heuristic baseline (`available_balance - min_balance`) | 6 / 30 | **20.0%** | **0.6015** | 6 / 30 | 0.02s | $0 |
| **`codex_gpt-6-luna`** | OpenAI Codex CLI (low reasoning effort, single trial per task) | 1 / 10 | **10.0%** | **0.4710** | 1 / 10 | 132.1s | — |

### Key Failure Modes in Frontier Models

1. **Failure to Reconcile Asynchronous State Overrides:** In Task 02, the model selected a valid installment option but reported `amount_safe_to_pay: 0` and misdated the earliest safe full-payment date by a month because it failed to correctly align the employer's salary effective date.
2. **Missing Liquidity Troughs:** In Task 05, when an employer moved payday from the 15th to the 23rd, the model opted to wait rather than selecting an available three-installment schedule that successfully smoothed cash flow above the floor.
3. **Non-Cash Asset Conflation:** In Task 09, models frequently treated unrealized equity investments as liquid spendable cash, recommending immediate full payment when the liquid ledger was below the safety margin.

---

## Quickstart & Reproduction

### Prerequisites
- Python 3.10 or newer.
- Standard library only (no external dependencies required for local baselines or grader validation).

### Installation

```bash
git clone https://github.com/anubhavjha084-web/finflow-eval.git
cd finflow-eval

# Optional: install dev dependencies for API evaluations
pip install -r requirements.txt
```

### 1. Validate All Graders (90 Automated Checks, $0 Cost)

```bash
python run_eval.py --mode validate-graders
```

### 2. Run Local Baseline Evaluations ($0 Cost)

```bash
python run_eval.py --mode baseline
```

### 3. Run Frontier API Evaluations (Optional)

Configure API keys in `.env` (copy from `.env.example`):

```bash
# OpenAI
python run_eval.py --mode api --provider openai --model gpt-4o --trials 3 --temperature 0.2

# Anthropic
python run_eval.py --mode api --provider anthropic --model claude-3-7-sonnet-20250219 --trials 3

# Google Gemini
python run_eval.py --mode api --provider gemini --model gemini-2.5-pro --trials 3
```

---

## Repository Structure

```
finflow-eval/
├── README.md                      # Benchmark documentation and results
├── requirements.txt               # Optional API runner dependencies
├── .env.example                   # Template for API credentials
├── LICENSE                        # MIT License
├── run_eval.py                    # Multi-provider evaluation harness & baseline runner
├── run_codex_eval.py              # Codex CLI execution runner
├── code/
│   └── main.py                    # Legacy static heuristic baseline
├── graders/
│   ├── grade_task.py              # Deterministic 6-dimension evaluation grader
│   └── test_graders.py            # 90-check unit & adversarial validation suite
├── reference/
│   ├── daily_simulator.py         # Deterministic 90-day cash-flow simulation engine
│   ├── solve_reference.py         # Reference solver entry point
│   └── task_*_solution.json       # Ground-truth solution specifications (Tasks 01–10)
├── results/
│   ├── benchmark_summary.json     # Aggregated benchmark metrics
│   ├── grader_validation_report.json # Detailed report of 90/90 validation checks
│   └── *_results.json             # Detailed run summaries per system
├── tasks/
│   ├── tasks_overview.json        # Index of all 10 task scenarios
│   └── task_*.json                # Complete task prompts & structured input data
└── transcripts/
    └── *_transcripts.json         # Raw model completions and reasoning traces
```

---

## Prior Work & Data Provenance

- **Prior Work:** This project builds upon an earlier algorithmic prototype (`FinDecision-v0`, located in `code/main.py`), which evaluated basic static cash balances without multi-day temporal simulation. FinFlow-Eval significantly expands this into a full 90-day forward simulator with date-anchored recurring cash flows, currency conversions, and natural language message parsing.
- **Data Provenance:** All 10 packaged tasks and financial profiles use 100% synthetic data. They contain zero confidential employer, client, or personal financial information. All balances and transaction dates were synthesized specifically to test edge cases in forward liquidity modeling.

---

## AI Assistance & Human Review Disclosure

- **AI Tools Used:** AI assistance was utilized to assist with drafting initial code structures, exploring scenario edge cases, and humanizing code comments and variable naming conventions.
- **Human Verification:** All mathematical simulation rules, ground-truth answer keys, rubric weightings, and adversarial grader assertions were directly authored, reviewed, and validated by the author to ensure 100% deterministic reproducibility and domain correctness.

---

## License

This project is licensed under the [MIT License](LICENSE).
