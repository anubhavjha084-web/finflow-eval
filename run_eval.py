#!/usr/bin/env python3
"""
FinFlow-Eval: Multi-Provider Evaluation Harness & Baseline Runner.
Modes: validate-graders, oracle, baseline, api (OpenAI/Anthropic/Gemini).
"""

import argparse
import datetime
import json
import math
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Dict, Any, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from graders.grade_task import grade_prediction
from graders.test_graders import run_grader_validation_suite
from reference.solve_reference import solve_task, simulate_decision


def load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and not v.startswith("your_") and k not in os.environ:
                os.environ[k] = v


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased estimator of pass@k (Chen et al., 2021)."""
    if n < k:
        k = n
    if c == 0:
        return 0.0
    if n - c < k:
        return 1.0
    return round(1.0 - (math.comb(n - c, k) / math.comb(n, k)), 4)


def compute_pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k reliability metric: (c/n)^k."""
    if n == 0:
        return 0.0
    p = c / n
    return round(p ** k, 4)


def simulate_naive_ledger_baseline(task_data: Dict[str, Any]) -> Tuple[str, str]:
    """Reproduces the original static-balance heuristic from code/main.py."""
    inp = task_data["input_data"]
    prof = inp["user_profile"]
    req = inp["request"]
    opts = inp["payment_options"]

    rid = req["request_id"]
    avail = float(prof["current_available_balance"])
    min_bal = float(prof["minimum_balance_to_keep"])
    amt = float(req["requested_amount"])
    rdate = req["request_date"]
    partial_ok = str(req["allows_partial_payment"]).lower() == "true"
    methods = [m.strip() for m in prof["payment_methods_user_will_consider"].split("|") if m.strip()]
    max_mo = int(prof["max_installment_months"]) if prof.get("max_installment_months") else 999

    headroom = max(0.0, avail - min_bal)
    safe = round(min(amt, headroom), 2)

    reasoning = (
        f"[Naive Ledger Baseline (`code/main.py`)] Static balance check only: "
        f"current_available_balance ({avail}) - minimum_balance_to_keep ({min_bal}) = {headroom:.2f} headroom. "
        f"Ignored 90-day recurring expenses, pending authorizations, FX rates, and employer messages."
    )

    if safe >= amt and "full_payment" in methods:
        pred = {
            "request_id": rid,
            "amount_safe_to_pay": safe,
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "payment_plan": f"{rdate}:{amt:g}",
            "earliest_date_for_full_payment": rdate,
            "spending_changes_needed": "none",
            "decision_explanation": (
                f"Static balance check: paying {prof['home_currency']} {amt:,.2f} today leaves "
                f"{avail - amt:,.2f} (>= {min_bal:,.2f} minimum)."
            ),
        }
        return json.dumps(pred, indent=2), reasoning

    valid_inst = [
        o for o in opts
        if o["payment_method"] == "installments"
        and "installments" in methods
        and int(o["number_of_payments"]) <= max_mo
        and float(o["payment_amount"]) <= headroom
    ]
    if valid_inst:
        best = sorted(valid_inst, key=lambda x: float(x["total_payable_amount"]))[0]
        n_pay = int(best["number_of_payments"])
        freq = int(best["payment_frequency_days"])
        dt0 = datetime.datetime.strptime(best["first_payment_date"], "%Y-%m-%d")
        parts = [
            f"{(dt0 + datetime.timedelta(days=i * freq)).strftime('%Y-%m-%d')}:{best['payment_amount']}"
            for i in range(n_pay)
        ]
        pred = {
            "request_id": rid,
            "amount_safe_to_pay": safe,
            "affordability_status": "affordable_with_plan",
            "recommended_payment_method": "installments",
            "payment_plan": "|".join(parts),
            "earliest_date_for_full_payment": rdate if safe >= amt else "",
            "spending_changes_needed": "none",
            "decision_explanation": (
                f"Static balance check: use {n_pay} installments of {prof['home_currency']} {best['payment_amount']} "
                f"starting {best['first_payment_date']} (keeping >= {min_bal:,.2f} minimum)."
            ),
        }
        return json.dumps(pred, indent=2), reasoning

    if partial_ok and "partial_payment" in methods and safe > 0:
        rem = round(amt - safe, 2)
        pred = {
            "request_id": rid,
            "amount_safe_to_pay": safe,
            "affordability_status": "affordable_with_plan",
            "recommended_payment_method": "partial_payment",
            "payment_plan": f"{rdate}:{safe:g}|{req['desired_completion_date']}:{rem:g}",
            "earliest_date_for_full_payment": req["desired_completion_date"],
            "spending_changes_needed": "none",
            "decision_explanation": (
                f"Static balance check: pay {safe:g} today and {rem:g} on {req['desired_completion_date']} (minimum {min_bal:,.2f})."
            ),
        }
        return json.dumps(pred, indent=2), reasoning

    pred = {
        "request_id": rid,
        "amount_safe_to_pay": safe,
        "affordability_status": "not_affordable",
        "recommended_payment_method": "not_recommended",
        "payment_plan": "none",
        "earliest_date_for_full_payment": "",
        "spending_changes_needed": "none",
        "decision_explanation": (
            f"Static balance check: insufficient headroom ({safe:g}) to cover {amt:g} while protecting {min_bal:,.2f}."
        ),
    }
    return json.dumps(pred, indent=2), reasoning


def simulate_rolling_mean_no_override_agent(task_data: Dict[str, Any]) -> Tuple[str, str]:
    """90-day cash-flow ablation simulator with message overrides and cuts disabled."""
    pred = simulate_decision(
        task_data,
        apply_messages=False,
        allow_spending_cuts=False,
        allow_partial_payment=False,
        filter_non_cash=False,
    )
    reasoning = (
        f"[Rolling-Mean 90-Day Simulator (Ablation: apply_messages=False, allow_spending_cuts=False)] "
        f"Computed 90-day recurring ledger projection from historical financial_events and exchange_rates "
        f"for {task_data['task_id']} ({pred['request_id']}): safe_to_pay={pred['amount_safe_to_pay']}, "
        f"status={pred['affordability_status']}, method={pred['recommended_payment_method']}."
    )
    return json.dumps(pred, indent=2), reasoning


def _urlopen_with_retry(req: urllib.request.Request, timeout: int = 120, max_retries: int = 3) -> Dict[str, Any]:
    """HTTP request with exponential backoff on 429/5xx."""
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            err_body = ""
            try:
                err_body = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            if e.code in (429, 500, 502, 503, 504) and attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [API Retry {attempt}/{max_retries}] HTTP {e.code} — retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {e.code} error from provider API: {err_body}") from e
        except urllib.error.URLError as e:
            if attempt < max_retries:
                wait = 2 ** attempt
                print(f"  [Network Retry {attempt}/{max_retries}] {e.reason} — retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"Network error calling provider API: {e.reason}") from e
    raise RuntimeError("Exhausted API retries.")


def call_live_llm_api(
    provider: str,
    model: str,
    prompt: str,
    temperature: float = 0.0,
) -> Tuple[str, Dict[str, Any]]:
    """Call OpenAI/Anthropic/Gemini REST API, return (text, usage_metadata)."""
    t0 = time.time()
    provider = provider.lower().strip()

    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment or .env file.")
        url = "https://api.openai.com/v1/chat/completions"
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if not model.startswith("o1") and not model.startswith("o3") and not model.startswith("gpt-5"):
            payload["temperature"] = temperature
            payload["response_format"] = {"type": "json_object"}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        data = _urlopen_with_retry(req)
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        in_tok = usage.get("prompt_tokens", 0)
        out_tok = usage.get("completion_tokens", 0)
        cost = round((in_tok * 2.5e-6) + (out_tok * 10.0e-6), 6)
        return content, {
            "latency_sec": round(time.time() - t0, 3),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_cost_usd": cost,
        }

    elif provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY in environment or .env file.")
        url = "https://api.anthropic.com/v1/messages"
        payload = {
            "model": model,
            "max_tokens": 2048,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
            method="POST",
        )
        data = _urlopen_with_retry(req)
        blocks = data.get("content", [])
        content = "\n".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage", {})
        in_tok = usage.get("input_tokens", 0)
        out_tok = usage.get("output_tokens", 0)
        cost = round((in_tok * 3.0e-6) + (out_tok * 15.0e-6), 6)
        return content, {
            "latency_sec": round(time.time() - t0, 3),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_cost_usd": cost,
        }

    elif provider == "gemini":
        api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY) in environment or .env file.")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "responseMimeType": "application/json",
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        data = _urlopen_with_retry(req)
        parts = data["candidates"][0]["content"]["parts"]
        content = "\n".join(p.get("text", "") for p in parts)
        usage = data.get("usageMetadata", {})
        in_tok = usage.get("promptTokenCount", 0)
        out_tok = usage.get("candidatesTokenCount", 0)
        cost = round((in_tok * 1.25e-6) + (out_tok * 5.0e-6), 6)
        return content, {
            "latency_sec": round(time.time() - t0, 3),
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "estimated_cost_usd": cost,
        }

    else:
        raise ValueError(f"Unsupported provider '{provider}'. Choose from: openai, anthropic, gemini.")


def update_benchmark_summary(new_report: Dict[str, Any]) -> None:
    """Merge new_report into results/benchmark_summary.json."""
    spath = os.path.join(BASE_DIR, "results", "benchmark_summary.json")
    if os.path.exists(spath):
        try:
            with open(spath, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            summary = {"systems": []}
    else:
        summary = {"systems": []}

    systems = [s for s in summary.get("systems", []) if s.get("system_id") != new_report["system_id"]]
    systems.append(new_report)
    summary["generated_at_utc"] = new_report["timestamp_utc"]
    summary["systems"] = systems

    with open(spath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def evaluate_system(
    system_id: str,
    system_type: str,
    model_version: str,
    settings: Dict[str, Any],
    trials_per_task: int,
    generator_fn,
) -> Dict[str, Any]:
    tasks_dir = os.path.join(BASE_DIR, "tasks")
    ref_dir = os.path.join(BASE_DIR, "reference")
    results_dir = os.path.join(BASE_DIR, "results")
    transcripts_dir = os.path.join(BASE_DIR, "transcripts")

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    t_start = time.time()

    task_results = []
    transcripts = []
    total_cost = 0.0
    passed_total = 0
    n_trials = 0
    sum_scores = 0.0
    sum_p1 = 0.0
    sum_pk = 0.0
    sum_phk = 0.0

    for i in range(1, 11):
        tid = f"task_{i:02d}"
        with open(os.path.join(tasks_dir, f"{tid}.json"), "r", encoding="utf-8") as f:
            task_data = json.load(f)
        with open(os.path.join(ref_dir, f"{tid}_solution.json"), "r", encoding="utf-8") as f:
            ref_sol = json.load(f)

        trials = []
        c = 0
        tsc_sum = 0.0

        for t in range(1, trials_per_task + 1):
            raw, meta = generator_fn(task_data, t)
            grade = grade_prediction(raw, ref_sol, task_data)
            if grade["passed"]:
                c += 1
                passed_total += 1
            n_trials += 1
            tsc_sum += grade["total_score"]
            sum_scores += grade["total_score"]
            total_cost += meta.get("estimated_cost_usd", 0.0)

            trials.append({
                "trial": t,
                "passed": grade["passed"],
                "total_score": grade["total_score"],
                "subscores": grade["subscores"],
                "checks": grade["checks"],
                "error": grade["error"],
                "metadata": meta,
                "raw_output": raw,
            })

            transcripts.append({
                "system_id": system_id,
                "model_version": model_version,
                "task_id": tid,
                "request_id": task_data["original_request_id"],
                "title": task_data["title"],
                "trial": t,
                "timestamp_utc": ts,
                "passed": grade["passed"],
                "total_score": grade["total_score"],
                "error": grade["error"],
                "expected_output": ref_sol["expected_output"],
                "model_raw_output": raw,
                "diagnostic_notes": meta.get("diagnostic_notes", ""),
            })

        p1 = compute_pass_at_k(trials_per_task, c, 1)
        pk = compute_pass_at_k(trials_per_task, c, trials_per_task)
        phk = compute_pass_hat_k(trials_per_task, c, trials_per_task)

        sum_p1 += p1
        sum_pk += pk
        sum_phk += phk

        task_results.append({
            "task_id": tid,
            "request_id": task_data["original_request_id"],
            "title": task_data["title"],
            "difficulty": task_data["difficulty"],
            "trials": trials_per_task,
            "trials_passed": c,
            "mean_score": round(tsc_sum / trials_per_task, 4),
            "pass_at_1": p1,
            "pass_at_k": pk,
            "pass_hat_k": phk,
            "primary_error": trials[0]["error"],
            "subscores_trial_1": trials[0]["subscores"],
        })

    ntasks = len(task_results)
    report = {
        "benchmark": "FinFlow-Eval v1.0",
        "system_id": system_id,
        "system_type": system_type,
        "model_version": model_version,
        "settings": settings,
        "timestamp_utc": ts,
        "num_tasks": ntasks,
        "trials_per_task": trials_per_task,
        "total_trials": n_trials,
        "total_passed_trials": passed_total,
        "aggregate_metrics": {
            "overall_pass_rate": round(passed_total / max(1, n_trials), 4),
            "mean_rubric_score": round(sum_scores / max(1, n_trials), 4),
            "mean_pass_at_1": round(sum_p1 / ntasks, 4),
            "mean_pass_at_k": round(sum_pk / ntasks, 4),
            "mean_pass_hat_k": round(sum_phk / ntasks, 4),
        },
        "runtime_seconds": round(time.time() - t_start, 4),
        "estimated_total_cost_usd": round(total_cost, 6),
        "per_task_results": task_results,
    }

    out_path = os.path.join(results_dir, f"{system_id}_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    tr_path = os.path.join(transcripts_dir, f"{system_id}_transcripts.json")
    with open(tr_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)

    return report


def main() -> None:
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    parser = argparse.ArgumentParser(description="FinFlow-Eval Benchmark Runner")
    parser.add_argument(
        "--mode",
        choices=["validate-graders", "oracle", "baseline", "api"],
        default="baseline",
        help="Execution mode (default: baseline)",
    )
    parser.add_argument("--provider", choices=["openai", "anthropic", "gemini"], default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o", help="Model ID for --mode api")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per task (k)")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for --mode api")
    args = parser.parse_args()

    if args.mode == "validate-graders":
        report = run_grader_validation_suite()
        val_path = os.path.join(BASE_DIR, "results", "grader_validation_report.json")
        with open(val_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"[Grader Validation] {report['passed_tests']}/{report['total_tests']} tests passed -> {val_path}")
        if report["failed_tests"] > 0:
            sys.exit(1)
        return

    if args.mode in ("oracle", "baseline"):
        val_report = run_grader_validation_suite()
        with open(os.path.join(BASE_DIR, "results", "grader_validation_report.json"), "w", encoding="utf-8") as f:
            json.dump(val_report, f, indent=2)

        oracle_rep = evaluate_system(
            system_id="oracle_reference",
            system_type="reference_solver",
            model_version="solve_reference.py (v1.1 daily-simulator self-consistency check)",
            settings={"temperature": 0.0, "deterministic": True},
            trials_per_task=1,
            generator_fn=lambda td, _: (
                json.dumps(solve_task(td), indent=2),
                {"latency_sec": 0.001, "estimated_cost_usd": 0.0, "diagnostic_notes": "Reference solver output for self-consistency checking."},
            ),
        )
        print(
            f"[Oracle Reference] Pass Rate: {oracle_rep['aggregate_metrics']['overall_pass_rate']*100:.1f}% "
            f"(Mean Score: {oracle_rep['aggregate_metrics']['mean_rubric_score']:.4f})"
        )
        if args.mode == "oracle":
            return

        naive_rep = evaluate_system(
            system_id="naive_ledger_baseline",
            system_type="heuristic_baseline",
            model_version="FinDecision-v0 (code/main.py static balance heuristic)",
            settings={"temperature": 0.0, "deterministic": True, "source": "code/main.py"},
            trials_per_task=3,
            generator_fn=lambda td, _: (
                simulate_naive_ledger_baseline(td)[0],
                {
                    "latency_sec": 0.001,
                    "estimated_cost_usd": 0.0,
                    "diagnostic_notes": simulate_naive_ledger_baseline(td)[1],
                },
            ),
        )
        print(
            f"[Naive Ledger Baseline] Pass Rate: {naive_rep['aggregate_metrics']['overall_pass_rate']*100:.1f}% "
            f"(Mean Score: {naive_rep['aggregate_metrics']['mean_rubric_score']:.4f}, "
            f"pass@3: {naive_rep['aggregate_metrics']['mean_pass_at_k']*100:.1f}%, "
            f"pass^3: {naive_rep['aggregate_metrics']['mean_pass_hat_k']*100:.1f}%)"
        )

        rolling_rep = evaluate_system(
            system_id="rolling_mean_no_override_agent",
            system_type="partial_simulation_baseline",
            model_version="RollingMean-90D-v1 (90-day ledger simulator without NL message overrides)",
            settings={"temperature": 0.0, "deterministic": True},
            trials_per_task=3,
            generator_fn=lambda td, _: (
                simulate_rolling_mean_no_override_agent(td)[0],
                {
                    "latency_sec": 0.002,
                    "estimated_cost_usd": 0.0,
                    "diagnostic_notes": simulate_rolling_mean_no_override_agent(td)[1],
                },
            ),
        )
        print(
            f"[Rolling-Mean 90D Simulator] Pass Rate: {rolling_rep['aggregate_metrics']['overall_pass_rate']*100:.1f}% "
            f"(Mean Score: {rolling_rep['aggregate_metrics']['mean_rubric_score']:.4f}, "
            f"pass@3: {rolling_rep['aggregate_metrics']['mean_pass_at_k']*100:.1f}%, "
            f"pass^3: {rolling_rep['aggregate_metrics']['mean_pass_hat_k']*100:.1f}%)"
        )

        systems_list = [oracle_rep, rolling_rep, naive_rep]
        res_dir = os.path.join(BASE_DIR, "results")
        for fn in sorted(os.listdir(res_dir)):
            if fn.endswith("_results.json") and fn not in (
                "oracle_reference_results.json",
                "rolling_mean_no_override_agent_results.json",
                "naive_ledger_baseline_results.json",
            ):
                try:
                    with open(os.path.join(res_dir, fn), "r", encoding="utf-8") as rf:
                        systems_list.append(json.load(rf))
                except Exception:
                    pass
        summary_bundle = {
            "generated_at_utc": oracle_rep["timestamp_utc"],
            "grader_validation": {
                "total_tests": val_report["total_tests"],
                "passed_tests": val_report["passed_tests"],
                "failed_tests": val_report["failed_tests"],
            },
            "systems": systems_list,
        }
        with open(os.path.join(BASE_DIR, "results", "benchmark_summary.json"), "w", encoding="utf-8") as f:
            json.dump(summary_bundle, f, indent=2)
        print("[Summary Saved] results/benchmark_summary.json")
        return

    if args.mode == "api":
        safe_model_id = args.model.replace("/", "_").replace(":", "_")
        rep = evaluate_system(
            system_id=f"{args.provider}_{safe_model_id}",
            system_type="frontier_llm_api",
            model_version=f"{args.provider}:{args.model}",
            settings={"provider": args.provider, "model": args.model, "temperature": args.temperature, "trials": args.trials},
            trials_per_task=args.trials,
            generator_fn=lambda td, _: call_live_llm_api(args.provider, args.model, td["prompt"], args.temperature),
        )
        update_benchmark_summary(rep)
        print(
            f"[Live API: {args.provider}:{args.model}] Pass Rate: {rep['aggregate_metrics']['overall_pass_rate']*100:.1f}% "
            f"| Mean Score: {rep['aggregate_metrics']['mean_rubric_score']:.4f} "
            f"| pass@{args.trials}: {rep['aggregate_metrics']['mean_pass_at_k']*100:.1f}% "
            f"| pass^{args.trials}: {rep['aggregate_metrics']['mean_pass_hat_k']*100:.1f}% "
            f"| Cost: ${rep['estimated_total_cost_usd']:.4f}"
        )


if __name__ == "__main__":
    main()
