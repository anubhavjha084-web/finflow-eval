#!/usr/bin/env python3
"""Run FinFlow tasks through Codex CLI without an API key."""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from run_eval import (
    BASE_DIR,
    evaluate_system,
    run_grader_validation_suite,
    update_benchmark_summary,
)


def parse_usage(events_text: str) -> dict:
    usage = {}
    for line in events_text.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "turn.completed":
            usage = event.get("usage", {})
    return usage


def codex_response(model: str, effort: str, prompt: str) -> tuple[str, dict]:
    codex = shutil.which("codex")
    if not codex:
        raise RuntimeError("Codex CLI was not found on PATH.")

    with tempfile.TemporaryDirectory(prefix="finflow-codex-") as workdir:
        work_path = Path(workdir)
        resp_file = work_path / "response.txt"
        cmd = [
            codex,
            "exec",
            "--ephemeral",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--model",
            model,
            "--config",
            f'model_reasoning_effort="{effort}"',
            "--json",
            "--output-last-message",
            str(resp_file),
            "--cd",
            str(work_path),
            "-",
        ]
        t0 = time.perf_counter()
        res = subprocess.run(
            cmd,
            input=prompt,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=600,
            check=False,
        )
        elapsed = round(time.perf_counter() - t0, 3)
        response = resp_file.read_text(encoding="utf-8") if resp_file.exists() else ""
        if res.returncode != 0 or not response.strip():
            raise RuntimeError(
                f"Codex CLI failed (exit={res.returncode}); stderr: {res.stderr[-2000:]}"
            )

    usage = parse_usage(res.stdout)
    ver = subprocess.run(
        [codex, "--version"], capture_output=True, text=True, encoding="utf-8", errors="replace"
    ).stdout.strip()

    return response, {
        "latency_sec": elapsed,
        "input_tokens": usage.get("input_tokens", 0),
        "output_tokens": usage.get("output_tokens", 0),
        "estimated_cost_usd": 0.0,
        "billing_note": (
            "No API key or API dollar billing used. This run used the signed-in ChatGPT/Codex "
            "plan allowance; plan limits apply and dollar cost telemetry is unavailable."
        ),
        "codex_cli_version": ver,
        "exit_code": res.returncode,
        "stderr": res.stderr[-4000:],
        "codex_event_stream": res.stdout,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-6-luna")
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--effort", default="low")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("--trials must be at least 1")

    validation = run_grader_validation_suite()
    if validation["failed_tests"]:
        raise SystemExit(
            f"Grader validation failed: {validation['failed_tests']}/{validation['total_tests']} checks."
        )

    system_id = "codex_" + "".join(c if c.isalnum() or c in "-_" else "_" for c in args.model)
    version = subprocess.run(
        [shutil.which("codex") or "codex", "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    ).stdout.strip()
    report = evaluate_system(
        system_id=system_id,
        system_type="chatgpt_plan_codex_cli",
        model_version=f"{args.model} via {version}",
        settings={
            "route": "Codex CLI signed in with ChatGPT; no API key",
            "model": args.model,
            "reasoning_effort": args.effort,
            "temperature": "Codex CLI default; not exposed by this runner",
            "trials_per_task": args.trials,
            "cost_note": "Uses ChatGPT/Codex plan allowance; no API dollar-cost telemetry.",
        },
        trials_per_task=args.trials,
        generator_fn=lambda task, trial: codex_response(args.model, args.effort, task["prompt"]),
    )
    report["grader_validation"] = {
        "total_tests": validation["total_tests"],
        "passed_tests": validation["passed_tests"],
        "failed_tests": validation["failed_tests"],
    }
    result_path = Path(BASE_DIR) / "results" / f"{system_id}_results.json"
    result_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    update_benchmark_summary(report)
    print(json.dumps(report["aggregate_metrics"], indent=2))
    print(f"Saved: {result_path}")
    print(f"Saved: {Path(BASE_DIR) / 'transcripts' / (system_id + '_transcripts.json')}")


if __name__ == "__main__":
    main()
