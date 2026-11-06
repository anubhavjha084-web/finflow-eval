#!/usr/bin/env python3
"""Daily cash-flow reference solver entry point."""

try:
    from .daily_simulator import parse_date, simulate_decision, solve_all_tasks, solve_task
except ImportError:
    from daily_simulator import parse_date, simulate_decision, solve_all_tasks, solve_task

if __name__ == "__main__":
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    print(json.dumps(solve_all_tasks(str(root / "tasks")), indent=2))
