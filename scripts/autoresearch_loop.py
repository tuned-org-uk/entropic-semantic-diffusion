#!/usr/bin/env python3
"""Autonomous autoresearch loop driver for ESDM.

Parses the loop configuration from AGENTS.md, establishes a baseline, then
iterates: modify -> commit -> verify -> guard -> keep/discard -> log.

The modification strategy is a bounded parameter sweep over the exp3
hyperparameters exposed in scripts/verify_fitness.py.  Each iteration applies
one atomic change to the verify script, tests it, and reverts if the composite
fitness score does not improve.

Usage:
    .venv/bin/python scripts/autoresearch_loop.py [--iterations N] [--evals]
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path


REPO = Path("/Users/tuned-silicon/papers/entropic-semantic-diffusion")
AGENTS = REPO / "AGENTS.md"
VERIFY_CMD = ".venv/bin/python scripts/verify_fitness.py"
GUARD_CMD = ".venv/bin/python -m pytest tests/ -q --tb=no"


def run(cmd: str, cwd: Path = REPO, check: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command and return the completed process."""
    return subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, check=check)


def git_commit(message: str) -> str:
    """Stage all changes and commit with the experiment prefix."""
    run("git add -A", check=True)
    run(
        f'git -c user.name=Mec-iS -c user.email=tunedconsulting@gmail.com '
        f'commit -m "experiment: {message}"',
        check=True,
    )
    return run("git rev-parse --short HEAD", check=True).stdout.strip()


def git_revert() -> None:
    """Revert the last commit without editing the message."""
    run("git revert HEAD --no-edit", check=True)


def extract_metric(output: str) -> float:
    """Extract the first float from command output."""
    for line in output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return float(line)
        except ValueError:
            continue
    raise ValueError(f"No numeric metric found in output:\n{output}")


def extract_guard_metric(output: str) -> str:
    """Extract number of passed tests from pytest output."""
    m = re.search(r"(\d+) passed", output)
    return m.group(1) if m else "-"


def parse_agents_config() -> dict:
    """Read the autoresearch loop configuration from AGENTS.md."""
    text = AGENTS.read_text()
    cfg = {
        "goal": "Maximize ESDM composite fitness score",
        "scope": "esdm/experiments/, esdm/models/, esdm/eval/fitness.py, scripts/verify_fitness.py",
        "metric": "Fitness score (higher is better)",
        "direction": "higher_is_better",
        "verify": VERIFY_CMD,
        "guard": GUARD_CMD,
        "iterations": 25,
    }
    # Table rows in the Loop configuration section.
    for line in text.splitlines():
        if "Goal" in line and "|" in line and "Maximize" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                cfg["goal"] = parts[2]
        if "Verify" in line and "|" in line and "verify_fitness" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                cfg["verify"] = parts[2]
        if "Guard" in line and "|" in line and "pytest" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                cfg["guard"] = parts[2]
        if "Direction" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3 and parts[2]:
                cfg["direction"] = parts[2]
        if "Iterations" in line and "|" in line and "25" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 3:
                try:
                    cfg["iterations"] = int(parts[2].split()[0])
                except ValueError:
                    pass
    return cfg


def build_strategy() -> list[dict]:
    """Return a list of atomic parameter-change strategies.

    Each strategy rewrites the verify_fitness.py call to `run_single` with a
    different hyperparameter value.  The regex targets the keyword argument
    name and replaces its current value.
    """
    changes: list[dict] = []

    def add(name: str, value, description: str) -> None:
        changes.append({
            "param": name,
            "value": value,
            "desc": description,
        })

    # Barrier height
    add("kl_S_weight", 0.8, "increase kl_S_weight to 0.8")
    add("kl_S_weight", 1.0, "increase kl_S_weight to 1.0")
    add("kl_S_weight", 0.2, "decrease kl_S_weight to 0.2")
    add("kl_S_weight", 0.05, "decrease kl_S_weight to 0.05")

    # Decay rate
    add("kl_decay", 0.08, "increase kl_decay to 0.08")
    add("kl_decay", 0.1, "increase kl_decay to 0.1")
    add("kl_decay", 0.01, "decrease kl_decay to 0.01")

    # Pump ceiling
    add("lambda_clip", 100.0, "increase lambda_clip to 100.0")
    add("lambda_clip", 5.0, "decrease lambda_clip to 5.0")

    # SNR / cluster separation
    add("rho", 5.0, "increase rho to 5.0")
    add("rho", 1.0, "decrease rho to 1.0")

    # Run length
    add("n_steps", 500, "increase n_steps to 500")
    add("n_steps", 200, "decrease n_steps to 200")

    # WHT filter treatments (compared against none baseline)
    add("wht_filter", '"basic"', "switch to basic WHT filter")
    add("wht_filter", '"bivariate"', "switch to bivariate WHT filter")

    return changes


def apply_change(verify_path: Path, param: str, value) -> bool:
    """Apply a single parameter change to scripts/verify_fitness.py.

    Returns True if the file was actually changed.
    """
    text = verify_path.read_text()
    # Match keyword argument with a numeric/string value on the same line.
    pattern = rf"(\b{param}\s*=\s*)([^,\n]+)"
    replacement = rf"\g<1>{value}"
    new_text, count = re.subn(pattern, replacement, text, count=1)
    if count == 0:
        return False
    verify_path.write_text(new_text)
    return True


def append_tsv(tsv: Path, row: dict) -> None:
    """Append one row to the results TSV."""
    with tsv.open("a") as f:
        f.write(
            f"{row['iteration']}\t{row['timestamp']}\t{row['commit']}\t"
            f"{row['metric']:.2f}\t{row['delta']:.2f}\t{row['guard']}\t"
            f"{row['guard_metric']}\t{row['status']}\t{row['description']}\n"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="ESDM autoresearch loop driver")
    parser.add_argument("--iterations", type=int, default=None,
                        help="Override iteration count from AGENTS.md")
    parser.add_argument("--evals", action="store_true",
                        help="Enable mid-loop checkpoint summaries")
    parser.add_argument("--evals-interval", type=int, default=None,
                        help="Override checkpoint frequency")
    args = parser.parse_args()

    cfg = parse_agents_config()
    max_iterations = args.iterations if args.iterations is not None else cfg["iterations"]

    print("[autoresearch] mode: classic")
    print(f"Goal: {cfg['goal']}")
    print(f"Verify: {cfg['verify']}")
    print(f"Guard:  {cfg['guard']}")
    print(f"Iterations: {max_iterations}")

    # -------------------- Precondition checks --------------------
    run("git rev-parse --git-dir", check=True)
    status = run("git status --porcelain").stdout.strip()
    if status:
        print("WARNING: working tree is dirty")
        print(status)
    branch = run("git branch --show-current", check=True).stdout.strip()
    if branch == "HEAD":
        print("ERROR: detached HEAD", file=sys.stderr)
        return 1

    # Guard baseline
    print("\n[guard] establishing baseline...")
    guard = run(cfg["guard"])
    if guard.returncode != 0:
        print("ERROR: guard baseline failed", file=sys.stderr)
        print(guard.stdout)
        print(guard.stderr)
        return 1
    guard_metric_baseline = extract_guard_metric(guard.stdout)
    print(f"[guard] baseline passed ({guard_metric_baseline} tests)")

    # -------------------- Establish baseline --------------------
    print("\n[baseline] running verify command...")
    baseline_out = run(cfg["verify"], check=True).stdout
    baseline_metric = extract_metric(baseline_out)
    print(f"[baseline] metric = {baseline_metric:.2f}")

    commit = run("git rev-parse --short HEAD", check=True).stdout.strip()
    ts = datetime.datetime.now().astimezone().isoformat()
    out_dir = REPO / f"autoresearch/loop-{datetime.datetime.now():%y%m%d-%H%M}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_dir / "results.tsv"

    with tsv.open("w") as f:
        f.write(f"# metric_direction: {cfg['direction']}\n")
        f.write("iteration\ttimestamp\tcommit\tmetric\tdelta\tguard\tguard-metric\tstatus\tdescription\n")
    append_tsv(tsv, {
        "iteration": 0,
        "timestamp": ts,
        "commit": commit,
        "metric": baseline_metric,
        "delta": 0.0,
        "guard": "pass",
        "guard_metric": guard_metric_baseline,
        "status": "baseline",
        "description": "initial state",
    })

    # -------------------- Iteration loop --------------------
    verify_path = REPO / "scripts" / "verify_fitness.py"
    strategy = build_strategy()
    current_metric = baseline_metric
    kept = 0
    discarded = 0
    crashed = 0
    eval_interval = args.evals_interval
    if eval_interval is None and args.evals:
        eval_interval = max(1, max_iterations // 3) if max_iterations > 0 else 10

    for iteration in range(1, max_iterations + 1):
        print(f"\n========== Iteration {iteration}/{max_iterations} ==========")

        # Phase 1: Review
        print(run("git log --oneline -10").stdout)

        # Phase 2: Modify
        strat = strategy[(iteration - 1) % len(strategy)]
        changed = apply_change(verify_path, strat["param"], strat["value"])
        if not changed:
            print(f"[modify] no change applied for {strat['desc']}")
            append_tsv(tsv, {
                "iteration": iteration,
                "timestamp": datetime.datetime.now().astimezone().isoformat(),
                "commit": "-",
                "metric": current_metric,
                "delta": 0.0,
                "guard": "pass",
                "guard_metric": guard_metric_baseline,
                "status": "no-op",
                "description": strat["desc"],
            })
            continue
        print(f"[modify] {strat['desc']}")

        # Phase 3: Commit
        try:
            commit = git_commit(strat["desc"])
        except subprocess.CalledProcessError as exc:
            print(f"[commit] failed: {exc}")
            # Undo any working-tree changes
            run("git reset --hard HEAD", check=True)
            append_tsv(tsv, {
                "iteration": iteration,
                "timestamp": datetime.datetime.now().astimezone().isoformat(),
                "commit": "-",
                "metric": current_metric,
                "delta": 0.0,
                "guard": "pass",
                "guard_metric": guard_metric_baseline,
                "status": "hook-blocked",
                "description": strat["desc"],
            })
            continue

        # Phase 4: Verify
        try:
            verify = run(cfg["verify"], check=True)
            metric = extract_metric(verify.stdout)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"[verify] failed: {exc}")
            git_revert()
            crashed += 1
            append_tsv(tsv, {
                "iteration": iteration,
                "timestamp": datetime.datetime.now().astimezone().isoformat(),
                "commit": "-",
                "metric": 0.0,
                "delta": -current_metric,
                "guard": "fail",
                "guard_metric": "-",
                "status": "crash",
                "description": strat["desc"],
            })
            continue
        delta = metric - current_metric
        print(f"[verify] metric = {metric:.2f} (delta = {delta:+.2f})")

        # Phase 5: Guard
        guard = run(cfg["guard"])
        guard_pass = guard.returncode == 0
        guard_metric = extract_guard_metric(guard.stdout)
        print(f"[guard] {'pass' if guard_pass else 'FAIL'} ({guard_metric} tests)")

        # Phase 6: Decide
        if not guard_pass:
            git_revert()
            status = "crash"
            crashed += 1
        elif delta > 0:
            status = "keep"
            current_metric = metric
            kept += 1
        else:
            git_revert()
            status = "discard"
            discarded += 1
        print(f"[decide] {status}")

        # Phase 7: Log
        append_tsv(tsv, {
            "iteration": iteration,
            "timestamp": datetime.datetime.now().astimezone().isoformat(),
            "commit": commit if status != "discard" else "-",
            "metric": metric,
            "delta": delta,
            "guard": "pass" if guard_pass else "fail",
            "guard_metric": guard_metric,
            "status": status,
            "description": strat["desc"],
        })

        # Eval checkpoint
        if args.evals and eval_interval and iteration % eval_interval == 0:
            print(f"--- Eval Checkpoint (iterations {iteration - eval_interval + 1}-{iteration}) ---")
            print(f"Metric: {baseline_metric:.2f} -> {current_metric:.2f} ({current_metric - baseline_metric:+.2f})")
            print(f"Kept: {kept}/{iteration} | Discarded: {discarded} | Crashed: {crashed}")
            print("---")

    # -------------------- Summary --------------------
    improvement = (current_metric - baseline_metric) / baseline_metric * 100 if baseline_metric else 0.0
    summary = (
        f"\nAutoresearch complete.\n"
        f"Total iterations: {max_iterations}\n"
        f"Kept: {kept} | Discarded: {discarded} | Crashed: {crashed}\n"
        f"Baseline metric: {baseline_metric:.2f}\n"
        f"Final metric:    {current_metric:.2f}\n"
        f"Improvement:     {improvement:+.2f}%\n"
        f"Results TSV:     {tsv}\n"
    )
    print(summary)

    handoff = {
        "version": "2.1.0",
        "source": "loop",
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "status": "COMPLETE",
        "results_tsv": str(tsv),
        "findings": [
            f"Baseline metric: {baseline_metric:.2f}",
            f"Final metric: {current_metric:.2f}",
            f"Kept {kept} of {max_iterations} iterations",
        ],
        "config": cfg,
    }
    handoff_path = out_dir / "handoff.json"
    handoff_path.write_text(json.dumps(handoff, indent=2))
    print(f"Handoff written to {handoff_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())