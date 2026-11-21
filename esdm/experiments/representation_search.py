"""Bounded, validation-selected representation search; see autoresearch.md."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from esdm.cli.generate import _restore_generation_model
from esdm.experiments.v2_readout import (
    FEATURES, HALF_POWER_MODE, RANK, SEED, _model, _rng, path_data,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "autoresearch/loop-261121-representation"
# Predeclared experimental budget, not model hyperparameters.
RATES = (0.00003, 0.0001, 0.0003, 0.001, 0.003)
BUDGETS = (16, 32, 64, 128, 256)
BATCH = 64  # Design training batch size.
BOOTSTRAPS = 2000  # Same interval resolution as the v2.0.0 diagnostic.


class TrialFailure(RuntimeError):
    """A failed trial with its completed-update evidence preserved."""

    def __init__(self, error, record):
        super().__init__(str(error))
        self.record = record


def fit_probe(code: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit an affine decoder on training cases alone."""
    return np.linalg.lstsq(np.column_stack((code, np.ones(len(code)))), target, rcond=None)[0]


def probe_errors(decoder, code, target):
    prediction = np.column_stack((code, np.ones(len(code)))) @ decoder
    return np.square(prediction - target).mean(axis=1)


def confirmation_accepts(comparisons):
    """Both controls must clear the frozen effect and uncertainty thresholds."""
    return len(comparisons) == 2 and all(
        np.isfinite(c["relative_improvement"]) and c["relative_improvement"] >= 0.01
        and np.isfinite(c["ci95"]).all() and c["ci95"][1] < 0
        for c in comparisons
    )


def draw_cases(basis, count, stream):
    std = (1 + (np.arange(FEATURES) / HALF_POWER_MODE) ** 2) ** -0.5
    return (_rng(stream).standard_normal((count, FEATURES)) * std) @ basis.T


def array_hash(array):
    return sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def model_for_arm(data, arm, baseline):
    model = _model(data, RANK)
    if arm == "random":
        with torch.no_grad():
            model.enc_mu.weight.copy_(torch.tensor(baseline["weight"]))
            model.enc_mu.bias.copy_(torch.tensor(baseline["bias"]))
    return model


def train(data, arm, rate, budget, baseline):
    model = model_for_arm(data, arm, baseline).train()
    inputs = torch.from_numpy(data.train.clean).float()
    rng = _rng(102)
    batches = np.concatenate([
        rng.permutation(len(inputs))
        for _ in range(max(BUDGETS) * BATCH // len(inputs))
    ]).reshape(-1, BATCH)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=rate, weight_decay=model.cfg["training"]["weight_decay"]
    )
    chart = model.U_q.clone()
    trace = []
    stop = "budget"
    completed = 0
    try:
        for step, indices in enumerate(batches[:budget], 1):
            optimizer.zero_grad(set_to_none=True)
            out = model(inputs[indices])
            if not torch.isfinite(out.loss):
                raise FloatingPointError(f"nonfinite loss at step {step}")
            out.loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            completed += 1
            if any(not torch.isfinite(t).all() for t in model.state_dict().values()):
                raise FloatingPointError(f"nonfinite state at step {step}")
            trace.append({
                "step": step, "loss": float(out.loss.detach()), "gradient_norm": float(norm),
                "tau": float(model.clock.tau_n),
                "terms": {key: float(value.detach()) for key, value in out.metrics.items()},
            })
            if model.clock.is_heat_death():
                stop = "heat_death"
                break
    except (FloatingPointError, RuntimeError) as error:
        raise TrialFailure(error, {
            "arm": arm, "lr": rate, "max_updates": budget, "updates": completed,
            "failure_step": step, "stop_reason": "crash", "trace": trace,
        }) from error
    # Invariant failures abort the whole search, rather than selecting around a defect.
    torch.testing.assert_close(model.U_q, chart, rtol=0, atol=0)
    weights = model.readout.weights().detach()
    assert ((weights >= 1 - model.readout.radius) & (weights <= 1 + model.readout.radius)).all()
    return model.eval(), {
        "arm": arm, "lr": rate, "max_updates": budget, "updates": len(trace),
        "stop_reason": stop, "trace": trace,
    }


@torch.no_grad()
def evaluate(model, training, heldout):
    train_x = torch.from_numpy(training).float()
    held_x = torch.from_numpy(heldout).float()
    # Local generator keeps evaluation from altering any subsequent training.
    generator = torch.Generator().manual_seed(SEED)
    train_mu, _, train_z = model.encode(train_x, generator=generator)
    mu, logvar, z = model.encode(held_x, generator=generator)
    train_code, code = train_mu.double().numpy(), mu.double().numpy()
    errors = probe_errors(fit_probe(train_code, training), code, heldout)
    sampled = probe_errors(
        fit_probe(train_z.double().numpy(), training), z.double().numpy(), heldout
    )
    singular = torch.linalg.svdvals(model.enc_mu.weight).double().numpy()
    return errors, {
        "mu_probe_mse": float(errors.mean()), "sample_probe_mse": float(sampled.mean()),
        "encoder_singular_values": singular.tolist(),
        "encoder_rank": int(torch.linalg.matrix_rank(model.enc_mu.weight)),
        "mu_variance_per_mode": code.var(axis=0).tolist(),
        "kl_z": float(model._kl_z(mu, logvar)),
        "readout_gains": model.readout.weights().tolist(),
    }


def checkpoint(model, path, record):
    torch.save({
        "model_version": "2.0.0",  # Compatible architecture, not a release claim.
        "proposed_release": "2.0.1",
        "model_config": {
            "model": {key: val for key, val in model.cfg["model"].items() if key != "L_base"},
            "clock": model.cfg["clock"],
        },
        "model_state_dict": model.state_dict(), "experiment": record,
    }, path)
    return sha256(path.read_bytes()).hexdigest()


def restore_checked(model, path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    restored = _restore_generation_model(saved["model_config"], saved["model_state_dict"]).eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    return restored


def contrast(left, right):
    delta = left - right
    rng = _rng(103)
    means = np.array([
        delta[rng.integers(len(delta), size=len(delta))].mean() for _ in range(BOOTSTRAPS)
    ])
    return {
        "mean_difference": float(delta.mean()),
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
        "relative_improvement": float(1 - left.mean() / right.mean()),
    }


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "autoresearch.jsonl"
    if log_path.exists():
        raise FileExistsError(f"Refusing to overwrite a research run: {log_path}")
    baseline_path = DEFAULT_OUTPUT / "baseline-encoder.json"
    baseline = json.loads(baseline_path.read_text())
    data = path_data()
    validation = draw_cases(data.basis, 512, 100)
    config = {
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_hashes": {
            str(path.relative_to(ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), ROOT / "esdm/models/esdm.py",
                         ROOT / "autoresearch.md", baseline_path, ROOT / "uv.lock")
        },
        "seed": SEED, "rates": RATES, "budgets": BUDGETS,
        "training_sha256": array_hash(data.train.clean),
        "validation_sha256": array_hash(validation),
        "confirmation_stream": 101, "confirmation_cases": 2048,
        "torch": torch.__version__, "numpy": np.__version__,
        "scope": "Synthetic smooth path; one training seed; conditional case intervals.",
    }
    write_json(output / "protocol.json", config)

    def log(record):
        with log_path.open("a") as f:
            f.write(json.dumps(record, allow_nan=False) + "\n")

    control, record = train(data, "random", 0.0003, 32, baseline)
    _, control_validation = evaluate(control, data.train.clean, validation)
    log({"kind": "baseline", **record, "validation": control_validation})
    control_digest = checkpoint(control, output / "baseline.pt", record)
    winner, winner_record, best = None, None, float("inf")
    rows = []
    for budget in BUDGETS:
        for rate in RATES:
            trial = len(rows) + 1
            try:
                model, record = train(data, "chart", rate, budget, baseline)
                _, scores = evaluate(model, data.train.clean, validation)
                score = scores["mu_probe_mse"]
                if not np.isfinite(score):
                    raise FloatingPointError("nonfinite validation score")
                decision = "validation_leader" if score < best else "discard"
                if score < best:
                    best, winner, winner_record = score, model, record
                log({"kind": "candidate", "trial": trial, **record,
                     "validation": scores, "decision": decision})
                rows.append((trial, rate, budget, record["updates"], score, decision))
                print(f"trial={trial:02d} lr={rate:g} budget={budget} "
                      f"validation_mse={score:.9f} {decision}", flush=True)
            except TrialFailure as error:
                log({"kind": "candidate", "trial": trial, "lr": rate,
                     **error.record, "decision": "crash", "error": str(error)})
                rows.append((trial, rate, budget, error.record["updates"], None, "crash"))
                print(f"trial={trial:02d} crash: {error}", flush=True)
    with (output / "results.tsv").open("w") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(("trial", "lr", "max_updates", "updates", "validation_mse", "decision"))
        writer.writerows(rows)
    if winner is None:
        raise RuntimeError("No finite candidate; no confirmation or release.")
    winner_digest = checkpoint(winner, output / "winner.pt", winner_record)
    matched, matched_record = train(
        data, "random", winner_record["lr"], winner_record["max_updates"], baseline
    )
    log({"kind": "matched_control", **matched_record})
    matched_digest = checkpoint(matched, output / "matched-control.pt", matched_record)
    selection = {
        "lr": winner_record["lr"], "max_updates": winner_record["max_updates"],
        "updates": winner_record["updates"], "validation_mse": best,
        "checkpoint_sha256": winner_digest,
        "baseline_checkpoint_sha256": control_digest,
        "matched_checkpoint_sha256": matched_digest,
    }
    # Freeze selection on disk before creating or looking at confirmation cases.
    write_json(output / "selection.json", selection)
    restored = []
    for model, name in ((winner, "winner"), (control, "baseline"), (matched, "matched-control")):
        loaded = restore_checked(model, output / f"{name}.pt")
        before, _ = evaluate(model, data.train.clean, validation)
        after, _ = evaluate(loaded, data.train.clean, validation)
        np.testing.assert_array_equal(before, after)
        restored.append(loaded)
    winner, control, matched = restored
    confirmation = draw_cases(data.basis, 2048, 101)
    errors, diagnostics = {}, {}
    for name, model in (
        ("winner", winner), ("baseline", control), ("matched_control", matched),
        ("untrained_chart", model_for_arm(data, "chart", baseline)),
    ):
        errors[name], diagnostics[name] = evaluate(model, data.train.clean, confirmation)
    chart = data.basis[:, :RANK]
    comparisons = {
        name: contrast(errors["winner"], errors[name])
        for name in ("baseline", "matched_control", "untrained_chart")
    }
    accepted = confirmation_accepts([comparisons["baseline"], comparisons["matched_control"]])
    result = {
        "accepted": accepted, "selection": selection,
        "checkpoint_round_trip": "all state tensors and validation errors exactly equal",
        "confirmation_sha256": array_hash(confirmation), "diagnostics": diagnostics,
        "comparisons": comparisons,
        "fixed_chart_mse": float(np.square(confirmation @ chart @ chart.T - confirmation).mean()),
        "interpretation": "Mean-code affine recoverability only. Compare the untrained chart "
                          "before attributing any gain to learning; not generation evidence.",
    }
    write_json(output / "confirmation.json", result)
    log({"kind": "confirmation", **result})
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.output_dir)


if __name__ == "__main__":
    main()
