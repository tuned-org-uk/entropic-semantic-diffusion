"""CPU path-graph diagnostics for the v2 encoding and decoding notebooks.

The rank sweep uses untrained encoders. A separate fixed-budget diagnostic
trains one synthetic ESDM; the readout comparison fits only per-mode gains.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import resources

import numpy as np
import torch
import yaml

from esdm.models import ESDM

SEED = 3407
FEATURES = 32  # A small path with five dyadic ranks up to full rank.
RANK = FEATURES // 4  # Design rank; only the encoding diagnostic varies it.
RANKS = (2, 4, RANK, 16, FEATURES)
N_TRAIN, N_TEST = 512, 512  # Enough independent cases for paired case intervals on CPU.
NOISE_SD = 0.15  # Fixed observation corruption, in original feature units.
HALF_POWER_MODE = 4  # Sets the signal's half-power bandwidth, independently of any readout.
STEPS, LR = 100, 0.05  # Identical fixed full-batch Adam budget for all trainable arms.
BOOTSTRAPS = 2000  # Nonparametric case-resampling replicates; no refit or reseeding.


@dataclass
class Split:
    clean: np.ndarray
    observed: np.ndarray


@dataclass
class PathData:
    basis: np.ndarray
    laplacian: np.ndarray
    train: Split
    test: Split


def _rng(stream: int) -> np.random.Generator:
    """Explicit stream identifiers keep each draw independent of call order."""
    return np.random.default_rng(np.random.SeedSequence(SEED, spawn_key=(stream,)))


def path_data() -> PathData:
    """Draw Gaussian graph signals in the analytic DCT-II basis of the path."""
    k = np.arange(FEATURES)
    basis = np.cos(np.pi * (k[:, None] + 0.5) * k / FEATURES) * np.sqrt(2 / FEATURES)
    basis[:, 0] = 1 / np.sqrt(FEATURES)
    edges = np.diff(np.eye(FEATURES), axis=0)
    std = (1 + (k / HALF_POWER_MODE) ** 2) ** -0.5

    def split(n, stream):
        rng = _rng(stream)
        clean = (rng.standard_normal((n, FEATURES)) * std) @ basis.T
        observed = clean + rng.normal(0, NOISE_SD, size=clean.shape)
        return Split(clean, observed)

    return PathData(basis, edges.T @ edges, split(N_TRAIN, 0), split(N_TEST, 1))


def protocol(data: PathData) -> dict:
    def digest(array):
        return sha256(array.astype("<f8").tobytes()).hexdigest()

    return {
        "seed": SEED, "device": "cpu", "features": FEATURES,
        "ranks": list(RANKS), "readout_rank": RANK, "n_train": N_TRAIN, "n_test": N_TEST,
        "data": "Unweighted path; independent Gaussian DCT-II coefficients, variance "
                "1/(1+(k/4)^2), k=0..31. Clean targets; iid feature-space observation noise.",
        "noise_sd": NOISE_SD, "half_power_mode": HALF_POWER_MODE,
        "metric": "MSE: mean squared error per original feature, then mean across held-out cases.",
        "optimizer": "Adam, full batch, default betas/eps, no weight decay or stopping rule",
        "updates": STEPS, "learning_rate": LR, "bootstrap_draws": BOOTSTRAPS,
        "interval": "Percentile 95% interval for mean paired per-case MSE differences. "
                    "Conditional on the single fixed split, fitted parameters and initialisation; "
                    "no refitting or multiplicity adjustment.",
        "train_clean_sha256": digest(data.train.clean), "test_clean_sha256": digest(data.test.clean),
        "train_observed_sha256": digest(data.train.observed),
        "test_observed_sha256": digest(data.test.observed),
    }


def paired_contrast(left: np.ndarray, right: np.ndarray) -> dict:
    delta = np.asarray(left) - np.asarray(right)
    indices = _rng(2).integers(len(delta), size=(BOOTSTRAPS, len(delta)))
    interval = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
    return {"mean": float(delta.mean()), "ci95": interval.tolist()}


def _model(data: PathData, rank: int) -> ESDM:
    torch.manual_seed(SEED)
    cfg = yaml.safe_load(resources.files("esdm").joinpath("_configs/default.yaml").read_text())
    cfg["model"].update(dim_input=FEATURES, n_nodes=FEATURES, dim_latent=rank, n_modes=rank,
                        L_base=data.laplacian)
    return ESDM(cfg).cpu().eval()


def _case_errors(prediction, target) -> np.ndarray:
    return np.square(prediction - target).mean(axis=1)


def _probe(train_code, test_code, data):
    """OLS affine probe fitted only on training codes and clean features."""
    def affine(code):
        return np.column_stack((code, np.ones(len(code))))

    decoder = np.linalg.lstsq(affine(train_code), data.train.clean, rcond=None)[0]
    return _case_errors(affine(test_code) @ decoder, data.test.clean)


@torch.no_grad()
def encoding_experiment(data: PathData) -> dict:
    """Compare the fixed spectral chart with recoverability from real ESDM codes."""
    rows = []
    for rank in RANKS:
        chart = data.basis[:, :rank]
        chart_error = _case_errors((data.test.clean @ chart) @ chart.T, data.test.clean)
        model = _model(data, rank)
        train_mu, _, train_z = model.encode(torch.from_numpy(data.train.clean).float())
        test_mu, logvar, test_z = model.encode(torch.from_numpy(data.test.clean).float())
        mu_error = _probe(train_mu.double().numpy(), test_mu.double().numpy(), data)
        z_error = _probe(train_z.double().numpy(), test_z.double().numpy(), data)
        rows.append({
            "rank": rank, "chart_mse": float(chart_error.mean()),
            "mu_probe_mse": float(mu_error.mean()), "sample_probe_mse": float(z_error.mean()),
            "mu_minus_chart": paired_contrast(mu_error, chart_error),
            "sample_minus_mu": paired_contrast(z_error, mu_error),
            "heldout_logvar_range": [float(logvar.min()), float(logvar.max())],
            "heldout_sample_norm_range": [float(test_z.norm(dim=1).min()),
                                          float(test_z.norm(dim=1).max())],
        })
    return {
        "scope": "Clean held-out inputs. Chart = projection and transpose synthesis, no fit. "
                 "ESDM.encode = independently initialised, untrained model per rank; separate "
                 "train-only affine least-squares probes for posterior mean and one sample per "
                 "case. Probe error is not ESDM.forward reconstruction or ELBO performance.",
        "rows": rows,
    }


def training_batches() -> np.ndarray:
    """Two seeded passes over the 512 training cases, 32 cases per update."""
    rng = _rng(4)
    return np.stack([rng.permutation(N_TRAIN) for _ in range(2)]).reshape(-1, 32)


def trained_encoder_diagnostic(data: PathData) -> dict:
    """Fit one fresh ESDM via its actual training loss, with no retries."""
    model = _model(data, RANK)
    train = torch.from_numpy(data.train.clean).float()
    test = torch.from_numpy(data.test.clean).float()
    config = model.cfg["training"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"],
                                 weight_decay=config["weight_decay"])
    batches = training_batches()

    @torch.no_grad()
    def mean_errors():
        # Prevent the unused posterior draws in encode from perturbing training randomness.
        with torch.random.fork_rng(devices=[]):
            model.eval()
            train_mu = model.encode(train)[0].double().numpy()
            test_mu = model.encode(test)[0].double().numpy()
        return _probe(train_mu, test_mu, data)

    @torch.no_grad()
    def gain_range():
        gains = model.readout.weights()
        return [float(gains.min()), float(gains.max())]

    before = mean_errors()
    logits_before = model.readout.logits.detach().clone()
    result = {
        "scope": "One full-model fit on clean training cases only. Fresh affine probes "
                 "separately map pre/post ESDM.encode posterior means to clean targets. "
                 "Held-out probe MSE measures linearly recoverable information, not full "
                 "forward reconstruction; raw features also feed the wave path.",
        "seed": SEED, "rank": RANK, "device": "cpu", "optimizer": "Adam",
        "learning_rate": config["lr"], "weight_decay": config["weight_decay"],
        "grad_clip": None, "dtype": "float32", "epochs": 2,
        "batch_size": batches.shape[1], "planned_updates": len(batches),
        "batch_order_sha256": sha256(batches.astype("<i8").tobytes()).hexdigest(),
        "before_mu_probe_mse": float(before.mean()), "after_mu_probe_mse": None,
        "after_minus_before": None, "before_gain_range": gain_range(), "after_gain_range": None,
        "max_abs_logit_change": None, "stop_reason": None, "nonfinite_step": None,
        "completed_updates": 0, "finite_gradient_steps": 0, "step_trace": [],
    }
    model.train()
    for step, indices in enumerate(batches, start=1):
        optimizer.zero_grad(set_to_none=True)
        out = model(train[indices])
        row = {"step": step, "loss": None, "grad_l2": None, "readout_grad_l2": None}
        result["step_trace"].append(row)
        if not torch.isfinite(out.loss.detach()):
            result.update(stop_reason="nonfinite_loss", nonfinite_step=step)
            break
        row["loss"] = float(out.loss.detach())
        out.loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        norm = torch.stack([g.double().norm() for g in grads]).norm()
        if not torch.isfinite(norm):
            result.update(stop_reason="nonfinite_gradient", nonfinite_step=step)
            break
        result["finite_gradient_steps"] += 1
        row["grad_l2"] = float(norm)
        row["n_grad_tensors"] = len(grads)
        if model.readout.logits.grad is not None:
            row["readout_grad_l2"] = float(model.readout.logits.grad.double().norm())
        optimizer.step()
        result["completed_updates"] += 1
        if any(not p.detach().isfinite().all() for p in model.parameters()):
            result.update(stop_reason="nonfinite_parameter", nonfinite_step=step)
            break
    else:
        result["stop_reason"] = "fixed_budget"
        after = mean_errors()
        result.update(after_mu_probe_mse=float(after.mean()),
                      after_minus_before=paired_contrast(after, before),
                      after_gain_range=gain_range(),
                      max_abs_logit_change=float((model.readout.logits.detach() - logits_before)
                                                 .abs().max()))
    return result


class _FreeReadout(torch.nn.Module):
    """Same diagonal chart and unit complement, with freely trainable gains."""

    def __init__(self):
        super().__init__()
        self.gains = torch.nn.Parameter(torch.ones(RANK, dtype=torch.float64))

    def weights(self):
        return self.gains

    def forward(self, x, basis):
        return x + ((x @ basis) * (self.gains - 1)) @ basis.T


def decoding_experiment(data: PathData) -> dict:
    from esdm.models.bounded_readout import BoundedSpectralReadout

    basis = torch.from_numpy(data.basis[:, :RANK].copy())
    train_in, train_y = map(torch.from_numpy, (data.train.observed, data.train.clean))
    test_in = torch.from_numpy(data.test.observed)
    permutation = _rng(3).permutation(RANK)
    arms, errors = {}, {}
    for name in ("identity", "smooth", "prime", "shuffled", "free"):
        if name == "free":
            module = _FreeReadout()
        else:
            module = BoundedSpectralReadout(RANK).double()
            radius = module.radius.detach().numpy().tolist()
            with torch.no_grad():
                if name in ("identity", "smooth"):
                    module.logits.zero_()
                elif name == "shuffled":
                    # Permute normalised residuals, NOT gains or absolute residuals.
                    # atanh and a permutation commute; radii stay in original order.
                    module.logits.copy_(module.logits[permutation].clone())

        def evaluate():
            with torch.no_grad():
                err = _case_errors(module(test_in, basis).numpy(), data.test.clean)
                record = {"mse": float(err.mean()), "gains": module.weights().tolist()}
            return record, err

        initial, e0 = evaluate()
        if name != "identity":
            opt = torch.optim.Adam(module.parameters(), lr=LR)
            for _ in range(STEPS):
                opt.zero_grad()
                (module(train_in, basis) - train_y).square().mean().backward()
                opt.step()
        trained, e1 = evaluate()
        arms[name] = {"initial": initial, "trained": trained,
                      "updates": 0 if name == "identity" else STEPS,
                      "trained_minus_initial": paired_contrast(e1, e0)}
        errors[name] = {"initial": e0, "trained": e1}

    pairs = [(name, "identity") for name in ("smooth", "prime", "shuffled", "free")]
    pairs += [("prime", name) for name in ("smooth", "shuffled", "free")]
    contrasts = {
        phase: {f"{a}_minus_{b}": paired_contrast(errors[a][phase], errors[b][phase])
                for a, b in pairs}
        for phase in ("initial", "trained")
    }
    return {
        "scope": "Noisy features Q to clean features via the isolated signal readout. "
                 "No diffusion noise, clock updates, full-model loss or end-to-end training.",
        "control": "Shuffled permutes the prime arm's normalised residuals across mode indices "
                   "but retains each mode's radius and the exact residual histogram. "
                   "This tests ordering, not the origin of the residual values.",
        "radius": radius, "shuffle_permutation": permutation.tolist(),
        "arms": arms, "contrasts": contrasts,
    }


@torch.no_grad()
def model_boundary(data: PathData) -> dict:
    """Check public encode, actual forward at two clock times, and readout homogeneity."""
    model = _model(data, RANK)
    weights = model.readout.weights()
    u0 = model.U_q[:, 0].reshape(1, -1)
    scales = torch.tensor([1.0, 10.0, 1000.0])
    wave = scales[:, None] * u0
    amplified = model.readout(wave, model.U_q)
    modal = (amplified @ u0.T).flatten()
    # Use a mode with a known positive orientation: read back in that same basis.
    torch.testing.assert_close(modal, scales * weights[0])
    x = torch.from_numpy(data.test.clean[:8]).float()
    mu, logvar, z = model.encode(x)
    records = []
    for batch in (x, torch.from_numpy(data.test.clean[8:16]).float()):
        out = model(batch)
        raw = model._last_Q  # The stored pre-readout state of this specific forward call.
        alpha = model.clock.alpha_bar()
        check = alpha.sqrt() * model.readout(raw, model.U_q)
        torch.testing.assert_close(out.x_hat, check)
        records.append({
            "tau": float(out.clock.tau_n), "alpha_bar": float(alpha),
            "raw_wave_norm_range": [float(raw.norm(dim=1).min()), float(raw.norm(dim=1).max())],
            "feature_mse": float((out.x_hat - batch).square().mean()), "loss": float(out.loss),
            "max_abs_explicit_readout_error": float((out.x_hat - check).abs().max()),
        })
    return {
        "scope": "Untrained full ESDM, eval mode, two consecutive batches of eight clean "
                 "held-out samples. Diagnostic only; input skips the latent bottleneck in forward.",
        "modal_input": scales.tolist(), "modal_output": modal.tolist(),
        "first_gain": float(weights[0]), "first_radius": float(model.readout.radius[0]),
        "mu_shape": list(mu.shape), "logvar_shape": list(logvar.shape),
        "z_norm_range": [float(z.norm(dim=1).min()), float(z.norm(dim=1).max())],
        "forward_passes": records,
    }
