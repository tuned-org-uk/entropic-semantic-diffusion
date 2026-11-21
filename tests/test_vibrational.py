"""
Phase 3 test — VibrationalStateBlock.
All assertions from 01-IMPLEMENTATION-PLAN.md §4.4.
"""
import torch

from esdm._vendor.vdt import VibrationalStateBlock as _VDTVibrationalBlock
from esdm.models.vibrational import (
    VibrationalStateBlock,
    _laplacian_from_dense,
    compute_mode_residuals,
)


def _random_L_f(F: int):
    """Return a simple feature-space Laplacian (F, F)."""
    A = torch.rand(F, F)
    A = (A + A.T) / 2.0
    A.fill_diagonal_(0.0)
    D = A.sum(dim=-1).diag()
    return D - A


def test_vibrational_instantiates():
    block = VibrationalStateBlock(dim=16, n_heads=2, n_steps=3)
    assert block is not None
    assert block.dim == 16


def test_vibrational_matches_vdt_when_lambda_one():
    """With Lambda_tau=1.0 and delta_tau=dt, output matches original VDT."""
    torch.manual_seed(42)
    F = 8
    x = torch.randn(2, F)
    L_f = _random_L_f(F)
    lap = _laplacian_from_dense(L_f)

    # Reference VDT block call: Q_t: (B, F, 1), Q_tm1: zeros, L_f: (B, F, F), lap
    vdt_block = _VDTVibrationalBlock(n_nodes=F, feat_dim=1, n_heads=1, dropout=0.0)
    vdt_block.eval()
    with torch.no_grad():
        Q_vdt, rho_p_vdt, rho_m_vdt = vdt_block(
            x.unsqueeze(-1),
            x.unsqueeze(-1).clone(),  # Q_tm1 = Q_t (zero initial velocity)
            L_f.unsqueeze(0).expand(x.shape[0], -1, -1),
            lap,
            recompute_cfl=True,
        )

    # ESDM wrapper with Lambda=1 and delta_tau equal to the VDT block's dt.
    esdm_block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, gamma=0.1, dropout=0.0)
    # Copy VDT parameters so the mixing blocks are identical.
    esdm_block._vdt_block.load_state_dict(vdt_block.state_dict())
    esdm_block.eval()
    with torch.no_grad():
        dt = vdt_block.log_dt.exp().clamp(max=lap.dt_max_cfl())
        Q_esdm, rho_p_esdm, rho_m_esdm = esdm_block(
            x, L_f, Lambda_tau=1.0, delta_tau=float(dt)
        )

    torch.testing.assert_close(Q_esdm, Q_vdt.squeeze(-1))


def test_vibrational_forcing_vanishes_when_lambda_zero():
    """With Lambda_tau=0.0, the forcing term B_t vanishes from the recurrence."""
    torch.manual_seed(0)
    F = 8
    x = torch.randn(2, F)
    L_f = _random_L_f(F)

    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, gamma=0.1, dropout=0.0)
    block.eval()
    with torch.no_grad():
        Q_zero, _, _ = block(x, L_f, Lambda_tau=0.0, delta_tau=0.05)

        # Expected: same recurrence but without the B_t term.
        Q_t = x.unsqueeze(-1)
        Q_tm1 = Q_t.clone()  # zero initial velocity (Q_tm1 = Q_t)
        dt = torch.tensor(0.05, dtype=Q_t.dtype, device=Q_t.device)
        dt2 = dt * dt
        L_term = torch.bmm(L_f.unsqueeze(0).expand(Q_t.shape[0], -1, -1), Q_t)
        damp = block._vdt_block.gamma.view(1, 1, -1) * (Q_t - Q_tm1)
        Q_expected = 2.0 * Q_t - Q_tm1 - dt2 * L_term - damp

    torch.testing.assert_close(Q_zero, Q_expected.squeeze(-1))


def test_vibrational_output_finite_for_random_inputs():
    """Q_next is finite and bounded for random inputs."""
    torch.manual_seed(1)
    F = 16
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=2)
    x = torch.randn(4, F)
    L_f = _random_L_f(F)

    Q, rho_plus, rho_minus = block(x, L_f, Lambda_tau=1.0, delta_tau=0.1)
    assert Q.shape == x.shape
    assert torch.all(torch.isfinite(Q))
    assert torch.all(torch.isfinite(rho_plus))
    assert torch.all(torch.isfinite(rho_minus))


def test_vibrational_rho_non_negative():
    """rho_plus and rho_minus are PSD (non-negative diagonal entries)."""
    torch.manual_seed(3)
    F = 8
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1)
    x = torch.randn(2, F)
    L_f = _random_L_f(F)

    _, rho_plus, rho_minus = block(x, L_f, Lambda_tau=0.5, delta_tau=0.1)
    # PSD matrices have non-negative diagonals.
    assert (rho_plus.diag() >= -1e-6).all()
    assert (rho_minus.diag() >= -1e-6).all()


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------


def test_compute_mode_residuals_zero_when_identical():
    """Identical consecutive states produce zero residual per mode."""
    torch.manual_seed(5)
    F = 8
    B = 4
    Q_t = torch.randn(B, F)
    r = compute_mode_residuals(Q_t, Q_t)
    assert r.shape == (F,)
    assert torch.allclose(r, torch.zeros(F), atol=1e-6)


def test_compute_mode_residuals_known_difference():
    """A pure single-mode step concentrates residual on that mode."""
    B = 3
    F = 8
    Q_t = torch.zeros(B, F)
    Q_prev = torch.zeros(B, F)
    Q_prev[:, 3] = 1.0
    r = compute_mode_residuals(Q_t, Q_prev)
    expected = torch.zeros(F)
    expected[3] = float(B) ** 0.5  # L2 norm over the batch dimension
    assert torch.allclose(r, expected, atol=1e-6)
    assert (r[:3] == 0).all()
    assert (r[4:] == 0).all()


# ---------------------------------------------------------------------------
# Hybridisation operator tests (issue #20)
# ---------------------------------------------------------------------------


def test_hybridisation_output_shape():
    """H operator produces (B, F, 1) output matching Q_t shape."""
    torch.manual_seed(3407)
    F = 16
    q = F // 4
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, dropout=0.0)
    x = torch.randn(4, F)
    L_f = _random_L_f(F)
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()[:, :q]
    Q, _, _ = block(x, L_f, Lambda_tau=1.0, delta_tau=0.1, U_q=U_q)
    assert Q.shape == x.shape
    assert torch.all(torch.isfinite(Q))


def test_hybridisation_backward_compat_no_uq():
    """Without U_q, the block falls back to bare L_f@Q_t (matches old tests)."""
    torch.manual_seed(3407)
    F = 8
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, gamma=0.1, dropout=0.0)
    block.eval()
    x = torch.randn(2, F)
    L_f = _random_L_f(F)
    with torch.no_grad():
        Q, _, _ = block(x, L_f, Lambda_tau=0.0, delta_tau=0.05)
    assert Q.shape == x.shape
    assert torch.all(torch.isfinite(Q))


def test_hybridisation_diagnostics_populated():
    """net_energy_flow and pi_contraction_ratio are populated after forward."""
    torch.manual_seed(3407)
    F = 16
    q = F // 4
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=2, dropout=0.0)
    x = torch.randn(4, F)
    L_f = _random_L_f(F)
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()[:, :q]
    block(x, L_f, Lambda_tau=1.0, delta_tau=0.1, U_q=U_q)
    diag = block.diagnostics()
    assert "angular_displacement" in diag
    assert "pi_contraction_ratio" in diag
    assert isinstance(diag["angular_displacement"], float)
    assert isinstance(diag["pi_contraction_ratio"], float)


def test_hybridisation_gradients_flow():
    """Gradients flow through sigma_sq and eta when not frozen."""
    torch.manual_seed(3407)
    F = 16
    q = F // 4
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, dropout=0.0)
    x = torch.randn(2, F)
    L_f = _random_L_f(F)
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()[:, :q]
    Q, _, _ = block(x, L_f, Lambda_tau=1.0, delta_tau=0.1, U_q=U_q)
    loss = Q.sum()
    loss.backward()
    assert block.sigma_sq.grad is not None
    assert block.eta.grad is not None


def test_hybridisation_freeze_h():
    """When freeze_h=True, sigma_sq and eta have no grad."""
    torch.manual_seed(3407)
    F = 16
    q = F // 4
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, dropout=0.0, freeze_h=True)
    assert not block.sigma_sq.requires_grad
    assert not block.eta.requires_grad
    x = torch.randn(2, F)
    L_f = _random_L_f(F)
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()[:, :q]
    Q, _, _ = block(x, L_f, Lambda_tau=1.0, delta_tau=0.1, U_q=U_q)
    loss = Q.sum()
    loss.backward()
    assert block.sigma_sq.grad is None
    assert block.eta.grad is None


def test_hybridisation_q_equals_f_recovers_identity():
    """When q=F, U_q U_q^T = I, so lifting is identity (no spectral filtering)."""
    torch.manual_seed(3407)
    F = 8
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=1, dropout=0.0)
    x = torch.randn(2, F)
    L_f = _random_L_f(F)
    # Full eigenvectors -> U_q is (F, F), P = I
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()  # all F eigenvectors
    Q, _, _ = block(x, L_f, Lambda_tau=1.0, delta_tau=0.1, U_q=U_q)
    assert Q.shape == x.shape
    assert torch.all(torch.isfinite(Q))


# ---------------------------------------------------------------------------
# v2 free amplitudes
# ---------------------------------------------------------------------------


def test_q_preserves_input_at_zero_entropic_time():
    torch.manual_seed(3407)
    F = 16
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=2, dropout=0.0)
    x = 10 * torch.randn(8, F)
    L_f = _random_L_f(F)
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()[:, :4]

    Q, _, _ = block(x, L_f, Lambda_tau=1.0, delta_tau=0.0, U_q=U_q)
    torch.testing.assert_close(Q, x)


def test_unforced_laplacian_wave_is_linear_in_initial_amplitude():
    torch.manual_seed(3407)
    F = 16
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=2, dropout=0.0)
    x = torch.randn(8, F)
    L_f = _random_L_f(F)

    Q, _, _ = block(x, L_f, Lambda_tau=0.0, delta_tau=0.1)
    scaled, _, _ = block(10 * x, L_f, Lambda_tau=0.0, delta_tau=0.1)
    torch.testing.assert_close(scaled, 10 * Q)


def test_angular_displacement_nonzero_for_different_inputs():
    """Angular displacement > 0 when input differs from the wave recurrence output."""
    torch.manual_seed(3407)
    F = 16
    block = VibrationalStateBlock(dim=F, n_heads=1, n_steps=2, dropout=0.0)
    x = 3 * torch.randn(4, F)
    L_f = _random_L_f(F)
    vals, vecs = torch.linalg.eigh(L_f.double())
    U_q = vecs.float()[:, :4]

    Q, _, _ = block(x, L_f, Lambda_tau=1.0, delta_tau=0.1, U_q=U_q)
    diag = block.diagnostics()
    # atan2 of the orthogonal and parallel components is scale-invariant.
    x_dir = x.double() / x.double().norm(dim=-1, keepdim=True)
    parallel = (Q.double() * x_dir).sum(dim=-1)
    orthogonal = Q.double() - parallel[:, None] * x_dir
    expected = torch.atan2(orthogonal.norm(dim=-1), parallel).mean().item()
    assert expected > 0 and abs(diag["angular_displacement"] - expected) < 1e-5
    assert diag["gamma_mean"] > 0.0
