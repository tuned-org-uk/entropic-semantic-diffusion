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
            torch.zeros_like(x.unsqueeze(-1)),
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

    assert torch.allclose(Q_esdm, Q_vdt.squeeze(-1), atol=1e-5)


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
        Q_tm1 = torch.zeros_like(Q_t)
        dt = torch.tensor(0.05, dtype=Q_t.dtype, device=Q_t.device)
        dt2 = dt * dt
        L_term = torch.bmm(L_f.unsqueeze(0).expand(Q_t.shape[0], -1, -1), Q_t)
        damp = block._vdt_block.gamma.view(1, 1, -1) * (Q_t - Q_tm1)
        Q_expected = 2.0 * Q_t - Q_tm1 - dt2 * L_term - damp

    assert torch.allclose(Q_zero, Q_expected.squeeze(-1), atol=1e-5)


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
