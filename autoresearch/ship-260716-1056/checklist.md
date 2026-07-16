# Ship Checklist — issue #29 baseline handoff

| # | Item | Result | Notes |
|---|---|---|---|
| 1 | pyproject.toml version matches latest tag | PASS | 0.2.0 = v0.2.0 tag |
| 2 | Working tree clean | PASS | git status clean on main |
| 3 | Baseline reproduces AGENTS.md score | PASS | 74.8954 ≈ 74.90 |
| 4 | tau_monotone | PASS | 10.00/10 |
| 5 | tau_range_bonus | PASS | 5.00/5 |
| 6 | alpha_bar_decrease | PASS | 10.00/10 |
| 7 | conservation | PASS (tautological) | 15.00/15 — enforced by construction |
| 8 | heat_death | PASS | 10.00/10, fires at 610/800 |
| 9 | heat_death_timing | WARN | 4.38/5 — tuned, not derived |
| 10 | rho_sign_stabilizing | PASS | 10.00/10 |
| 11 | recon_decrease | WARN | 6.58/10 — decoder near-collapsed |
| 12 | net_energy_flow | FAIL | 0.00/5 — Lambda dominates H (#29 Phase 2) |
| 13 | pi_contraction_ratio | WARN | 3.94/5 — H too strong early, inert late |
| 14 | runs/ cleanup of pre-v0.1.0 dirs | PASS | 56 dirs removed, 0 tracked files affected |
| 15 | handoff.json written | PASS | autoresearch/ship-260716-1056/handoff.json |
