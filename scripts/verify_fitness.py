#!/usr/bin/env python3
"""Verify command: output ESDM fitness score as a single number."""
import sys
from esdm.eval.fitness import compute_fitness
from esdm.experiments.exp3 import run_single

m = run_single(n_steps=300, seed=3407)
r = compute_fitness(m)
print(f'{r["total"]:.2f}')
