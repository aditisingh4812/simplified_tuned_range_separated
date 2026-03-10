#!/usr/bin/env python3

from gdd_pbe import gdd_tuned_pbe

atom = "He 0 0 0"

results = gdd_tuned_pbe(
    atom=atom,
    basis="def2-tzvpp",
    charge=0,
    spin=0,
    unit="Angstrom",
    C=0.9
)

print(results)

