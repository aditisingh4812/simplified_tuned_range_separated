#!/usr/bin/env python3
"""
GDD tuning for the PBE(ω) functional in PySCF.

This script implements the GDD scheme from Eqs. (7)–(15) of
Modrzejewski et al., J. Phys. Chem. A 2013, 117, 11580–11586,
and specializes it to PBE(ω) = LRC-ωPBE.

Implemented protocol
--------------------
1. Run a reference SCF with PBE(0.40), as recommended in the paper.
2. Compute ω_GDD[ρ] from the converged density.
3. Run a final self-consistent PBE(ω) calculation with ω = ω_GDD.
4. Optionally do one extra update cycle (off by default).

Theory used
-----------
Eq. (7):  <d_X,σ^2> = ∫ ρσ(r) wσ(r) d_X,σ(r)^2 dr / ∫ ρσ(r) wσ(r) dr
Eq. (8):  wσ(r) = 1 if tσ(r) <= μ, else 0
Eq. (9):  tσ(r) = τ_UEG,σ(r) / τσ(r)
Eq. (10): τ_UEG,σ(r) = (3/5) (6π^2)^(2/3) ρσ(r)^(5/3)
Eq. (11): τσ(r) = Σ_i |∇ψ_iσ(r)|^2
Eq. (12): choose μ such that ∫ ρσ(r) wσ(r; μ) dr = N_tail, with μ >= 0.07
Eq. (13)-(14): d_X,σ(r) from the occupied-orbital dipole matrix
Eq. (15): ω_GDD = C / sqrt(<d_X,σ^2>)

For PBE(ω), the paper gives C = 0.90.

Notes
-----
* This implementation currently targets closed-shell RKS references.
* Coordinates are interpreted in Bohr.
* PBE(ω) is represented as RSH(ω, 1, -1) + PBE,PBE, i.e. full long-range
  HF exchange and zero short-range HF exchange on top of PBE exchange/correlation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Any

import numpy as np
from pyscf import dft, gto, lib
from pyscf.dft import numint

# =============================================================================
# Constants
# =============================================================================

DEFAULT_C_PBE_OMEGA = 0.90
"""Constant C in Eq. (15) for PBE(ω) as given in the paper."""

DEFAULT_OMEGA0 = 0.30
"""Initial ω value for the reference PBE(0.40) calculation."""

DEFAULT_MU_FLOOR = 0.07
"""Minimum allowed μ value from Eq. (12)."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class GDDMetrics:
    """Results from a GDD calculation."""
    mu: float
    """Tail parameter μ from Eq. (8)."""
    n_tail_eff: float
    """Effective tail charge after applying μ threshold."""
    d2_avg: float
    """Average squared dipole <d_X^2> in bohr²."""
    omega_gdd: float
    """Computed ω_GDD value in bohr⁻¹."""


@dataclass
class SCFResult:
    """Results from a single SCF calculation."""
    xc: str
    """Exchange-correlation functional used."""
    energy: float
    """Total energy in Hartree."""
    converged: bool
    """Whether SCF converged."""
    omega_input: Optional[float]
    """Input ω value used (if applicable)."""
    omega_gdd: Optional[float]
    """ω_GDD computed from this density."""
    mu: Optional[float]
    """μ parameter from GDD calculation."""
    n_tail: Optional[float]
    """Effective tail charge."""
    d2_avg: Optional[float]
    """Average squared dipole."""
    mo_energy: Optional[np.ndarray] = None
    """Molecular orbital energies."""


# =============================================================================
# Functional Definitions
# =============================================================================

def pbe_xc(omega: float) -> str:
    return f'GGA_X_PBE, GGA_C_PBE'


# =============================================================================
# Molecule Setup
# =============================================================================

def build_mol(
    atom: str, 
    basis: str, 
    charge: int = 0, 
    spin: int = 0, 
    unit: str = 'Bohr'
) -> gto.Mole:
    """
    Build a PySCF molecule.
    
    Parameters
    ----------
    atom : str
        PySCF atom specification.
    basis : str
        Basis set name.
    charge : int, optional
        Molecular charge.
    spin : int, optional
        Number of unpaired electrons (2S).
    unit : str, optional
        Unit for coordinates ('Bohr' or 'Angstrom').
    
    Returns
    -------
    gto.Mole
        Constructed molecule.
    """
    mol = gto.Mole()
    mol.atom = atom
    mol.unit = unit
    mol.basis = basis
    mol.ecp = basis
    mol.charge = charge
    mol.spin = spin
    mol.verbose = 4
    mol.build()
    return mol


def run_rks(
    mol: gto.Mole, 
    xc: str, 
    grid_level: int = 5, 
    conv_tol: float = 1e-10
) -> dft.RKS:
    """
    Run a restricted Kohn-Sham calculation.
    
    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule.
    xc : str
        Exchange-correlation functional.
    grid_level : int, optional
        Grid level for numerical integration.
    conv_tol : float, optional
        Convergence tolerance for SCF.
    
    Returns
    -------
    dft.RKS
        Converged RKS object.
    """
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = grid_level
    mf.conv_tol = conv_tol
    mf.conv_tol_grad = 1e-8
    mf.kernel()
    return mf


# =============================================================================
# Grid Data Extraction
# =============================================================================

def closed_shell_grid_data(
    mol: gto.Mole, 
    mf: dft.RKS
) -> Dict[str, Any]:
    """
    Extract grid data needed for GDD calculation.
    
    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule.
    mf : dft.RKS
        Converged RKS object.
    
    Returns
    -------
    dict
        Dictionary containing grid coordinates, weights, density,
        kinetic energy density, and occupied orbital information.
    
    Raises
    ------
    NotImplementedError
        If spin is not zero (open-shell not supported).
    ValueError
        If occupation pattern doesn't match closed-shell.
    """
    if mol.spin != 0:
        raise NotImplementedError(
            'This script currently supports only closed-shell RKS systems (spin=0).'
        )

    # Build grids if not already built
    grids = mf.grids
    if grids.coords is None:
        grids.build(with_non0tab=True)

    coords = grids.coords
    weights = grids.weights

    # Evaluate AO basis and density
    ao = numint.eval_ao(mol, coords, deriv=1)
    dm_tot = mf.make_rdm1()
    rho_full = numint.eval_rho(mol, ao, dm_tot, xctype='GGA')

    # For closed-shell, spin density is half of total density
    rho_sigma = 0.5 * rho_full[0]

    # Get occupied orbitals
    occ_idx = np.where(mf.mo_occ > 1e-12)[0]
    if 2 * len(occ_idx) != mol.nelectron:
        raise ValueError('Unexpected occupation pattern for closed-shell RKS.')

    mo_coeff_occ = mf.mo_coeff[:, occ_idx]

    # Compute orbital values and gradients
    ao0 = ao[0]
    aox, aoy, aoz = ao[1], ao[2], ao[3]
    
    mo_val = ao0 @ mo_coeff_occ
    mo_x = aox @ mo_coeff_occ
    mo_y = aoy @ mo_coeff_occ
    mo_z = aoz @ mo_coeff_occ

    # Kinetic energy density: τ_σ = Σ_i |∇ψ_iσ|²
    tau_sigma = (
        np.einsum('pi,pi->p', mo_x, mo_x) +
        np.einsum('pi,pi->p', mo_y, mo_y) +
        np.einsum('pi,pi->p', mo_z, mo_z)
    )

    return {
        'coords': coords,
        'weights': weights,
        'rho_sigma': rho_sigma,
        'tau_sigma': tau_sigma,
        'mo_coeff_occ': mo_coeff_occ,
        'mo_val': mo_val,
    }


# =============================================================================
# GDD Core Calculations
# =============================================================================

def choose_mu(
    rho_sigma: np.ndarray,
    tau_sigma: np.ndarray,
    weights: np.ndarray,
    n_tail: float = 1.0,
    mu_floor: float = DEFAULT_MU_FLOOR
) -> Tuple[float, np.ndarray, float]:
    """
    Determine μ parameter from Eq. (12).
    
    Parameters
    ----------
    rho_sigma : np.ndarray
        Spin density at grid points.
    tau_sigma : np.ndarray
        Kinetic energy density at grid points.
    weights : np.ndarray
        Grid integration weights.
    n_tail : float, optional
        Target tail charge N_tail.
    mu_floor : float, optional
        Minimum allowed μ value.
    
    Returns
    -------
    mu : float
        Chosen μ parameter.
    w : np.ndarray
        Weight function w(r) (1 for t ≤ μ, 0 otherwise).
    n_tail_eff : float
        Effective tail charge after applying μ threshold.
    """
    eps = 1e-18
    
    # UEG kinetic energy density from Eq. (10)
    tau_ueg = (3.0 / 5.0) * (6.0 * np.pi**2) ** (2.0 / 3.0) * np.maximum(rho_sigma, eps) ** (5.0 / 3.0)
    
    # t(r) from Eq. (9)
    t = tau_ueg / np.maximum(tau_sigma, eps)

    # Sort by t and find μ that gives desired tail charge
    order = np.argsort(t)
    cumulative = np.cumsum(rho_sigma[order] * weights[order])
    idx = np.searchsorted(cumulative, n_tail, side='left')
    
    if idx >= len(order):
        mu_raw = float(t[order[-1]])
    else:
        mu_raw = float(t[order[idx]])

    # Apply floor
    mu = max(mu_floor, mu_raw)
    
    # Compute weight function and effective tail charge
    w = (t <= mu).astype(float)
    n_tail_eff = float(np.dot(rho_sigma * w, weights))
    
    return mu, w, n_tail_eff


def occupied_dipole_matrices(
    mol: gto.Mole, 
    mo_coeff_occ: np.ndarray
) -> np.ndarray:
    """
    Compute dipole matrices in the occupied orbital basis.
    
    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule.
    mo_coeff_occ : np.ndarray
        Occupied MO coefficients.
    
    Returns
    -------
    np.ndarray
        Dipole matrices f_ij^a with shape (3, nocc, nocc).
    """
    # int1e_r gives AO dipole integrals in Bohr
    r_ao = mol.intor_symmetric('int1e_r', comp=3)
    return np.einsum('xuv,ui,vj->xij', r_ao, mo_coeff_occ, mo_coeff_occ)


def average_dx2(
    mol: gto.Mole,
    data: Dict[str, Any],
    w_mask: np.ndarray
) -> float:
    """
    Compute average squared dipole <d_X^2> from Eq. (7).
    
    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule.
    data : dict
        Grid data from closed_shell_grid_data().
    w_mask : np.ndarray
        Weight function w(r) from Eq. (8).
    
    Returns
    -------
    float
        Average squared dipole in bohr².
    
    Raises
    ------
    ZeroDivisionError
        If tail region normalization is zero.
    """
    coords = data['coords']
    weights = data['weights']
    rho_sigma = data['rho_sigma']
    mo_val = data['mo_val']
    mo_coeff_occ = data['mo_coeff_occ']

    # Dipole matrices in occupied basis
    f = occupied_dipole_matrices(mol, mo_coeff_occ)
    
    # Eq. (13)-(14): d_X(r) = [Σ_ij f_ij ψ_i(r) ψ_j(r)] / ρ_σ(r) - r
    center_num = np.einsum('aij,pi,pj->pa', f, mo_val, mo_val, optimize=True)
    center = center_num / np.maximum(rho_sigma[:, None], 1e-18)
    
    dvec = center - coords
    d2 = np.einsum('pa,pa->p', dvec, dvec)

    numerator = np.dot(rho_sigma * w_mask * d2, weights)
    denominator = np.dot(rho_sigma * w_mask, weights)
    
    if denominator <= 0.0:
        raise ZeroDivisionError('Tail-region normalization is zero.')
    
    return float(numerator / denominator)


def compute_omega_gdd_from_mf(
    mol: gto.Mole,
    mf: dft.RKS,
    C: float = DEFAULT_C_PBE_OMEGA,
    n_tail: float = 1.0,
    mu_floor: float = DEFAULT_MU_FLOOR
) -> GDDMetrics:
    """
    Compute ω_GDD from a converged SCF density.
    
    Parameters
    ----------
    mol : gto.Mole
        PySCF molecule.
    mf : dft.RKS
        Converged RKS object.
    C : float, optional
        Constant from Eq. (15).
    n_tail : float, optional
        Target tail charge N_tail.
    mu_floor : float, optional
        Minimum allowed μ value.
    
    Returns
    -------
    GDDMetrics
        Computed GDD metrics.
    """
    data = closed_shell_grid_data(mol, mf)
    mu, w, n_tail_eff = choose_mu(
        data['rho_sigma'], 
        data['tau_sigma'], 
        data['weights'],
        n_tail=n_tail, 
        mu_floor=mu_floor
    )
    d2_avg = average_dx2(mol, data, w)
    omega_gdd = C / np.sqrt(d2_avg)
    
    return GDDMetrics(
        mu=mu,
        n_tail_eff=n_tail_eff,
        d2_avg=d2_avg,
        omega_gdd=float(omega_gdd)
    )


# =============================================================================
# Main Workflow
# =============================================================================

def gdd_tuned_pbe(
    atom: str,
    basis: str = 'cc-pvtz',
    charge: int = 0,
    spin: int = 0,
    unit: str = 'Bohr',
    grid_level: int = 5,
    conv_tol: float = 1e-10,
    C: float = DEFAULT_C_PBE_OMEGA,
    omega0: float = DEFAULT_OMEGA0,
    n_tail: float = 1.0,
    mu_floor: float = DEFAULT_MU_FLOOR,
    extra_cycle: bool = False
) -> Dict[str, SCFResult]:
    """
    Perform GDD-tuned PBE(ω) calculation.
    
    Parameters
    ----------
    atom : str
        PySCF atom specification.
    basis : str, optional
        Basis set name.
    charge : int, optional
        Molecular charge.
    spin : int, optional
        Number of unpaired electrons (2S).
    unit : str, optional
        Unit for coordinates ('Bohr' or 'Angstrom').
    grid_level : int, optional
        Grid level for numerical integration.
    conv_tol : float, optional
        Convergence tolerance for SCF.
    C : float, optional
        Constant from Eq. (15).
    omega0 : float, optional
        Initial ω value for reference calculation.
    n_tail : float, optional
        Target tail charge N_tail.
    mu_floor : float, optional
        Minimum allowed μ value.
    extra_cycle : bool, optional
        Whether to perform an extra update cycle.
    
    Returns
    -------
    dict
        Dictionary with results for each calculation step.
    
    Raises
    ------
    RuntimeError
        If any SCF calculation fails to converge.
    """
    mol = build_mol(atom=atom, basis=basis, charge=charge, spin=spin, unit=unit)

    # Step 1: Initial density from PBE(ω0) [recommended ω0 = 0.40]
    xc0 = pbe_xc(omega0)
    mf0 = run_rks(mol, xc0, grid_level=grid_level, conv_tol=conv_tol)
    
    if not mf0.converged:
        raise RuntimeError('Initial PBE(ω0) calculation did not converge.')
    
    gdd0 = compute_omega_gdd_from_mf(
        mol, mf0, C=C, n_tail=n_tail, mu_floor=mu_floor
    )

    # Step 2: Final self-consistent PBE(ω_GDD)
    xc1 = pbe_xc(gdd0.omega_gdd)
    mf1 = run_rks(mol, xc1, grid_level=grid_level, conv_tol=conv_tol)
    
    if not mf1.converged:
        raise RuntimeError('Final PBE(ω_GDD) calculation did not converge.')
    
    gdd1 = compute_omega_gdd_from_mf(
        mol, mf1, C=C, n_tail=n_tail, mu_floor=mu_floor
    )

    results = {
        'initial': SCFResult(
            xc=xc0,
            energy=float(mf0.e_tot),
            converged=bool(mf0.converged),
            omega_input=float(omega0),
            omega_gdd=float(gdd0.omega_gdd),
            mu=float(gdd0.mu),
            n_tail=float(gdd0.n_tail_eff),
            d2_avg=float(gdd0.d2_avg),
            mo_energy=np.array(mf0.mo_energy, copy=True),
        ),
        'final': SCFResult(
            xc=xc1,
            energy=float(mf1.e_tot),
            converged=bool(mf1.converged),
            omega_input=float(gdd0.omega_gdd),
            omega_gdd=float(gdd1.omega_gdd),
            mu=float(gdd1.mu),
            n_tail=float(gdd1.n_tail_eff),
            d2_avg=float(gdd1.d2_avg),
            mo_energy=np.array(mf1.mo_energy, copy=True),
        ),
    }

    # Optional extra cycle
    if extra_cycle:
        xc2 = pbe_xc(gdd1.omega_gdd)
        mf2 = run_rks(mol, xc2, grid_level=grid_level, conv_tol=conv_tol)
        
        if not mf2.converged:
            raise RuntimeError('Extra-cycle PBE(ω_GDD) calculation did not converge.')
        
        gdd2 = compute_omega_gdd_from_mf(
            mol, mf2, C=C, n_tail=n_tail, mu_floor=mu_floor
        )
        
        results['extra_cycle'] = SCFResult(
            xc=xc2,
            energy=float(mf2.e_tot),
            converged=bool(mf2.converged),
            omega_input=float(gdd1.omega_gdd),
            omega_gdd=float(gdd2.omega_gdd),
            mu=float(gdd2.mu),
            n_tail=float(gdd2.n_tail_eff),
            d2_avg=float(gdd2.d2_avg),
            mo_energy=np.array(mf2.mo_energy, copy=True),
        )

    return results


# =============================================================================
# Output and Testing
# =============================================================================

def print_report(results: Dict[str, SCFResult]) -> None:
    """Print formatted results report."""
    print('\n=== GDD-tuned PBE(ω) report ===')
    
    for label, res in results.items():
        print(f'\n[{label}]')
        print(f'xc                 : {res.xc}')
        
        if res.omega_input is not None:
            print(f'omega_input (bohr⁻¹): {res.omega_input:.10f}')
        if res.omega_gdd is not None:
            print(f'omega_GDD   (bohr⁻¹): {res.omega_gdd:.10f}')
        
        print(f'E_tot       (Eh)     : {res.energy:.12f}')
        
        if res.mu is not None:
            print(f'mu                    : {res.mu:.10f}')
        if res.n_tail is not None:
            print(f'N_tail_eff            : {res.n_tail:.10f}')
        if res.d2_avg is not None:
            print(f'<d_X^2>      (bohr²) : {res.d2_avg:.10f}')
        
        if res.mo_energy is not None and len(res.mo_energy) > 0:
            homo_idx = (len(res.mo_energy) - 1) // 2
            print(f'HOMO energy  (Eh)     : {res.mo_energy[homo_idx]:.10f}')


TEST_SYSTEMS = {
    'He': 'He 0.0 0.0 0.0',
    'Ne': 'Ne 0.0 0.0 0.0',
}


def main() -> None:
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description='GDD tuning for PBE(ω) in PySCF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python %(prog)s --atom "He 0 0 0" --basis cc-pvtz
  python %(prog)s --run-tests --basis cc-pvdz
  python %(prog)s --atom "O 0 0 0; H 0 1 0; H 0 0 1" --unit Angstrom --extra-cycle
        """
    )
    
    parser.add_argument('--atom', help='PySCF atom specification, e.g. "He 0 0 0"')
    parser.add_argument('--basis', default='cc-pvtz', help='Basis set name')
    parser.add_argument('--charge', type=int, default=0, help='Molecular charge')
    parser.add_argument('--spin', type=int, default=0, help='Number of unpaired electrons (2S)')
    parser.add_argument('--unit', default='Bohr', choices=['Bohr', 'Angstrom'],
                       help='Unit for coordinates')
    parser.add_argument('--grid-level', type=int, default=5, 
                       help='Grid level for numerical integration')
    parser.add_argument('--conv-tol', type=float, default=1e-10,
                       help='SCF convergence tolerance')
    parser.add_argument('--C', type=float, default=DEFAULT_C_PBE_OMEGA,
                       help='Constant C in Eq. (15). For PBE(ω), the paper uses C=0.90.')
    parser.add_argument('--omega0', type=float, default=DEFAULT_OMEGA0,
                       help='Initial ω value for reference PBE(ω) calculation.')
    parser.add_argument('--n-tail', type=float, default=1.0,
                       help='Target tail charge N_tail in Eq. (12). Use 1 for isolated systems.')
    parser.add_argument('--mu-floor', type=float, default=DEFAULT_MU_FLOOR,
                       help='Minimum allowed μ value')
    parser.add_argument('--extra-cycle', action='store_true',
                       help='Do one additional PBE(ω_GDD) update cycle')
    parser.add_argument('--run-tests', action='store_true',
                       help='Run built-in He and Ne tests')
    
    args = parser.parse_args()

    # Run tests if requested
    if args.run_tests:
        for name, atom in TEST_SYSTEMS.items():
            print(f'\n{"":#>60}')
            print(f'# Testing {name}')
            print(f'{"":#<60}')
            
            res = gdd_tuned_pbe(
                atom=atom,
                basis=args.basis,
                charge=0,
                spin=0,
                unit=args.unit,
                grid_level=args.grid_level,
                conv_tol=args.conv_tol,
                C=args.C,
                omega0=args.omega0,
                n_tail=args.n_tail,
                mu_floor=args.mu_floor,
                extra_cycle=args.extra_cycle,
            )
            print_report(res)
        return

    # Validate atom input
    if not args.atom:
        parser.error('Either --atom or --run-tests is required')

    # Run main calculation
    res = gdd_tuned_pbe(
        atom=args.atom,
        basis=args.basis,
        charge=args.charge,
        spin=args.spin,
        unit=args.unit,
        grid_level=args.grid_level,
        conv_tol=args.conv_tol,
        C=args.C,
        omega0=args.omega0,
        n_tail=args.n_tail,
        mu_floor=args.mu_floor,
        extra_cycle=args.extra_cycle,
    )
    print_report(res)


if __name__ == '__main__':
    # Limit threads for reproducibility
    lib.num_threads(1)
    main()
