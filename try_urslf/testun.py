# Import necessary libraries
import numpy as np
from scipy.special import erf  # Error function used in the model
from pyscf import gto, dft  
from pyscf.dft import numint  

# ------------------------------------------------------------------------------
# Define the molecule 
# ------------------------------------------------------------------------------
mol = gto.M(atom='''
 N     0.000000     0.000000    -1.266782
 C     0.000000     0.000000    -0.038533
 O     0.000000     0.000000     1.137334
        ''',
        basis='cc-pvdz',
        spin=1)  

# ------------------------------------------------------------------------------
# Compute number of alpha and beta electrons
# ------------------------------------------------------------------------------
n = mol.nelectron
n_alpha = (n + mol.spin) // 2
print("n_alpha", n_alpha)
n_beta = n - n_alpha
print("n_beta", n_beta)

# ------------------------------------------------------------------------------
# Run unrestricted Kohn-Sham DFT with a hybrid functional (PBE0)
# ------------------------------------------------------------------------------
mf = dft.UKS(mol)  
mf.xc = 'PBE'     
mf.run()

# ------------------------------------------------------------------------------
# Generate a numerical integration grid for real-space evaluation
# ------------------------------------------------------------------------------
grids = dft.gen_grid.Grids(mol)
grids.level = 5  # Grid density level (fine grid)
grids.build()

# ------------------------------------------------------------------------------
# Evaluate atomic orbital (AO) values and gradients on the grid
# ------------------------------------------------------------------------------
ao = numint.eval_ao(mol, grids.coords, deriv=1)

# ------------------------------------------------------------------------------
# Get 1-electron density matrix for both alpha and beta channels
# ------------------------------------------------------------------------------
dm = mf.make_rdm1()  # dm[0] = alpha, dm[1] = beta

# ------------------------------------------------------------------------------
# Constants used in model
# ------------------------------------------------------------------------------
rho_in = 0.000696
epsilon = 1e-20
a1 = 1.91718
a2 = -0.02817
a3 = 0.14954

# ------------------------------------------------------------------------------
# ----------------------- ALPHA CHANNEL ----------------------------------------
# ------------------------------------------------------------------------------
# Evaluate spin-up (alpha) electron density and its gradient
rho_total_a, grad_x_a, grad_y_a, grad_z_a = numint.eval_rho(mol, ao, dm[0], xctype='GGA')
print("rho_total_a", rho_total_a)

# Compute erf term scaled by number of alpha electrons
erf_term_a = erf(n_alpha * rho_total_a / rho_in)

# Avoid numerical issues with zero/near-zero density
mask_a = rho_total_a > epsilon

# Initialize integrand
integrand_tg_a = np.zeros_like(rho_total_a)
# Compute the custom integrand only where density is meaningful
integrand_tg_a[mask_a] = (
    (3 / (4 * np.pi * rho_total_a[mask_a])) ** (1/3) * erf_term_a[mask_a]
)

# Perform real-space integration
V_a = np.dot(erf_term_a, grids.weights)
tilde_g_integral_a = np.dot(integrand_tg_a, grids.weights)

# Normalize
tilde_g_a = tilde_g_integral_a / V_a if V_a > 1e-14 else 0.0

# Compute mu_eff for alpha using empirical model
mu_eff_a = a1 / tilde_g_a + a2 * tilde_g_a / (1 + a3 * tilde_g_a**2)

# ------------------------------------------------------------------------------
# ----------------------- BETA CHANNEL -----------------------------------------
# ------------------------------------------------------------------------------
# Same as alpha, but using beta density matrix
rho_total_b, grad_x_b, grad_y_b, grad_z_b = numint.eval_rho(mol, ao, dm[1], xctype='GGA')
print("rho_total_b", rho_total_b)

erf_term_b = erf(n_beta * rho_total_b / rho_in)
mask_b = rho_total_b > epsilon

integrand_tg_b = np.zeros_like(rho_total_b)
integrand_tg_b[mask_b] = (
    (3 / (4 * np.pi * rho_total_b[mask_b])) ** (1/3) * erf_term_b[mask_b]
)

V_b = np.dot(erf_term_b, grids.weights)
tilde_g_integral_b = np.dot(integrand_tg_b, grids.weights)
tilde_g_b = tilde_g_integral_b / V_b if V_b > 1e-14 else 0.0

mu_eff_b = a1 / tilde_g_b + a2 * tilde_g_b / (1 + a3 * tilde_g_b**2)

# ------------------------------------------------------------------------------
# Final result: total effective mu as sum of alpha and beta channels
# ------------------------------------------------------------------------------
mu_eff = mu_eff_a + mu_eff_b

# Print result
print("mu_eff", mu_eff)

