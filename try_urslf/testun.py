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
rho_in = 0.0164
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


# ------------------------------------------------------------------------------
# ----------------------- BETA CHANNEL -----------------------------------------
# ------------------------------------------------------------------------------
# Same as alpha, but using beta density matrix
rho_total_b, grad_x_b, grad_y_b, grad_z_b = numint.eval_rho(mol, ao, dm[1], xctype='GGA')
print("rho_total_b", rho_total_b)

# ------------------------------------------------------------------------------
# -----------------------      TOTAL   -----------------------------------------
# ------------------------------------------------------------------------------

rho_total = rho_total_b + rho_total_a
erf_term = erf(n * rho_total / rho_in)
mask = rho_total > epsilon

integrand_tg = np.zeros_like(rho_total)
integrand_tg[mask] = (
    (3 / (4 * np.pi * rho_total[mask])) ** (1/3) * erf_term[mask]
)

V = np.dot(erf_term, grids.weights)
tilde_g_integral = np.dot(integrand_tg, grids.weights)
tilde_g = tilde_g_integral / V if V > 1e-14 else 0.0


# ------------------------------------------------------------------------------
# Final result: total effective mu 
# ------------------------------------------------------------------------------
mu_eff = a1 / tilde_g + a2 * tilde_g / (1 + a3 * tilde_g**2)

# Print result
print("mu_eff", mu_eff)

# ------------------------------------------------------------------------------
# Final n_c cutoff
# ------------------------------------------------------------------------------

n_c = rho_in/n

print("n_c", n_c)

# ------------------------------------------------------------------------------
# Final cutoff radius r_c
# ------------------------------------------------------------------------------

r_c = (3 / (4 * np.pi * n_c)) ** (1 / 3)

print("r_c", r_c)

