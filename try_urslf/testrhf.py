# Import necessary libraries
import numpy as np  
from scipy.special import erf  # Error function used in the model
from pyscf import gto, dft  
from pyscf.dft import numint

# ------------------------------------------------------------------------------
# Define the molecule with coordinates and custom nuclear charges
# ------------------------------------------------------------------------------
mol = gto.M(atom='''
Ne 0 0 0        
        ''',
        basis='cc-pvdz',  
        spin=0)  

# ------------------------------------------------------------------------------
# Compute number of alpha and beta electrons
# ------------------------------------------------------------------------------
n = mol.nelectron  

# ------------------------------------------------------------------------------
# Set up a restricted Kohn-Sham DFT calculation using PBE functional
# ------------------------------------------------------------------------------
mf = dft.RKS(mol)  
mf.xc = 'PBE'  
mf.run()  

# ------------------------------------------------------------------------------
# Generate a numerical integration grid for DFT
# ------------------------------------------------------------------------------
grids = dft.gen_grid.Grids(mol)  
grids.level = 5  # Grid level: higher value = finer grid
grids.build()  

# ------------------------------------------------------------------------------
# Evaluate atomic orbital (AO) values and their derivatives on the grid
# ------------------------------------------------------------------------------
ao = numint.eval_ao(mol, grids.coords, deriv=1)  # AO values and gradients

# ------------------------------------------------------------------------------
# Get the one-electron density matrix from the SCF calculation
# ------------------------------------------------------------------------------
dm = mf.make_rdm1()  # 1-electron reduced density matrix

# ------------------------------------------------------------------------------
# Constants used 
# ------------------------------------------------------------------------------
rho_in = 0.0164  # Normalization constant for density
epsilon = 1e-20  # Threshold to avoid division by zero
a1 = 1.91718
a2 = -0.02817
a3 = 0.14954

# ------------------------------------------------------------------------------
# Compute spin density and its gradient (GGA-level evaluation)
# ------------------------------------------------------------------------------
rho_total_a, grad_x_a, grad_y_a, grad_z_a = numint.eval_rho(mol, ao, dm, xctype='GGA')

print("rho_total_a", rho_total_a)  # Optional: print raw α-density values

# ------------------------------------------------------------------------------
# Compute the error function term for the custom model
# ------------------------------------------------------------------------------
erf_term_a = erf(n * rho_total_a / rho_in)  # erf scaled by density
mask_a = rho_total_a > epsilon  # Mask to skip near-zero densities

# ------------------------------------------------------------------------------
# Build the custom integrand involving density and erf term
# ------------------------------------------------------------------------------
integrand_tg_a = np.zeros_like(rho_total_a)  # Initialize integrand
# Only compute where density is meaningful (above epsilon)
integrand_tg_a[mask_a] = (
    (3 / (4 * np.pi * rho_total_a[mask_a])) ** (1/3) * erf_term_a[mask_a]
)

# ------------------------------------------------------------------------------
# Perform real-space integration: sum values weighted by grid weights
# ------------------------------------------------------------------------------
V_a = np.dot(erf_term_a, grids.weights)  # Normalization factor
tilde_g_integral_a = np.dot(integrand_tg_a, grids.weights)  # Integrated value

# ------------------------------------------------------------------------------
# Compute the average tilde_g and plug into the μ_eff model
# ------------------------------------------------------------------------------
tilde_g_a = tilde_g_integral_a / V_a if V_a > 1e-14 else 0.0  # Avoid division by small number

# Empirical formula for effective range-separation parameter μ_eff
mu_eff_a = (
    a1 / tilde_g_a + a2 * tilde_g_a / (1 + a3 * tilde_g_a ** 2)
)

# ------------------------------------------------------------------------------
# Final μ_eff 
# ------------------------------------------------------------------------------
mu_eff = mu_eff_a 

# Output the result
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


