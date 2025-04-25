import numpy as np
from scipy.special import erf
from pyscf import gto, dft  # Changed from scf to dft
from pyscf.dft import numint

# Define the molecule
mol = gto.M(atom='''  
C         0.0000  0.0000  5.7854 0.744860
O        -0.6754 -0.6754  5.1099 0.888124
O         0.6754  0.6754  6.4607 0.888065
        ''', basis='cc-pvdz',spin=0)

n = mol.nelectron
n_alpha = (mol.nelectron + mol.spin) // 2
print("n_alpha",n_alpha)
n_beta = mol.nelectron - n_alpha
print("n_beta",n_beta)
# Use DFT UKS instead of HF UKS (critical fix)
mf = dft.RKS(mol)
mf.xc = 'PBE'
mf.run()

# Generate integration grid
grids = dft.gen_grid.Grids(mol)
grids.level = 5
grids.build()

# Evaluate AO values and density gradient
ao = numint.eval_ao(mol, grids.coords, deriv=1)

# Get total density (alpha + beta)
dm = mf.make_rdm1()

rho_in = 0.000696
epsilon = 1e-20
a1 = 1.91718
a2 = -0.02817
a3 = 0.14954

#################alpha part##########################
rho_total_a, grad_x_a, grad_y_a, grad_z_a = numint.eval_rho(mol, ao, dm, xctype='GGA')
rho_total_a = 0.5*rho_total_a
print("rho_total_a",rho_total_a)
erf_term_a = erf(n_alpha*rho_total_a/rho_in)
mask_a = rho_total_a > epsilon

integrand_tg_a = np.zeros_like(rho_total_a)
integrand_tg_a[mask_a] = ((3/(4*np.pi*rho_total_a[mask_a]))**(1/3) * (erf_term_a[mask_a]))

# Compute integrals
V_a = np.dot(erf_term_a, grids.weights)
tilde_g_integral_a = np.dot(integrand_tg_a, grids.weights)

# Final result
tilde_g_a = tilde_g_integral_a / V_a if V_a > 1e-14 else 0.0

mu_eff_a = a1/tilde_g_a + a2*tilde_g_a/(1 + a3*tilde_g_a**2)




mu_eff = (mu_eff_a)*2

print("mu_eff", mu_eff)
