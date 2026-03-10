# Simplified Tuned Range Separated

This initiative offers practical guidance on implementing the Simplified Tuned Range-Separated Functional (STRSF). The methodology is demonstrated through the try_urslf GitHub repository, which hosts a customized extension of the PySCF computational chemistry framework. Within this repository, users will find dedicated modules for both restricted and unrestricted calculations, tailored explicitly for optimizing the ω_eff parameter in range-separated hybrid functionals.

# Repository Overview:

### ω_eff Implementation
Computes the **effective range-separation parameter (ω_eff)** based on the formulation introduced in:

**J. Phys. Chem. Lett. 2025, 16, 32, 8198–8208**

### ω_GDD Implementation
Computes the range-separation parameter using the **Global Density-Dependent (GDD)** approach from 

**J. Phys. Chem. A 2013, 117, 45, 11580–11586**

The repository archives computational workflows and raw data supporting the figures and tables in the associated paper. It includes:

Input files from quantum chemistry packages (NWChem) for reproducibility.

Calculation details for transparency and peer validation.

Separately, the geom repository houses all molecular geometries (optimized structures) used in the study, ensuring seamless reconstruction of the paper’s computational experiments.


## 🚀 Features
### Dual Range-Separation Parameter Methods
Two independent parameter tuning approaches are implemented:

- **ω_eff** – Effective range-separation parameter derived from the STRSF formulation.
- **ω_GDD** – Global Density-Dependent tuning method.

### PySCF Integration
The implementations extend the **PySCF** framework to allow efficient calculations of range-separated hybrid functionals.

### Reproducible Workflows
Complete input files and computational outputs are provided for transparency and reproducibility.

### Molecular Geometry Library
Optimized molecular structures used in the study are available for direct use in quantum chemistry calculations.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/aditisingh4812/simplified_tuned_range_separated.git
cd simplified_tuned_range_separated
pip install -r requirements.txt
```


## **©️ Copyright**
**Copyright © 2025, Szymon Śmiga Group**

## 📚 Citing Simplified Tuned Range Separated

If you use **Simplified Tuned Range Separated** in your research, please cite the following work:

https://pubs.acs.org/doi/10.1021/acs.jpclett.5c01441

**Simplified Tuned Range Separated Toolkit**  



