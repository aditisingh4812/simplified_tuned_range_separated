1. Dataset title

Computational Data and Software for Simplified, Physically Motivated, and Broadly Applicable Range-Separation Tuning

This repository contains the computational data, input files, workflows, molecular geometries, and software implementations supporting the study:

A. Singh, S. Jana, L. A. Constantin, F. Della Sala, P. Samal, and S. Śmiga, “Simplified, Physically Motivated, and Broadly Applicable Range-Separation Tuning,” The Journal of Physical Chemistry Letters 16, 8198–8208 (2025).

DOI: https://doi.org/10.1021/acs.jpclett.5c01441

2. Authors and affiliations

Aditi Singh
Institute of Physics, Faculty of Physics, Astronomy and Informatics, Nicolaus Copernicus University, ul. Grudziądzka 5, 87-100 Toruń, Poland.

Subrata Jana
Institute of Physics, Faculty of Physics, Astronomy and Informatics, Nicolaus Copernicus University, ul. Grudziądzka 5, 87-100 Toruń, Poland.

Lucian A. Constantin
Institute for Microelectronics and Microsystems (CNR-IMM), 73100 Lecce, Italy.

Fabio Della Sala
Institute for Microelectronics and Microsystems (CNR-IMM), 73100 Lecce, Italy; Center for Biomolecular Nanotechnologies, Istituto Italiano di Tecnologia, 73010 Arnesano, LE, Italy.

Prasanjit Samal
School of Physical Sciences, National Institute of Science Education and Research, An OCC of Homi Bhabha National Institute, Bhubaneswar 752050, India.

Szymon Śmiga
Institute of Physics, Faculty of Physics, Astronomy and Informatics, Nicolaus Copernicus University, ul. Grudziądzka 5, 87-100 Toruń, Poland.

These affiliations correspond to the publication and its Supporting Information.

3. Date of data creation and collection

2025

The computational data were generated as part of the research underlying the associated publication, which was published in 2025. The repository contains the computational workflows and data supporting the figures and tables of the study.

Repository release: v1.0.0, archived in Zenodo in 2026.

For the purposes of this metadata, 2025 should be used as the principal data-creation year unless you have a documented, more precise calculation period.

4. Description of research methodology and tools

The dataset contains computational results supporting the development and assessment of a simplified approach for determining the range-separation parameter in range-separated hybrid (RSH) density-functional calculations.

The ω_eff parameter is determined from the electron density using the formulation introduced in the associated publication. The approach is based on the compressibility sum rule of density functional theory and avoids the multiple self-consistent calculations normally required by conventional tuning procedures.

The repository also contains an implementation of the Global Density-Dependent (GDD) range-separation parameter, ω_GDD, for comparison.

The computational workflows include:

Density Functional Theory (DFT) calculations.
Range-separated hybrid (RSH) calculations.
Calculation of the effective range-separation parameter, ω_eff.
Calculation of the GDD range-separation parameter, ω_GDD.
Molecular and periodic-system calculations.
Ground-state electron-density calculations.
Excited-state calculations using time-dependent DFT/range-separated hybrid approaches.
Supporting calculations performed using PySCF and NWChem where applicable.

The repository provides customized PySCF functionality for calculating ω_eff and implementing the corresponding range-separated hybrid calculations. It also contains NWChem input files and computational workflows required to reproduce the reported calculations.

For the periodic calculations reported in the publication, VASP calculations used PBE-generated electron densities; the reported setups include Γ-centered k-point meshes and specified plane-wave energy cutoffs.

5. Data format and structure

The repository contains both computational software and research data.

Software

Python source code and PySCF-based implementations are provided for the calculation of range-separation parameters.

Dependencies are specified in:

requirements.txt
Computational input files

Input files for quantum-chemistry calculations, including NWChem input files, are provided to facilitate reproduction of the calculations.

Computational results

The repository contains numerical results and computational outputs associated with the figures and tables of the publication.

Molecular geometries

Optimized molecular geometries used in the calculations are provided through the geometry-related repository/directories.

Repository organization

The principal directories include:

fig2_calc/
geom/
tab1_calc/
tab2_calc/
try_urslf/

The try_urslf component contains the customized computational implementation, while the other directories contain calculation workflows, geometries, and data associated with the reported results.

Typical computational data formats include:

.py       Python source code
.in       Quantum-chemistry input files
.xyz      Molecular geometries
.out      Computational output files
.txt      Numerical/text data

You should remove any format from this list that isn't actually present in your repository. The repository itself confirms Python code, NWChem inputs, computational outputs and molecular geometries, but I wouldn't claim a specific file extension unless it is actually present.

6. Data sharing and licensing

The repository is publicly available through GitHub:

GitHub repository — simplified_tuned_range_separated

The software repository is distributed under the BSD 3-Clause License, as indicated by the repository's existing LICENSE file.

The archived version of the repository is available through Zenodo under the persistent identifier:

DOI: 10.5281/zenodo.22657188

Users are requested to cite both the software DOI and the associated publication when using the software or computational data.

7. Funding

This work was supported by the National Science Centre (Narodowe Centrum Nauki), Poland:

Project registration number: 2021/38/B/HS1/00001

Project title: Quantum Chemistry under Spatial Confinement

Principal Investigator: Dr. Szymon Filip Śmiga

8. Persistent identifiers

Software/data repository DOI:

10.5281/zenodo.22657188

Associated publication DOI:

10.1021/acs.jpclett.5c01441

GitHub repository:

https://github.com/aditisingh4812/simplified_tuned_range_separated
