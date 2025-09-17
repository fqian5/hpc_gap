# HPC Gap Analysis: Quantum Chemistry Molecular Hamiltonian Study

This project implements a comprehensive analysis of molecular Hamiltonians using Matrix Product States (MPS) and adiabatic quantum computation to study energy gaps across different bond dimensions.

## Project Overview

The pipeline performs the following analysis for molecules H₂, H₄, BeH₃, CH₄, NH₃, and H₂O:

1. **Molecular Setup**: Define molecular geometries with STO-3G basis set
2. **Hamiltonian Construction**: Convert molecules to qubit Hamiltonians using OpenFermion-PySCF
3. **Eigenvalue Analysis**: Compute ground and excited states
4. **MPS Decomposition**: Approximate ground states using SVD-based MPS with varying bond dimensions
5. **Mean Field Approximation**: Generate mean field Hamiltonians
6. **Adiabatic Analysis**: Study energy gaps in adiabatic evolution from mean field to full Hamiltonian
7. **Gap Visualization**: Plot gap evolution and bond dimension dependencies

## File Structure

```
hpc_gap/
├── molecules/               # Molecular geometry definitions
│   ├── H2.py
│   ├── H4.py
│   ├── BeH3.py
│   ├── CH4.py
│   ├── NH3.py
│   └── H2O.py
├── ez_adiabatic.py         # State preparation utilities
├── mps_prep.py             # MPS circuit preparation (requires quimb/qiskit)
├── quantum_chemistry_utils.py  # Core quantum chemistry functions
├── adiabatic_analysis.py   # Gap analysis and plotting
├── main.py                 # Main execution script
├── run_hpc_gap.sh          # SLURM cluster submission script
└── README.md               # This file
```

## Usage

### Local Testing

Test with H₂ molecule only:
```bash
python main.py --test
```

### Full Analysis

Run complete analysis for all molecules:
```bash
python main.py --processes 4
```

### Cluster Execution

Submit to SLURM cluster:
```bash
# Test run
sbatch run_hpc_gap.sh test

# Full run
sbatch run_hpc_gap.sh
```

## Command Line Options

- `--molecules`: Specify molecules to analyze (default: all)
- `--test`: Run test mode with H₂ only
- `--processes`: Number of parallel processes
- `--skip-processing`: Skip molecular processing, only run gap analysis
- `--skip-gaps`: Skip gap analysis, only run molecular processing

## Dependencies

### Required Python Packages
- numpy
- scipy
- matplotlib
- pyscf
- openfermion
- openfermion-pyscf

### Optional (for MPS circuit preparation)
- quimb
- qiskit

## Output Files

### Per Molecule
- `{molecule}_hamiltonian.txt`: Pickled qubit Hamiltonian matrix
- `{molecule}_eigen.txt`: JSON with eigenvalues and eigenvectors
- `{molecule}_mps.txt`: JSON with MPS decomposition results
- `{molecule}_meanfield.txt`: Pickled mean field Hamiltonian

### Gap Analysis
- `{bond_dim}/mf{molecule}`: JSON with gap evolution data
- `{bond_dim}/mf{molecule}_gap.png`: Gap vs adiabatic parameter plot
- `{molecule}_bond_gap.png`: Bond dimension vs minimum gap plot

### Summary
- `summary_plots/all_molecules_bond_gap.png`: Combined analysis plot
- `processing_results.json`: Molecular processing summary
- `gap_analysis_results.json`: Gap analysis summary

## Molecular Geometries

- **H₂**: Linear diatomic (0.74 Å)
- **H₄**: Linear chain (1.0 Å spacing)
- **BeH₃**: Trigonal planar (1.34 Å Be-H bonds)
- **CH₄**: Tetrahedral (1.09 Å C-H bonds)
- **NH₃**: Pyramidal (1.01 Å N-H bonds, 107° angle)
- **H₂O**: Bent (0.96 Å O-H bonds, 104.5° angle)

## Implementation Details

### MPS Decomposition
- Custom SVD-based implementation from scratch
- Linear progression of bond dimensions from 2 to maximum
- 5 steps in bond dimension progression
- Overlap calculation with exact ground state

### Adiabatic Evolution
- 20 steps in adiabatic parameter s ∈ [0,1]
- H(s) = (1-s)H_meanfield + s*H_full
- Gap normalization using initial and final gap average

### Parallelization
- Molecule processing parallelized across available cores
- Gap analysis parallelized per molecule
- Configurable process count

## Cluster Configuration

The SLURM script is configured for:
- 8 CPU cores
- 32 GB RAM
- 12 hour time limit
- Batch partition

Modify `run_hpc_gap.sh` for your specific cluster requirements.

## Error Handling

The pipeline includes comprehensive error handling:
- Individual molecule failures don't stop the entire analysis
- Missing dependencies are caught with informative messages
- Results are saved incrementally to prevent data loss

## Performance Notes

- Small molecules (H₂, H₄): Complete analysis in minutes
- Larger molecules: May require hours depending on system size
- Memory usage scales exponentially with qubit count
- Gap analysis is the most computationally intensive step

## Troubleshooting

### Common Issues

1. **OpenFermion Import Error**: Ensure openfermion and openfermion-pyscf are installed
2. **PySCF Convergence**: Some molecules may require adjusted SCF settings
3. **Memory Issues**: Reduce number of parallel processes or use cluster with more RAM
4. **Qubit Limit**: Very large molecules may exceed practical qubit limits

### Debug Mode

Run with single process for detailed error messages:
```bash
python main.py --test --processes 1
```