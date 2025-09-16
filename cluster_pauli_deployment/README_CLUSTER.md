# CLUSTER DEPLOYMENT INSTRUCTIONS

## 📁 Files in this directory
- `main_cluster_pauli.py` - Main execution script
- `quantum_chemistry_utils_pauli.py` - Symmer-based utilities
- `adiabatic_analysis.py` - Gap analysis functions
- `run_cluster_pauli.sh` - SLURM submission script
- `*_pauli_hamiltonian.json` - Pre-generated Pauli Hamiltonians
- `pauli_hamiltonians_summary.json` - Generation summary

## 🚀 Quick Start

### 1. Upload to cluster
```bash
scp -r cluster_pauli_deployment/ user@cluster:/path/to/workdir/
```

### 2. Test run (H2 only)
```bash
cd cluster_pauli_deployment
sbatch run_cluster_pauli.sh test
```

### 3. Full analysis (all molecules)
```bash
sbatch run_cluster_pauli.sh
```

## 📊 Expected outputs
- `pauli_analysis_results.json` - Main results
- `gap_analysis_results.json` - Gap analysis data
- `*_mps.txt` - MPS decomposition results
- `hpc_gap_pauli_output.out` - SLURM stdout
- `hpc_gap_pauli_error.out` - SLURM stderr

## 🔧 Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Symmer (optional, fallback available)
- 4GB memory, 8 cores

## ✅ Molecules included
- H2: 4 qubits, 3 Pauli terms
- H4: 8 qubits, 9 Pauli terms
- H2O: 14 qubits, 22 Pauli terms
- BeH3: Failed locally (will skip on cluster)

Total memory usage: <1GB vs 64GB for dense matrices!