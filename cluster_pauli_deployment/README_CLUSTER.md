# CLUSTER DEPLOYMENT INSTRUCTIONS

## 📁 Files in this directory
- `main_cluster_pauli.py` - Main execution script
- `quantum_chemistry_utils_pauli.py` - Symmer-based utilities
- `adiabatic_analysis.py` - Gap analysis functions
- `run_cluster_pauli.sh` - SLURM submission script (combined execution)
- `run_cluster_pauli_separate.sh` - SLURM submission script (separate execution)
- `*_pauli_hamiltonian.json` - Pre-generated Pauli Hamiltonians
- `pauli_hamiltonians_summary.json` - Generation summary

## 🚀 Execution Options

### Option 1: Combined Execution (Original)
All molecules processed together in one job.

```bash
# Upload to cluster
scp -r cluster_pauli_deployment/ user@cluster:/path/to/workdir/

# Test run (H2 only)
sbatch run_cluster_pauli.sh test

# Full analysis (all molecules together)
sbatch run_cluster_pauli.sh
```

### Option 2: Separate Execution (NEW!)
Each molecule processed individually with separate output files.

```bash
# Upload to cluster
scp -r cluster_pauli_deployment/ user@cluster:/path/to/workdir/

# Test run (first molecule only)
sbatch run_cluster_pauli_separate.sh test

# Full analysis (each molecule separately)
sbatch run_cluster_pauli_separate.sh
```

## 📊 Expected Outputs

### Combined Execution Output:
- `pauli_analysis_results.json` - Main results (all molecules)
- `*_mps.txt` - MPS decomposition results
- `hpc_gap_pauli_output.out` - SLURM stdout
- `hpc_gap_pauli_error.out` - SLURM stderr

### Separate Execution Output:
- `individual_results/` directory containing:
  - `{molecule}_output.out` - Individual stdout for each molecule
  - `{molecule}_error.out` - Individual stderr for each molecule
  - `{molecule}_results.json` - Individual results for each molecule
- `master_pauli_results.json` - Combined master results file
- `completion_timestamp_separate.txt` - Final completion summary
- `hpc_gap_pauli_separate_output.out` - SLURM stdout
- `hpc_gap_pauli_separate_error.out` - SLURM stderr

## 🔧 Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Symmer (optional, fallback available)
- 4GB memory, 8 cores

## ✅ Molecules included
- H2: 4 qubits, 3 Pauli terms
- H4: 8 qubits, 9 Pauli terms
- H2O: 14 qubits, 22 Pauli terms
- **N2: 20 qubits, 23 Pauli terms** (NEW!)
- **Auto-detection**: Script automatically finds available molecules

## 🔧 Auto-Detection Feature
The cluster script automatically detects which Pauli Hamiltonian files are present and processes only those molecules. No manual configuration needed!

Total memory usage: <1GB vs 64GB for dense matrices!