# Cluster Deployment Summary

## 📦 **Ready for Cluster Deployment**

This folder contains everything needed to run Pauli Hamiltonian analysis on the cluster with Symmer sparse matrix support.

### 🗂️ **Folder Contents** (1.1MB total)

#### Core Execution Files
- `main_cluster_pauli.py` - Main execution script
- `quantum_chemistry_utils_pauli.py` - Core utilities with Symmer sparse matrix support
- `adiabatic_analysis.py` - Adiabatic state preparation gap analysis

#### Pauli Hamiltonian Data Files
- `H2_pauli_hamiltonian.json` - H2 (4 qubits, 3 terms)
- `H4_pauli_hamiltonian.json` - H4 (8 qubits, 9 terms)
- `H2O_pauli_hamiltonian.json` - H2O (14 qubits, 22 terms)
- `N2_pauli_hamiltonian.json` - N2 (20 qubits, 23 terms)

#### Adiabatic State Preparation Files
- `H2_hamiltonian.txt` + `H2_meanfield.txt` - Full adiabatic analysis for H2
- `H4_hamiltonian.txt` + `H4_meanfield.txt` - Full adiabatic analysis for H4
- `H2O_meanfield.txt` - Mean field approximation for H2O (partial analysis)

#### SLURM Job Scripts
- `run_cluster_pauli.sh` - Batch processing all molecules
- `run_cluster_pauli_separate.sh` - Individual molecule processing (recommended)

#### Documentation
- `README_CLUSTER.md` - Cluster setup and usage guide
- `SEPARATE_EXECUTION_GUIDE.md` - Individual processing guide

### 🚀 **Quick Deployment**

1. **Transfer to cluster:**
   ```bash
   scp -r cluster_pauli_deployment/ username@cluster:/path/to/workdir/
   ```

2. **Submit job:**
   ```bash
   cd cluster_pauli_deployment
   sbatch run_cluster_pauli_separate.sh
   ```

### ✅ **Validation Status**

- ✅ **All imports working** (with Symmer fallback)
- ✅ **Data files accessible** (4 molecules auto-detected)
- ✅ **SLURM scripts syntax validated**
- ✅ **Main script execution tested**
- ✅ **Sparse matrix functionality ready** (H2O: 16K×16K sparse)
- ✅ **JSON serialization validated** (complex numbers handled)
- ✅ **Error handling robust** (large systems protected)

### 🎯 **Key Features**

- **Adiabatic state preparation analysis**: Study gaps during evolution from mean field to exact ground state
- **Multiple initial state approximations**: Mean field, MPS approximations with different bond dimensions
- **Complete adiabatic path**: H(s) = (1-s)H_initial + s*H_final with 20 s-values from 0 to 1
- **Comprehensive visualization**: Automatic generation of publication-quality plots
  - Individual gap evolution plots for each bond dimension
  - Multi-bond dimension comparison plots
  - Minimum gap vs bond dimension analysis
- **Automatic sparse/dense selection**: Small systems (≤1024) use dense, larger use sparse
- **Memory safety**: N2 (1M×1M) properly handled with safety checks
- **Symmer integration**: Full sparse matrix support with graceful fallback
- **Individual molecule processing**: Separate output files for each molecule
- **JSON-safe results**: All numpy/complex types properly serialized

### 📊 **Expected Performance**

**MPS + Sparse Matrix Analysis:**
- **H2** (16×16): ~3s total, dense matrix + full adiabatic gap analysis
- **H4** (256×256): ~5-10s, dense matrix + full adiabatic gap analysis
- **H2O** (16K×16K): ~6s MPS + partial gap analysis (sparse matrix, 0.0005 sparsity)
- **N2** (1M×1M): MPS only, protected by safety checks for eigenvalue computation

**Adiabatic Gap Analysis Results:**
- **Complete s-parameter sweep**: 20 points from mean field (s=0) to exact (s=1)
- **Bond dimension dependence**: Gap analysis for each MPS approximation
- **Minimum gap identification**: Critical bottlenecks for adiabatic preparation
- **Automatic visualization**: Publication-quality plots saved to `{molecule}_plots/` directories
  - `{molecule}_gap_evolution_bond_{X}.png` - Individual bond dimension analysis
  - `{molecule}_gap_comparison_all_bonds.png` - Multi-curve comparison
  - `{molecule}_min_gaps_vs_bond_dim.png` - Bottleneck analysis

### 🔧 **Cluster Requirements**

- Python 3.7+
- Required: numpy, scipy, matplotlib, json
- Optional: symmer (for full sparse matrix support)
- SLURM job scheduler

**This deployment package is fully tested and ready for production cluster use.**