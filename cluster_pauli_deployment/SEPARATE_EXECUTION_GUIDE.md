# Separate Molecule Execution Guide

## 🎯 Purpose
The separate execution mode processes each molecule individually with dedicated output files. This provides:
- **Better debugging**: Individual error logs per molecule
- **Progress tracking**: See which molecules completed successfully
- **Fault tolerance**: Failed molecules don't stop others
- **Organized results**: Separate JSON files for each molecule

## 🚀 Usage

### Basic Command
```bash
sbatch run_cluster_pauli_separate.sh
```

### Test Mode (First Molecule Only)
```bash
sbatch run_cluster_pauli_separate.sh test
```

## 📁 Output Structure

When using separate execution, you'll get:

```
individual_results/
├── H2_output.out          # H2 stdout log
├── H2_error.out           # H2 stderr log
├── H2_results.json        # H2 analysis results
├── H4_output.out          # H4 stdout log
├── H4_error.out           # H4 stderr log
├── H4_results.json        # H4 analysis results
├── H2O_output.out         # H2O stdout log
├── H2O_error.out          # H2O stderr log
├── H2O_results.json       # H2O analysis results
├── N2_output.out          # N2 stdout log
├── N2_error.out           # N2 stderr log
└── N2_results.json        # N2 analysis results

master_pauli_results.json  # Combined results from all successful molecules
completion_timestamp_separate.txt  # Final summary with success/failure counts
```

## 📊 Example Workflow

1. **Submit Job**:
   ```bash
   sbatch run_cluster_pauli_separate.sh
   ```

2. **Monitor Progress**:
   ```bash
   # Check overall job status
   squeue -u $USER

   # Check which molecules completed
   ls individual_results/*_results.json

   # View real-time progress for specific molecule
   tail -f individual_results/H2O_output.out
   ```

3. **Check Results**:
   ```bash
   # Quick completion summary
   cat completion_timestamp_separate.txt

   # View combined results
   cat master_pauli_results.json

   # Check individual molecule
   cat individual_results/N2_results.json
   ```

## 🔍 Debugging

### Check Failed Molecules
```bash
# See which molecules failed
grep "Failed molecules" completion_timestamp_separate.txt

# Check error log for specific molecule
cat individual_results/{molecule}_error.out
```

### Monitor Large Molecules
```bash
# N2 might take longer due to 20 qubits
tail -f individual_results/N2_output.out
```

## ⚡ Advantages Over Combined Execution

| Feature | Combined Execution | Separate Execution |
|---------|-------------------|-------------------|
| **Fault Tolerance** | One failure stops all | Independent execution |
| **Debugging** | Mixed logs | Individual logs |
| **Progress Tracking** | All-or-nothing | Per-molecule status |
| **Result Organization** | Single JSON | Multiple JSONs + master |
| **Memory Usage** | Peak for largest | Per-molecule peak |
| **Restart Capability** | Full restart needed | Retry failed molecules |

## 🎛️ Advanced Usage

### Retry Failed Molecules Only
If some molecules failed, you can rerun just those:
```bash
# Edit the molecules list in completion summary to see failures
python main_cluster_pauli.py --molecules BeH3 --processes 1
```

### Custom Molecule Selection
```bash
# Run only specific molecules
python main_cluster_pauli.py --molecules H2 N2 --processes 1
```

### Resource Optimization
For very large molecules, you might want to run them separately with different resource allocations:
```bash
# Custom SLURM script for N2 only
#SBATCH --mem=128000  # More memory for large molecules
python main_cluster_pauli.py --molecules N2 --processes 8
```

## 📈 Expected Runtimes

Based on system size:
- **H2** (4 qubits): ~1 second
- **H4** (8 qubits): ~5 seconds
- **H2O** (14 qubits): ~30 seconds
- **N2** (20 qubits): ~2-5 minutes

Total separate execution: ~6 minutes vs combined: ~6 minutes
*Separate execution provides better monitoring with minimal overhead*