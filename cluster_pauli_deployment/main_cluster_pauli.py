#!/usr/bin/env python3
"""
Main cluster execution script for Pauli Hamiltonian analysis
Uses Symmer to read pre-generated Pauli Hamiltonians
"""

import argparse
import json
import time
import multiprocessing as mp
from quantum_chemistry_utils_pauli import process_molecule
from adiabatic_analysis import process_molecule_gaps

def run_analysis(molecules, processes=1, test_mode=False):
    """Run the full analysis pipeline"""
    print(f"Starting analysis with {processes} processes...")
    print(f"Test mode: {test_mode}")
    print(f"Molecules: {molecules}")

    start_time = time.time()

    if processes == 1:
        # Single-threaded execution
        results = []
        for molecule in molecules:
            result = process_molecule(molecule, test_mode)
            results.append(result)
    else:
        # Multi-threaded execution
        with mp.Pool(processes=processes) as pool:
            # Create tasks
            tasks = [(molecule, test_mode) for molecule in molecules]

            # Run tasks
            results = pool.starmap(process_molecule, tasks)

    # Filter successful results
    successful_results = [r for r in results if r.get('success', False)]

    print(f"\n{'='*50}")
    print(f"MOLECULAR PROCESSING COMPLETE")
    print(f"Successful: {len(successful_results)}/{len(molecules)}")

    # Run adiabatic gap analysis
    print(f"\nStarting adiabatic gap analysis...")
    gap_results = []

    for result in successful_results:
        try:
            # Analyze adiabatic gaps
            gap_analysis = process_molecule_gaps(
                result['molecule_name'],
                result['bond_dimensions']
            )
            gap_results.append(gap_analysis)
        except Exception as e:
            print(f"Gap analysis failed for {result['molecule_name']}: {e}")

    # Save final results
    final_results = {
        'analysis_complete': True,
        'total_molecules': len(molecules),
        'successful_molecules': len(successful_results),
        'processing_time': time.time() - start_time,
        'molecular_results': successful_results,
        'gap_results': gap_results
    }

    with open('pauli_analysis_results.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    print(f"\n{'='*50}")
    print(f"PAULI HAMILTONIAN ANALYSIS COMPLETE")
    print(f"Total time: {final_results['processing_time']:.2f} seconds")
    print(f"Results saved to: pauli_analysis_results.json")

    return final_results

def main():
    parser = argparse.ArgumentParser(description='Pauli Hamiltonian cluster analysis')
    parser.add_argument('--molecules', nargs='+',
                        default=['H2', 'H4', 'BeH3', 'H2O'],
                        help='Molecules to analyze')
    parser.add_argument('--processes', type=int, default=1,
                        help='Number of processes to use')
    parser.add_argument('--test', action='store_true',
                        help='Test mode (H2 only)')

    args = parser.parse_args()

    print("PAULI HAMILTONIAN CLUSTER ANALYSIS")
    print("=" * 50)
    print("Using Symmer to read pre-generated Pauli Hamiltonians")

    if args.test:
        molecules = ['H2']
        print("Running in TEST mode")
    else:
        molecules = args.molecules
        print("Running FULL analysis")

    print(f"Molecules: {molecules}")
    print(f"Processes: {args.processes}")

    # Check for required Pauli Hamiltonian files
    missing_files = []
    for molecule in molecules:
        pauli_file = f'{molecule}_pauli_hamiltonian.json'
        try:
            with open(pauli_file, 'r') as f:
                data = json.load(f)
            print(f"✓ Found {pauli_file}: {data['n_qubits']} qubits, {len(data['pauli_terms'])} terms")
        except FileNotFoundError:
            missing_files.append(pauli_file)
            print(f"✗ Missing {pauli_file}")

    if missing_files:
        print(f"\nERROR: Missing Pauli Hamiltonian files:")
        for file in missing_files:
            print(f"  {file}")
        print(f"\nPlease run 'python generate_pauli_hamiltonians.py' locally first")
        print(f"and transfer the *_pauli_hamiltonian.json files to the cluster")
        return 1

    print(f"\n✓ All required Pauli Hamiltonian files found")

    try:
        results = run_analysis(molecules, args.processes, args.test)
        return 0

    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())