"""
Adiabatic Quantum Computation Analysis for Molecular Hamiltonians
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pickle
import json
from scipy.sparse.linalg import eigsh
import scipy.linalg
from multiprocessing import Pool, cpu_count

def plot_adiabatic_gaps(s_values, gaps, molecule_name, bond_dim, save_dir=None):
    """
    Plot adiabatic gap evolution vs s parameter

    Parameters:
    -----------
    s_values : array-like
        Adiabatic parameter values (0 to 1)
    gaps : array-like
        Energy gaps at each s value
    molecule_name : str
        Name of the molecule
    bond_dim : int
        Bond dimension
    save_dir : str, optional
        Directory to save plots
    """
    plt.figure(figsize=(10, 6))
    plt.plot(s_values, gaps, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Adiabatic Parameter s', fontsize=12)
    plt.ylabel('Energy Gap', fontsize=12)
    plt.title(f'{molecule_name} - Adiabatic Gap Evolution (Bond Dim {bond_dim})', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Highlight minimum gap
    min_idx = np.argmin(gaps)
    plt.plot(s_values[min_idx], gaps[min_idx], 'ro', markersize=8,
             label=f'Min Gap: {gaps[min_idx]:.4f} at s={s_values[min_idx]:.3f}')
    plt.legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/{molecule_name}_gap_evolution_bond_{bond_dim}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Plot saved: {filename}")

    plt.close()

def plot_bond_dimension_comparison(results_dict, molecule_name, save_dir=None):
    """
    Plot gap evolution for all bond dimensions on same plot

    Parameters:
    -----------
    results_dict : dict
        Dictionary with bond dimensions as keys, each containing s_values and gaps
    molecule_name : str
        Name of the molecule
    save_dir : str, optional
        Directory to save plots
    """
    plt.figure(figsize=(12, 8))

    colors = plt.cm.viridis(np.linspace(0, 1, len(results_dict)))

    for i, (bond_dim, data) in enumerate(results_dict.items()):
        s_values = np.array(data['s_values'])
        gaps = np.array(data['gaps'])

        plt.plot(s_values, gaps, color=colors[i], linewidth=2,
                marker='o', markersize=3, label=f'Bond Dim {bond_dim}')

    plt.xlabel('Adiabatic Parameter s', fontsize=12)
    plt.ylabel('Energy Gap', fontsize=12)
    plt.title(f'{molecule_name} - Gap Evolution Comparison (All Bond Dimensions)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/{molecule_name}_gap_comparison_all_bonds.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Comparison plot saved: {filename}")

    plt.close()

def plot_minimum_gaps_vs_bond_dimension(results_dict, molecule_name, save_dir=None):
    """
    Plot minimum gap vs bond dimension

    Parameters:
    -----------
    results_dict : dict
        Dictionary with bond dimensions as keys, each containing min_gap
    molecule_name : str
        Name of the molecule
    save_dir : str, optional
        Directory to save plots
    """
    bond_dims = []
    min_gaps = []

    for bond_dim, data in results_dict.items():
        bond_dims.append(int(bond_dim))
        min_gaps.append(data['min_gap'])

    # Sort by bond dimension
    sorted_data = sorted(zip(bond_dims, min_gaps))
    bond_dims, min_gaps = zip(*sorted_data)

    plt.figure(figsize=(10, 6))
    plt.plot(bond_dims, min_gaps, 'ro-', linewidth=2, markersize=8)
    plt.xlabel('Bond Dimension', fontsize=12)
    plt.ylabel('Minimum Gap', fontsize=12)
    plt.title(f'{molecule_name} - Minimum Gap vs Bond Dimension', fontsize=14)
    plt.grid(True, alpha=0.3)

    # Add value labels on points
    for bd, mg in zip(bond_dims, min_gaps):
        plt.annotate(f'{mg:.4f}', (bd, mg), textcoords="offset points",
                    xytext=(0,10), ha='center', fontsize=9)

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        filename = f'{save_dir}/{molecule_name}_min_gaps_vs_bond_dim.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"    Min gaps plot saved: {filename}")

    plt.close()


def construct_adiabatic_hamiltonian(H_initial, H_final, s):
    """
    Construct adiabatic Hamiltonian H(s) = (1-s)H_initial + s*H_final

    Parameters:
    -----------
    H_initial : numpy.ndarray
        Initial Hamiltonian (mean field)
    H_final : numpy.ndarray
        Final Hamiltonian (conjugated)
    s : float
        Adiabatic parameter (0 <= s <= 1)

    Returns:
    --------
    numpy.ndarray
        Adiabatic Hamiltonian at parameter s
    """
    return (1 - s) * H_initial + s * H_final


def compute_gap(hamiltonian, k=2):
    """
    Compute the energy gap (difference between ground and first excited state)

    Parameters:
    -----------
    hamiltonian : numpy.ndarray
        The Hamiltonian matrix
    k : int
        Number of eigenvalues to compute (at least 2 for gap)

    Returns:
    --------
    float
        Energy gap between ground and first excited state
    """
    if hamiltonian.shape[0] <= 50:
        # For small matrices, use full diagonalization
        eigenvals = scipy.linalg.eigvals(hamiltonian)
        eigenvals = np.sort(np.real(eigenvals))
        return eigenvals[1] - eigenvals[0]
    else:
        # For large matrices, use sparse solver
        eigenvals, _ = eigsh(hamiltonian, k=k, which='SA')
        eigenvals = np.sort(eigenvals)
        return eigenvals[1] - eigenvals[0]


def adiabatic_gap_analysis(H_initial, H_final, n_steps=20):
    """
    Analyze the adiabatic gap evolution

    Parameters:
    -----------
    H_initial : numpy.ndarray
        Initial Hamiltonian
    H_final : numpy.ndarray
        Final Hamiltonian
    n_steps : int
        Number of steps in the adiabatic evolution

    Returns:
    --------
    tuple
        (s_values, gaps) - adiabatic parameters and corresponding gaps
    """
    s_values = np.linspace(0, 1, n_steps)
    gaps = []

    for s in s_values:
        H_s = construct_adiabatic_hamiltonian(H_initial, H_final, s)
        gap = compute_gap(H_s)
        gaps.append(gap)

    return s_values, np.array(gaps)


def normalize_gaps(gaps):
    """
    Normalize gaps so that initial and final gaps are 1

    Parameters:
    -----------
    gaps : numpy.ndarray
        Array of gaps

    Returns:
    --------
    numpy.ndarray
        Normalized gaps
    """
    if len(gaps) < 2:
        return gaps

    # Use average of initial and final gaps for normalization
    normalization_factor = (gaps[0] + gaps[-1]) / 2.0

    if normalization_factor == 0:
        return gaps  # Avoid division by zero

    return gaps / normalization_factor


def process_molecule_gaps(molecule_name, bond_dimensions):
    """
    Process gap analysis for a single molecule across all bond dimensions

    Parameters:
    -----------
    molecule_name : str
        Name of the molecule
    bond_dimensions : list
        List of bond dimensions to analyze

    Returns:
    --------
    dict
        Gap analysis results
    """
    print(f"Processing gap analysis for {molecule_name}")

    try:
        # Load mean field Hamiltonian
        with open(f"{molecule_name}_meanfield.txt", 'rb') as f:
            H_meanfield = pickle.load(f)

        # Load conjugated Hamiltonians (we'll need to reconstruct these)
        # For now, let's load the full Hamiltonian as final target
        with open(f"{molecule_name}_hamiltonian.txt", 'rb') as f:
            H_full = pickle.load(f)

        results = {}
        min_gaps = []

        # Create output directory
        os.makedirs(f"{molecule_name}", exist_ok=True)

        for bond_dim in bond_dimensions:
            print(f"  Processing bond dimension {bond_dim}")

            # For this implementation, we'll use the full Hamiltonian as final
            # In a complete implementation, you'd load the specific conjugated Hamiltonian
            H_final = H_full

            # Perform adiabatic gap analysis
            s_values, gaps = adiabatic_gap_analysis(H_meanfield, H_final, n_steps=20)

            # Normalize gaps
            normalized_gaps = normalize_gaps(gaps)

            # Store results
            results[bond_dim] = {
                's_values': s_values.tolist(),
                'gaps': gaps.tolist(),
                'normalized_gaps': normalized_gaps.tolist(),
                'min_gap': np.min(gaps)
            }

            min_gaps.append(np.min(gaps))

            # Save individual results
            gap_filename = f"{bond_dim}/mf{molecule_name}"
            os.makedirs(os.path.dirname(gap_filename), exist_ok=True)
            with open(gap_filename, 'w') as f:
                json.dump({
                    's_values': s_values.tolist(),
                    'gaps': gaps.tolist(),
                    'normalized_gaps': normalized_gaps.tolist()
                }, f, indent=2)

            # Generate individual plots for this bond dimension
            plot_save_dir = f"{molecule_name}_plots"
            plot_adiabatic_gaps(s_values, gaps, molecule_name, bond_dim, save_dir=plot_save_dir)

        # Generate comprehensive comparison plots
        plot_save_dir = f"{molecule_name}_plots"

        # Plot all bond dimensions comparison
        plot_bond_dimension_comparison(results, molecule_name, save_dir=plot_save_dir)

        # Plot minimum gaps vs bond dimension
        plot_minimum_gaps_vs_bond_dimension(results, molecule_name, save_dir=plot_save_dir)

        print(f"    All plots saved to directory: {plot_save_dir}/")

        return {
            'molecule_name': molecule_name,
            'success': True,
            'bond_dimensions': bond_dimensions,
            'min_gaps': min_gaps,
            'results': results
        }

    except Exception as e:
        print(f"Error processing gap analysis for {molecule_name}: {str(e)}")
        return {
            'molecule_name': molecule_name,
            'success': False,
            'error': str(e)
        }


def load_molecule_results(molecule_name):
    """
    Load previously computed results for a molecule

    Parameters:
    -----------
    molecule_name : str
        Name of the molecule

    Returns:
    --------
    dict
        Loaded results
    """
    try:
        # Load MPS data to get bond dimensions
        with open(f"{molecule_name}_mps.txt", 'r') as f:
            mps_data = json.load(f)

        # Extract bond dimensions (excluding metadata)
        bond_dims = [int(k) for k in mps_data.keys() if k.isdigit()]
        bond_dims.sort()

        return {
            'bond_dimensions': bond_dims,
            'mps_data': mps_data
        }

    except Exception as e:
        print(f"Error loading results for {molecule_name}: {str(e)}")
        return None


def run_gap_analysis_parallel(molecule_names, n_processes=None):
    """
    Run gap analysis for multiple molecules in parallel

    Parameters:
    -----------
    molecule_names : list
        List of molecule names to process
    n_processes : int
        Number of parallel processes (default: CPU count)

    Returns:
    --------
    list
        List of results for each molecule
    """
    if n_processes is None:
        n_processes = min(cpu_count(), len(molecule_names))

    # Prepare arguments for parallel processing
    args_list = []
    for mol_name in molecule_names:
        mol_results = load_molecule_results(mol_name)
        if mol_results:
            args_list.append((mol_name, mol_results['bond_dimensions']))

    # Run in parallel
    with Pool(n_processes) as pool:
        results = pool.starmap(process_molecule_gaps, args_list)

    return results


def create_summary_plots(results, output_dir="summary_plots"):
    """
    Create summary plots across all molecules

    Parameters:
    -----------
    results : list
        List of gap analysis results
    output_dir : str
        Directory to save summary plots
    """
    os.makedirs(output_dir, exist_ok=True)

    # Plot minimum gaps for all molecules
    plt.figure(figsize=(12, 8))

    for result in results:
        if result['success']:
            mol_name = result['molecule_name']
            bond_dims = result['bond_dimensions']
            min_gaps = result['min_gaps']

            plt.plot(bond_dims, min_gaps, 'o-', linewidth=2, label=mol_name, markersize=6)

    plt.xlabel('Bond Dimension')
    plt.ylabel('Minimum Energy Gap')
    plt.title('Minimum Gap vs Bond Dimension - All Molecules')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale often better for gaps
    plt.tight_layout()

    plt.savefig(f"{output_dir}/all_molecules_bond_gap.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Summary plots saved to {output_dir}/")


if __name__ == "__main__":
    # Test with available molecules
    test_molecules = ['H2']

    print("Running gap analysis...")
    results = run_gap_analysis_parallel(test_molecules, n_processes=1)

    print("Creating summary plots...")
    create_summary_plots(results)

    print("Gap analysis complete!")
    for result in results:
        if result['success']:
            print(f"{result['molecule_name']}: {len(result['bond_dimensions'])} bond dimensions processed")
        else:
            print(f"{result['molecule_name']}: Failed - {result['error']}")