import numpy as np
from pyscf import gto, scf
from openfermionpyscf import run_pyscf,generate_molecular_hamiltonian
from openfermion.linalg import get_sparse_operator
from openfermion import MolecularData, general_basis_change
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigsh
from quimb.tensor import MatrixProductState
from mps_prep import *
from ez_adiabatic import *
import numpy as np


spacing = 5
geometry = [['H', (i*spacing, 0.0, 0.0)] for i in range(7)]
charge=1
basis='sto3g'
multiplicity=1
molecule = MolecularData(
    geometry = geometry,
    charge = charge,
    basis = basis,
    multiplicity = multiplicity,
)

# geometry = [['H', (i*spacing, 0.0, 0.0)] for i in range(8)]
# charge=0
# basis='sto3g'
# multiplicity=1
# molecule = MolecularData(
#     geometry = geometry,
#     charge = charge,
#     basis = basis,
#     multiplicity = multiplicity,
# )
molecule = run_pyscf(molecule)
H = generate_molecular_hamiltonian(molecule.geometry, molecule.basis, multiplicity=molecule.multiplicity, charge=molecule.charge)
H_sparse = get_sparse_operator(H, n_qubits=molecule.n_qubits)
eigenvalues, eigenvectors = eigsh(H_sparse, k=5, which='SA')

gs = eigenvectors[:, 0]
gs_e = eigenvalues[0]



s_array = np.linspace(0,1,8)
min_diff_list = []
overlap_list = []
e_diff_list = []
mps = MatrixProductState.from_dense(gs)
mps.show()
max_bond = mps.max_bond()
max_bond_list = np.round(np.linspace(2, max_bond, 6)).astype(int).tolist()
gap_dict = {}
for mb in max_bond_list:
    mps = MatrixProductState.from_dense(gs)
    n_qubits = int(np.log2(len(gs)))
    
    state = mps.compress(max_bond=mb, cutoff=1e-10, method='svd')
    state = state.to_dense().flatten()
    householder = dummy_state_prep(state)
    H_z = sum_of_z(n_qubits)
    H_conj = householder.conj().T@H_z@householder
    overlap = np.linalg.norm(np.dot(gs,state))
    print("Overlap between Truncated State and True Ground State is ",overlap)
    overlap_list.append(overlap)
    eigenvalues_conj, eigenvectors_conj = np.linalge.eigh(H_conj)
    diff_array = []
    for s in s_array:
        H_s = (1-s)*H_conj/(eigenvalues_conj[1]-eigenvalues_conj[0]) + s*H_sparse/(eigenvalues[1]-eigenvalues[0])
        eigenvalues_s, eigenvectors_s = eigsh(H_s, k=5, which='SA')
        diff = eigenvalues_s[1] - eigenvalues_s[0]
        diff_array.append(diff)
    gap_dict[i] = diff_array
    min_diff_list.append(min(diff_array))


import matplotlib.pyplot as plt
x = np.linspace(1,len(max_bond_list),len(max_bond_list))
plt.plot(x,overlap_list,marker = 'v', label = 'Overlap')
plt.plot(x,min_diff_list,marker = 'o', label = 'Min Gap')
plt.title('Min Gap and Overlap vs max layer of entangler for H7 chain')
plt.legend()

plt.savefig("h7.png")  