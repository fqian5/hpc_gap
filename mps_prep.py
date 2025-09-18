import numpy as np
from numpy.typing import NDArray
import quimb.tensor as qtn
from qiskit import QuantumCircuit
from qiskit.circuit.library import UnitaryGate

from qiskit.quantum_info import Operator
def gram_schmidt(matrix: NDArray[np.complex128]) -> NDArray[np.complex128]:
    """ Perform Gram-Schmidt orthogonalization on the columns of a matrix
    to define the unitary block to encode the MPS.

    Notes
    -----
    If a column is (approximately) zero, it is replaced with a random vector.

    Parameters
    ----------
    `matrix` : NDArray[np.complex128]
        Input matrix with complex entries.

    Returns
    -------
    `unitary` : NDArray[np.complex128]
        A unitary matrix with orthonormal columns derived from the input matrix.
        If a column is (approximately) zero, it is replaced with a random vector.
    """
    num_rows, num_columns = matrix.shape
    unitary = np.zeros((num_rows, num_columns), dtype=np.complex128)
    orthonormal_basis: list[NDArray[np.complex128]] = []

    for j in range(num_columns):
        column = matrix[:, j]
        column = column.astype(np.complex128) ## fix dytpe issues
        # If column is (approximately) zero, replace with random
        if np.allclose(column, 0):
            column = np.random.uniform(-1, 1, num_rows)
            if np.iscomplexobj(matrix):
                column = column + 1j * np.random.uniform(-1, 1, num_rows)

        # Gram-Schmidt orthogonalization
        for basis_vector in orthonormal_basis:
            column = column.astype(np.complex128) ## fix dytpe issues
            column -= (basis_vector.conj().T @ column) * basis_vector

        # Handle near-zero vectors (linear dependence)
        norm = np.linalg.norm(column)
        if norm < 1e-12:
            is_complex = np.iscomplexobj(matrix)
            column = np.random.uniform(-1, 1, num_rows)
            if is_complex:
                column += 1j * np.random.uniform(-1, 1, num_rows)
            for basis_vector in orthonormal_basis:
                column -= (basis_vector.conj().T @ column) * basis_vector

        unitary[:, j] = column / np.linalg.norm(column)
        orthonormal_basis.append(unitary[:, j])

    return unitary

def generate_layer(mps: qtn.MatrixProductState) -> list[tuple[list[int], NDArray[np.complex128]]]:
    """ Convert a Matrix Product State (MPS) to a circuit representation
    using a single unitary layer.

    Args:
        mps (qtn.MatrixProductState): The MPS to convert.

    Returns:
        unitary_layer (list[tuple[list[int], NDArray[np.complex128]]]): A list of
        tuples representing the unitary layer of the circuit.
            Each tuple contains:
            - A list of qubit indices (in LSB order) that the unitary acts on.
            - A unitary matrix (as a 2D NumPy array) that encodes the MPS.
    """
    num_sites = mps.L

    unitary_layer: list[tuple[list[int], NDArray[np.complex128]]] = []

    for i, tensor in enumerate(reversed(mps.arrays)):
        i = num_sites - i - 1

        # MPS representation uses 1D entanglement, thus we need to define
        # the range of the indices via the tensor shape
        # i.e., if q0 and q3 are entangled, then regardless of q1 and q2 being
        # entangled the entanglement range would be q0-q3
        if i == 0:
            d_right, d = tensor.shape
            tensor = tensor.reshape((1, d_right, d))
        if i == num_sites - 1:
            d_left, d = tensor.shape
            tensor = tensor.reshape((d_left, 1, d))

        tensor = np.swapaxes(tensor, 1, 2)

        # Combine the physical index and right-virtual index of the tensor to construct an isometry
        # matrix
        d_left, d, d_right = tensor.shape
        isometry = tensor.reshape((d * d_left, d_right))

        # Put qubit ordering in LSB (we provide this so that users can modify between LSB and MSB)
        # To put into MSB, comment the second line below
        qubits = reversed(range(i - int(np.ceil(np.log2(d_left))), i + 1))
        qubits = [abs(qubit - num_sites + 1) for qubit in qubits] # type: ignore

        # Create all-zero matrix and add the isometry columns
        matrix = np.zeros((isometry.shape[0], isometry.shape[0]), dtype=isometry.dtype)

        # Keep columns for which all ancillas are in the zero state
        matrix[:, : isometry.shape[1]] = isometry

        # Perform Gram-Schmidt orthogonalization to ensure the columns are orthonormal
        unitary = gram_schmidt(matrix)

        # Store the unitary layer for the circuit
        unitary_layer.append((qubits, unitary)) # type: ignore

    return unitary_layer



def mps_to_circuit_approx(
        mps: qtn.MatrixProductState,
        max_num_layers: int,
        chi_max: int,
        target_fidelity:np.float64
    ) -> QuantumCircuit:
    r""" Approximately encodes the MPS into a circuit via multiple layers
    of exact encoding of bond 2 truncated MPS.

    Notes
    -----
    Whilst we can encode the MPS exactly in a single layer, we require
    $log(chi) + 1$ qubits for each tensor, which results in larger circuits.
    This function uses bond 2 which allows us to encode the MPS using one and
    two qubit gates, which results in smaller circuits, and easier to run on
    hardware.

    This is the core idea of Ran's paper [1].

    [1] https://arxiv.org/abs/1908.07958

    Parameters
    ----------
    `statevector` : NDArray[np.complex128]
        The statevector to convert, which should be normalized.
    `max_num_layers` : int
        The maximum number of layers to use in the circuit.
    `chi_max` : int
        The maximum bond dimension for the MPS compression.

    Returns
    -------
    `circuit` : QuantumCircuit in Qiskit
    """
    mps.permute_arrays()

    mps.compress(form="left", max_bond=chi_max)
    mps.left_canonicalize(normalize=True)

    compressed_mps = mps.copy(deep=True)
    disentangled_mps = mps.copy(deep=True)
    num_qubits = mps.L
    circuit = QuantumCircuit(num_qubits)

    unitary_layers = []

    # Initialize the zero state |00...0> to serve as comparison
    # for the disentangled MPS
    zero_state = np.zeros((2**mps.L,), dtype=np.complex128)
    zero_state[0] = 1.0  # |00...0> state

    # Ran's approach uses a iterative disentanglement of the MPS
    # where each layer compresses the MPS to a maximum bond dimension of 2
    # and applies the inverse of the layer to disentangle the MPS
    # After a few layers we are adequately close to |00...0> state
    # after which we can simply reverse the layers (no inverse) and apply them
    # to the |00...0> state to obtain the MPS state
    for layer_index in range(max_num_layers):
        # Compress the MPS from the previous layer to a maximum bond dimension of 2,
        # |ψ_k> -> |ψ'_k>
        compressed_mps = disentangled_mps.copy(deep=True)

        # Normalize improves fidelity of the encoding
        compressed_mps.normalize()
        compressed_mps.compress(form="left", max_bond=2)

        # Generate unitary layer
        unitary_layer = generate_layer(compressed_mps)
        unitary_layers.append(unitary_layer)

        # To update the MPS definition, apply the inverse of U_k to disentangle |ψ_k>,
        # |ψ_(k+1)> = inv(U_k) @ |ψ_k>
        for i, _ in enumerate(unitary_layer):
            inverse = unitary_layer[-(i + 1)][1].conj().T

            if inverse.shape[0] == 4:
                disentangled_mps.gate_split_(
                    inverse, (i - 1, i)
                )
            else:
                disentangled_mps.gate_(
                    inverse, (i), contract=True
                )

        # Compress the disentangled MPS to a maximum bond dimension of chi_max
        # This is to ensure that the disentangled MPS does not grow too large
        # and improves the fidelity of the encoding
        disentangled_mps: qtn.MatrixProductState = qtn.tensor_1d_compress.tensor_network_1d_compress( # type: ignore
            disentangled_mps, max_bond=chi_max
        )

        fidelity = np.linalg.norm(np.vdot(disentangled_mps.to_dense(), zero_state))
        
        if fidelity >= target_fidelity:
            # If the disentangled MPS is close enough to the zero state,
            # we can stop the disentanglement process
            print(f"Reached target fidelity in {layer_index + 1} layers.")
            break

    if layer_index == max_num_layers - 1:
        print(f"Reached {fidelity} with the maximum number of layers {max_num_layers}.")

    # The layers disentangle the MPS to a state close to |00...0>
    # inv(U_k) ... inv(U_1) |ψ> = |00...0>
    # So, we have to reverse the layers and apply them to the |00...0> state
    # to obtain the MPS state
    # |ψ> = U_1 ... U_k |00...0>
    unitary_layers.reverse()

    # Apply the unitary layers to the |00...0> state
    gate_index = 0
    for unitary_layer in unitary_layers:
        for qubits, unitary in unitary_layer:

            gate = UnitaryGate(unitary)
            circuit.append(gate, qubits)
            gate_index += 1

    return circuit


def state_prep_circuit(mps: qtn.MatrixProductState,
        max_num_layers: int,
        chi_max: int,
        target_fidelity:np.float64):

    circuit = mps_to_circuit_approx(mps, max_num_layers, chi_max,target_fidelity)
    unitary = Operator(circuit).data
    return unitary