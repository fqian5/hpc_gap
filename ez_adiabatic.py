import numpy as np
def single_operator_embedding(operators, n,location):
    op = np.array([1])
    for i in range(n):
        if i == location:
            op = np.kron(op, operators)
        else:
            op = np.kron(op, np.eye(2))
    return op # checked 

def sum_of_z(n):
    """
    Output the sum of  single -Z pauli operator on n qubits. make sure |0000>is gs
    The Z operator is defined as:
    Z = [[1, 0],
         [0, -1]]
    input:
    n: int, number of qubits
    output:
    sum_z: np.ndarray, shape (2**n, 2**n)
    """
    z = np.array([[1, 0], [0, -1]])
    sum_z = np.zeros((2**n, 2**n), dtype=np.complex128)
    for i in range(n):
        sum_z += single_operator_embedding(-z, n, i)
    return sum_z# checked 


def flatten_state(state):
    """
    Ensure the quantum state is a 1D array of shape (k,).
    Accepts input in shape (k,), (1, k), or (k, 1).
    """
    state = np.asarray(state)
    if state.ndim == 1:
        return state
    elif state.ndim == 2:
        if 1 in state.shape:
            return state.flatten()
        else:
            raise ValueError("Input state has shape (k, k) or higher. Expected a vector.")
    else:
        raise ValueError("Input state must be 1D or 2D with one singleton dimension.")

def dummy_state_prep(state):
    """
    Using householder matrix to construct a unitary that prepares a given state.
    input:
    state: np.ndarray, shape (2**n,), the target state to prepare
    output:
    unitary: np.ndarray, shape (2**n, 2**n), the unitary matrix that prepares the state
    """
    state = flatten_state(state)
    eye = np.eye(len(state), dtype=np.complex128)
    k = np.zeros(len(state), dtype=np.complex128)
    
    k[0] = 1
    dot_product = np.abs(k.conj().T@ state)

    if dot_product== 0:
        phase = 1
    else:
        phase = k.conj().T@ state/np.abs(k.conj().T@ state)

    w = phase*k - state
    w = w / np.linalg.norm(w)  # Normalize w
    u = eye - 2 * np.outer(w,w.conj()) # Householder reflection
    u = phase * u
    return u # problem
def random_state(n):
    """
    Generate a random quantum state of n qubits.
    input:
    n: int, number of qubits
    output:
    state: np.ndarray, shape (2**n,), the random quantum state
    """
    state = np.random.rand(2**n) + 1j * np.random.rand(2**n)
    state = state/np.linalg.norm(state)  # Normalize the state
    return state# checked 

    
import numpy as np

def complete_unitary(v, random_state=None):
    """
    Given a vector v of length n, returns an n×n unitary U whose
    first column equals v/||v||.

    Parameters
    ----------
    v : array_like, shape (n,)
        Input vector (can be complex). Does not need to be normalized.
    random_state : int or np.random.Generator, optional
        Seed or Generator for reproducible random columns.

    Returns
    -------
    U : ndarray, shape (n, n)
        A unitary matrix whose first column is v/||v||.
    """
    v = np.asarray(v, dtype=complex)
    n = v.shape[0]

    # Normalize the first column
    v = v / np.linalg.norm(v)

    # Build a full n×n matrix: first column v, others random
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    # random complex matrix for the remaining columns
    X = rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))
    X[:, 0] = v

    # QR decomposition: Q is unitary (up to phase conventions)
    Q, R = np.linalg.qr(X)

    # Fix global phases so that Q[:,0] exactly matches v
    # (QR may flip sign/phase on the first column if R[0,0] is negative/complex)
    phase = np.vdot(v, Q[:, 0]) / abs(np.vdot(v, Q[:, 0]))
    Q *= phase.conjugate()

    return Q
