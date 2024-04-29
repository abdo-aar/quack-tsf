import pennylane as qml
from torch import Tensor
from pennylane.wires import Wires


def u_z_operator(x: Tensor, alpha: Tensor, wires: Wires, qq_inter: bool = False):
    """
    Implements the U_z encoding operator for the IQP-like feature map. It encodes an input vector x into a quantum
    state using linear and quadratic terms scaled by a bandwidth parameter alpha. The encoding involves two steps:
    1. Linear Term Encoding: Applies RZ gates scaled by alpha for each input element, encoding individual features into
    the quantum state.
    2. Quadratic Term Encoding: Uses CNOT-RZ-CNOT sequences for each qubit pair to simulate ZZ
    interactions, scaled by alpha^2, encoding feature interactions into the quantum state.
    This method prepares the quantum state to reflect the input vector's individual and interacting components.

    :param x: Input matrix of shape (c, n), where c is the number of points, and n is the dimension
    :param alpha: Bandwidth parameter to scale the input elements.
    :param wires: Qubits to apply the encoding operations.
    :param qq_inter: Whether to apply qubit-to-qubit interactions only.
    :return: None. Encodes x through the U_z operator across the specified 'wires'.
    """
    n = len(wires)  # Ensure operations are applied based on the given wires

    # First term: alpha * sum(x_j * Z_j)
    for j in range(n):
        qml.RZ(alpha * x[j], wires=wires[j])

    # Second term: alpha^2 * sum(x_j * x_j' * Z_j Z_j')
    for j in range(n):
        for j_prime in range(j + 1, n):  # Optimized to avoid redundancy and only apply for j < j_prime
            if qq_inter and j_prime > j + 1:
                break   # Go to the next qubit if we are considering only sequential qubit to qubit interactions

            qml.CNOT(wires=[wires[j], wires[j_prime]])
            qml.RZ(alpha ** 2 * x[j] * x[j_prime], wires=wires[j_prime])
            qml.CNOT(wires=[wires[j], wires[j_prime]])


def iqp_feature_map(x: Tensor, alpha: Tensor, wires: Wires, qq_inter: bool = False):
    """
    IQP-like feature map.

    :param x: input matrix of shape (c, n), where c is the number of points, and n is the dimension
    :param alpha: bandwidth parameter
    :param wires: circuit wires
    :param qq_inter: Only use sequential qubit-to-qubit interactions or not
    :return: None. Encodes x into the quantum state across specified 'wires'.
    """
    for wire in wires:
        qml.Hadamard(wires=wire)

    u_z_operator(x, alpha, wires, qq_inter)

    for wire in wires:
        qml.Hadamard(wires=wire)

    u_z_operator(x, alpha, wires, qq_inter)
