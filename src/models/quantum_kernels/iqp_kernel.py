import torch
import pennylane as qml
from gpytorch.constraints import Interval
from torch import Tensor
from src.models.quantum_kernels.quantum_utils import iqp_feature_map
from gpytorch.kernels import Kernel
from builtins import ValueError


class IQPKernel(Kernel):
    # the sinc kernel is stationary
    is_stationary = False

    def __init__(
            self,
            num_features: int,
            shots: int = None,
            initial_alpha: float = 0.5,
            alpha_constraint=Interval(lower_bound=0, upper_bound=1),  # The recommended bandwidth interval = [0,1]
            mod: str = 'default.qubit.torch',
            torch_device: str = 'cpu',
            diff_method: str = "best",
            qq_inter: bool = False,
            provider: str = None,
            backend: str = None,
            **kwargs):
        """
        Initializes the IQP kernel class

        :param num_features: length of the subsequence (i.e, n)
        :param shots: number of circuit shots to run
        :param initial_alpha: The init value of alpha
        :param alpha_constraint: constraint on alpha
        :param mod: mod to be used on the quantum device, default is 'default.qubit'
               Fastest mod when using gradient descent hyperparameter tuning + a simulator:
               - <<default.qubit.torch + torch_device = 'cpu'>> --> for few number of qubits : because of the direct
                        gradients computations

               Fastest mod when using a hyperparameter tuning approach other than the gradient descent one:
               - <<lightning.gpu + torch_device = 'cpu'>> => significant speed increase (x2.6) compared to other mods :
                        the reason is that in this mod, we don't need to compute param-shift rule based gradients which
                        are costly.
        :param torch_device: device used by torch
        :param diff_method: differentiation method used for gradient computation when gradient optimization is enabled
        :param qq_inter: Only use sequential qubit-to-qubit interactions or not
        :param provider: The provider to use in the quantum device
        :param backend: The backend to use in the quantum device
        :param kwargs: other arguments to be passed to the super class
        """
        super().__init__(**kwargs)

        """
        Todo in Future: Allow multi-batch processing and allow setting parameters as
        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.ones(*self.batch_shape, 1, 1)))
        """

        self.register_parameter(name="raw_alpha", parameter=torch.nn.Parameter(torch.ones(1)))

        self.register_constraint("raw_alpha", alpha_constraint)

        self.alpha = initial_alpha

        if not shots and qq_inter:
            shots = num_features ** 3  # Set shots = w ** 3 when working with qq_inter type of model

        if mod == 'default.qubit.torch':
            quantum_device = qml.device(mod, wires=num_features, torch_device=torch_device, shots=shots)
        elif mod == 'lightning.gpu':
            quantum_device = qml.device(mod, wires=num_features, shots=shots)
        elif mod == 'default.qubit':
            if torch_device != 'cpu':
                raise ValueError(f"Quantum device of mod = `{mod}` should be run with torch_device = 'cpu'")
            quantum_device = qml.device(mod, wires=num_features, torch_device=torch_device, shots=shots)
        elif mod == 'qiskit.ibmq':  # to run on IBM's Quantum Hardware
            quantum_device = qml.device(mod, wires=num_features, torch_device='cpu', shots=shots,
                                        backend=backend, provider=provider)
        else:
            raise ValueError(f"Unhandled quantum device: `{mod}`")

        @qml.qnode(quantum_device, interface='torch', diff_method=diff_method)
        def circuit(x1, x2):
            """
            computes the fidelity overlap between two vectors x1 and x2

            :param x1: of shape (num_features)
            :param x2: of shape (num_features)
            :return: Measurement probabilities of shape (2^num_features,)
            """
            iqp_feature_map(x1, self.alpha, wires=quantum_device.wires, qq_inter=qq_inter)
            qml.adjoint(iqp_feature_map)(x2, self.alpha, wires=quantum_device.wires, qq_inter=qq_inter)
            return qml.probs(wires=quantum_device.wires)

        # .squeeze() is added before [0] to make both default and lightning mods be compatible as the qubit
        self.kernel = lambda x1, x2: circuit(x1, x2).squeeze()[0]

    @property
    def alpha(self):
        return self.raw_alpha_constraint.transform(self.raw_alpha)

    @alpha.setter
    def alpha(self, value):
        self._set_alpha(value)

    def _set_alpha(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_alpha)
        self.initialize(raw_alpha=self.raw_alpha_constraint.inverse_transform(value))

    # This is the kernel function
    def forward(self, x1: Tensor, x2: Tensor, assume_normalized_kernel: bool = True, **params):
        """
        Implements the gpytorch.kernels.Kernel forward method

        :param x1: of shape (num_points, num_features)
        :param x2: of shape (num_points, num_features)
        :param assume_normalized_kernel: (bool, optional). Assume that the kernel is normalized, in
            which case the diagonal of the kernel matrix is set to 1, avoiding unnecessary
            computations.
        :return: the piecewise kernel matrix
        """
        if x1.shape == x2.shape and torch.equal(x1, x2):  # For fast and robust computation of the square kernel matrix
            k = qml.kernels.square_kernel_matrix(x1, self.kernel, assume_normalized_kernel=True).to(self.device)
        else:
            k = qml.kernels.kernel_matrix(x1, x2, self.kernel).to(self.device)
        return k
