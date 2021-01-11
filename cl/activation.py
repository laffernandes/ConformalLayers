from .module import ConformalModule
from .utils import ScalarTensor, SparseTensor, SizeAny
from abc import abstractmethod
from collections import OrderedDict
from typing import Optional, Tuple
import math, numpy, torch


class BaseActivation(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(None, name=name)

    @abstractmethod
    def to_tensor(self, previous: SparseTensor) -> Tuple[Optional[ScalarTensor], Optional[ScalarTensor]]:
        pass


class NoActivation(BaseActivation):
    def __init__(self) -> None:
        super(NoActivation, self).__init__()

    def to_tensor(self, previous: SparseTensor) -> Tuple[None, None]:
        matrix_scalar = None
        tensor_scalar = None
        return matrix_scalar, tensor_scalar


class SRePro(BaseActivation):
    def __init__(self,
                 alpha: Optional[float]=None,
                 *, name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name=name)
        self._alpha = alpha

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['alpha'] = 'Automatic' if self._alpha is None else self._alpha
        return entries

    def output_dims(self, *in_dims: int) -> SizeAny:
        return in_dims

    def to_tensor(self, previous: SparseTensor) -> Tuple[ScalarTensor, ScalarTensor]:
        # Compute the alpha parameter
        if self._alpha is None:
            min_dim = numpy.argmin(previous.shape)
            symmetric = torch.sparse.mm(previous, previous.t()) if min_dim == 0 else torch.sparse.mm(previous.t(), previous)
            alpha = torch.sqrt(math.sqrt(symmetric._nnz()) * symmetric._values().max())  # See https://mathoverflow.net/questions/111633/upper-bound-on-largest-eigenvalue-of-a-real-symmetric-nn-matrix-with-all-main-d
        else:
            alpha = torch.as_tensor(self.alpha, dtype=previous.dtype, device=previous.device)
        # Compute the last coefficient of the matrix
        matrix_scalar = alpha / 2
        # Compute the coefficient on the main diagonal of the last slice of the tensor
        tensor_scalar = 1 / (2 * alpha)
        # Return the scalars of the tensor representation of the activation function
        return matrix_scalar, tensor_scalar
    
    @property
    def alpha(self) -> Optional[float]:
        return self._alpha
