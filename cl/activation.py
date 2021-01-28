from .module import ConformalModule, ForwardMinkowskiData, ForwardTorchData
from .utils import DenseTensor, SizeAny
from abc import abstractmethod
from collections import OrderedDict
from collections.abc import Iterable
from typing import Optional, Tuple, Union
import math, numpy, torch
import os
import glob


class WrappedMinkowskiSRePro(torch.nn.Module):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedMinkowskiSRePro, self).__init__()
        self._owner = (owner,) # We use a tuple to avoid infinite recursion while PyTorch traverses the module's tree

    def forward(self, input: ForwardMinkowskiData) -> ForwardMinkowskiData:
        raise RuntimeError('Illegal call to the forward function. ConformalLayers was developed to evaluate the SRePro activation function differently in this module.')

    def to_tensor(self, alpha_upper: DenseTensor) -> Tuple[DenseTensor, DenseTensor]:
        # Get the alpha parameter
        alpha = alpha_upper if self.owner.alpha is None else torch.as_tensor(self.owner.alpha, dtype=alpha_upper.dtype, device=alpha_upper.device)
        # Compute the last coefficient of the matrix
        matrix_scalar = alpha / 2
        # Compute the coefficient on the main diagonal of the last slice of the tensor
        tensor_scalar = 1 / (2 * alpha)
        # Return the scalars of the tensor representation of the activation function
        return matrix_scalar, tensor_scalar

    @property
    def owner(self) -> ConformalModule:
        return self._owner[0]


class WrappedTorchSRePro(torch.nn.Module):
    def __init__(self,
                 owner: ConformalModule) -> None:
        super(WrappedTorchSRePro, self).__init__()
        self._owner = (owner,) # We use a tuple to avoid infinite recursion while PyTorch traverses the module's tree

    def forward(self, input: ForwardTorchData) -> ForwardTorchData:
        (input, input_extra), alpha_upper = input
        # Get the alpha parameter
        if self.owner.alpha is None:
            alpha = alpha_upper
        elif isinstance(self.owner.alpha, Iterable):
            alpha = torch.as_tensor(self.owner.alpha, dtype=input.dtype, device=input.device)
        else:
            alpha = torch.full_like(alpha_upper, self.owner.alpha)
        # Apply the activation function
        output = input
        in_channels = len(alpha)
        if input.shape[1] == in_channels:
            batches, *in_volume = input.shape[0], input.shape[2:]
        elif in_channels == 1:
            batches, *in_volume = input.shape
        else:
            raise NotImplementedError()
        flatted_input = input.view(batches, in_channels, -1, 1)
        input_sqr_norm = torch.linalg.norm(flatted_input, ord='fro', dim=(2, 3))
        output_extra = (input_sqr_norm + input_extra * (alpha * alpha)) / (2 * alpha)
        alpha_upper = torch.full_like(alpha_upper, 1)
        # Return the result

        # path = os.path.join('Experiments', 'Tensors')
        # idx = len(glob.glob(os.path.join(path, '*.pth')))
        # torch.save({
        #     'input' : input.detach().cpu(),
        #     'input_extra' : input_extra.detach().cpu(),
        #     'alpha' : alpha.detach().cpu(),
        #     'output_extra' : output_extra.detach().cpu(),
        # }, os.path.join(path, 'input_batch_{}_layer_{}.pth'.format(idx//3, idx%3)))

        return (output, output_extra), alpha_upper

    def to_tensor(self, alpha_upper: DenseTensor) -> Tuple[DenseTensor, DenseTensor]:
        raise RuntimeError('Illegal call to the forward function. ConformalLayers was developed to evaluate the SRePro activation function differently in this module.')

    @property
    def owner(self) -> ConformalModule:
        return self._owner[0]


class BaseActivation(ConformalModule):
    def __init__(self,
                 *, name: Optional[str]=None) -> None:
        super(BaseActivation, self).__init__(name=name)

    @abstractmethod
    def to_tensor(self, alpha_upper: DenseTensor) -> Tuple[Optional[DenseTensor], Optional[DenseTensor]]:
        pass


class NoActivation(BaseActivation):
    def __init__(self) -> None:
        super(NoActivation, self).__init__()
        self._identity_module = torch.nn.Identity()

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        return input

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (*in_dims,)

    def to_tensor(self, alpha_upper: DenseTensor) -> Tuple[None, None]:
        matrix_scalar = None
        tensor_scalar = None
        return matrix_scalar, tensor_scalar


class SRePro(BaseActivation):
    def __init__(self,
                 alpha: Optional[Union[float, Tuple[float, ...]]]=None,
                 *, name: Optional[str]=None) -> None:
        super(SRePro, self).__init__(name=name)
        self._alpha = alpha
        self._torch_module = WrappedTorchSRePro(self)
        self._minkowski_module = WrappedMinkowskiSRePro(self)

    def _repr_dict(self) -> OrderedDict:
        entries = super()._repr_dict()
        entries['alpha'] = 'Automatic' if self._alpha is None else self._alpha
        return entries

    def forward(self, input: Union[ForwardMinkowskiData, ForwardTorchData]) -> Union[ForwardMinkowskiData, ForwardTorchData]:
        native_module = self._torch_module if self.training else self._minkowski_module
        return native_module(input)

    def output_dims(self, *in_dims: int) -> SizeAny:
        return (*in_dims,)

    def to_tensor(self, alpha_upper: DenseTensor) -> Tuple[DenseTensor, DenseTensor]:
        native_module = self._torch_module if self.training else self._minkowski_module
        return native_module.to_tensor(alpha_upper)
    
    @property
    def alpha(self) -> Optional[Union[float, Tuple[float, ...]]]:
        return self._alpha
