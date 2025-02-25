# ConformalLayers: A non-linear sequential neural network with associative layers

ConformalLayers is a conformal embedding of sequential layers of Convolutional Neural Networks (CNNs) that allows associativity between operations like convolution, average pooling, dropout, flattening, padding, dilation, grouping, and stride. Such embedding allows associativity between layers of CNNs, considerably reducing the number of operations to perform inference in neural networks.

This repository is a implementation of ConformalLayers written in Python using [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) and [PyTorch](https://pytorch.org/) as backend. This implementation is a first step into the usage of activation functions, like ReSPro, that can be represented as tensors, depending on the geometry model.

Please cite our [SIBGRAPI'21](http://arxiv.org/abs/2110.12108) paper if you use this code in your research. The paper presents a complete description of the library:

```txt
@InProceedings{sousa_et_al-sibgrapi-2021,
  author    = {Sousa, Eduardo V. and Fernandes, Leandro A. F. and Vasconcelos, Cristina N.},
  title     = {{C}onformal{L}ayers: a non-linear sequential neural network with associative layers},
  booktitle = {Proceedings of the 2021 34th SIBGRAPI Conference on Graphics, Patterns and Images},
  year      = {2021},
  pages     = {386--393},
  doi       = {https://doi.org/10.1109/SIBGRAPI54419.2021.00059}
}
```

Please, let [Eduardo Vera Sousa](http://www.ic.uff.br/~eduardovera), [Leandro A. F. Fernandes](http://www.ic.uff.br/~laffernandes) and [Cristina Nader Vasconcelos](http://www.ic.uff.br/~crisnv/index.php) know if you want to contribute to this project. Also, do not hesitate to contact them if you encounter any problems.

**Contents:**

- [ConformalLayers: A non-linear sequential neural network with associative layers](#conformallayers-a-non-linear-sequential-neural-network-with-associative-layers)
  - [Requirements](#requirements)
  - [Installation](#installation)
    - [Python System](#python-system)
    - [Docker](#docker)
  - [Running Examples](#running-examples)
  - [Running Unit Tests](#running-unit-tests)
  - [Documentation](#documentation)
    - [Modules](#modules)
      - [<u>Activation</u>](#uactivationu)
      - [<u>Convolution</u>](#uconvolutionu)
      - [<u>Pooling</u>](#upoolingu)
      - [<u>Regularization</u>](#uregularizationu)
      - [<u>Utility</u>](#uutilityu)
  - [License](#license)

## Requirements

Make sure that you have the following tools and Python modules before attempting to use ConformalLayers:

- [Python 3.8+](https://www.python.org) interpreter
- [PyTorch](https://pytorch.org/) (version >= 1.10.1) with CUDA
- [Minkowski Engine](https://github.com/NVIDIA/MinkowskiEngine) (version >= 0.5.4)

The following tool is optional:

- [Docker](https://www.docker.com/)

The following Python modules are optional, but necessary for running the experiments:

- [CodeCarbon](https://codecarbon.io/) (version >= 1.2.0)
- [PyTorch Lightning](https://www.pytorchlightning.ai/) (version >= 1.5.9)
- [scikit-learn](https://scikit-learn.org/)
- [SciPy](https://scipy.org/) (version >= 1.8.0)
- [TorchMetrics](https://torchmetrics.readthedocs.io/) (version >= 0.7.0)
- [tqdm](https://tqdm.github.io/) (version >= 4.61.2)
- [tzlocal](https://pypi.org/project/tzlocal/) (version == 2.1)
- [Weights & Biases](https://wandb.ai/) (version >= 0.12.9)

The complete set of required Python modules will be installed automatically by following the instructions presented below.

## Installation

### Python System

No magic needed here. Just run:

```bash
git clone https://github.com/Prograf-UFF/ConformalLayers
cd ConformalLayers
python setup.py install
```

But the Minkowski Engine may need some special libraries like MKL or OpenBLAS. See its [Quick Start Tutorial](https://nvidia.github.io/MinkowskiEngine/quick_start.html) for details. Also, make sure that Minkowski Engine is not using the `CPU_ONLY` build set! It happens when you have installed the CPU version of PyTorch. So, first you have to install the [CUDA Toolkit/SDK](https://developer.nvidia.com/cuda-zone) and [PyTorch with CUDA](https://pytorch.org/get-started/locally) according to your computer platform.

### Docker

Just run:

```bash
git clone https://github.com/Prograf-UFF/ConformalLayers
cd ConformalLayers
docker build -t clayers .
```

Once the docker is built, check it loads ConformalLayers correctly:

```bash
docker run clayers python3 -c "import cl; print(cl.__version__)"
```

and run it (detached) with GPUs:

```bash
docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it -d clayers
```

## Running Experiments

First, you have to go to the folder `<ConformalLayers-dir>/experiments`. Run all experiments using the following commands:

```bash
python run_all_bechmarks.py --wandb_entity NAME
python run_all_sweeps.py --wandb_entity NAME
```

Results will be available after summarization:

```bash
python summarize_bechmarks.py --wandb_entity NAME
python summarize_sweeps.py --wandb_entity NAME
```

You have to set `NAME` to the name of your entity in [Weights & Biases](https://wandb.ai/).

Individual benchmarks and sweeps can be run using the scripts `benchmark.py` and `sweep.py`. Use `--help` with these scripts to get help on the available arguments.

The files in the `<ConformalLayers-dir>/experiments/stuff/networks` folder contains the description of each architecture used in our experiments and present the usage of the classes and methods of our library.

## Running Unit Tests

First, you have to go to the folder `<ConformalLayers-dir>/tests`. Run all tests using the following command:

```bash
python run_all_tests.py
```

To run the tests for each module, run:

```bash
python test-<module_name>.py
```

## Documentation

Here you find a brief description of the classes available for the user. The detailed documentation is not ready yet.

Contents:

- [Modules](#modules)
  - [Convolution](#convolution)
  - [Pooling](#pooling)
  - [Activation](#activation)
  - [Regularization](#regularization)

### Modules

Here we present the main modules implemented in our framework. They can be found inside `cl`. Most of the modules are used just like in PyTorch, so users with some background on this framework benefits from this implementation. For users not familiar with PyTorch, the usage still quite simple and intuitive.  

| Module | Description |
| --- | --- |
| `cl.ConformalLayers` | This class is equivalent to the `nn.Sequential` module from PyTorch |
| `cl.Conv1d`, `cl.Conv2d`, `cl.Conv3d` | Convolution operation implemented for *n*-D signals |
| `cl.AvgPool1d`, `cl.AvgPool2d`, `cl.AvgPool3d` | Average pooling operation implemented for *n*-D signals |
| `cl.BaseActivation` | The abstract class for the activation function layer. To extend the library, one shall implement this class |
| `cl.Dropout` | In this version, `cl.Dropout` is only regularization available. In this approach, during the training phase, we randomly shut down some neurons with a probability `p`, passed as argument to this module |
| `cl.Flatten` | Flattens a contiguous range of dims into a tensor. |
| `cl.ReSPro` | The layer that corresponds to the `ReSPro` activation function. Such function is a linear function with non-linear behavior that can be encoded as a tensor. The non-linearity of this function is controlled by a parameter <span>&alpha;</span> (passed as argument) that can be provided or inferred from the data |
<br>

To define a sequential network, you need to queue the layers in an instance of `cl.ConformalLayers`. This class in a very similar to the `nn.Sequential` module from PyTorch and plays an important role in this task, as you can see by comparing the code snippets below:

```python
# This one is built with pure PyTorch
import torch.nn as nn

class D3ModNet(nn.Module):
    
    def __init__(self) -> None:
        super(D3ModNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

```python
# This one is built with ConformalLayers
import cl
import torch.nn as nn

class D3ModNetCL(nn.Module):

    def __init__(self) -> None:
        super(D3ModNetCL, self).__init__()
        self.features = cl.ConformalLayers(
            cl.Conv2d(3, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(32, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
            cl.Conv2d(32, 32, kernel_size=3),
            cl.ReSPro(),
            cl.AvgPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(128, 10)

    def forward(self, images: ImageBatch) -> Logits:
        x = self.features(images)
        x = torch.flatten(x, 1)
        return self.classifier(x)
```

They look pretty much the same code, right? That's because we've implemented `cl.ConformalLayers` to be a transition smoothest as possible to the PyTorch user. Most of the modules has almost the same method signatures of the ones provided by PyTorch.

#### <u>Activation</u>

Our activation module has `cl.ReSPro` activation function implemented natively. By using <u>Re</u>flections, <u>S</u>calings and <u>Pro</u>jections on an hypersphere in higher dimensions, we created a non-linear differentiable associative activation function that can be represented in tensor form. It has only one parameter, that controls how close to linear or non-linear is the curve. More details are available on the main paper.

- `alpha` (`float`, optional) - controls the non-linearity of the curve. If it is not provided, it's automatically estimated.

#### <u>Convolution</u>

The convolution operation implemented in ConformalLayers on the modules `cl.Conv1d`, `cl.Conv2d` and `cl.Conv3d` is almost the same one implemented on PyTorch but we do not allow bias. This is mostly due to the construction of our logic when building the representation with tensors. Although we have a few ideas on how to include bias on this representation, they are not included in the current version. The parameters are detailed below and are originally available in [PyTorch convolution documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html). The exception here relies on the `padding_mode` parameter, that is always set to '`zeros`' in our implementation.

- `in_channels` (`int`) – Number of channels in the input image.
- `out_channels` (`int`) – Number of channels produced by the convolution.
- `kernel_size` (`int` or `tuple`) – Size of the convolving kernel.
- `stride` (`int` or `tuple`, optional) – Stride of the convolution. Default: 1.
- `padding` (`int`, `tuple` or `str`, optional) – Padding added to both sides of the input. Default: 0.
- ` ` (`int` or `tuple`, optional) – Spacing between kernel elements. Default: 1.
- `groups` (`int`, optional) – Number of blocked connections from input channels to output channels. Default: 1.

#### <u>Pooling</u>

In our current implementation, we only support average pooling, which is implemented on modules `cl.AvgPool1d`, `cl.AvgPool2d` and `cl.AvgPool3d`. The parameters list, originally available in [PyTorch average pooling documentation page](https://pytorch.org/docs/stable/generated/torch.nn.AvgPool1d.html), is described below:

- `kernel_size` – the size of the window.
- `stride` – the stride of the window. Default value is `kernel_size`.
- `padding` – implicit zero padding to be added on both sides.
- `ceil_mode` – when `True`, will use ceil instead of floor to compute the output shape.
- `count_include_pad` – when `True`, will include the zero-padding in the averaging calculation.

#### <u>Regularization</u>

On regularization module we have `cl.Dropout` implemented in this version. It is based on the idea of randomly shutting down some neurons in order to prevent overfitting. It takes only two parameters, listed below. This list was originally available in [PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html).  

- `p` – probability of an element to be zeroed. Default: 0.5.
- `inplace` – If set to True, will do this operation in-place. Default: False.

#### <u>Utility</u>

The `cl.Flatten` class behaves like `nn.Flatten(start_dim=1, end_dim=-1)`. See [PyTorch documentation page](https://pytorch.org/docs/stable/generated/torch.nn.Flatten.html).

## License

This software is licensed under the GNU General Public License v3.0. See the [`LICENSE`](LICENSE) file for details.
