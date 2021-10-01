<a href="https://github.com/christinahedges/scatterbrain/workflows/tests.yml"><img src="https://github.com/christinahedges/scatterbrain/workflows/pytest/badge.svg" alt="Test status"/></a>

<a href="https://github.com/christinahedges/scatterbrain/workflows/flake8.yml"><img src="https://github.com/christinahedges/scatterbrain/workflows/flake8/badge.svg" alt="flake8 status"/></a>

# scatterbrain

`scatterbrain` is our GPU hack for processing TESS images, see [tess-backdrop](https://ssdatalab.github.io/tess-backdrop/) for our current tool.

# TODO

* Add documentation
* Importing `cupy` or `numpy` in the way we are currently is breaking flake8
* loading and saving
* add radial spline dm
* add cholesky batch solve

# MPI instructions

`scatterbrain` includes the option to use MPI to do parallel IO of FITS files. This feature is only available when computing new background models from TESS full frame images using the `run_backdrop_mpi.py` script. Note that only IO is done in parallel, other functionalities (e.g. BackDrop) run in a single CPU.

## Installation

#### OpenMPI
This package uses `mpi4py` library to interface Python code with MPI. You will need a working version of MPI, for example OpenMPI. Here are basic instructions of how to install OpenMPI on a OSX machine with [Home Brew](https://formulae.brew.sh/formula/open-mpi):

```
brew install openmpi
```
This will enable the `mpirun` and `mpiexec` commands. To test it, use the following `hello.c` example

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);
    int rank;
    int world;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world);
    printf("Hello: rank %d, world: %d\n",rank, world);
    MPI_Finalize();
}
```
Compile and the run the script with `mpirun`:

```sh
foo@bar:~$ mpicc -o hello ./hello.c
foo@bar:~$ mpiexec -n 2 ./hello
Hello: rank 0, world: 2
Hello: rank 1, world: 2
```

#### MPI for Python

Then install `mpi4py` following the [documentation](https://mpi4py.readthedocs.io/en/stable/install.html). Using PyPI:

```
pip install mpi4py
```
To test it:

```sh
foo@bar:~$ mpiexec -n 5 python -m mpi4py.bench helloworld
Hello, World! I am process 0 of 5 on localhost.
Hello, World! I am process 1 of 5 on localhost.
Hello, World! I am process 2 of 5 on localhost.
Hello, World! I am process 3 of 5 on localhost.
Hello, World! I am process 4 of 5 on localhost.
```

Now you are ready to run `scatterbrain` with MPI.

## Usage

To run one of the `run_backdrop*.py` scripts that fit background models to TESS FFIs using MPI use the following command on your local terminal:

```sh
foo@bar:~$ mpiexec -n 4 python run_backdrop_mpi.py [program arguments]
```
where the `-n 4` flag sets the number of ranks to be used.

# CUDA instructions

## Installation
`scatterbrain` can also run in GPU hardware for maximum speed up! For this, we use [CuPy](https://docs.cupy.dev/en/stable/) library that implements NumPy multi-dimensional arrays on CUDA.
In order to run `scatterbrain` in a GPU you will need:
  * NVIDIA GPU with CUDA
  * CUDA Toolkit according to your GPU hardware (e.g. v11.0)
  * CuPy version according to your CUDA Toolkit version

Follow [CuPy instructions](https://docs.cupy.dev/en/stable/install.html) to install it.
We recommend using the `pip` command, as example for CUDA v11.0:

```
pip install cupy-cuda110
```

## Usage

To import CuPy instead of NumPy you just need to set a new environmental variable `USE_CUPY` before importing `scatterbrain` modules so all of them can import the right packages. For default (also if `USE_CUPY == False`) `scatterbrain` imports only NumPy.

```python
import os

os.environ["USE_CUPY"] = str(True)
import scatterbrain
```

We opted for the use of an environmental variable, instead of just checking for GPU availability, in the weird case you want to run `scatterbrain` CPU-only on a GPU-available machine.

If using one of the `run_backdrop*.py` scripts that fit background models to TESS FFIs, you just have to add the `--cupy` flag to enable the GPU, this will automatically set the environmental variable.

To fit new FFI-background model you can combine the power of parallel IO and GPU computing by using the following command:

```sh
mpiexec -n 4 python run_backdrop_mpi.py --cupy [program arguments]
```
