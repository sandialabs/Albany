# PyAlbany

PyAlbany is a Python wrapper for Albany.

### Prerequisites

PyAlbany requires additional required dependencies:
* A python executable (either Python 3 recommended),
* Two python packages: numpy and mpi4py (Important: both should be installed such that they are linked with the MPI library used during the compilation of Albany):
```
env MPICC=$MY_MPI_COMPILER pip install --no-cache-dir numpy --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user
env MPICC=$MY_MPI_COMPILER pip install --no-cache-dir mpi4py --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user
```
* Pybind11:
```
pip install pybind11 --trusted-host=pypi.org --trusted-host=files.pythonhosted.org --user
```

### Configuration

In order to configure PyAlbany, both Trilinos and Albany have to be configured accordingly.

PyAlbany can be configured with either Python 2 (tested with 2.7) or Python 3 (tested with 3.6).
To switch from one python version to the other requires to recompile both Trilinos and Albany.

##### Trilinos configuration
PyAlbany does not add any particular Trilinos requirement.

##### Albany configuration
To select the python version, the easiest way is to specify the desired python executable during the configuration of Albany as illustrated as follows:
```
-D PYTHON_EXECUTABLE="/usr/bin/python3" \
```

### Testing

PyAlbany can be tested using the following command from the Albany build directory:
```
ctest -R PyAlbany
```

### Using PyAlbany

##### PYTHONPATH
In order to use PyAlbany, the `PYTHONPATH` environment variable should be set to allow Python to find the required Python packages.

It is recommended to install PyAlbany installing Albany:
```
make install
```
All the PyAlbany python files will be located to a python subfolder of the Albany lib folder.

The `PYTHONPATH` should then be updated as follows (for Python 3.8):
```
export PYTHONPATH=${Albany_LIB_DIRS}/python3.8/site-packages:${PYTHONPATH}
```
Where `Albany_LIB_DIRS` is the Albany installed library directory.

##### Examples
Examples on how to use PyAlbany can be found in the examples folder of this directory.
