# PyAlbany

PyAlbany is a Python wrapper for Albany.

PyAlbany can be installed using spack as described in https://github.com/sandialabs/Albany/wiki/Building-Albany-using-Spack or by compiling Albany as described in this README.

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

In order to configure PyAlbany, both Albany has to be configured accordingly.

PyAlbany has been tested with Python 3 (tested with 3.7 and 3.8) but should work with Python 2.

To switch from one python version to the other requires to recompile both Trilinos and Albany.

##### Trilinos configuration
PyAlbany does not add any particular Trilinos requirement.

##### Albany configuration
The first required option is:
```
-D ENABLE_ALBANY_PYTHON=ON \
```
to enable the build of PyAlbany.

Finally, to select the python version, the easiest way is to specify the desired python executable during the configuration of Albany as illustrated as follows:
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
