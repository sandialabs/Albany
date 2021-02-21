# PyAlbany

PyAlbany is a Python wrapper for Albany.

### Prerequisites

PyAlbany requires additional required dependencies:
* A python executable (either Python 2 or 3),
* Two python packages: numpy and mpi4py (Important: mpi4py should be installed such that mpi4py is linked with the MPI library used during the compilation of Trilinos),
* Swig 3.0.11 or higher (PyAlbany has been tested with swig 3.0.11 and 3.0.12).

### Configuration

In order to configure PyAlbany, both Trilinos and Albany have to be configured accordingly.

PyAlbany can be configured with either Python 2 (tested with 2.7) or Python 3 (tested with 3.6).
To switch from one python version to the other requires to recompile both Trilinos and Albany.

##### Trilinos configuration
PyAlbany adds one more Trilinos dependency to the already existing Albany dependencies: PyTrilinos.
PyTrilinos is a Python wrapper of some of the capabilities of Trilinos packages.

To enable PyAlbany, it is required to first build Trilinos with PyTrilinos enabled.
To do so, Trilinos must be configured with the following options:
```
-D Trilinos_ENABLE_PyTrilinos:BOOL=ON \
-D PyTrilinos_DOCSTRINGS:BOOL=ON \
```
The first option enables PyTrilinos and the second one enables PyTrilinos docstrings.

Moreover, setting the `MPI_BASE_DIR` is required to compile PyTrilinos for PyAlbany:
```
-D MPI_BASE_DIR:PATH=${MY_MPI_PATH} \
```

To select the python version, the easiest way is to specify the desired python executable during the configuration of Trilinos (and Albany) as illustrated as follows:
```
-D PYTHON_EXECUTABLE="/usr/bin/python3" \
```

Moreover, if swig cannot be found by cmake, the swig executable must be specified both for the configuration of Trilinos and Albany as illustrated as follows:
```
-D SWIG_EXECUTABLE="/usr/bin/swig" \
```
##### Albany configuration
The current version of PyAlbany relies on PyTrilinos source files and files generated during the building process of Trilinos which are not currently installed with the make command:
```
make install
```

To work around this, if PyAlbany is enabled, both the source (called `MY_TRILINOS_SOURCE_DIR` in the remaining of this readme) and build directory (called `MY_TRILINOS_BUILD_DIR` in the remaining of this readme) of Trilinos have to be specified.
This should be improved in the future by either installing those files while installing PyTrilinos or find another way to build against PyTrilinos.

Albany should be configured with the following extra options:
```
-D ENABLE_ALBANY_PYTHON:BOOL=ON \
-D TRILINOS_SOURCE_DIR=${MY_TRILINOS_SOURCE_DIR} \
-D TRILINOS_BUILD_DIR=${MY_TRILINOS_BUILD_DIR} \
```

CMake should be enabled to find the Python include path by itself. However, it has been observed that the found path does not necessarily include the required Python header file. To overcome this issue, it is possible to specify manually the Python include path as follows:
```
-D PYTHON_INCLUDE_PATH=${MY_PYTHON_DIR} \
```

As said in the previous section on the configuration of Trilinos, the following options should be used to specify the python executable and the swig executable:
```
-D PYTHON_EXECUTABLE="/usr/bin/python3" \
-D SWIG_EXECUTABLE="/usr/bin/swig" \
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

The `PYTHONPATH` should then be updated as follows (for Python 2.7):
```
export PYTHONPATH=${Trilinos_LIB_DIRS}/python2.7/site-packages:${Albany_LIB_DIRS}/python2.7/site-packages:${PYTHONPATH}
```
Where `Trilinos_LIB_DIRS` and `Albany_LIB_DIRS` are the Trilinos installed library directory and the Albany installed library directory respectively.

An alternative option is to set the `PYTHONPATH` as set in the PyAlbany tests:
```
export PYTHONPATH=${Trilinos_LIB_DIRS}/python2.7/site-packages:${ALBANY_BINARY_DIR}/PyAlbany/swig:${ALBANY_SOURCE_DIR}/PyAlbany:${PYTHONPATH}
```
Where `Trilinos_LIB_DIRS`, `ALBANY_BINARY_DIR`, and `ALBANY_SOURCE_DIR` are the Trilinos installed library directory, the Albany build directory, and the Albany source directory respectively.

##### Examples
Examples on how to use PyAlbany can be found in the examples folder of this directory.
