<img src="https://github.com/SNLComputation/Albany/wiki/images/albany5.png" width="300">

# Albany

Albany is an implicit, unstructured grid, finite element code for the solution and analysis of multiphysics
problems. The Albany repository on the GitHub site contains over 100 regression tests and examples
that demonstrate the code's capabilities on a wide variety of problems including fluid mechanics, ice-sheet flow and other applications.  In particular, Albany houses the land-ice component of the U.S. Department of Energy's [Energy Exascale Earth System Model (E3SM)](https://e3sm.org/) known as [MPAS-Albany Land Ice (MALI)](https://mpas-dev.github.io/land_ice/land_ice.html).  

The Albany web page is located at [http://sandialabs.github.io/Albany](http://sandialabs.github.io/Albany)

## Features

### Analysis of complex multiphysics problems

![von Karman Vortex Street](https://github.com/SNLComputation/Albany/wiki/images/vonKarman.png)

Illustration of a von Karman vortex street that forms around a heated tube bundle under certain conditions

### Software architecture

Albany heavily leverages the [Trilinos](https://trilinos.org) Framework, available at:

	git clone https://github.com/trilinos/Trilinos.git

Albany supports the solution of very large problems (those over 2.1 billion degrees of freedom) using MPI, and
also demonstrates the use of the [Kokkos](https://github.com/kokkos) hardware abstraction package to support 
generic manycore computing across a variety of platforms - MPI + [threads, OpenMP, Cuda, Intel MIC].

## Building Albany

Detailed build instructions for both Trilinos and Albany are maintained on the Albany wiki at [https://github.com/sandialabs/Albany/wiki/Building-Albany-and-supporting-tools](https://github.com/sandialabs/Albany/wiki/Building-Albany-and-supporting-tools).  We note that it is also possible to build Albany using spack; for details on how to do this, please see the [Building Albany using Spack](https://github.com/sandialabs/Albany/wiki/Building-Albany-using-SPACK) site.

We note also that there exists a supported Python interface to Albany, known as [PyAlbany](https://github.com/sandialabs/Albany/wiki/PyAlbany).  Please see the following [slides on PyAlbany](https://drive.google.com/file/d/1VQwHbnDeuuiOrwY_yMXfuVirdLhu5VZF/view) for more information.

## Nightly Build and Test Results

Ths nightly build results for the Trilinos and SCOREC libraries along with Albany and the status of the Albany regression tests are posted on the world-viewable Albany CDash site at [http://my.cdash.org/index.php?project=Albany](http://my.cdash.org/index.php?project=Albany), as well as an additional CDash site internal to Sandia National Laboratories.

The regression test suite is contained within the Albany repository in the directories:

	/tests

These tests are stand-alone and also serve as nice examples about how to describe various PDEs discretized in Albany.  They also serve as a template for developing new simulations.

Once Albany is built, the default test suite is executed by typing `ctest` within the build directory. Any individual test can be executed by
moving into its sub-directory, and executing `ctest` in that sub-directory. Many Albany tests run in parallel using up to 4 MPI ranks.

## Documentation

An [HTML user guide](http://sandialabs.github.io/Albany/user-guide/guide.html) can be found inside the Albany repository at:

	/doc/user-guide/guide.html

The LaTeX Developer's Guide is located at:

	/doc/developersGuide

Note that these documents are not maintained and may be out-of-date with respect to the current version of Albany.  For specific questions about using or developing Albany, please contact <a href="https://www.sandia.gov/-ikalash/staff/irina-tezaur/">Irina Tezaur</a> at ikalash@sandia.gov.




## Note on unsupported code

Prior version of Albany included additional capabilities not present in the current version of the code, such as Quantum Computer Aided Design (QCAD), Model Order Reduction (MOR), Advanced Topology Optimization (ATO), etc.  These capabilities are still available as [Albany tags](https://github.com/sandialabs/Albany/tags).  Each tag has documentation about the version of Trilinos that can be used to build the tag.  

The Laboratory for Computational Mechanics (LCM) capabilities within Albany have been moved to separate repositories: [Albany-LCM](https://github.com/sandialabs/LCM) and [Albany-SCOREC](https://github.com/scorec/Albany).  The latter repository contains capabilities for adaptive mesh refinement (AMR) using the [Parallel Unstructured Mesh Interface (PUMI)](https://scorec.rpi.edu/~seol/PUMI.pdf) library developed at the Rennselaer Polytechnic Institute (RPI).  While Albany-LCM is developed/maintained by a Sandia team, the status of the Albany-SCOREC team is unknown at the present time.
