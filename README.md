# Albany

Albany is an implicit, unstructured grid, finite element code for the solution and analysis of partial differential equations.

## Features

### Analysis of complex multiphysics problems

![von Karman Vortex Street](https://github.com/gahansen/Albany/wiki/images/vonKarman.png)

Illustration of a von Karman vortex street that forms around a heated tube bundle under certain conditions

### Repo branches

The Git repository contains two branches, *master* and *tpetra*. The master branch contains the stable version of Albany that is 
based on Epetra distributed solver data structures.

The *tpetra* branch is under development to replace the Epetra architecture with
the Tpetra and Thyra Trilinos packages. With Tpetra, the ordinal size of the
distributed data structures are a template parameter which direcly supports the
fundamental word size of the target computing platform. This enables the calculation of
very large problems (over 2.1 billion degrees of freedom). Additionally, Tpetra internally
uses the Kokkos multicore package to support generic multithreaded multicore computing across
a wide variety of platforms.

### Software architecture

Albany heavily leverages the Trilinos Framework, available at:

	git clone https://software.sandia.gov/trilinos/repositories/publicTrilinos

and optionally depends on the SCOREC Parallel Unstructured Mesh Infrastructure 
[http://www.scorec.rpi.edu/pumi](http://www.scorec.rpi.edu/pumi)

## Building Albany

Detailed build instructions for both Trilinos and Albany are maintained at 
[http://redmine.scorec.rpi.edu/projects/albany-rpi/wiki](http://redmine.scorec.rpi.edu/projects/albany-rpi/wiki)

## Nightly Build and Test Results

Ths nightly build results for the Trilinos and SCOREC libraries and the *master* and *tpetra* branches
of Albany are posted on the Albany CDash site. Further, the results of the regression test suite of both
branches on an SMP server are presented on the CDash site at 
[http://my.cdash.org/index.php?project=Albany](http://my.cdash.org/index.php?project=Albany)

The regression test suite is contained within the Albany repository in the directory:

	/examples

These tests are stand-alone and also serve as nice examples about how to describe various multiphysics problems.
They also serve as a template for developing new simulations.

## Documentation

The HTML user guide is maintained inside the Albany repository at:

	/doc/user-guide/guide.html

The LaTeX Developer's Guide is located at:

	/doc/developersGuide
