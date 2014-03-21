# Albany

Albany is an implicit, unstructured grid, finite element code for the solution and analysis of partial differential equations.

## Features

### Analysis of complex multiphysics problems

![von Karman Vortex Street](https://github.com/gahansen/Albany/wiki/images/vonKarman.png)

Illustration of a von Karman vortex street that forms around a heated tube bundle under certain conditions

### Software architecture

Albany heavily leverages the Trilinos Framework, available at:

	git clone https://software.sandia.gov/trilinos/repositories/publicTrilinos

and optionally depends on the SCOREC Parallel Unstructured Mesh Infrastructure:

	http://www.scorec.rpi.edu/pumi/

## Building Albany

Detailed build instructions for both Trilinos and Albany are maintained at:

	http://redmine.scorec.rpi.edu/projects/albany-rpi/wiki

## Documentation

The HTML user guide is maintained in this repository at:

	~Albany/doc/user-guide/guide.html

The LaTeX Developer's Guide is located at:

	~Albany/doc/developersGuide
