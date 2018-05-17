# This directory contains various tools for reading and visualizing data in Paraview.

## Topic 1: Albany normally names the solution output field "solution\_" Paraview will apply displacements if this output field is called "displacement\_" To rename the field in Albany, please add the line to the Exodus block in the input yaml file: 

		Exodus Solution Name: displacement

  Also note that the tractions are called "residual\_" You can also rename these:

		Exodus Residual Name: tractions

  Note that you can manually apply displacements by using "Warp Vector" operating on the "solution\_" field.

## Topic 2: Visualize Maximum Cauchy Stress in LCM problems:

  1. Load up LCM dataset as usual.
  2. Apply any filters as usual (warp vector for displacements, etc.)
  3. Load the Filters->Alphabetical->MergeBlocks filter
  3. Go to Filters->Alphabetical->Programmable Filters
  4. Copy the contents of the file - MaxStressProgrammableFilter.py - into the "Script" window in the ProgrammableFilter window.
  5. Hit Apply.
  6. Select "Cauchy\_Stress\_Max" in view menu.

## Topic 3: Note that you can change this from an element field to a node field by applying the:

   Filters->Alphabetical->CellDataToPointData

  filter.

## Topic 4: View nodesets:

  1. Please see file Nodesets.txt

## Topic 5: Plot load vs. displacement

  Cameron Smith developed a nice [Paraview python utility](https://github.com/cwsmith/pvloadvsdisplacement) to plot 
  load vs. displacement curves on LCM problems.
