Dec 15, 2019

These are test cases for the TDM (3D manufacturing) project.

TDM project is a thermal-based analysis, consists of additive manufacturing (Additive3D) and subtractive manufacturing (Subtractive3D), where the physical model and its implementation is described in the Overleaf project:

https://www.overleaf.com/project/5bb3c1d0b60841103c73a50d

Both test cases include a serial simulation and a parallel simulation, with the same physics, material properties and other input variable. For subtractive 3D cases, it might take couple of time steps for the laser source to be calculated.

Additive3D - Bare plate: the bare plate simulation calculates the temperature field over only a solid subtrate, where melt pool size is of interest. This is the base case for the TDM project.

Additive3D - Single layer: the single layer simulation calculates the temperature field in a more practical set up, where material powder is placed over its substrate. This case is testing the incoming laser distribution for the powder region. The relative calcuation is in TDM/evaluators/Laser_Source_Def.hpp

Subtractive 3D - First pulse: this simulation simulates a pulsed laser running on a solid subtrate, but with a much higher fluence. After the first pulse, material will be removed, where those elements will be set to have conductivity of 0, with no further energy going out.

Subtractive 3D - Depth: this case follows the first pulse simulation, which accounts for the following incoming pulse, putting energy onto a grooved solid subtrate. By applying level set method onto the calculated result from first pulse simulation, a grooved geometry could be acquired. This case currently depends on the restart functionality, and mainly tests the calculation of Depth state variable. 
    
