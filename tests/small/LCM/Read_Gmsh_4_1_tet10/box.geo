// This script creates a unit cube.
// Run with:
// gmsh_bin box.geo -3 -order 2
//                  |  |
//                  |  - Create order 2 mesh
//                  - Create 3D mesh


// Set the geometry kernel
SetFactory("OpenCASCADE");

// Set mesh output style: 
// 1 for .msh, 16 for .vtk (for use with paraview)
Mesh.Format = 1;

// Characteristic length
l = 1.0;

// Create the unit cube
box_corner  = {0, 0, 0};
box_extents = {l, l, l};
box_v       = newv;
Box (box_v) = { box_corner[0],  box_corner[1],  box_corner[2], 
                box_extents[0], box_extents[1], box_extents[2]};

// Label surfaces to use with Albany boundary conditions
Physical Surface("max_x") = {2};
Physical Surface("min_x") = {1};
Physical Surface("min_z") = {5};
Physical Surface("max_z") = {6};
Physical Surface("min_y") = {3};
Physical Surface("max_y") = {4};


// Save all elements to file, not just labeled ones
Mesh.SaveAll=1;
