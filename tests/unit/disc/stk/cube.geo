// GMSH .geo file
Mesh.MshFileVersion = 4.1;

C = 10;
nC = 4;

lc = 1;

Point(1) = { 0, 0, 0, lc};
Point(2) = { C, 0, 0, lc};
Point(3) = { C, C, 0, lc};
Point(4) = { 0, C, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Transfinite Line {1,2,3,4} = nC+1 Using Progression 1;

Line Loop(11) = {1,2,3,4};
Plane Surface(12) = {11};
Transfinite Surface {12};
Recombine Surface {12};

Extrude {0, 0, C}
{
  Surface{12}; Layers{nC}; Recombine;
}

// physical entities

Physical Volume("Body_1") = {1};

Physical Surface("Bottom") = {12};
Physical Surface("Top") = {34};

Physical Curve("Bottom_boundary") = {1, 2, 3, 4};
