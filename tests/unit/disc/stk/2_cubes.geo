// GMSH .geo file
Mesh.MshFileVersion = 4.1;

C = 10;
nC = 4;

lc = 1;

Point(1) = {   0, 0, 0, lc};
Point(2) = {   C, 0, 0, lc};
Point(3) = { 2*C, 0, 0, lc};
Point(4) = {   0, C, 0, lc};
Point(5) = {   C, C, 0, lc};
Point(6) = { 2*C, C, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {1, 4};
Line(4) = {2, 5};
Line(5) = {3, 6};
Line(6) = {4, 5};
Line(7) = {5, 6};

Transfinite Line {1,3,4,5,6} = nC+1 Using Progression 1;
Transfinite Line {2,7} = 2*nC+1 Using Progression 1;

Line Loop(11) = {1,4,-6,-3};
Plane Surface(12) = {11};
Transfinite Surface {12};
Recombine Surface {12};

Line Loop(13) = {2,5,-7,-4};
Plane Surface(14) = {13};
Transfinite Surface {14};
Recombine Surface {14};

Extrude {0, 0, C}
{
  Surface{12,14}; Layers{nC}; Recombine;
}

// physical entities

Physical Volume("Body_1") = {1};
Physical Volume("Body_2") = {2};

Physical Surface("Bottom") = {12, 14};
Physical Surface("Top") = {36, 58};
Physical Surface("Intersection") = {27};

Physical Curve("Bottom_boundary") = {1,2,3,5,6,7};