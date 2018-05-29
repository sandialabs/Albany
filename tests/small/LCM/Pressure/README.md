# Test Description: Analytic Pressures and Tractions

## Cantilever Beam - Force at end

* Test: input\_tetra4\_tip.yaml, input\_tetra10\_tip.yaml, and input\_hex8\_tip.yaml

* &delta; = F L<sup>3</sup> / (3 E I)

  F - traction at tip of beam (-0.1)  
  L - length of beam (beam is 10 long with a 1 x 1 cross section)  
  E - modulus of elasticity (4000.0)  
  I - moment of inertia (a<sup>4</sup> / 12 = 0.083333)  

  &delta; = -0.1 (analytic)

          = -0.0990 (using composite tets)

          = -0.0386 (using linear tets)

          = -0.0994 (using hex8)

## Cantilever Beam - Distributed vertical traction

* Test: input\_tetra4\_trac.yaml, input\_tetra10\_trac.yaml, and input\_hex8\_trac.yaml

* &delta; = q L<sup>4</sup> / (8 E I)

  F - distributed vertical traction across top of beam (-0.1)  
  L - length of beam (beam is 10 long with a 1 x 1 cross section)  
  E - modulus of elasticity (4000.0)  
  I - moment of inertia (a<sup>4</sup> / 12 = 0.083333)  

  &delta; = -0.3750 (analytic)

          = -0.3704 (using composite tets)

          = -0.1454 (using linear tets)

          = -0.3784 (using hex8)

## Cantilever Beam - Pressure along beam

* Test: input\_tetra4.yaml, input\_tetra10.yaml, and input\_hex8.yaml

* &delta; = q L<sup>4</sup> / (8 E I)

  q - pressure along beam (1.0 on top of beam - 0.1 per unit area)  
  L - length of beam (beam is 10 long with a 1 x 1 cross section)  
  E - modulus of elasticity (4000.0)  
  I - moment of inertia (a<sup>4</sup> / 12 = 0.083333)  

  &delta; = -0.3750 (analytic)

          = -0.3704 (using composite tets)

          = -0.1454 (using linear tets)

          = -0.0009 (using hex8)
