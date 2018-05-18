# Test Description: Analytic Pressures and Tractions

## Cantilever Beam - Force at end

* Test: input\_comp.yaml

* deflection = F L^3 / (3 E I)

  F - traction at tip of beam (-0.1)
  L - length of beam (beam is 10 long with a 1 x 1 cross section)
  E - modulus of elasticity (4000.0)
  I - moment of inertia (a^4 / 12 = 0.083333)

  deflection = 0.1

## Cantilever Beam - Pressure along beam

* Test: input\_tetra4.yaml and input\_tetra10.yaml

* deflection = q L^4 / (8 E I)

  q - pressure along beam (1.0 on top of beam - 0.1 per unit area)
  L - length of beam (beam is 10 long with a 1 x 1 cross section)
  E - modulus of elasticity (4000.0)
  I - moment of inertia (a^4 / 12 = 0.083333)

  deflection = -0.375
