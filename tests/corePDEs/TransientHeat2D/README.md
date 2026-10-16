# Transient Heat 2D tests

`tempus_be_nox_solver.yaml` is a forward Backward Euler solve. The `transient_opt_*.yaml`
inputs test the transient optimization capability (`AlbanyAnalysis`, ROL), with problems
whose optima are known, for the two ways Piro computes the gradient:

- ROL driver (`Tempus Driver: false`): ROL integrates the problem with Backward Euler and
  computes the exact gradient of the discrete objective (discrete adjoint);
- Tempus driver (`Tempus Driver: true`): Tempus integrates the problem and computes the
  gradient with its adjoint sensitivities (continuous adjoint, first order accurate in time);

and the three response types (`Response Type`): `Final Time` (FT), `Time Integral` (TI,
trapezoidal rule) and `Sum Over Times` (SOT, times 0.25, 0.5, 0.75).

## Problems

All the problems are on the unit square, 5x5 Q1 mesh (STK2D), conductivity 1, t in (0, 1],
dt = 0.05, and use the response g = 100 * int (T - T*)^2 (`Squared L2 Difference Source ST
Target MST`), where the target T* is given by the `Target Temperature` problem sublist
(`Type: Linear Space-Time`, T* = c0 + cx x + cy y + ct t).

- **A (flux and source controls)**: dT/dt - Laplacian(T) = s, dT/dn = q_L on x = 0 (`SideSet0`),
  dT/dn = q_R on x = 1 (`SideSet1`), dT/dn = 0 on y = 0, 1, T(0) = 1 + x, controls
  (q_L, q_R, s) in [-4, 4]^3, initial guess 0, T* = 1 + x + 0.5 t. T* is the solution for
  (q_L, q_R, s) = (-1, 1, 0.5), and it is in the finite element space and linear in time, so
  this is the exact optimum of the discrete problem, with J = 0, for every response type and
  both drivers.
- **B (distributed Dirichlet control)**: no source, T = d on y = 0 (weak Dirichlet condition on
  `NodeSet2`, d = `dirichlet_field` in [0, 5], initial guess 3), dT/dn = 1 on y = 1 (`SideSet3`),
  dT/dn = 0 on x = 0, 1, T(0) = T* = 1 + y. Exact optimum d = 1 (|d|_2 = sqrt(6)), J = 0.
- **BR (regularized Dirichlet control)**: B with the objective of g + 1/2 |d|^2 (`Sum Of
  Responses` with a `Quadratic Linear Operator Based` response, A = D = identity, given by
  `transient_opt_identity_6x6.mm`). The misfit at the optimum is not zero. Optimum (exact
  minimizer of the discrete quadratic objective), symmetric about x = 1/2:
  - FT: d = (0.740743095089, 1.062996605588, 0.984812717012, ...), |d|_2 = 2.301529145591
  - SOT: d = (0.728008534163, 1.044725499161, 0.967891380947, ...), |d|_2 = 2.261973262319
- **C (active bound)**: A with the upper bound of s lowered to 0.3. TI optimum (exact minimizer
  of the discrete quadratic objective): (-0.900388147325, 1.099611852675, 0.3).

The Dirichlet control is on the bottom edge because the `Quadratic Linear Operator Based` response
reads the matrices with the map of the parameter, which must be contiguous: the nodes of
`NodeSet2` are 0, ..., 5.

## Dirichlet conditions and the Tempus adjoint

With weak Dirichlet conditions (rows f_i = x_i - d_i) the mass matrix df/dx_dot is singular,
and the Tempus adjoint sensitivities need special treatment for the jumps of the adjoint of the
FT and SOT responses. When the Tempus adjoint is used, Albany puts `Perturb Dirichlet` (1 if not
set) on the diagonal of these rows of the mass matrix and gives the vector of these values to
Tempus (`Sensitivities`->`Algebraic Equations Mass Diagonal`), which corrects the gradient.
The Tempus-driven BR tests fail without this correction (|d|_2 = 2.2823 for FT, 2.2366 for SOT).

## Tests and tolerances

| Problem | ROL driver | Tempus driver | Check |
|---|---|---|---|
| A  | FT, TI, SOT | FT, TI, SOT | parameters, 1e-6 |
| B  | TI, SOT     | FT, TI, SOT | two-norm and uniform value of d, 1e-6 |
| BR | FT, SOT     | FT, SOT     | two-norm of d, 1e-6 (ROL), 5e-3 (Tempus) |
| C  | TI          | (TI, not registered) | parameters, 1e-6 (1e-4 for Tempus) |

The Tempus gradient is not the exact gradient of the discrete objective, but it vanishes where
the misfit vanishes, so for A and B both drivers have the exact optimum. For BR and C it is
inconsistent with the objective at O(dt), hence the looser tolerances; the Tempus-driven C test
is not registered until it is confirmed that the trust region converges with that gradient.

`performGradientChecks.py` (PyAlbany) runs the gradient checks of all these inputs, before and
after the optimization.
