//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Unit tests of the MPAS-Albany interface (src/landIce/interfaceWithMPAS/Interface.cpp).
//
// The interface is called the way MALI calls it, through the mock in MockMPAS.hpp. Most tests solve
// problems whose discrete solution is known exactly, so they need no baseline:
//  - an ice slab of uniform thickness on an inclined plane, with no slip at the bed and the exact
//    solution prescribed on the lateral boundary (see SlabGeometry for the solution). The solution only
//    depends on the depth. With Glen's law exponent n=1 and a viscosity constant in each layer, the FO
//    solution with wedges linear along the vertical is exact at the nodes; with n=3, the
//    depth-integrated (MOLHO) solution is exact, since the exact profile 1-(1-sigma)^4 belongs to its
//    vertical basis;
//  - quantities that do not depend on the solve: beta = mu*N for a linear power law, the body force
//    rho*g*grad(s), the effective pressure passed in.
// Since the fields vary in space (and with the layer), these tests also check that the interface maps
// the MPAS data to the right nodes/elements and back, for different partitions, with shuffled local
// orderings, non-contiguous global ids and ranks with no triangles.
//
// The tests write albany_input.yaml (the file the interface reads) in the working directory.
//
// The interface can use two meshes, selected by the Discretization Method in albany_input.yaml: the
// logically extruded mesh ("Extruded", the default) and the 3D STK mesh ("STKExtruded"). The option
// --disc-method sets the Method the tests write (by default, none: the interface default). The tests
// that compare the two meshes, or check what only one of them does, set the Method themselves, and
// run only in the run with the default Method.

#include "MockMPAS.hpp"

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_UnitTestRepository.hpp>
#include <Teuchos_CommandLineProcessor.hpp>
#include <Teuchos_Array.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <vector>

// Parameter list built by velocity_solver_extrude_3d_grid (a global variable in Interface.cpp)
extern Teuchos::RCP<Teuchos::ParameterList> paramList;

namespace {

using namespace mock_mpas;

const MPI_Comm comm = MPI_COMM_WORLD;

// Discretization Method written in albany_input.yaml (empty: none, i.e., the interface default)
std::string discMethod;

TEUCHOS_STATIC_SETUP()
{
  Teuchos::UnitTestRepository::getCLP().setOption("disc-method", &discMethod,
      "Discretization Method written in albany_input.yaml: Extruded or STKExtruded (default: none, i.e., the "
      "interface default, Extruded)");
}

// Whether the tests use the logically extruded mesh
bool usingExtrudedMesh () { return discMethod.empty() || discMethod=="Extruded"; }

// Whether this is the run with the default Discretization Method (see the comment at the top)
bool isDefaultMethodRun () { return discMethod.empty(); }

// ============================================ Helpers ============================================ //

// Calls velocity_solver_init_mpi at construction and velocity_solver_finalize at destruction (on all ranks,
// as MALI does), so that each test goes through the whole lifecycle of the interface.
// Declare it after the MockMPAS object: the Albany objects released by velocity_solver_finalize use the
// reduced communicator (not a duplicate of it), which MockMPAS frees when destroyed.
struct InterfaceSession {
  InterfaceSession () { velocity_solver_init_mpi__(comm); }
  ~InterfaceSession () { velocity_solver_finalize__(); }
};

// Nonlinear and linear solvers. The tolerances are tight, since the tests compare with exact solutions.
// Backtracking keeps Newton from accepting steps that increase the residual (Glen's law with n=3).
const std::string piroInput = R"(  Piro:
    NOX:
      Nonlinear Solver: Line Search Based
      Line Search:
        Method: Backtrack
        Full Step:
          Full Step: 1.0
      Status Tests:
        Test Type: Combo
        Combo Type: OR
        Number of Tests: 3
        Test 0:
          Test Type: NormF
          Norm Type: Two Norm
          Scale Type: Unscaled
          Tolerance: 1.0e-09
        Test 1:
          Test Type: NormWRMS
          Relative Tolerance: 1.0e-11
          Absolute Tolerance: 1.0e-11
          Tolerance: 1.0
        Test 2:
          Test Type: MaxIters
          Maximum Iterations: 50
      Direction:
        Method: Newton
        Newton:
          Forcing Term Method: Constant
          Rescue Bad Newton Solve: true
          Linear Solver:
            Tolerance: 1.0e-12
          Stratimikos Linear Solver:
            Stratimikos:
              Linear Solver Type: Belos
              Linear Solver Types:
                Belos:
                  Solver Type: Block GMRES
                  Solver Types:
                    Block GMRES:
                      Maximum Iterations: 1000
                      Num Blocks: 500
                      Block Size: 1
                      Output Frequency: 50
                      Verbosity: 33
              Preconditioner Type: Ifpack2
              Preconditioner Types:
                Ifpack2:
                  Overlap: 1
                  Prec Type: ILUT
                  Ifpack2 Settings:
                    'fact: ilut level-of-fill': 2.0
      Printing:
        Output Precision: 3
        Output Processor: 0
        Output Information:
          Error: true
          Warning: true
          Outer Iteration: true
          Parameters: false
          Details: false
          Linear Solver Details: true
      Solver Options:
        Status Test Check Type: Minimal
)";

// Writes albany_input.yaml in the working directory. problemBody holds the entries of the "Problem"
// sublist, indented by 4 spaces. method is the Discretization Method (if empty, none is written). Collective.
void writeAlbanyInput (const std::string& problemBody, const std::string& method = discMethod)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  if (rank==0) {
    std::ofstream ofs("albany_input.yaml");
    ofs << "%YAML 1.1\n---\nANONYMOUS:\n"
        << "  Problem:\n"
        << "    Solution Method: Steady\n"
        << problemBody;
    if (method.empty()) {
      ofs << "  Discretization: {}\n";
    } else {
      ofs << "  Discretization:\n"
          << "    Method: " << method << "\n";
    }
    ofs << piroInput;
  }
  MPI_Barrier(comm);
}

double globalMax (double value) {
  double result;
  MPI_Allreduce(&value, &result, 1, MPI_DOUBLE, MPI_MAX, comm);
  return result;
}

int globalMax (int value) {
  int result;
  MPI_Allreduce(&value, &result, 1, MPI_INT, MPI_MAX, comm);
  return result;
}

double maxAbs (const std::vector<double>& v) {
  double m = 0;
  for (double x : v) m = std::max(m, std::abs(x));
  return m;
}

double maxAbsDiff (const std::vector<double>& a, const std::vector<double>& b) {
  if (a.size()!=b.size()) return std::numeric_limits<double>::infinity();
  double m = 0;
  for (std::size_t i=0; i<a.size(); ++i) m = std::max(m, std::abs(a[i]-b[i]));
  return m;
}

// Relative difference (w.r.t. the global max of |b|) between two fields, maximized over all ranks
double globalRelDiff (const std::vector<double>& a, const std::vector<double>& b) {
  return globalMax(maxAbsDiff(a,b)) / std::max(globalMax(maxAbs(b)), std::numeric_limits<double>::min());
}

// Entry of "Required Fields Info" with the given field name, or null
const Teuchos::ParameterList* findRequiredField (const Teuchos::ParameterList& rfi, const std::string& name) {
  const int n = rfi.get<int>("Number Of Fields");
  for (int i=0; i<n; ++i) {
    const auto& field = rfi.sublist("Field " + std::to_string(i));
    if (field.get<std::string>("Field Name")==name) return &field;
  }
  return nullptr;
}

// ---------------------------------- Slab on an inclined plane ---------------------------------- //

// The FE mesh is the rectangle [2,10]x[1,7] km of a larger 13x10 km grid, so that global ids are not
// contiguous. Five layers of different thickness.
MeshOptions slabMesh (PartitionType partition, int numIdleRanks, bool noSlipBed)
{
  MeshOptions o;
  o.nx = 13;
  o.ny = 10;
  o.dx = 1.0;
  o.numLayers = 5;
  o.levels = {0.0, 0.1, 0.25, 0.45, 0.7, 1.0};
  o.partition = partition;
  o.numIdleRanks = numIdleRanks;
  o.shuffleLocalOrdering = true;
  o.isActiveTriangle = [](const std::array<Point,3>& v) {
    const double eps = 1e-10;
    for (const auto& p : v) {
      if (p.x<2.0-eps || p.x>10.0+eps || p.y<1.0-eps || p.y>7.0+eps) return false;
    }
    return true;
  };
  // Velocity prescribed on the lateral boundary and, optionally, at the bed (no slip)
  o.isDirichletNode = [noSlipBed](const Point&, int il, bool isBoundaryVertex) {
    return isBoundaryVertex || (noSlipBed && il==0);
  };
  o.isIceMarginEdge = [](const Point&, const Point&) { return false; };
  return o;
}

// Slab with surface s = s0 - slope*x and bed s - H. In Albany units (km, kPa, m/yr), the FO solution
// depends on the depth d = s - z = H*(1-sigma) only. Since then u_x = -slope*u_d, the longitudinal
// stresses do not vanish: the FO x-momentum equation becomes kappa*(mu*u_d)_d = -rho*g*slope, with
// kappa = 1+4*slope^2, and the effective strain rate is eps^2 = kappa/4*u_d^2. For Glen's law with
// flow rate factor A (in Albany units, including the stiffening factor phi as A/phi^n):
//   u(d) = 2*A*(rho*g*slope)^n*(H^(n+1)-d^(n+1))/(n+1) / kappa^((n+1)/2)
//   4*mu*eps^2 = 2*A*(rho*g*slope*d)^(n+1) / kappa^((n+1)/2)   (Albany divides it by the seconds per year)
// In the FE space, the problem reduces to a 1D problem along the vertical with constant coefficients in
// each layer (the mapping of each wedge is affine), which is why the discrete solutions are exact.
struct SlabGeometry {
  double s0    = 2.0;   // surface elevation at x=0 [km]
  double slope = 0.01;  // surface (and bed) slope along x
  double H     = 1.0;   // thickness [km]
  double surface (const Point& p) const { return s0 - slope*p.x; }
  double kappa () const { return 1.0 + 4.0*slope*slope; }
};

// Basal fields that vary in space (they do not affect the solution when the bed is no slip)
double muField (const Point& p) { return 1e-3*(1.0 + 0.5*std::sin(0.7*p.x)*std::cos(0.9*p.y)); }
double effectivePressureField (const Point& p) { return 500.0*(1.0 + 0.3*std::cos(0.5*p.x + 0.3*p.y)); }

// Sets the geometry and the basal fields of a slab
void setSlabFields (const MockMPAS& mpas, const SlabGeometry& g, FieldData& f)
{
  mpas.allocate(f);
  mpas.setVertexField(f.elevation,         [&](const Point& p) { return g.surface(p); });
  mpas.setVertexField(f.thickness,         [&](const Point&)   { return g.H; });
  mpas.setVertexField(f.bedTopography,     [&](const Point& p) { return g.surface(p) - g.H; });
  mpas.setVertexField(f.mu,                muField);
  mpas.setVertexField(f.effectivePressure, effectivePressureField);
}

// Prescribes the velocity u(sigma) (along x) at the Dirichlet nodes, and initialGuessFactor*u(sigma) elsewhere
void setSlabVelocity (const MockMPAS& mpas, const std::function<double(double)>& u, double initialGuessFactor, FieldData& f)
{
  const int nV3D = mpas.numVertices3D();
  for (int ib=0; ib<mpas.numVertices(); ++ib) {
    for (int il=0; il<=mpas.numLayers(); ++il) {
      const int j = mpas.vertex3D(il,ib);
      const double factor = mpas.isDirichletNode(il,ib) ? 1.0 : initialGuessFactor;
      f.velocityOnVertices[j] = factor*u(mpas.levels()[il]);
      f.velocityOnVertices[j+nV3D] = 0.0;
    }
  }
}

// Max error of the exported velocity w.r.t. the exact solution u(sigma) (along x), over all ranks
double slabVelocityError (const MockMPAS& mpas, const std::function<double(double)>& u, const FieldData& f)
{
  double err = 0;
  const int nV3D = mpas.numVertices3D();
  for (int ib=0; ib<mpas.numVertices(); ++ib) {
    for (int il=0; il<=mpas.numLayers(); ++il) {
      const int j = mpas.vertex3D(il,ib);
      err = std::max(err, std::abs(f.velocityOnVertices[j] - u(mpas.levels()[il])));
      err = std::max(err, std::abs(f.velocityOnVertices[j+nV3D]));
    }
  }
  return globalMax(err);
}

// Max relative errors of the outputs that the slab tests check in the same way
struct SlabOutputErrors {
  double bodyForce = 0;          // |body force| = rho*g*slope on every basal cell
  double beta = 0;               // beta = mu*N (power law with exponent 1, grounded ice)
  double effectivePressure = 0;  // returned unchanged when it is an input
};

SlabOutputErrors slabOutputErrors (const MockMPAS& mpas, const PhysicalParameters& p, const SlabGeometry& g, const FieldData& f)
{
  SlabOutputErrors e;
  const double bf = p.iceDensity*p.gravity*g.slope;
  for (int it=0; it<mpas.numTriangles(); ++it) {
    e.bodyForce = std::max(e.bodyForce, std::abs(f.bodyForceOnBasalCell[it]-bf)/bf);
  }
  for (int ib=0; ib<mpas.numVertices(); ++ib) {
    const auto pt = mpas.vertex(ib);
    const double N = effectivePressureField(pt);
    const double beta = muField(pt)*N;
    e.beta = std::max(e.beta, std::abs(f.beta[ib]-beta)/beta);
    e.effectivePressure = std::max(e.effectivePressure, std::abs(f.effectivePressure[ib]-N)/N);
  }
  e.bodyForce = globalMax(e.bodyForce);
  e.beta = globalMax(e.beta);
  e.effectivePressure = globalMax(e.effectivePressure);
  return e;
}

// Max relative error of the exported dissipation w.r.t. D(il) (layer il), over all ranks
double slabDissipationError (const MockMPAS& mpas, const std::function<double(int)>& D, const FieldData& f)
{
  double err = 0;
  for (int it=0; it<mpas.numTriangles(); ++it) {
    for (int il=0; il<mpas.numLayers(); ++il) {
      err = std::max(err, std::abs(f.dissipationHeatOnPrisms[mpas.prism(il,it)]-D(il))/std::abs(D(il)));
    }
  }
  return globalMax(err);
}

// =============================================== Tests =============================================== //

// The mock itself: the data passed to Albany satisfies the invariants Albany relies on
TEUCHOS_UNIT_TEST(MockMPAS, MeshConsistency)
{
  for (auto partition : {PartitionType::Strips, PartitionType::Blocks, PartitionType::Scattered}) {
    for (int numIdleRanks : {0, 1}) {
      for (int ordering : {0, 1}) {
        auto options = slabMesh(partition, numIdleRanks, true);
        options.ordering = ordering;
        MockMPAS mpas(comm, options);
        const auto msg = mpas.checkConsistency();
        TEST_ASSERT(msg.empty());
        if (!msg.empty()) out << msg;
        // 8x6 quads in the FE mesh, embedded in a larger grid
        TEST_EQUALITY(mpas.numGlobalTriangles(), 96);
        TEST_EQUALITY(mpas.numGlobalVertices(), 63);
        TEST_COMPARE(mpas.globalVerticesStride(), >, mpas.numGlobalVertices());
      }
    }
  }
}

// Physical parameters passed by MPAS end up in the Albany parameter list, with the flow rate factor
// converted to [Pa^-n s^-1] using the seconds per year passed by MPAS.
TEUCHOS_UNIT_TEST(MPASInterface, PhysicalParametersFromMPAS)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Strips, 0, true));
  InterfaceSession session;
  PhysicalParameters p;
  p.gravity = 9.81;
  p.iceDensity = 917.0;
  p.oceanDensity = 1027.0;
  p.flowParamA = 2.0e-4;
  p.flowLawExponent = 3.0;
  p.useGLP = false;
  p.clausiusClapeyronCoeff = 8.7e-8;
  p.meltingTemperature = 273.0;
  p.secondsPerYear = 3.15576e7;
  p.gasConstant = 8.3145;
  setPhysicalParameters(p);
  writeAlbanyInput("");

  mpas.computeGrid();
  mpas.extrudeGrid();

  TEST_ASSERT(Teuchos::nonnull(paramList));
  const auto& problem = paramList->sublist("Problem");
  const auto& phys = problem.sublist("LandIce Physical Parameters");
  TEST_FLOATING_EQUALITY(phys.get<double>("Gravity Acceleration"), 9.81, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Ice Density"), 917.0, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Water Density"), 1027.0, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Clausius-Clapeyron Coefficient"), 8.7e-8, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Atmospheric Pressure Melting Temperature"), 273.0, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Seconds per Year"), 3.15576e7, 1e-14);
  TEST_EQUALITY(phys.get<bool>("Use GLP"), false);

  const auto& visc = problem.sublist("LandIce Viscosity");
  TEST_FLOATING_EQUALITY(visc.get<double>("Glen's Law n"), 3.0, 1e-14);
  TEST_FLOATING_EQUALITY(visc.get<double>("Glen's Law A"), 2.0e-4/std::pow(1000.0,4)/3.15576e7, 1e-13);
  TEST_FLOATING_EQUALITY(visc.sublist("Arrhenius Relationship").get<double>("Gas Constant"), 8.3145, 1e-14);
  TEST_EQUALITY(visc.get<std::string>("Flow Rate Type"), std::string("Temperature Based"));
  TEST_EQUALITY(visc.get<bool>("Use P0 Temperature"), true);

  // Without GLP: no beta zeroed on floating ice, effective pressure zeroed at floating nodes instead
  const auto& beta = problem.sublist("LandIce BCs").sublist("BC 0").sublist("Basal Friction Coefficient");
  TEST_EQUALITY(beta.get<bool>("Zero Beta On Floating Ice"), false);
  TEST_EQUALITY(beta.get<bool>("Zero Effective Pressure On Floating Ice At Nodes"), true);
  TEST_EQUALITY(problem.get<int>("Basal Cubature Degree"), 4);
  TEST_COMPARE_ARRAYS(problem.get<Teuchos::Array<int>>("Cubature Degrees (Horiz Vert)"), Teuchos::tuple<int>(4,3));
}

// The deprecated overload (still called by E3SM's MALI) keeps the values Albany used before MPAS could set them
TEUCHOS_UNIT_TEST(MPASInterface, DeprecatedPhysicalParametersUseLegacyDefaults)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Strips, 0, true));
  InterfaceSession session;
  PhysicalParameters p;
  p.flowParamA = 2.0e-4;
  p.meltingTemperature = 270.0;   // ignored by the deprecated overload
  p.secondsPerYear = 3.0e7;       // ignored
  p.gasConstant = 8.0;            // ignored
  setPhysicalParametersDeprecated(p);
  writeAlbanyInput("");

  mpas.computeGrid();
  mpas.extrudeGrid();

  TEST_ASSERT(Teuchos::nonnull(paramList));
  const auto& problem = paramList->sublist("Problem");
  const auto& phys = problem.sublist("LandIce Physical Parameters");
  TEST_FLOATING_EQUALITY(phys.get<double>("Atmospheric Pressure Melting Temperature"), 273.15, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Seconds per Year"), 3.1536e7, 1e-14);
  TEST_EQUALITY(phys.get<bool>("Use GLP"), true);
  const auto& visc = problem.sublist("LandIce Viscosity");
  TEST_FLOATING_EQUALITY(visc.sublist("Arrhenius Relationship").get<double>("Gas Constant"), 8.314, 1e-14);
  TEST_FLOATING_EQUALITY(visc.get<double>("Glen's Law A"), 2.0e-4/std::pow(1000.0,4)/3.1536e7, 1e-13);

  // With GLP: beta zeroed on floating ice, higher basal cubature degree
  const auto& beta = problem.sublist("LandIce BCs").sublist("BC 0").sublist("Basal Friction Coefficient");
  TEST_EQUALITY(beta.get<bool>("Zero Beta On Floating Ice"), true);
  TEST_EQUALITY(beta.get<bool>("Zero Effective Pressure On Floating Ice At Nodes"), false);
  TEST_EQUALITY(problem.get<int>("Basal Cubature Degree"), 8);
}

// Values set in albany_input.yaml take precedence over those passed by MPAS
TEUCHOS_UNIT_TEST(MPASInterface, InputFileTakesPrecedence)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Strips, 0, true));
  InterfaceSession session;
  PhysicalParameters p;
  p.iceDensity = 917.0;
  p.oceanDensity = 1027.0;
  p.secondsPerYear = 3.15576e7;
  p.gasConstant = 8.3145;
  setPhysicalParameters(p);
  writeAlbanyInput(
    "    LandIce Physical Parameters:\n"
    "      Water Density: 1000.0\n"
    "      Seconds per Year: 3.0e+07\n"
    "    LandIce Viscosity:\n"
    "      Type: \"Glen's Law\"\n"
    "      Flow Rate Type: Uniform\n"
    "      \"Glen's Law A\": 2.0e-24\n"
    "      Arrhenius Relationship:\n"
    "        Gas Constant: 8.0\n");

  mpas.computeGrid();
  mpas.extrudeGrid();

  TEST_ASSERT(Teuchos::nonnull(paramList));
  const auto& problem = paramList->sublist("Problem");
  const auto& phys = problem.sublist("LandIce Physical Parameters");
  TEST_FLOATING_EQUALITY(phys.get<double>("Water Density"), 1000.0, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Ice Density"), 917.0, 1e-14);
  TEST_FLOATING_EQUALITY(phys.get<double>("Seconds per Year"), 3.0e7, 1e-14);
  const auto& visc = problem.sublist("LandIce Viscosity");
  TEST_EQUALITY(visc.get<std::string>("Flow Rate Type"), std::string("Uniform"));
  TEST_FLOATING_EQUALITY(visc.get<double>("Glen's Law A"), 2.0e-24, 1e-14);
  TEST_FLOATING_EQUALITY(visc.sublist("Arrhenius Relationship").get<double>("Gas Constant"), 8.0, 1e-14);
}

// The fields the interface registers depend on the sliding law and on the effective pressure type
TEUCHOS_UNIT_TEST(MPASInterface, RequiredFields)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Blocks, 0, true));
  InterfaceSession session;
  setPhysicalParameters(PhysicalParameters());

  auto has = [](const Teuchos::ParameterList* field, const std::string& name, const std::string& value) {
    return field->isParameter(name) && field->get<std::string>(name)==value;
  };
  auto isInput = [&](const Teuchos::ParameterList* field, const std::string& type) {
    return field!=nullptr && has(field,"Field Type",type) && has(field,"Field Origin","Mesh");
  };
  auto isOutput = [&](const Teuchos::ParameterList* field, const std::string& type) {
    return field!=nullptr && has(field,"Field Type",type) && has(field,"Field Usage","Output");
  };

  // Default: power law with effective pressure from MPAS
  writeAlbanyInput("");
  mpas.computeGrid();
  mpas.extrudeGrid();
  {
    TEST_ASSERT(Teuchos::nonnull(paramList));
    const auto& disc = paramList->sublist("Discretization");
    const auto& rfi = disc.sublist("Required Fields Info");
    TEST_ASSERT(isInput(findRequiredField(rfi,"temperature"), "Elem Scalar"));
    TEST_ASSERT(isInput(findRequiredField(rfi,"ice_thickness"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(rfi,"surface_height"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(rfi,"dirichlet_field"), "Node Vector"));
    TEST_ASSERT(isInput(findRequiredField(rfi,"stiffening_factor_log"), "Node Scalar"));
    TEST_ASSERT(isOutput(findRequiredField(rfi,"body_force"), "Elem Vector"));

    const auto& basal = disc.sublist("Side Set Discretizations").sublist("basalside").sublist("Required Fields Info");
    TEST_ASSERT(isInput(findRequiredField(basal,"bed_topography"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(basal,"mu"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(basal,"effective_pressure"), "Node Scalar"));
    TEST_ASSERT(isOutput(findRequiredField(basal,"beta"), "Node Scalar"));
    TEST_ASSERT(findRequiredField(basal,"bed_roughness")==nullptr);
    TEST_ASSERT(findRequiredField(basal,"bulk_friction")==nullptr);
    TEST_ASSERT(findRequiredField(basal,"basal_debris_factor")==nullptr);
  }

  // Regularized Coulomb with hydrostatic effective pressure: bed roughness is an input, N an output
  writeAlbanyInput(
    "    LandIce BCs:\n"
    "      BC 0:\n"
    "        Basal Friction Coefficient:\n"
    "          Type: Regularized Coulomb\n"
    "          Power Exponent: 0.3333333333\n"
    "          Effective Pressure Type: Hydrostatic Computed At Nodes\n");
  mpas.computeGrid();
  mpas.extrudeGrid();
  {
    TEST_ASSERT(Teuchos::nonnull(paramList));
    const auto& basal = paramList->sublist("Discretization").sublist("Side Set Discretizations")
                                  .sublist("basalside").sublist("Required Fields Info");
    TEST_ASSERT(isInput(findRequiredField(basal,"bed_roughness"), "Node Scalar"));
    TEST_ASSERT(isOutput(findRequiredField(basal,"effective_pressure"), "Node Scalar"));
    TEST_ASSERT(findRequiredField(basal,"bulk_friction")==nullptr);
  }

  // Debris friction: also bulk friction and basal debris factor. The interface does not set the two
  // regularization values, which the sliding law requires.
  writeAlbanyInput(
    "    LandIce BCs:\n"
    "      BC 0:\n"
    "        Basal Friction Coefficient:\n"
    "          Type: Debris Friction\n"
    "          Power Exponent: 0.3333333333\n"
    "          Effective Pressure Regularization: 1.0e+03\n"
    "          Sliding Velocity Regularization: 5.0e+02\n");
  mpas.computeGrid();
  mpas.extrudeGrid();
  {
    TEST_ASSERT(Teuchos::nonnull(paramList));
    const auto& basal = paramList->sublist("Discretization").sublist("Side Set Discretizations")
                                  .sublist("basalside").sublist("Required Fields Info");
    TEST_ASSERT(isInput(findRequiredField(basal,"bed_roughness"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(basal,"bulk_friction"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(basal,"basal_debris_factor"), "Node Scalar"));
    TEST_ASSERT(isInput(findRequiredField(basal,"effective_pressure"), "Node Scalar"));
  }
}

// Defaults of the depth-integrated model, and rejection of an inconsistent temperature setting
TEUCHOS_UNIT_TEST(MPASInterface, DepthIntegratedSetup)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Strips, 0, true));
  InterfaceSession session;
  setPhysicalParameters(PhysicalParameters());

  writeAlbanyInput("    Depth Integrated Model: true\n");
  mpas.computeGrid();
  mpas.extrudeGrid();
  TEST_ASSERT(Teuchos::nonnull(paramList));
  const auto& problem = paramList->sublist("Problem");
  TEST_EQUALITY(problem.sublist("LandIce Viscosity").get<bool>("Use P0 Temperature"), false);
  TEST_COMPARE_ARRAYS(problem.get<Teuchos::Array<int>>("Cubature Degrees (Horiz Vert)"), Teuchos::tuple<int>(4,7));
  const auto& rfi = paramList->sublist("Discretization").sublist("Required Fields Info");
  const auto temperature = findRequiredField(rfi,"temperature");
  TEST_ASSERT(temperature!=nullptr && temperature->get<std::string>("Field Type")=="QuadPoint Scalar");
  if (usingExtrudedMesh()) {
    // Albany's mesh has one layer (and the MPAS temperature is evaluated at the cubature points)
    TEST_EQUALITY(paramList->sublist("Discretization").get<int>("NumLayers"), 1);
  }

  writeAlbanyInput(
    "    Depth Integrated Model: true\n"
    "    LandIce Viscosity:\n"
    "      Use P0 Temperature: true\n");
  mpas.computeGrid();
  TEST_THROW(mpas.extrudeGrid(), std::runtime_error);
}

// FO slab, n=1, uniform flow rate from MPAS: u(sigma) = A*rho*g*slope*H^2*(1-(1-sigma)^2)/(phi*kappa),
// where A is the flow rate factor in MPAS units ([k^-1 kPa^-n yr^-1], which are Albany's) and phi the
// stiffening factor. Checks velocity, dissipation, body force, beta and effective pressure.
// The Dirichlet nodes are those of options (the exact solution is prescribed there). problemInput is added
// to the Problem sublist of albany_input.yaml, and method is the Discretization Method. If given,
// checkParameters is called after the solve, while the interface still holds its parameter list.
void checkSlabFOUniformFlowRate (const MeshOptions& options, const std::string& problemInput,
                                 const std::string& method, Teuchos::FancyOStream& out, bool& success,
                                 const std::function<void()>& checkParameters = {})
{
  MockMPAS mpas(comm, options);
  InterfaceSession session;
  PhysicalParameters p;
  p.gravity = 9.81;
  p.iceDensity = 917.0;
  p.flowLawExponent = 1.0;
  p.flowParamA = 0.5;           // [k^-1 kPa^-1 yr^-1]
  p.secondsPerYear = 3.1536e7;
  setPhysicalParameters(p);
  writeAlbanyInput(
    "    LandIce Viscosity:\n"
    "      Flow Rate Type: Uniform\n"
    "      \"Glen's Law Homotopy Parameter\": 1.0\n" + problemInput, method);

  const SlabGeometry g;
  const double phi = 1.25;
  const double rgs = p.iceDensity*p.gravity*g.slope;
  const double uS = p.flowParamA*rgs*g.H*g.H/(phi*g.kappa());
  auto u = [&](double sigma) { return uS*(1.0-(1.0-sigma)*(1.0-sigma)); };
  // Dissipation: the FE strain rate in a layer is the exact one at mid layer
  auto D = [&](int il) {
    const double depth = g.H*(1.0 - 0.5*(mpas.levels()[il]+mpas.levels()[il+1]));
    return 2.0*p.flowParamA*rgs*rgs*depth*depth/(phi*g.kappa()*p.secondsPerYear);
  };

  FieldData f;
  int error = 0;
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlabFields(mpas, g, f);
    mpas.setVertexField(f.stiffeningFactor, [&](const Point&) { return phi; });
    mpas.setPrismField(f.temperatureOnPrisms, [](const Point&, double, int) { return 250.0; });
    setSlabVelocity(mpas, u, 0.0, f);
    error = mpas.solve(f, true, 0.0);
  }
  TEST_EQUALITY(globalMax(error), 0);

  const double velErr = slabVelocityError(mpas, u, f)/uS;
  out << "  relative velocity error: " << velErr << "\n";
  TEST_COMPARE(velErr, <, 1e-8);

  const double dissErr = slabDissipationError(mpas, D, f);
  out << "  relative dissipation error: " << dissErr << "\n";
  TEST_COMPARE(dissErr, <, 1e-6);

  const auto e = slabOutputErrors(mpas, p, g, f);
  out << "  relative errors: body force " << e.bodyForce << ", beta " << e.beta
      << ", effective pressure " << e.effectivePressure << "\n";
  TEST_COMPARE(e.bodyForce, <, 1e-10);
  TEST_COMPARE(e.beta, <, 1e-12);
  TEST_COMPARE(e.effectivePressure, <, 1e-14);

  if (checkParameters && mpas.isActive()) {
    checkParameters();
  }
}

// FO slab with uniform flow rate, with a scattered partition
TEUCHOS_UNIT_TEST(MPASInterface, SlabFOUniformFlowRate)
{
  checkSlabFOUniformFlowRate(slabMesh(PartitionType::Scattered, 0, true), "", discMethod, out, success);
}

// FO slab with uniform flow rate, with the column-wise ordering of the 3D entities passed by MPAS (MALI only
// uses the layer-wise one), a block partition and (in parallel) a rank without triangles
TEUCHOS_UNIT_TEST(MPASInterface, SlabFOColumnwiseOrdering)
{
  auto options = slabMesh(PartitionType::Blocks, 1, true);
  options.ordering = 1;
  checkSlabFOUniformFlowRate(options, "", discMethod, out, success);
}

// Input files written for the 3D STK mesh refer to its node set "dirichlet" (in the Dirichlet BCs) and side
// set "ice_margin_side" (in the LandIce BCs). With the logically extruded mesh, the interface applies them on
// the corresponding node sets and side set of that mesh: the FO slab (whose Dirichlet nodes are the lateral
// columns and the bed) is still solved exactly.
TEUCHOS_UNIT_TEST(MPASInterface, BackwardCompatibleInput)
{
  const std::string problemInput =
    "    Dirichlet BCs:\n"
    "      SDBC on NS dirichlet for DOF U0 prescribe Field: dirichlet_field\n"
    "      SDBC on NS dirichlet for DOF U1 prescribe Field: dirichlet_field\n"
    "    LandIce BCs:\n"
    "      BC 1:\n"
    "        Side Set Name: ice_margin_side\n";
  auto checkParameters = [&]() {
    TEST_ASSERT(Teuchos::nonnull(paramList));
    if (Teuchos::is_null(paramList)) return;
    const auto& problem = paramList->sublist("Problem");
    const auto& dbcs = problem.sublist("Dirichlet BCs");
    const auto& lateralSideSet = problem.sublist("LandIce BCs").sublist("BC 1").get<std::string>("Side Set Name");
    for (const std::string dof : {"U0", "U1"}) {
      const std::string oldName = "SDBC on NS dirichlet for DOF " + dof + " prescribe Field";
      if (usingExtrudedMesh()) {
        TEST_ASSERT(!dbcs.isParameter(oldName));
        for (const std::string ns : {"extruded_dirichlet", "basal_dirichlet_bed"}) {
          const std::string newName = "SDBC on NS " + ns + " for DOF " + dof + " prescribe Field";
          TEST_ASSERT(dbcs.isParameter(newName) && dbcs.get<std::string>(newName)=="dirichlet_field");
        }
      } else {
        TEST_ASSERT(dbcs.isParameter(oldName));
      }
    }
    TEST_EQUALITY(lateralSideSet, std::string(usingExtrudedMesh() ? "extruded_ice_margin_side" : "ice_margin_side"));
  };
  checkSlabFOUniformFlowRate(slabMesh(PartitionType::Strips, 0, true), problemInput, discMethod, out, success,
                             checkParameters);
}

// The logically extruded mesh supports, at each MPAS vertex, Dirichlet nodes on the whole column or at the
// bed only. Other patterns (here: the bed and the surface of the interior vertices) are rejected (on all
// ranks), while the 3D STK mesh supports them: with it, the FO slab is still solved exactly.
TEUCHOS_UNIT_TEST(MPASInterface, UnsupportedDirichletPattern)
{
  if (!isDefaultMethodRun()) {
    out << "  Skipped: this test sets the Discretization Method itself.\n";
    return;
  }
  auto options = slabMesh(PartitionType::Strips, 0, false);
  const int numLayers = options.numLayers;
  options.isDirichletNode = [numLayers](const Point&, int il, bool isBoundaryVertex) {
    return isBoundaryVertex || il==0 || il==numLayers;
  };
  {
    MockMPAS mpas(comm, options);
    InterfaceSession session;
    setPhysicalParameters(PhysicalParameters());
    writeAlbanyInput("", "Extruded");
    mpas.computeGrid();
    TEST_THROW(mpas.extrudeGrid(), std::runtime_error);
  }
  checkSlabFOUniformFlowRate(options, "", "STKExtruded", out, success);
}

// The settings that depend on the mesh. With the logically extruded mesh, the 3D fields set by MPAS have
// basal counterparts (extruded or interpolated), and the default BCs use the node sets and side sets of
// that mesh. An unknown Method is rejected.
TEUCHOS_UNIT_TEST(MPASInterface, MeshParameters)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Strips, 0, true));
  InterfaceSession session;
  setPhysicalParameters(PhysicalParameters());

  writeAlbanyInput("");
  mpas.computeGrid();
  mpas.extrudeGrid();
  TEST_ASSERT(Teuchos::nonnull(paramList));
  {
    const auto& problem = paramList->sublist("Problem");
    const auto& disc = paramList->sublist("Discretization");
    const auto& dbcs = problem.sublist("Dirichlet BCs");
    const auto& basal = disc.sublist("Side Set Discretizations").sublist("basalside");
    const auto& basalRFI = basal.sublist("Required Fields Info");
    const auto& lateralSideSet = problem.sublist("LandIce BCs").sublist("BC 1").get<std::string>("Side Set Name");
    if (usingExtrudedMesh()) {
      TEST_EQUALITY(disc.get<std::string>("Method"), std::string("Extruded"));
      TEST_EQUALITY(disc.get<int>("NumLayers"), mpas.numLayers());
      TEST_EQUALITY(disc.get<bool>("Columnwise Ordering"), true);
      TEST_ASSERT(!basal.isParameter("Method"));  // the interface builds the basal mesh

      using Names = Teuchos::Array<std::string>;
      auto contains = [](const Names& names, const std::string& name) {
        return std::find(names.begin(), names.end(), name)!=names.end();
      };
      const auto& extruded = disc.get<Names>("Extrude Basal Fields");
      const auto& interpolated = disc.get<Names>("Interpolate Basal Layered Fields");
      for (const std::string name : {"ice_thickness", "surface_height", "stiffening_factor_log"}) {
        const auto field = findRequiredField(basalRFI, name);
        TEST_ASSERT(field!=nullptr && field->get<std::string>("Field Type")=="Node Scalar");
        TEST_ASSERT(contains(extruded, name));
      }
      // The temperature is given on the MPAS layers, the Dirichlet velocity on the MPAS levels
      const auto temperature = findRequiredField(basalRFI, "temperature");
      TEST_ASSERT(temperature!=nullptr && temperature->get<std::string>("Field Type")=="Elem Layered Scalar" &&
                  temperature->get<int>("Number Of Layers")==mpas.numLayers());
      TEST_ASSERT(contains(interpolated, "temperature"));
      const auto dirichlet = findRequiredField(basalRFI, "dirichlet_field");
      TEST_ASSERT(dirichlet!=nullptr && dirichlet->get<std::string>("Field Type")=="Node Layered Vector" &&
                  dirichlet->get<int>("Number Of Layers")==mpas.numLayers()+1 && dirichlet->get<int>("Vector Dim")==2);
      TEST_ASSERT(contains(interpolated, "dirichlet_field"));

      for (const std::string ns : {"extruded_dirichlet", "basal_dirichlet_bed"}) {
        for (const std::string dof : {"U0", "U1"}) {
          TEST_ASSERT(dbcs.isParameter("SDBC on NS " + ns + " for DOF " + dof + " prescribe Field"));
        }
      }
      TEST_EQUALITY(lateralSideSet, std::string("extruded_ice_margin_side"));
    } else {
      TEST_EQUALITY(disc.get<std::string>("Method"), std::string("STKExtruded"));
      TEST_EQUALITY(basal.get<std::string>("Method"), std::string("SideSetSTK"));
      TEST_ASSERT(findRequiredField(basalRFI, "temperature")==nullptr);
      TEST_ASSERT(dbcs.isParameter("SDBC on NS dirichlet for DOF U0 prescribe Field"));
      TEST_ASSERT(dbcs.isParameter("SDBC on NS dirichlet for DOF U1 prescribe Field"));
      TEST_EQUALITY(lateralSideSet, std::string("ice_margin_side"));
    }
  }

  writeAlbanyInput("", "Ioss");
  mpas.computeGrid();
  TEST_THROW(mpas.extrudeGrid(), std::runtime_error);
}

// FO slab, n=1, temperature-based flow rate with a temperature that changes with the layer, and the gas
// constant, Clausius-Clapeyron coefficient and melting temperature passed by MPAS. With a Clausius-Clapeyron
// coefficient beta, the viscosity uses the pressure-corrected temperature min(T + beta*1000*rho*g*d, Tm),
// evaluated (with P0 temperature) at the mid-layer depth d. So the viscosity is still constant in each
// layer, and the FE solution is still exact at the nodes. The melting temperature is lowered so that the
// cap is active in the bottom layer. Uses a block partition and (in parallel) a rank without triangles.
TEUCHOS_UNIT_TEST(MPASInterface, SlabFOTemperatureBased)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Blocks, 1, true));
  InterfaceSession session;
  PhysicalParameters p;
  p.gravity = 9.81;
  p.iceDensity = 917.0;
  p.flowLawExponent = 1.0;
  p.gasConstant = 8.0;
  p.clausiusClapeyronCoeff = 9.7546e-8;
  p.meltingTemperature = 258.5;
  p.secondsPerYear = 3.1536e7;
  setPhysicalParameters(p);
  // A- is increased so that velocities have realistic values with n=1
  const double Aminus = 0.1, Qminus = 6.0e4;
  writeAlbanyInput(
    "    LandIce Viscosity:\n"
    "      Flow Rate Type: Temperature Based\n"
    "      \"Glen's Law Homotopy Parameter\": 1.0\n"
    "      Arrhenius Relationship:\n"
    "        \"A-\": 0.1\n");

  const SlabGeometry g;
  const double phi = 1.25;
  const double rgs = p.iceDensity*p.gravity*g.slope;
  const std::vector<double> T = {258.0, 255.0, 251.0, 247.0, 243.0};  // below 263.15 K: A- branch
  const auto& levels = mpas.levels();

  // Viscosity in each layer, in Albany units (ViscosityFO scales A by 3.1536e7*1000^(n+1)), using the
  // pressure-corrected temperature (the correction is in K per km of depth)
  const double ccPerKm = p.clausiusClapeyronCoeff*1000.0*p.iceDensity*p.gravity;
  std::vector<double> visc(mpas.numLayers());
  for (int il=0; il<mpas.numLayers(); ++il) {
    const double depth = g.H*(1.0 - 0.5*(levels[il]+levels[il+1]));
    const double Tcorr = std::min(T[il] + ccPerKm*depth, p.meltingTemperature);
    const double A = Aminus*std::exp(-Qminus/(p.gasConstant*Tcorr));
    visc[il] = phi/(2.0*A*3.1536e7*1.0e6);
  }
  TEST_COMPARE(T[0] + ccPerKm*g.H*(1.0-0.5*levels[1]), >, p.meltingTemperature);  // the cap is tested
  // u at level k: integral from the bed of u_d = rho*g*slope*d/(kappa*mu)
  std::vector<double> uLevel(mpas.numLayers()+1, 0.0);
  for (int il=0; il<mpas.numLayers(); ++il) {
    const double d0 = g.H*(1.0-levels[il]), d1 = g.H*(1.0-levels[il+1]);
    uLevel[il+1] = uLevel[il] + rgs/(g.kappa()*visc[il])*(d0*d0-d1*d1)/2.0;
  }
  auto u = [&](double sigma) {
    const auto it = std::find(levels.begin(), levels.end(), sigma);
    return uLevel[it-levels.begin()];
  };
  // 4*mu*eps^2 = kappa*mu*u_d^2
  auto D = [&](int il) {
    const double depth = g.H*(1.0 - 0.5*(levels[il]+levels[il+1]));
    return rgs*rgs*depth*depth/(g.kappa()*visc[il]*p.secondsPerYear);
  };

  FieldData f;
  int error = 0;
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlabFields(mpas, g, f);
    mpas.setVertexField(f.stiffeningFactor, [&](const Point&) { return phi; });
    mpas.setPrismField(f.temperatureOnPrisms, [&](const Point&, double, int il) { return T[il]; });
    setSlabVelocity(mpas, u, 0.0, f);
    error = mpas.solve(f, true, 0.0);
  }
  TEST_EQUALITY(globalMax(error), 0);

  const double uS = uLevel.back();
  const double velErr = slabVelocityError(mpas, u, f)/uS;
  out << "  surface velocity: " << uS << ", relative velocity error: " << velErr << "\n";
  TEST_COMPARE(velErr, <, 1e-8);

  const double dissErr = slabDissipationError(mpas, D, f);
  out << "  relative dissipation error: " << dissErr << "\n";
  TEST_COMPARE(dissErr, <, 1e-6);

  const auto e = slabOutputErrors(mpas, p, g, f);
  TEST_COMPARE(e.bodyForce, <, 1e-10);
  TEST_COMPARE(e.beta, <, 1e-12);
}

// Depth-integrated (MOLHO) slab, n=3, temperature-based flow rate with the gas constant passed by MPAS.
// The exact profile u_s*(1-(1-sigma)^4) is in the vertical basis, so the solution is exact, and the
// interface must return it at all the MPAS levels. Newton starts from 0.9 times the exact solution
// away from the Dirichlet nodes.
// The Clausius-Clapeyron coefficient is set to 0: the pressure correction of the temperature would make
// the flow rate factor depend on the depth, and the exact profile would no longer be in the basis.
TEUCHOS_UNIT_TEST(MPASInterface, SlabDepthIntegrated)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Strips, 1, true));
  InterfaceSession session;
  PhysicalParameters p;
  p.gravity = 9.81;
  p.iceDensity = 917.0;
  p.flowLawExponent = 3.0;
  p.gasConstant = 8.0;
  p.clausiusClapeyronCoeff = 0.0;
  p.secondsPerYear = 3.1536e7;
  setPhysicalParameters(p);
  writeAlbanyInput(
    "    Depth Integrated Model: true\n"
    "    LandIce Viscosity:\n"
    "      Flow Rate Type: Temperature Based\n"
    "      \"Glen's Law Homotopy Parameter\": 1.0\n");

  SlabGeometry g;
  g.slope = 0.05;
  const double phi = 0.8;
  const double T = 253.15;
  const double n = p.flowLawExponent;
  const double rgs = p.iceDensity*p.gravity*g.slope;
  // Flow rate factor in Albany units, including the stiffening factor
  const double A = 3.613e-13*std::exp(-6.0e4/(p.gasConstant*T))*3.1536e7*std::pow(1000.0,n+1)/std::pow(phi,n);
  const double kappaFactor = std::pow(g.kappa(), -(n+1)/2);
  const double uS = 2.0*A*std::pow(rgs,n)*std::pow(g.H,n+1)/(n+1)*kappaFactor;
  auto u = [&](double sigma) { return uS*(1.0-std::pow(1.0-sigma,n+1)); };
  // Albany's mesh has one layer, so every MPAS layer gets the column average of the dissipation:
  // the average of d^(n+1) is H^(n+1)/(n+2)
  const double Davg = 2.0*A*std::pow(rgs*g.H,n+1)/(n+2)*kappaFactor/p.secondsPerYear;

  FieldData f;
  int error = 0;
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlabFields(mpas, g, f);
    mpas.setVertexField(f.stiffeningFactor, [&](const Point&) { return phi; });
    mpas.setPrismField(f.temperatureOnPrisms, [&](const Point&, double, int) { return T; });
    setSlabVelocity(mpas, u, 0.9, f);
    error = mpas.solve(f, true, 0.0);
  }
  TEST_EQUALITY(globalMax(error), 0);

  const double velErr = slabVelocityError(mpas, u, f)/uS;
  out << "  surface velocity: " << uS << ", relative velocity error: " << velErr << "\n";
  TEST_COMPARE(velErr, <, 1e-6);

  const double dissErr = slabDissipationError(mpas, [&](int) { return Davg; }, f);
  out << "  relative dissipation error: " << dissErr << "\n";
  TEST_COMPARE(dissErr, <, 1e-6);

  const auto e = slabOutputErrors(mpas, p, g, f);
  TEST_COMPARE(e.bodyForce, <, 1e-10);
  TEST_COMPARE(e.beta, <, 1e-12);
}

// ------------------------------- Tests without an exact solution ------------------------------- //

// A slab sliding on its bed, with no slip on the lateral boundary, so that the solution is 3D and depends
// on all the inputs. State 1 and state 2 differ in every input.
struct SlidingSlabState {
  SlabGeometry geometry;
  double muFactor;
  std::function<double(const Point&)> stiffening;
};

SlidingSlabState slidingSlabState (int state)
{
  SlidingSlabState s;
  if (state==1) {
    s.muFactor = 0.4;
    s.stiffening = [](const Point& p) { return 1.0 + 0.2*std::sin(0.6*p.x); };
  } else {
    s.geometry.s0 = 2.1;
    s.geometry.slope = 0.015;
    s.geometry.H = 0.9;
    s.muFactor = 0.6;
    s.stiffening = [](const Point& p) { return 1.0 + 0.2*std::cos(0.4*p.y); };
  }
  return s;
}

void setSlidingSlabFields (const MockMPAS& mpas, const SlidingSlabState& s, int state, FieldData& f)
{
  setSlabFields(mpas, s.geometry, f);
  for (auto& mu : f.mu) mu *= s.muFactor;
  mpas.setVertexField(f.effectivePressure, [&](const Point& p) {
    return 0.8*917.0*9.81*s.geometry.H*(1.0 + 0.1*std::cos(0.3*p.x+0.2*p.y));
  });
  mpas.setVertexField(f.stiffeningFactor, s.stiffening);
  mpas.setPrismField(f.temperatureOnPrisms, [state](const Point& p, double sigma, int) {
    return 250.0 + 5.0*sigma + 0.2*p.x + (state==2 ? 2.0 : 0.0);
  });
  std::fill(f.velocityOnVertices.begin(), f.velocityOnVertices.end(), 0.0);
}

const std::string slidingSlabInput =
    "    LandIce Viscosity:\n"
    "      Flow Rate Type: Temperature Based\n"
    "      \"Glen's Law Homotopy Parameter\": 1.0\n"
    "      Arrhenius Relationship:\n"
    "        \"A-\": 0.1\n";

PhysicalParameters slidingSlabParameters ()
{
  PhysicalParameters p;
  p.gravity = 9.81;
  p.iceDensity = 917.0;
  p.flowLawExponent = 1.0;
  return p;
}

// Repeated solves on the same FE grid (as at each MALI time step) must give the same results as a solve
// on a newly created grid: nothing computed for previous inputs (e.g., memoized fields) may be reused.
TEUCHOS_UNIT_TEST(MPASInterface, RepeatedSolves)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Scattered, 0, false));
  InterfaceSession session;
  setPhysicalParameters(slidingSlabParameters());
  writeAlbanyInput(slidingSlabInput);

  const auto state1 = slidingSlabState(1);
  const auto state2 = slidingSlabState(2);

  FieldData f1a, f2a, f1b, f2b;
  int error = 0;
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlidingSlabFields(mpas, state1, 1, f1a);
    error = std::max(error, mpas.solve(f1a, true, 1.0));
    setSlidingSlabFields(mpas, state2, 2, f2a);
    error = std::max(error, mpas.solve(f2a, false, 1.0));
    setSlidingSlabFields(mpas, state1, 1, f1b);
    error = std::max(error, mpas.solve(f1b, false, 1.0));

    // Regenerate the FE grid (as MALI does when the ice extent changes), then solve state 2 again
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlidingSlabFields(mpas, state2, 2, f2b);
    error = std::max(error, mpas.solve(f2b, false, 1.0));
  }
  TEST_EQUALITY(globalMax(error), 0);

  // The two states give different results
  const double change = globalRelDiff(f2a.velocityOnVertices, f1a.velocityOnVertices);
  out << "  relative velocity change between the two states: " << change << "\n";
  TEST_COMPARE(change, >, 1e-2);

  auto compare = [&](const FieldData& a, const FieldData& b, const std::string& name) {
    const double velDiff  = globalRelDiff(a.velocityOnVertices, b.velocityOnVertices);
    const double betaDiff = globalRelDiff(a.beta, b.beta);
    const double bfDiff   = globalRelDiff(a.bodyForceOnBasalCell, b.bodyForceOnBasalCell);
    const double dissDiff = globalRelDiff(a.dissipationHeatOnPrisms, b.dissipationHeatOnPrisms);
    out << "  " << name << ": relative differences: velocity " << velDiff << ", beta " << betaDiff
        << ", body force " << bfDiff << ", dissipation " << dissDiff << "\n";
    TEST_COMPARE(velDiff, <, 1e-8);
    TEST_COMPARE(betaDiff, <, 1e-12);
    TEST_COMPARE(bfDiff, <, 1e-10);
    TEST_COMPARE(dissDiff, <, 1e-6);
  };
  compare(f2a, f2b, "state 2, same grid vs new grid");
  compare(f1a, f1b, "state 1, first vs third solve");
}

// The deprecated overloads (still called by E3SM's MALI) give the same results as the current ones
TEUCHOS_UNIT_TEST(MPASInterface, DeprecatedAPI)
{
  MockMPAS mpas(comm, slabMesh(PartitionType::Blocks, 1, false));
  InterfaceSession session;
  writeAlbanyInput(slidingSlabInput);
  const auto state = slidingSlabState(1);

  // The values the deprecated overload of velocity_solver_set_physical_parameters uses
  auto p = slidingSlabParameters();
  p.meltingTemperature = 273.15;
  p.secondsPerYear = 3.1536e7;
  p.gasConstant = 8.314;

  FieldData fCurrent, fDeprecated;
  int error = 0;
  setPhysicalParameters(p);
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlidingSlabFields(mpas, state, 1, fCurrent);
    error = std::max(error, mpas.solve(fCurrent, true, 1.0));
  }
  setPhysicalParametersDeprecated(p);
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlidingSlabFields(mpas, state, 1, fDeprecated);
    error = std::max(error, mpas.solveDeprecated(fDeprecated, true, 1.0));
  }
  TEST_EQUALITY(globalMax(error), 0);

  const double velDiff  = globalRelDiff(fDeprecated.velocityOnVertices, fCurrent.velocityOnVertices);
  const double betaDiff = globalRelDiff(fDeprecated.beta, fCurrent.beta);
  const double dissDiff = globalRelDiff(fDeprecated.dissipationHeatOnPrisms, fCurrent.dissipationHeatOnPrisms);
  const double bfDiff   = globalRelDiff(fDeprecated.bodyForceOnBasalCell, fCurrent.bodyForceOnBasalCell);
  out << "  relative differences: velocity " << velDiff << ", beta " << betaDiff
      << ", dissipation " << dissDiff << ", body force " << bfDiff << "\n";
  TEST_COMPARE(velDiff, <, 1e-10);
  TEST_COMPARE(betaDiff, <, 1e-12);
  TEST_COMPARE(dissDiff, <, 1e-8);
  TEST_COMPARE(bfDiff, <, 1e-12);
}

// The logically extruded mesh and the 3D STK mesh discretize the problem in the same way, so they give the
// same results (up to the solver tolerances, since the dofs are numbered differently), with the FO and the
// depth-integrated models.
TEUCHOS_UNIT_TEST(MPASInterface, MeshMethodsAgree)
{
  if (!isDefaultMethodRun()) {
    out << "  Skipped: this test sets the Discretization Method itself.\n";
    return;
  }
  for (bool depthIntegrated : {false, true}) {
    out << "  " << (depthIntegrated ? "Depth-integrated" : "FO") << " model:\n";
    FieldData fSTK, fExtruded;
    int error = 0;
    for (const std::string method : {"STKExtruded", "Extruded"}) {
      MockMPAS mpas(comm, slabMesh(PartitionType::Scattered, 0, false));
      InterfaceSession session;
      setPhysicalParameters(slidingSlabParameters());
      writeAlbanyInput(slidingSlabInput + (depthIntegrated ? "    Depth Integrated Model: true\n" : ""), method);
      auto& f = (method=="Extruded") ? fExtruded : fSTK;
      if (mpas.isActive()) {
        mpas.computeGrid();
        mpas.extrudeGrid();
        setSlidingSlabFields(mpas, slidingSlabState(1), 1, f);
        error = std::max(error, mpas.solve(f, true, 1.0));
      }
    }
    TEST_EQUALITY(globalMax(error), 0);

    // The grids of the two runs are the same (same options and seed), and so are the layouts of the fields
    const double velDiff  = globalRelDiff(fExtruded.velocityOnVertices, fSTK.velocityOnVertices);
    const double betaDiff = globalRelDiff(fExtruded.beta, fSTK.beta);
    const double bfDiff   = globalRelDiff(fExtruded.bodyForceOnBasalCell, fSTK.bodyForceOnBasalCell);
    const double dissDiff = globalRelDiff(fExtruded.dissipationHeatOnPrisms, fSTK.dissipationHeatOnPrisms);
    out << "    relative differences: velocity " << velDiff << ", beta " << betaDiff
        << ", body force " << bfDiff << ", dissipation " << dissDiff << "\n";
    TEST_COMPARE(velDiff, <, 1e-7);
    TEST_COMPARE(betaDiff, <, 1e-10);
    TEST_COMPARE(bfDiff, <, 1e-10);
    TEST_COMPARE(dissDiff, <, 1e-6);
  }
}

// Values of a vertex field at the owned vertices of all ranks, by global vertex id. Collective.
std::map<int,double> gatherOwnedVertexValues (const MockMPAS& mpas, const std::vector<double>& field)
{
  std::vector<double> local;  // (id, value) pairs
  if (mpas.isActive()) {
    for (int ib=0; ib<mpas.numVertices(); ++ib) {
      if (mpas.ownsVertex(ib)) {
        local.push_back(mpas.indexToVertexID()[ib]);
        local.push_back(field[ib]);
      }
    }
  }
  int size;
  MPI_Comm_size(comm, &size);
  int n = local.size();
  std::vector<int> counts(size), displs(size,0);
  MPI_Allgather(&n, 1, MPI_INT, counts.data(), 1, MPI_INT, comm);
  for (int r=1; r<size; ++r) displs[r] = displs[r-1]+counts[r-1];
  std::vector<double> all(displs.back()+counts.back());
  MPI_Allgatherv(local.data(), n, MPI_DOUBLE, all.data(), counts.data(), displs.data(), MPI_DOUBLE, comm);
  std::map<int,double> values;
  for (std::size_t i=0; i<all.size(); i+=2) {
    values[static_cast<int>(all[i])] = all[i+1];
  }
  return values;
}

// Max difference, over all ranks, between the field at every local vertex (owned or not) and the reference
// values, relative to the max of the reference values
double vertexValuesRelDiff (const MockMPAS& mpas, const std::vector<double>& field, const std::map<int,double>& ref)
{
  double scale = 0, diff = 0;
  for (const auto& kv : ref) scale = std::max(scale, std::abs(kv.second));
  if (mpas.isActive()) {
    for (int ib=0; ib<mpas.numVertices(); ++ib) {
      diff = std::max(diff, std::abs(field[ib] - ref.at(mpas.indexToVertexID()[ib])));
    }
  }
  return globalMax(diff)/std::max(scale, std::numeric_limits<double>::min());
}

// The beta returned to MPAS must not depend on the partition, even when Albany computes it with element
// quantities: with Regularized Coulomb and "Viscosity Flow Rate", the nodal beta uses the flow rate of the
// basal cell, and here the temperature changes from triangle to triangle. The same problem is solved on two
// partitions with different local orderings, and beta at every copy of every vertex is compared with beta
// at the owned vertices of the first partition. (In serial, the element order may be the same for the two
// orderings, so the parallel run is the meaningful one.)
TEUCHOS_UNIT_TEST(MPASInterface, BetaIndependentOfPartition)
{
  auto options1 = slabMesh(PartitionType::Strips, 0, false);
  auto options2 = slabMesh(PartitionType::Scattered, 0, false);
  options2.seed = 777;
  // Both must outlive the session. They have the same (non-idle) ranks, so that the Albany objects built
  // for the first grid are released by all the ranks that built them.
  MockMPAS mpas1(comm, options1);
  MockMPAS mpas2(comm, options2);
  InterfaceSession session;
  setPhysicalParameters(slidingSlabParameters());
  writeAlbanyInput(slidingSlabInput +
    "    LandIce BCs:\n"
    "      BC 0:\n"
    "        Basal Friction Coefficient:\n"
    "          Type: Regularized Coulomb\n"
    "          Power Exponent: 1.0\n");

  auto solve = [](MockMPAS& mpas, FieldData& f) {
    if (!mpas.isActive()) return 0;
    mpas.computeGrid();
    mpas.extrudeGrid();
    setSlidingSlabFields(mpas, slidingSlabState(1), 1, f);
    mpas.setVertexField(f.mu, [](const Point& p) { return 0.02*(1.0 + 0.2*std::cos(0.5*p.x)); });
    mpas.setVertexField(f.bedRoughness, [](const Point& p) { return 1e-3*(1.0 + 0.3*std::sin(0.8*p.y)); });
    mpas.setPrismField(f.temperatureOnPrisms, [](const Point& c, double sigma, int) {
      return 250.0 + 4.0*std::sin(2.1*c.x)*std::cos(1.7*c.y) + 3.0*sigma;
    });
    return mpas.solve(f, true, 1.0);
  };
  FieldData f1, f2;
  const int error1 = solve(mpas1, f1);
  const int error2 = solve(mpas2, f2);
  TEST_EQUALITY(globalMax(std::max(error1,error2)), 0);

  // The two solves agree (up to the solver tolerances), so beta differences can only come from its evaluation
  auto basalSpeed = [](const MockMPAS& mpas, const FieldData& f) {
    std::vector<double> speed(mpas.numVertices());
    for (int ib=0; ib<mpas.numVertices(); ++ib) {
      const int j = mpas.vertex3D(0,ib);
      speed[ib] = std::hypot(f.velocityOnVertices[j], f.velocityOnVertices[j+mpas.numVertices3D()]);
    }
    return speed;
  };
  const auto speedRef = gatherOwnedVertexValues(mpas1, basalSpeed(mpas1,f1));
  const double speedDiff = vertexValuesRelDiff(mpas2, basalSpeed(mpas2,f2), speedRef);

  const auto betaRef = gatherOwnedVertexValues(mpas1, f1.beta);
  const double betaDiff1 = vertexValuesRelDiff(mpas1, f1.beta, betaRef);
  const double betaDiff2 = vertexValuesRelDiff(mpas2, f2.beta, betaRef);
  out << "  relative differences: basal speed " << speedDiff << ", beta (copies, same partition) " << betaDiff1
      << ", beta (other partition) " << betaDiff2 << "\n";
  TEST_COMPARE(speedDiff, <, 1e-8);
  TEST_COMPARE(betaDiff1, <, 1e-12);
  TEST_COMPARE(betaDiff2, <, 1e-7);
}

} // anonymous namespace
