//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// Regression driver for the MPAS-Albany interface.
//
// Usage: MockMALI <mock_mpas.yaml>
//
// It mimics a MALI run: it builds the FE mesh and the fields of a test case with MockMPAS, calls the
// interface functions in the order MALI does (velocity_solver_init_mpi, set_physical_parameters,
// compute_2d_grid, extrude_3d_grid, one solve_fo per "time step", export_fo_velocity, finalize) and
// computes metrics of the fields Albany returns to MPAS (velocity, beta, effective pressure,
// dissipation heat, body force). The metrics are compared with the baseline in the "Regression"
// sublist of the input file. If the baseline is missing, the driver prints it, ready to be pasted in
// the input file, and fails.
//
// Like MALI, Albany reads its own input from albany_input.yaml in the working directory. As in MALI,
// Kokkos is initialized and finalized by the interface.
//
// The metrics are averages over the owned vertices (or over the triangles, which are not shared), so
// they do not depend on the partition: serial and parallel runs can share the same baseline.

#include "MockMPAS.hpp"

#include <Teuchos_Array.hpp>
#include <Teuchos_DefaultMpiComm.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>

#include <algorithm>
#include <cmath>
#include <exception>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace mock_mpas;

constexpr double pi = 3.14159265358979323846;

// ========================================== Test cases ========================================== //

struct TestCase {
  MeshOptions mesh;
  // Sets the inputs of solve number k (0-based). The velocity already in f (the solution of the
  // previous solve, as in MALI) is used as initial guess and must hold the Dirichlet values.
  std::function<void(const MockMPAS&, int k, FieldData&)> setFields;
};

// Minimum thickness MALI uses at the margin of the FE mesh [km]
constexpr double minThickness = 1e-3;

// Land-based dome, as in MALI's dome test: H = H0*sqrt(1-(r/R)^2) on a flat bed. The FE mesh contains
// the triangles touching the ice, and its whole boundary is ice margin (lateral BC), with no Dirichlet
// nodes. Power-law sliding with exponent 1 and effective pressure from MPAS; temperature and stiffening
// factor vary in space. At each solve the dome gets slightly thinner and warmer.
TestCase domeCase (Teuchos::ParameterList& tcParams, const PhysicalParameters& phys)
{
  const int nx         = tcParams.get("Number Of Grid Vertices X", 31);
  const int ny         = tcParams.get("Number Of Grid Vertices Y", 31);
  const double dx      = tcParams.get("Grid Spacing", 2.0);                    // [km]
  const double R       = tcParams.get("Dome Radius", 60.0*std::sqrt(0.125));   // [km]
  const double H0      = tcParams.get("Dome Thickness", 2.0*std::sqrt(0.125)); // [km]
  const double xc      = 0.5*(nx-1)*dx;
  const double yc      = 0.5*(ny-1)*dx;

  auto thickness = [=](const Point& p, int k) {
    const double r = std::hypot(p.x-xc, p.y-yc);
    return r<R ? H0*std::sqrt(1.0-r*r/(R*R))*(1.0-0.03*k) : 0.0;
  };

  TestCase tc;
  tc.mesh.nx = nx;
  tc.mesh.ny = ny;
  tc.mesh.dx = dx;
  tc.mesh.isActiveTriangle = [=](const std::array<Point,3>& v) {
    for (const auto& p : v) {
      if (thickness(p,0)>0) return true;
    }
    return false;
  };
  tc.mesh.isIceMarginEdge = [](const Point&, const Point&) { return true; };

  const double rhoG = phys.iceDensity*phys.gravity;
  tc.setFields = [=](const MockMPAS& mpas, int k, FieldData& f) {
    mpas.setVertexField(f.thickness,     [&](const Point& p) { return std::max(thickness(p,k), minThickness); });
    mpas.setVertexField(f.bedTopography, [&](const Point&)   { return 0.0; });
    mpas.setVertexField(f.elevation,     [&](const Point& p) { return std::max(thickness(p,k), minThickness); });
    mpas.setVertexField(f.mu, [&](const Point& p) {
      return 2e-3*(1.0 + 0.5*std::cos(pi*(p.x-xc)/R)*std::cos(pi*(p.y-yc)/R));
    });
    mpas.setVertexField(f.effectivePressure, [&](const Point& p) {
      return 0.8*rhoG*std::max(thickness(p,k), minThickness);   // [kPa]
    });
    mpas.setVertexField(f.stiffeningFactor, [&](const Point& p) { return 1.0 + 0.2*std::sin(pi*(p.x-xc)/R); });
    mpas.setPrismField(f.temperatureOnPrisms, [&](const Point& p, double sigma, int) {
      const double r = std::min(std::hypot(p.x-xc, p.y-yc)/R, 1.0);
      return 258.0 - 15.0*sigma - 5.0*r + 1.0*k;
    });
  };
  return tc;
}

// Marine ice sheet in a channel: the bed deepens along x and the ice goes afloat before reaching the
// calving front at the end of the channel (ice margin, lateral BC with ocean pressure). No slip at the
// ice divide (x=0) and on the channel walls. The sliding law is set in albany_input.yaml (e.g.,
// Regularized Coulomb with hydrostatic effective pressure, which Albany computes and returns to MPAS).
// At each solve the ice gets thinner, so the grounding line moves.
TestCase marineCase (Teuchos::ParameterList& tcParams, const PhysicalParameters& phys)
{
  const int nx    = tcParams.get("Number Of Grid Vertices X", 21);
  const int ny    = tcParams.get("Number Of Grid Vertices Y", 11);
  const double dx = tcParams.get("Grid Spacing", 2.0);   // [km]
  const double Lx = (nx-1)*dx;
  const double Ly = (ny-1)*dx;
  const double eps = 1e-6*dx;

  TestCase tc;
  tc.mesh.nx = nx;
  tc.mesh.ny = ny;
  tc.mesh.dx = dx;
  tc.mesh.isDirichletNode = [=](const Point& p, int, bool) {
    return p.x<eps || p.y<eps || p.y>Ly-eps;
  };
  tc.mesh.isIceMarginEdge = [=](const Point& p0, const Point& p1) {
    return p0.x>Lx-eps && p1.x>Lx-eps;
  };

  const double ri = phys.iceDensity, rw = phys.oceanDensity;
  tc.setFields = [=](const MockMPAS& mpas, int k, FieldData& f) {
    auto bed   = [&](const Point& p) { return 0.3 - 0.9*p.x/Lx; };
    auto thick = [&](const Point& p) { return (1.0 - 0.6*p.x/Lx)*(1.0-0.02*k); };
    mpas.setVertexField(f.bedTopography, bed);
    mpas.setVertexField(f.thickness, thick);
    mpas.setVertexField(f.elevation, [&](const Point& p) {
      const bool floating = ri*thick(p) < -rw*bed(p);
      return floating ? (1.0-ri/rw)*thick(p) : bed(p)+thick(p);
    });
    mpas.setVertexField(f.mu, [&](const Point& p) { return 0.3*(1.0 + 0.2*std::cos(pi*p.y/Ly)); });
    mpas.setVertexField(f.bedRoughness, [&](const Point& p) { return 5e-3*(1.0 + 0.3*std::sin(2*pi*p.x/Lx)); });
    mpas.setVertexField(f.effectivePressure, [&](const Point&) { return 0.0; });
    mpas.setPrismField(f.temperatureOnPrisms, [&](const Point& p, double sigma, int) {
      return 258.0 - 10.0*sigma + 2.0*p.y/Ly;
    });
    // Zero velocity at the Dirichlet nodes, previous solution elsewhere
    const int nV3D = mpas.numVertices3D();
    for (int ib=0; ib<mpas.numVertices(); ++ib) {
      for (int il=0; il<=mpas.numLayers(); ++il) {
        if (mpas.isDirichletNode(il,ib)) {
          f.velocityOnVertices[mpas.vertex3D(il,ib)] = f.velocityOnVertices[mpas.vertex3D(il,ib)+nV3D] = 0.0;
        }
      }
    }
  };
  return tc;
}

// ============================================ Metrics ============================================ //

using Metrics = std::vector<std::pair<std::string,double>>;

// Metrics of the fields returned by Albany. Collective on the world communicator.
Metrics computeMetrics (const MockMPAS& mpas, const FieldData& f)
{
  enum { SurfSpeed, BasalSpeed, SpeedSq, Beta, EffPress, Owned, Owned3D, Diss, BodyForce, Prisms, Triangles, NumSums };
  std::vector<double> sums(NumSums, 0.0), gsums(NumSums);
  double maxSpeed = 0, gmaxSpeed;
  if (mpas.isActive()) {
    const int nV3D = mpas.numVertices3D();
    const int nL = mpas.numLayers();
    auto speed = [&](int il, int ib) {
      const int j = mpas.vertex3D(il,ib);
      return std::hypot(f.velocityOnVertices[j], f.velocityOnVertices[j+nV3D]);
    };
    for (int ib=0; ib<mpas.numVertices(); ++ib) {
      for (int il=0; il<=nL; ++il) maxSpeed = std::max(maxSpeed, speed(il,ib));
      if (!mpas.ownsVertex(ib)) continue;
      sums[SurfSpeed]  += speed(nL,ib);
      sums[BasalSpeed] += speed(0,ib);
      for (int il=0; il<=nL; ++il) sums[SpeedSq] += std::pow(speed(il,ib),2);
      sums[Beta]     += f.beta[ib];
      sums[EffPress] += f.effectivePressure[ib];
      sums[Owned]    += 1;
      sums[Owned3D]  += nL+1;
    }
    for (double d : f.dissipationHeatOnPrisms) sums[Diss] += d;
    for (double b : f.bodyForceOnBasalCell) sums[BodyForce] += b;
    sums[Prisms]    += mpas.numPrisms();
    sums[Triangles] += mpas.numTriangles();
  }
  MPI_Allreduce(sums.data(), gsums.data(), NumSums, MPI_DOUBLE, MPI_SUM, mpas.worldComm());
  MPI_Allreduce(&maxSpeed, &gmaxSpeed, 1, MPI_DOUBLE, MPI_MAX, mpas.worldComm());

  return {
    {"Max Speed",               gmaxSpeed},
    {"Mean Surface Speed",      gsums[SurfSpeed]/gsums[Owned]},
    {"Mean Basal Speed",        gsums[BasalSpeed]/gsums[Owned]},
    {"RMS Speed",               std::sqrt(gsums[SpeedSq]/gsums[Owned3D])},
    {"Mean Beta",               gsums[Beta]/gsums[Owned]},
    {"Mean Effective Pressure", gsums[EffPress]/gsums[Owned]},
    {"Mean Dissipation Heat",   gsums[Diss]/gsums[Prisms]},
    {"Mean Body Force",         gsums[BodyForce]/gsums[Triangles]}
  };
}

// =========================================== Driver =========================================== //

PhysicalParameters readPhysicalParameters (Teuchos::ParameterList& pl)
{
  PhysicalParameters p;
  p.gravity                = pl.get("Gravity", p.gravity);
  p.iceDensity             = pl.get("Ice Density", p.iceDensity);
  p.oceanDensity           = pl.get("Ocean Density", p.oceanDensity);
  p.seaLevel               = pl.get("Sea Level", p.seaLevel);
  p.flowParamA             = pl.get("Flow Rate Factor", p.flowParamA);  // [k^-1 kPa^-n yr^-1]
  p.flowLawExponent        = pl.get("Flow Law Exponent", p.flowLawExponent);
  p.dynamicThickness       = pl.get("Dynamic Thickness", p.dynamicThickness);
  p.useGLP                 = pl.get("Use GLP", p.useGLP);
  p.clausiusClapeyronCoeff = pl.get("Clausius-Clapeyron Coefficient", p.clausiusClapeyronCoeff);
  p.meltingTemperature     = pl.get("Melting Temperature", p.meltingTemperature);
  p.secondsPerYear         = pl.get("Seconds Per Year", p.secondsPerYear);
  p.gasConstant            = pl.get("Gas Constant", p.gasConstant);
  return p;
}

int run (const std::string& inputFile)
{
  const MPI_Comm world = MPI_COMM_WORLD;
  int rank, size;
  MPI_Comm_rank(world, &rank);
  MPI_Comm_size(world, &size);
  const bool root = rank==0;

  Teuchos::ParameterList params;
  {
    Teuchos::MpiComm<int> tcomm(world);
    Teuchos::updateParametersFromYamlFileAndBroadcast(inputFile, Teuchos::ptrFromRef(params), tcomm);
  }
  auto& mockParams = params.sublist("Mock MPAS");
  const auto phys = readPhysicalParameters(params.sublist("Physical Parameters"));

  const std::string caseName = mockParams.get<std::string>("Test Case");
  auto& tcParams = mockParams.sublist("Test Case Parameters");
  TestCase tc;
  if (caseName=="Dome") {
    tc = domeCase(tcParams, phys);
  } else if (caseName=="Marine") {
    tc = marineCase(tcParams, phys);
  } else {
    if (root) std::cerr << "Error! Unknown test case '" << caseName << "'. Valid cases: Dome, Marine.\n";
    return 1;
  }

  auto& mesh = tc.mesh;
  mesh.numLayers = mockParams.get("Number Of Layers", 5);
  if (mockParams.isParameter("Levels")) {
    const auto levels = mockParams.get<Teuchos::Array<double>>("Levels");
    mesh.levels.assign(levels.begin(), levels.end());
  }
  mesh.partition = partitionTypeFromString(mockParams.get<std::string>("Partition", "Strips"));
  mesh.numIdleRanks = mockParams.get("Idle Ranks", 0);
  mesh.shuffleLocalOrdering = mockParams.get("Shuffle Local Ordering", true);
  mesh.seed = mockParams.get("Seed", 12345);
  const int numSolves = mockParams.get("Number Of Solves", 1);
  const double deltat = mockParams.get("Time Step", 1.0);   // [yr]
  const bool writeExodus = mockParams.get("Write Exodus Output", false);

  MockMPAS mpas(world, mesh);
  const auto msg = mpas.checkConsistency();
  if (!msg.empty()) {
    std::cerr << msg;
    return 1;
  }
  if (root) {
    std::cout << "MockMALI: test case " << caseName << ", " << mpas.numGlobalTriangles() << " triangles, "
              << mpas.numGlobalVertices() << " vertices, " << mesh.numLayers << " layers, "
              << size << " ranks (" << mpas.numIdleRanks() << " without triangles)\n";
  }

  // ---- The MALI run ---- //
  std::vector<Metrics> metrics(numSolves);
  int error = 0;
  velocity_solver_init_mpi__(world);
  setPhysicalParameters(phys);
  FieldData f;
  if (mpas.isActive()) {
    mpas.computeGrid();
    mpas.extrudeGrid();
    mpas.allocate(f);
  }
  for (int k=0; k<numSolves; ++k) {
    if (mpas.isActive()) {
      tc.setFields(mpas, k, f);
      error = std::max(error, mpas.solve(f, k==0, deltat));
    }
    metrics[k] = computeMetrics(mpas, f);
  }
  if (writeExodus && mpas.isActive()) {
    mpas.exportVelocity();
  }
  velocity_solver_finalize__();

  int gerror;
  MPI_Allreduce(&error, &gerror, 1, MPI_INT, MPI_MAX, world);

  // ---- Comparison with the baseline ---- //
  auto& regParams = params.sublist("Regression");
  const double relTol = regParams.get("Relative Tolerance", 1.0e-5);
  const double absTol = regParams.get("Absolute Tolerance", 1.0e-8);
  // Optional relative tolerances for specific metrics (e.g., for metrics that depend on the partition)
  const bool hasMetricTols = regParams.isSublist("Metric Relative Tolerances");
  auto& metricTols = regParams.sublist("Metric Relative Tolerances");
  const bool hasBaseline = regParams.isSublist("Baseline");
  auto& baseline = regParams.sublist("Baseline");

  int numFailed = 0, numMissing = 0;
  std::ostringstream report;
  report << std::setprecision(10) << std::scientific;
  for (int k=0; k<numSolves; ++k) {
    const std::string solveName = "Solve " + std::to_string(k);
    report << "  " << solveName << "\n";
    for (const auto& m : metrics[k]) {
      report << "    " << std::left << std::setw(26) << m.first << std::right << std::setw(18) << m.second;
      if (!baseline.isSublist(solveName) || !baseline.sublist(solveName).isParameter(m.first)) {
        report << "   (no baseline)\n";
        ++numMissing;
        continue;
      }
      const double ref = baseline.sublist(solveName).get<double>(m.first);
      const double diff = std::abs(m.second-ref);
      const double tol = metricTols.isParameter(m.first) ? metricTols.get<double>(m.first) : relTol;
      const bool pass = diff <= absTol + tol*std::abs(ref);
      report << "   baseline " << std::setw(18) << ref << "   rel. diff " << std::setw(18)
             << diff/std::max(std::abs(ref),std::numeric_limits<double>::min()) << (pass ? "   passed" : "   FAILED") << "\n";
      numFailed += pass ? 0 : 1;
    }
  }

  if (root) {
    std::cout << "\nMockMALI results (relative tolerance " << relTol << ", absolute tolerance " << absTol << "):\n"
              << report.str();
    if (gerror) {
      std::cout << "\nFAILED: Albany reported an error (e.g., the nonlinear solver did not converge).\n";
    }
    if (numFailed>0) {
      std::cout << "\nFAILED: " << numFailed << " metric(s) differ from the baseline.\n";
    }
    if (numMissing>0) {
      std::cout << "\nFAILED: " << numMissing << " metric(s) have no baseline"
                << (hasBaseline ? "" : " (no \"Baseline\" sublist)") << ".\n"
                << "To set the baseline, replace the \"Regression\" sublist of " << inputFile << " with:\n\n"
                << std::setprecision(15) << std::scientific
                << "  Regression:\n"
                << "    Relative Tolerance: " << relTol << "\n"
                << "    Absolute Tolerance: " << absTol << "\n";
      if (hasMetricTols) {
        std::cout << "    Metric Relative Tolerances:\n";
        for (auto it=metricTols.begin(); it!=metricTols.end(); ++it) {
          std::cout << "      " << metricTols.name(it) << ": " << metricTols.get<double>(metricTols.name(it)) << "\n";
        }
      }
      std::cout << "    Baseline:\n";
      for (int k=0; k<numSolves; ++k) {
        std::cout << "      Solve " << k << ":\n";
        for (const auto& m : metrics[k]) {
          std::cout << "        " << m.first << ": " << m.second << "\n";
        }
      }
      std::cout << std::endl;
    }
    if (!gerror && numFailed==0 && numMissing==0) {
      std::cout << "\nPASSED\n";
    }
  }

  return (gerror || numFailed>0 || numMissing>0) ? 1 : 0;
}

} // anonymous namespace

int main (int argc, char** argv)
{
  MPI_Init(&argc, &argv);
  int status = 1;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (argc!=2) {
    if (rank==0) std::cerr << "Usage: " << argv[0] << " <mock_mpas.yaml>\n";
  } else {
    try {
      status = run(argv[1]);
    } catch (const std::exception& e) {
      std::cerr << "MockMALI: exception on rank " << rank << ":\n" << e.what() << std::endl;
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }
  MPI_Finalize();
  return status;
}
