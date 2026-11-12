//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef LANDICE_MOCK_MPAS_HPP
#define LANDICE_MOCK_MPAS_HPP

// A mock of the MPAS (MALI) side of the MPAS-Albany interface
// (src/landIce/interfaceWithMPAS/Interface.cpp).
//
// MALI calls Albany through the velocity_solver_* functions, passing the FE mesh (the triangulation
// dual to the MPAS Voronoi mesh) and the fields with the data layout of
// components/mpas-albany-landice/src/mode_forward/Interface_velocity_solver.cpp in E3SM.
// This mock builds that same data from a structured triangulation, so the interface can be tested
// without MPAS:
//  - global ids are 1-based and, as in MALI, are those of a larger grid, so they are not contiguous
//    and the global strides are larger than the number of entities;
//  - each rank owns a set of triangles and stores all the vertices of its triangles; vertices shared
//    by several ranks are owned by one of the ranks having a triangle containing them;
//  - ranks without triangles are left out of the reduced communicator, as MALI does;
//  - optionally, the local ordering of triangles and vertices is shuffled, so that the interface
//    cannot rely on it being sorted.
//
//  Duality between MPAS and Albany entities:
//     MPAS     |  Albany (this mock)
//    ----------|--------------------
//     cell     |  vertex (node)
//     vertex   |  triangle
//     edge     |  edge

#include <mpi.h>

#include <array>
#include <functional>
#include <limits>
#include <string>
#include <vector>

// ===================================================================================================
// Declarations of the functions defined in src/landIce/interfaceWithMPAS/Interface.cpp.
// That file has no header (MALI declares them in Interface_velocity_solver.hpp). They are declared here
// with the same signatures (including the "__" suffix added by the macros in Interface.cpp), so a
// signature change that would break MALI's build also breaks the build of these tests.
// ===================================================================================================

int velocity_solver_init_mpi__ (MPI_Comm comm);

void velocity_solver_finalize__ ();

void velocity_solver_compute_2d_grid__ (MPI_Comm reducedComm);

void velocity_solver_set_physical_parameters__ (double const& gravity, double const& ice_density, double const& ocean_density,
    double const& sea_level, double const& flowParamA, double const& flowLawExponent, double const& dynamic_thickness,
    bool const& use_GLP, double const& clausiusClapeyoronCoeff, double const& melting_temperature,
    double const& seconds_per_year, double const& gas_constant);

// Deprecated overload, still called by E3SM's MALI
void velocity_solver_set_physical_parameters__ (double const& gravity, double const& ice_density, double const& ocean_density,
    double const& sea_level, double const& flowParamA, double const& flowLawExponent, double const& dynamic_thickness,
    bool const& use_GLP, double const& clausiusClapeyoronCoeff);

void velocity_solver_extrude_3d_grid__ (int nLayers, int globalTrianglesStride,
    int globalVerticesStride, int globalEdgesStride, int Ordering, MPI_Comm reducedComm,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& vertexProcIDs,
    const std::vector<double>& verticesCoords,
    const std::vector<int>& verticesOnTria,
    const std::vector<std::vector<int>> procsSharingVertices,
    const std::vector<bool>& isBoundaryEdge,
    const std::vector<int>& trianglesOnEdge,
    const std::vector<int>& verticesOnEdge,
    const std::vector<int>& indexToEdgeID,
    const std::vector<int>& indexToTriangleID,
    const std::vector<int>& dirichletNodesIds,
    const std::vector<int>& iceMarginEdgesIds);

void velocity_solver_solve_fo__ (int nLayers, int globalVerticesStride,
    int globalTrianglesStride, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double minBeta,
    const std::vector<double>& regulThk,
    const std::vector<double>& levelsNormalizedThickness,
    const std::vector<double>& elevationData,
    const std::vector<double>& thicknessData,
          std::vector<double>& betaData,
    const std::vector<double>& bedTopographyData,
    const std::vector<double>& smbData,
    const std::vector<double>& stiffeningFactorData,
          std::vector<double>& effectivePressureData,
    const std::vector<double>& muData,
    const std::vector<double>& bedRoughnessData,
    const std::vector<double>& bulkFrictionData,
    const std::vector<double>& basalDebrisData,
    const std::vector<double>& temperatureDataOnPrisms,
    std::vector<double>& bodyForceMagnitudeOnBasalCell,
    std::vector<double>& dissipationHeatOnPrisms,
    std::vector<double>& velocityOnVertices,
    int& error,
    const double& deltat);

// Deprecated overload, still called by E3SM's MALI
void velocity_solver_solve_fo__ (int nLayers, int globalVerticesStride,
    int globalTrianglesStride, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double minBeta,
    const std::vector<double>& regulThk,
    const std::vector<double>& levelsNormalizedThickness,
    const std::vector<double>& elevationData,
    const std::vector<double>& thicknessData,
          std::vector<double>& betaData,
    const std::vector<double>& bedTopographyData,
    const std::vector<double>& smbData,
    const std::vector<double>& stiffeningFactorData,
    const std::vector<double>& effectivePressureData,
    const std::vector<double>& muData,
    const std::vector<double>& temperatureDataOnPrisms,
    std::vector<double>& bodyForceMagnitudeOnBasalCell,
    std::vector<double>& dissipationHeatOnPrisms,
    std::vector<double>& velocityOnVertices,
    int& error,
    const double& deltat);

void velocity_solver_export_fo_velocity__ (MPI_Comm reducedComm);

namespace mock_mpas {

// Value MALI uses for "no entity"
constexpr int NotAnId = std::numeric_limits<int>::max();

struct Point {
  double x, y;
};

enum class PartitionType {
  Strips,     // contiguous strips of triangles along x
  Blocks,     // 2D blocks (strips along x, then along y)
  Scattered   // triangles dealt round robin after a random shuffle: many shared vertices
};

PartitionType partitionTypeFromString (const std::string& name);

struct MeshOptions {
  // Structured grid of (nx x ny) vertices (MPAS cells), with spacing dx [km], starting at (x0,y0).
  // Each quad is split in two triangles, alternating the diagonal.
  int nx = 11;
  int ny = 11;
  double dx = 1.0;
  double x0 = 0.0;
  double y0 = 0.0;

  // Vertical discretization. levels are the normalized levels, 0 at the bed and 1 at the surface
  // (MALI's levelsNormalizedThickness). If empty, uniform levels are used.
  int numLayers = 5;
  std::vector<double> levels;

  // 0: layer-wise ordering of the 3D entities (the only one MALI uses), 1: column-wise
  int ordering = 0;

  PartitionType partition = PartitionType::Strips;

  // Number of ranks (the first ones) that own no triangle, which MALI leaves out of the reduced
  // communicator. Ignored in serial runs (at least one rank must own triangles).
  int numIdleRanks = 0;

  bool shuffleLocalOrdering = true;
  unsigned int seed = 12345;

  // A triangle of the grid belongs to the FE mesh if this returns true. Default: all triangles.
  std::function<bool(const std::array<Point,3>& vertices)> isActiveTriangle;

  // Whether the 3D node at level il (0 at the bed) of the vertex at p is a Dirichlet node.
  // isBoundaryVertex is true if the vertex is on the boundary of the FE mesh. Default: none.
  std::function<bool(const Point& p, int il, bool isBoundaryVertex)> isDirichletNode;

  // Whether a boundary edge of the FE mesh belongs to the ice margin (where the lateral BC is applied).
  // Default: all boundary edges.
  std::function<bool(const Point& p0, const Point& p1)> isIceMarginEdge;
};

// Physical parameters, in the units MALI passes them to Albany. In particular, flowParamA is in
// [k^-1 kPa^-n yr^-1], i.e., MALI's A [Pa^-n s^-1] times 1000^(n+1) times the seconds per year.
struct PhysicalParameters {
  double gravity = 9.80616;
  double iceDensity = 910.0;
  double oceanDensity = 1028.0;
  double seaLevel = 0.0;
  double flowParamA = 1.0e-4;
  double flowLawExponent = 3.0;
  double dynamicThickness = 1.0e-2;
  bool   useGLP = true;
  double clausiusClapeyronCoeff = 9.7546e-8;
  double meltingTemperature = 273.15;
  double secondsPerYear = 3.1536e7;
  double gasConstant = 8.314;
};

// Calls the current (12 arguments) overload of velocity_solver_set_physical_parameters
void setPhysicalParameters (const PhysicalParameters& p);
// Calls the deprecated (9 arguments) overload, which does not take the last three parameters
void setPhysicalParametersDeprecated (const PhysicalParameters& p);

// Inputs and outputs of velocity_solver_solve_fo, with MALI's layout and units ([km] for lengths,
// [kPa] for the effective pressure, [m/yr] for velocities, [K] for the temperature).
//  - 2D fields have one entry per local vertex;
//  - prism fields (temperature, dissipation) have entry prism(il,it) for layer il and triangle it;
//  - velocityOnVertices holds the x component at vertex3D(il,ib) and the y component at
//    vertex3D(il,ib) + numVertices3D. On input it contains the initial guess and the Dirichlet values.
// Empty optional fields are not passed to Albany (as MALI does when a field is not available).
struct FieldData {
  std::vector<double> regulThickness;
  std::vector<double> elevation;
  std::vector<double> thickness;
  std::vector<double> beta;
  std::vector<double> bedTopography;
  std::vector<double> smb;
  std::vector<double> stiffeningFactor;
  std::vector<double> effectivePressure;
  std::vector<double> mu;
  std::vector<double> bedRoughness;   // optional
  std::vector<double> bulkFriction;   // optional
  std::vector<double> basalDebris;    // optional
  std::vector<double> temperatureOnPrisms;
  std::vector<double> bodyForceOnBasalCell;
  std::vector<double> dissipationHeatOnPrisms;
  std::vector<double> velocityOnVertices;
};

class MockMPAS {
public:
  // Builds the FE mesh and its partition. Collective on worldComm.
  MockMPAS (MPI_Comm worldComm, const MeshOptions& options);
  // Frees the reduced communicator. Albany keeps using it (it does not duplicate it) until
  // velocity_solver_finalize, so the object must outlive that call.
  ~MockMPAS ();

  MockMPAS (const MockMPAS&) = delete;
  MockMPAS& operator= (const MockMPAS&) = delete;

  // ---- MPI ---- //
  // Whether this rank owns triangles (only these ranks call the grid and solve functions)
  bool isActive () const { return m_active; }
  MPI_Comm worldComm () const { return m_worldComm; }
  // MPI_COMM_NULL on idle ranks
  MPI_Comm reducedComm () const { return m_reducedComm; }
  int numIdleRanks () const { return m_numIdleRanks; }

  // ---- Sizes (local) ---- //
  int numLayers () const { return m_numLayers; }
  int ordering () const { return m_ordering; }
  int numVertices () const { return static_cast<int>(m_indexToVertexID.size()); }
  int numTriangles () const { return static_cast<int>(m_indexToTriangleID.size()); }
  int numEdges () const { return static_cast<int>(m_indexToEdgeID.size()); }
  int numVertices3D () const { return (m_numLayers+1)*numVertices(); }
  int numPrisms () const { return m_numLayers*numTriangles(); }
  const std::vector<double>& levels () const { return m_levels; }

  // ---- Global sizes ---- //
  int numGlobalTriangles () const { return m_numGlobalTriangles; }
  int numGlobalVertices () const { return m_numGlobalVertices; }
  int globalVerticesStride () const { return m_globalVerticesStride; }
  int globalTrianglesStride () const { return m_globalTrianglesStride; }
  int globalEdgesStride () const { return m_globalEdgesStride; }

  // ---- Local entities ---- //
  Point vertex (int ib) const { return {m_verticesCoords[3*ib], m_verticesCoords[3*ib+1]}; }
  Point triangleCentroid (int it) const;
  bool ownsVertex (int ib) const { return m_vertexProcIDs[ib]==m_reducedRank; }
  bool isBoundaryVertex (int ib) const { return m_isBoundaryVertex[ib]; }
  const std::vector<int>& indexToVertexID () const { return m_indexToVertexID; }
  const std::vector<int>& indexToTriangleID () const { return m_indexToTriangleID; }
  const std::vector<int>& verticesOnTria () const { return m_verticesOnTria; }
  const std::vector<int>& vertexProcIDs () const { return m_vertexProcIDs; }
  const std::vector<std::vector<int>>& procsSharingVertices () const { return m_procsSharingVertices; }
  const std::vector<int>& dirichletNodesIds () const { return m_dirichletNodesIds; }
  const std::vector<int>& iceMarginEdgesIds () const { return m_iceMarginEdgesIds; }
  // Whether the node at level il of vertex ib is in dirichletNodesIds
  bool isDirichletNode (int il, int ib) const { return m_isDirichlet[vertex3D(il,ib)]; }

  // ---- Indices in MALI's arrays ---- //
  int vertex3D (int il, int ib) const {
    return m_ordering==0 ? il*numVertices() + ib : ib*(m_numLayers+1) + il;
  }
  int prism (int il, int it) const {
    return m_ordering==0 ? il*numTriangles() + it : it*m_numLayers + il;
  }

  // Resizes all the fields in f with the sizes MALI uses. Inputs are set to 0 except the stiffening
  // factor (set to 1); optional fields are cleared.
  void allocate (FieldData& f) const;

  // Helpers to fill fields from functions of the position
  void setVertexField (std::vector<double>& field, const std::function<double(const Point&)>& f) const;
  // f is evaluated at the centroid of the triangle and at the normalized height of the middle of the layer
  void setPrismField (std::vector<double>& field, const std::function<double(const Point&, double sigma, int il)>& f) const;
  // f returns the velocity components at the vertex p and normalized level sigma
  void setVelocity (std::vector<double>& velocity, const std::function<std::array<double,2>(const Point&, double sigma, int il)>& f) const;

  // ---- Calls to the interface, as MALI makes them (only on active ranks) ---- //
  void computeGrid ();
  void extrudeGrid ();
  // Return the error flag set by Albany (0 if the solve converged)
  int solve (FieldData& f, bool firstTimeStep, double deltat = 0.0);
  int solveDeprecated (FieldData& f, bool firstTimeStep, double deltat = 0.0);
  void exportVelocity ();

  // ---- Checks ---- //
  // Checks the invariants Albany relies on (unique ids, consistent sharing and ownership across ranks,
  // positively oriented triangles, Dirichlet nodes and ice margin edges stored locally, ...).
  // Collective on worldComm. Returns an empty string if all checks pass, a description otherwise.
  std::string checkConsistency () const;

private:
  void buildGlobalMesh (const MeshOptions& options);
  void partition (const MeshOptions& options, int numActiveRanks);
  void buildLocalMesh (const MeshOptions& options);

  MPI_Comm m_worldComm;
  MPI_Comm m_reducedComm = MPI_COMM_NULL;
  int m_worldRank = 0;
  int m_numIdleRanks = 0;
  int m_reducedRank = -1;
  bool m_active = false;

  int m_numLayers;
  int m_ordering;
  std::vector<double> m_levels;

  // Global (replicated) FE mesh
  int m_nx, m_ny;
  std::vector<Point> m_gridPoints;                 // all grid vertices
  std::vector<int> m_gTriangleID;                  // active triangles: global id
  std::vector<std::array<int,3>> m_gTriangleVerts; // active triangles: grid vertex indices (CCW)
  std::vector<int> m_gTriangleOwner;               // active triangles: owner (reduced rank)
  std::vector<std::vector<int>> m_gVertexTriangles;// grid vertex -> active triangles containing it
  std::vector<int> m_gVertexOwner;                 // grid vertex -> owner (reduced rank), -1 if not in FE mesh
  std::vector<bool> m_gIsBoundaryVertex;
  struct GlobalEdge {
    int id;
    std::array<int,2> verts;   // grid vertex indices
    std::vector<int> triangles;// active triangles (1 or 2)
  };
  std::vector<GlobalEdge> m_gEdges;
  int m_numGlobalTriangles = 0;
  int m_numGlobalVertices = 0;
  int m_globalVerticesStride = 0;
  int m_globalTrianglesStride = 0;
  int m_globalEdgesStride = 0;

  // Local data passed to the interface (MALI's names)
  std::vector<int> m_indexToVertexID;
  std::vector<int> m_vertexProcIDs;
  std::vector<double> m_verticesCoords;
  std::vector<int> m_verticesOnTria;
  std::vector<std::vector<int>> m_procsSharingVertices;
  std::vector<bool> m_isBoundaryEdge;
  std::vector<int> m_trianglesOnEdge;
  std::vector<int> m_verticesOnEdge;
  std::vector<int> m_indexToEdgeID;
  std::vector<int> m_indexToTriangleID;
  std::vector<int> m_dirichletNodesIds;
  std::vector<int> m_iceMarginEdgesIds;

  // Extra local data
  std::vector<int> m_localVertexToGrid;
  std::vector<int> m_localTriangleToGlobal;  // index in the m_gTriangle* arrays
  std::vector<bool> m_isBoundaryVertex;
  std::vector<bool> m_isDirichlet;            // indexed by vertex3D
};

} // namespace mock_mpas

#endif // LANDICE_MOCK_MPAS_HPP
