//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#define velocity_solver_init_mpi velocity_solver_init_mpi__
#define velocity_solver_finalize velocity_solver_finalize__
#define velocity_solver_init_fo velocity_solver_init_fo__
#define velocity_solver_solve_fo velocity_solver_solve_fo__
#define velocity_solver_compute_2d_grid velocity_solver_compute_2d_grid__
#define velocity_solver_set_grid_data velocity_solver_set_grid_data__
#define velocity_solver_extrude_3d_grid velocity_solver_extrude_3d_grid__
#define velocity_solver_export_fo_velocity velocity_solver_export_fo_velocity__
#define velocity_solver_set_physical_parameters velocity_solver_set_physical_parameters__

// ===================================================
//! Includes
// ===================================================

#include "LandIce_ProblemFactory.hpp"
#include "LandIce_StokesFO.hpp"
#include "LandIce_DepthIntegratedBasisUtils.hpp"

#include "Albany_SideSetSTKMeshStruct.hpp"
#include "Albany_MpasSTKMeshStruct.hpp"
#include "Albany_MpasBasalSTKMeshStruct.hpp"
#include "Albany_ExtrudedMesh.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"

#include "Albany_GlobalLocalIndexer.hpp"
#include "Albany_ThyraUtils.hpp"

#include "Albany_StringUtils.hpp" // for 'upper_case'
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKDiscretization.hpp"

#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <Piro_PerformSolve.hpp>
#include <Teuchos_CommHelpers.hpp>
#include <Teuchos_ParameterList.hpp>
#include <Teuchos_RCP.hpp>
#include <Teuchos_XMLParameterListHelpers.hpp>
#include <Teuchos_StandardCatchMacros.hpp>
#include <Thyra_DetachedVectorView.hpp>
#include <Teuchos_YamlParameterListHelpers.hpp>

#include <Teuchos_StackedTimer.hpp>
#include <Teuchos_TimeMonitor.hpp>

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <Ionit_Initializer.h>
#endif

#include <algorithm>
#include <array>
#include <fstream>
#include <vector>
#include <mpi.h>
#include <list>
#include <iostream>
#include <limits>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

// The mesh used by Albany, set by the 'Method' of the Discretization sublist of albany_input.yaml:
//  - "Extruded" (default): logically extruded mesh, whose basal mesh is the 2D triangulation passed by MPAS;
//  - "STKExtruded": 3D STK mesh, extruded explicitly from the MPAS triangulation.
bool useExtrudedMesh(true);
// Logically extruded mesh
Teuchos::RCP<Albany::MpasBasalSTKMeshStruct> basalMesh;
Teuchos::RCP<Albany::ExtrudedMesh> extrudedMesh;
std::vector<int> mpasVerticesOnTria; // verticesOnTria of the current FE grid
// 3D STK mesh
Teuchos::RCP<Albany::MpasSTKMeshStruct> meshStruct;

Teuchos::RCP<Albany::Application> albanyApp;
Teuchos::RCP<Teuchos::ParameterList> paramList;
Teuchos::RCP<const Teuchos_Comm> mpiComm, mpiCommMPAS;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
Teuchos::RCP<double> MPAS_dt;
Teuchos::RCP<Teuchos::StackedTimer> stackedTimer;

double MPAS_gravity(9.8), MPAS_rho_ice(910.0), MPAS_rho_seawater(1028.0), MPAS_sea_level(0),
    MPAS_flowParamA(1e-4), MPAS_flowLawExponent(3), MPAS_dynamic_thickness(1e-2),
    MPAS_ClausiusClapeyoronCoeff(9.7546e-8);
// Defaults match the values Albany used before MPAS could set them
double MPAS_meltingTemperature(273.15), MPAS_secondsPerYear(3.1536e7), MPAS_gasConstant(8.314);
bool MPAS_useGLP(true);

bool depthIntegratedModel(false);

std::vector<int> dirichletNodesIdsDepthInt;

std::set<std::string> noupdate_fields;

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;

bool keptMesh =false;
bool kokkosInitializedByAlbany = false;

typedef struct TET_ {
  int verts[4];
  int neighbours[4];
  char bound_type[4];
} TET;

namespace {

// Number of nested calls into the interface functions currently active
// (e.g., the deprecated velocity_solver_solve_fo calls the current one).
int interfaceCallDepth = 0;

// RAII guard used at the top of each interface function called by MPAS.
// The base timer ("Albany Velocity Solver") runs only while MPAS is inside
// one of these functions. If it ran from velocity_solver_init_mpi to
// velocity_solver_finalize, it would include all the time spent in MPAS
// between velocity solves, and that time would show up as (Remainder) in the report.
class InterfaceCallTimer {
public:
  InterfaceCallTimer () : active_(Teuchos::nonnull(stackedTimer)) {
    if (active_ && interfaceCallDepth++ == 0) {
      stackedTimer->startBaseTimer();
    }
  }
  ~InterfaceCallTimer () {
    if (active_ && --interfaceCallDepth == 0) {
      stackedTimer->stopBaseTimer();
    }
  }
  InterfaceCallTimer (const InterfaceCallTimer&) = delete;
  InterfaceCallTimer& operator= (const InterfaceCallTimer&) = delete;
private:
  const bool active_;
};

// Times consecutive phases of a function without opening a new scope for
// each phase: start() stops the current phase (if any) and starts the next
// one; the destructor stops the current phase.
class PhaseTimer {
public:
  explicit PhaseTimer (const std::string& name) { start(name); }
  ~PhaseTimer () { stop(); }
  void start (const std::string& name) {
    stop();
    monitor_.emplace(*Teuchos::TimeMonitor::getNewTimer(name));
  }
  void stop () { monitor_.reset(); }
  PhaseTimer (const PhaseTimer&) = delete;
  PhaseTimer& operator= (const PhaseTimer&) = delete;
private:
  std::optional<Teuchos::TimeMonitor> monitor_;
};

// Releases the solver of the previous solve.
void releasePreviousSolver () {
  if (Teuchos::is_null(solver)) return;
  auto releaseTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Release Previous Solver"));
  solver = Teuchos::null;
}

using STKFieldType = Albany::AbstractSTKFieldContainer::STKFieldType;
constexpr auto NODE_RANK = stk::topology::NODE_RANK;
constexpr auto ELEM_RANK = stk::topology::ELEMENT_RANK;

// The fields passed by MPAS to velocity_solver_solve_fo (same names), with MPAS's layout:
//  - 2D fields have one entry per local vertex (MPAS cell);
//  - 3D vertex fields have entry vertex3D(il,ib) for level il (0 at the bed) of vertex ib. The velocity
//    stores the x component at vertex3D(il,ib), and the y component at vertex3D(il,ib)+numVertices3D();
//  - prism fields have entry prism(il,it) for layer il (0 at the bed) of triangle it (MPAS vertex).
struct MpasFields {
  int nLayers;
  int globalVerticesStride;
  int globalTrianglesStride;
  int ordering; // 0: layer-wise, 1: column-wise ordering of the 3D entities
  const std::vector<int>&    indexToVertexID;
  const std::vector<int>&    indexToTriangleID;
  const std::vector<double>& levelsNormalizedThickness;
  const std::vector<double>& elevationData;
  const std::vector<double>& thicknessData;
        std::vector<double>& betaData;
  const std::vector<double>& bedTopographyData;
  const std::vector<double>& stiffeningFactorData;
        std::vector<double>& effectivePressureData;
  const std::vector<double>& muData;
  const std::vector<double>& bedRoughnessData;
  const std::vector<double>& bulkFrictionData;
  const std::vector<double>& basalDebrisData;
  const std::vector<double>& temperatureDataOnPrisms;
        std::vector<double>& bodyForceMagnitudeOnBasalCell;
        std::vector<double>& dissipationHeatOnPrisms;
        std::vector<double>& velocityOnVertices;

  int numVertices () const { return indexToVertexID.size(); }
  int numTriangles () const { return indexToTriangleID.size(); }
  int numVertices3D () const { return (nLayers+1)*numVertices(); }
  int numPrisms () const { return nLayers*numTriangles(); }
  int vertex3D (const int il, const int ib) const {
    return ordering==0 ? il*numVertices() + ib : ib*(nLayers+1) + il;
  }
  int prism (const int il, const int it) const {
    return ordering==0 ? il*numTriangles() + it : it*nLayers + il;
  }
};

// Whether a mesh field can be set with the data passed by MPAS: the data is given, the field is in the
// mesh, and it is not read from file (see noupdate_fields)
bool canUpdate (const std::vector<double>& data, const STKFieldType* field, const std::string& name) {
  return !data.empty() && (field != nullptr) && (noupdate_fields.find(name) == noupdate_fields.end());
}

// Entry of a "Required Fields Info" list with the given field name, or null
Teuchos::ParameterList* findRequiredField (Teuchos::ParameterList& rfi, const std::string& name) {
  const int n = rfi.get<int>("Number Of Fields",0);
  for (int i=0; i<n; ++i) {
    auto& field = rfi.sublist(util::strint("Field",i));
    if (field.get<std::string>("Field Name")==name) return &field;
  }
  return nullptr;
}

// Vertical profile of the depth-integrated velocity at the MPAS levels:
// u(level il) = phiBed[il]*u(bed) + phiTop[il]*u(surface)
void depthIntegratedVerticalProfile (const std::vector<double>& levelsNormalizedThickness,
                                     std::vector<double>& phiBed, std::vector<double>& phiTop) {
  const auto problem = Teuchos::rcp_dynamic_cast<LandIce::StokesFO>(albanyApp->getProblem());
  TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(problem), std::runtime_error,
    "Error! Stokes FO Problem not defined. At the moment the depth integrated model only works for Stokes FO.\n");
  const auto basis = problem->getCellDepthIntegratedBasis();
  TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(basis), std::runtime_error,
      "Error! The depth-integrated basis of the Stokes FO problem has not been built.\n");
  LandIce::evalDepthIntegratedVerticalProfile(*basis, levelsNormalizedThickness, phiBed, phiTop);
}

// ======================== Fields on the basal mesh (both meshes) ======================== //

// Imports the basal fields that are 2D for both meshes. basalNodeIds[ib] is the STK id of the node of vertex ib
// in the basal mesh. Returns the STK nodes of the basal mesh, ordered like the MPAS vertices, so that the export
// of beta and effective pressure does not need to look them up.
std::vector<stk::mesh::Entity> importBasalFields (const Albany::AbstractSTKMeshStruct& basal,
                                                  const std::vector<int>& basalNodeIds, const MpasFields& f)
{
  STKFieldType* bedTopographyField = basal.metaData->get_field <double> (NODE_RANK, "bed_topography");
  STKFieldType* muField = basal.metaData->get_field <double> (NODE_RANK, "mu");
  STKFieldType* bedRoughnessField = basal.metaData->get_field <double> (NODE_RANK, "bed_roughness");
  STKFieldType* bulkFrictionField = basal.metaData->get_field <double> (NODE_RANK, "bulk_friction");
  STKFieldType* basalDebrisField = basal.metaData->get_field <double> (NODE_RANK, "basal_debris_factor");
  STKFieldType* effectivePressureField = basal.metaData->get_field <double> (NODE_RANK, "effective_pressure");

  const bool update_bedTopography = canUpdate(f.bedTopographyData, bedTopographyField, "bed_topography");
  const bool update_mu = canUpdate(f.muData, muField, "mu");
  const bool update_bedRoughness = canUpdate(f.bedRoughnessData, bedRoughnessField, "bed_roughness");
  const bool update_bulkFriction = canUpdate(f.bulkFrictionData, bulkFrictionField, "bulk_friction");
  const bool update_basalDebris = canUpdate(f.basalDebrisData, basalDebrisField, "basal_debris_factor");
  const bool update_effectivePressure = canUpdate(f.effectivePressureData, effectivePressureField, "effective_pressure");

  std::vector<stk::mesh::Entity> basalNodes(f.numVertices());
  for(int ib = 0; ib < f.numVertices(); ++ib) {
    stk::mesh::Entity node = basal.bulkData->get_entity(NODE_RANK, basalNodeIds[ib]);
    TEUCHOS_TEST_FOR_EXCEPTION(!basal.bulkData->is_valid(node), std::logic_error,
        "Error! The basal mesh has no node with id " << basalNodeIds[ib] << " (MPAS vertex " << f.indexToVertexID[ib] << ").\n");
    basalNodes[ib] = node;

    if (update_bedTopography) {
      stk::mesh::field_data(*bedTopographyField, node)[0] = f.bedTopographyData[ib];
    }
    if (update_mu) {
      stk::mesh::field_data(*muField, node)[0] = f.muData[ib];
    }
    if (update_bedRoughness) {
      stk::mesh::field_data(*bedRoughnessField, node)[0] = f.bedRoughnessData[ib];
    }
    if (update_bulkFriction) {
      stk::mesh::field_data(*bulkFrictionField, node)[0] = f.bulkFrictionData[ib];
    }
    if (update_basalDebris) {
      stk::mesh::field_data(*basalDebrisField, node)[0] = f.basalDebrisData[ib];
    }
    if (update_effectivePressure) {
      stk::mesh::field_data(*effectivePressureField, node)[0] = f.effectivePressureData[ib];
    }
  }
  return basalNodes;
}

// Exports beta and the effective pressure (which can be an output, e.g., when computed as hydrostatic)
void exportBasalFields (const Albany::AbstractSTKMeshStruct& basal, const std::vector<stk::mesh::Entity>& basalNodes,
                        MpasFields& f)
{
  STKFieldType* betaField = basal.metaData->get_field <double> (NODE_RANK, "beta");
  STKFieldType* effectivePressureField = basal.metaData->get_field <double> (NODE_RANK, "effective_pressure");
  const bool export_effectivePressure = !f.effectivePressureData.empty() && (effectivePressureField!=nullptr);
  const bool export_beta = !f.betaData.empty() && (betaField!=nullptr);
  if (!export_effectivePressure && !export_beta) return;

  for(int ib = 0; ib < f.numVertices(); ++ib) {
    const stk::mesh::Entity node = basalNodes[ib];
    if (export_effectivePressure) {
      f.effectivePressureData[ib] = stk::mesh::field_data(*effectivePressureField,node)[0];
    }
    if (export_beta) {
      f.betaData[ib] = stk::mesh::field_data(*betaField,node)[0];
    }
  }
}

// ============================= 3D STK mesh (Method "STKExtruded") ============================= //

// Ids of the nodes of the basal mesh (the SideSetSTK mesh of the basalside), which are the ids of the 3D nodes
// at the bed: vertexID with the layer-wise ordering, but not with the column-wise one.
// With the depth-integrated model the 3D mesh has one layer.
std::vector<int> basalNodeIdsSTK3D (const MpasFields& f)
{
  const int meshLayers = depthIntegratedModel ? 1 : f.nLayers;
  const int vertexLayerShift = (f.ordering == 0) ? 1 : meshLayers + 1;
  std::vector<int> ids(f.numVertices());
  for (int ib = 0; ib < f.numVertices(); ++ib) {
    ids[ib] = vertexLayerShift * (f.indexToVertexID[ib]-1) + 1;
  }
  return ids;
}

// Imports the geometry, the stiffening factor, the temperature, the Dirichlet velocity and the initial
// guess (which, with the depth-integrated model, are only set at the bed and surface levels).
// Returns the STK nodes of the 3D mesh, ordered like the MPAS 3D vertices (the entries of the levels
// without a node are not set), so that the export of the solution does not need to look them up again.
std::vector<stk::mesh::Entity> importFieldsSTK3D (const MpasFields& f)
{
  const int nLayers = f.nLayers;
  const int ordering = f.ordering;
  const auto& indexToVertexID = f.indexToVertexID;
  const auto& indexToTriangleID = f.indexToTriangleID;
  const auto& levelsNormalizedThickness = f.levelsNormalizedThickness;
  const auto& velocityOnVertices = f.velocityOnVertices;
  const auto& temperatureDataOnPrisms = f.temperatureDataOnPrisms;

  int numVertices3D = f.numVertices3D();
  int vertexColumnShift = (ordering == 1) ? 1 : f.globalVerticesStride;
  int lVertexColumnShift = (ordering == 1) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (ordering == 0) ? 1 : nLayers + 1;

  int elemColumnShift = (ordering == 1) ? 1 : f.globalTrianglesStride;
  int lElemColumnShift = (ordering == 1) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (ordering == 0) ? 1 : nLayers;

  auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(albanyApp->getDiscretization(),true);
  int neq = stk_disc->getNumEq();

  auto& layerThicknessRatio = meshStruct->layers_data.dz_ref;

  if(depthIntegratedModel)
    layerThicknessRatio[0] = 1.0;
  else {
    for (int i = 0; i < nLayers; i++)
      layerThicknessRatio[i] = levelsNormalizedThickness[i+1]-levelsNormalizedThickness[i];
  }

  using SolFldContainerType = Albany::OrdinarySTKFieldContainer;

  auto fld_container = stk_disc->getSolutionFieldContainer();
  auto sol_fld_container = Teuchos::rcp_dynamic_cast<SolFldContainerType>(fld_container);
  auto solutionField = sol_fld_container->getSolutionField();

  STKFieldType* surfaceHeightField = meshStruct->metaData->get_field <double> (NODE_RANK, "surface_height");
  STKFieldType* thicknessField = meshStruct->metaData->get_field <double> (NODE_RANK, "ice_thickness");
  STKFieldType* dirichletField = meshStruct->metaData->get_field <double> (NODE_RANK, "dirichlet_field");
  STKFieldType* stiffeningFactorLogField = meshStruct->metaData->get_field <double> (NODE_RANK, "stiffening_factor_log");

  bool update_dirichlet = canUpdate(velocityOnVertices, dirichletField, "dirichlet_field");
  bool update_stiffeningFactorLog = canUpdate(f.stiffeningFactorData, stiffeningFactorLogField, "stiffening_factor_log");

  std::vector<stk::mesh::Entity> nodes3D(numVertices3D);

  for (int j = 0; j < numVertices3D; ++j) {
    int ib = (ordering == 0) * (j % lVertexColumnShift)
            + (ordering == 1) * (j / vertexLayerShift);
    int il = (ordering == 0) * (j / lVertexColumnShift)
            + (ordering == 1) * (j % vertexLayerShift);

    int gId(0);
    // When using depthIntegratedModel, Albany mesh has one layer, however MPAS mesh can have multiple layers
    if(depthIntegratedModel) {
       if ((il != 0) && (il != nLayers)) continue;
       int layer = il/nLayers; // 0 or 1
       int depthVertexLayerShift = (ordering == 0) ? 1 : 2;
       gId = layer * vertexColumnShift + depthVertexLayerShift * (indexToVertexID[ib]-1) + 1;
    } else {
      gId = il * vertexColumnShift + vertexLayerShift * (indexToVertexID[ib]-1) + 1;
    }

    stk::mesh::Entity node = meshStruct->bulkData->get_entity(NODE_RANK, gId);
    nodes3D[j] = node;
    double* coord = stk::mesh::field_data(*meshStruct->getCoordinatesField(), node);
    coord[2] = f.elevationData[ib] + (levelsNormalizedThickness[il]-1.0) * f.thicknessData[ib];

    double* thickness = stk::mesh::field_data(*thicknessField, node);
    thickness[0] = f.thicknessData[ib];
    double* sHeight = stk::mesh::field_data(*surfaceHeightField, node);
    sHeight[0] = f.elevationData[ib];

    if(update_stiffeningFactorLog) {
      double* stiffeningFactorLog = stk::mesh::field_data(*stiffeningFactorLogField, node);
      stiffeningFactorLog[0] = std::log(f.stiffeningFactorData[ib]);
    }

    double* sol = stk::mesh::field_data(*solutionField, node);
    sol[0] = velocityOnVertices[j];
    sol[1] = velocityOnVertices[j + numVertices3D];
    if(neq==3) {
      sol[2] = f.thicknessData[ib];
    }
    if(update_dirichlet) {
      double* dirichletVel = stk::mesh::field_data(*dirichletField, node);
      dirichletVel[0]=velocityOnVertices[j]; //velocityOnVertices stores initial guess and dirichlet velocities.
      dirichletVel[1]=velocityOnVertices[j + numVertices3D];
    }
  }

  //In the following we import the temperature field temperatureDataOnPrisms from MPAS,
  //which is stored as a constant in each Prism (wedge), into a temperature filed in Albany mesh.
  STKFieldType* temperatureField = meshStruct->metaData->get_field<double>(ELEM_RANK, "temperature");
  bool update_temperature = canUpdate(temperatureDataOnPrisms, temperatureField, "temperature");
  if(depthIntegratedModel && update_temperature) {
    // In this case the Albany mesh only has one layer, but the MPAS mesh can still have multiple layers
    // We compute the temperature at Albany quadrature points.
    // For each quadrature point we identify what MPAS wedge/layer it belongs to,
    // and assign the temperature at that wedge to Albany temperature QuadPoint field at that quad point.
    const auto problem = Teuchos::rcp_dynamic_cast<LandIce::StokesFO>(albanyApp->getProblem());
    TEUCHOS_TEST_FOR_EXCEPTION(Teuchos::is_null(problem), std::runtime_error,
        "Error! Stokes FO Problem not defined. At the moment the depth integrated model only works for Stokes FO.\n");
    const auto cellCubature = problem->getCellCubature();

    int numQPs = cellCubature->getNumPoints();
    Kokkos::DynRankView<double, PHX::Device> quadPointCoords("refPoints", numQPs, 3);
    Kokkos::DynRankView<double, PHX::Device> quadPointWeights("refWeights", numQPs);

    // Pre-Calculate reference element quantities
    cellCubature->getCubature(quadPointCoords, quadPointWeights);

    //Compute mesh layers associated to quad points
    std::vector<int> layerVec(numQPs);
    auto quadPointCoordsHost = Kokkos::create_mirror_view(quadPointCoords);
    Kokkos::deep_copy(quadPointCoordsHost, quadPointCoords);
    for(int qp=0; qp<numQPs; qp++) {
      int il=0;
      auto z = (quadPointCoordsHost(qp,2)+1.0)/2; //quad points are defined on [-1,1]
      while ((z > levelsNormalizedThickness[il+1]) && (il<nLayers)) il++;
      layerVec[qp] = il;
    }

    // Populate the temperature mesh field, defined at quad points
    for(int ib=0; ib < (int) indexToTriangleID.size(); ++ib ) {
      stk::mesh::Entity elem = meshStruct->bulkData->get_entity(ELEM_RANK, indexToTriangleID[ib]);
      double* temperature = stk::mesh::field_data(*temperatureField, elem);
      for(int qp=0; qp<numQPs; qp++) {
        int lId = layerVec[qp] * lElemColumnShift + elemLayerShift * ib;
        temperature[qp] = temperatureDataOnPrisms[lId];
      }
    }
  } else if(update_temperature){ // Here we copy the temperature on Prisms from MPAS into a P0 field in the Albany mesh.
    for(int ib=0; ib < (int) indexToTriangleID.size(); ++ib ) {
      for(int il=0; il<nLayers; il++) {
        int lId = il * lElemColumnShift + elemLayerShift * ib;
        int gId = il * elemColumnShift + elemLayerShift * (indexToTriangleID[ib]-1) + 1;
        stk::mesh::Entity elem = meshStruct->bulkData->get_entity(ELEM_RANK, gId);
        double* temperature = stk::mesh::field_data(*temperatureField, elem);
        temperature[0] = temperatureDataOnPrisms[lId];
      }
    }
  }

  return nodes3D;
}

// Exports the velocity, the dissipation heat and the body force
void exportFieldsSTK3D (MpasFields& f, const std::vector<stk::mesh::Entity>& nodes3D)
{
  const int nLayers = f.nLayers;
  const int ordering = f.ordering;
  const auto& indexToVertexID = f.indexToVertexID;
  const auto& indexToTriangleID = f.indexToTriangleID;
  auto& velocityOnVertices = f.velocityOnVertices;
  auto& dissipationHeatOnPrisms = f.dissipationHeatOnPrisms;
  auto& bodyForceMagnitudeOnBasalCell = f.bodyForceMagnitudeOnBasalCell;

  int numVertices3D = f.numVertices3D();
  int numPrisms = f.numPrisms();
  int lVertexColumnShift = (ordering == 1) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (ordering == 0) ? 1 : nLayers + 1;

  int elemColumnShift = (ordering == 1) ? 1 : f.globalTrianglesStride;
  int lElemColumnShift = (ordering == 1) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (ordering == 0) ? 1 : nLayers;

  auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(albanyApp->getDiscretization(),true);
  auto fld_container = stk_disc->getSolutionFieldContainer();
  auto sol_fld_container = Teuchos::rcp_dynamic_cast<Albany::OrdinarySTKFieldContainer>(fld_container);
  auto solutionField = sol_fld_container->getSolutionField();

  // The velocity is read directly from the STK solution field, using the node entities cached
  // during the import. After the solve, the Albany observer writes the overlapped solution
  // (owned and shared nodes) into this field. Before, at every call we built a new
  // CombineAndScatterManager (i.e., a Tpetra Import, which requires communication), an owned
  // and an overlapped vector and a global-to-local indexer, and synced the vector to host,
  // only to read back the same values that are stored in the STK field.
  // If the observer is not called (e.g., the solve failed and only converged solutions are
  // written), the field holds the initial guess set above, as the previous code also returned.
  if(depthIntegratedModel) {
    // Vertical profile of the depth-integrated velocity at the MPAS levels
    std::vector<double> phiBed, phiTop;
    depthIntegratedVerticalProfile(f.levelsNormalizedThickness, phiBed, phiTop);

    for(int ib = 0; ib < (int) indexToVertexID.size(); ++ib) {
      const double* solBed = stk::mesh::field_data(*solutionField, nodes3D[vertexLayerShift * ib]);
      const double* solTop = stk::mesh::field_data(*solutionField, nodes3D[nLayers * lVertexColumnShift + vertexLayerShift * ib]);

      for(int il=0; il<nLayers+1; ++il) {
        int j = il * lVertexColumnShift + vertexLayerShift * ib;
        velocityOnVertices[j] = phiBed[il] * solBed[0] + phiTop[il] * solTop[0];
        velocityOnVertices[j + numVertices3D]  = phiBed[il] * solBed[1] + phiTop[il] * solTop[1];
      }
    }
  } else {
    for (int j = 0; j < numVertices3D; ++j) {
      const double* sol = stk::mesh::field_data(*solutionField, nodes3D[j]);
      velocityOnVertices[j] = sol[0];
      velocityOnVertices[j + numVertices3D] = sol[1];
    }
  }

  STKFieldType* dissipationHeatField = meshStruct->metaData->get_field <double> (ELEM_RANK, "dissipation_heat");
  STKFieldType* bodyForceField  = meshStruct->metaData->get_field <double> (ELEM_RANK, "body_force");
  if(!dissipationHeatOnPrisms.empty())
    std::fill(dissipationHeatOnPrisms.begin(), dissipationHeatOnPrisms.end(), 0.0);
  for (int j = 0; j < numPrisms; ++j) {
    int ib = (ordering == 0) * (j % (lElemColumnShift))
            + (ordering == 1) * (j / (elemLayerShift));
    int il = (ordering == 0) * (j / (lElemColumnShift))
            + (ordering == 1) * (j % (elemLayerShift));
    int gId = depthIntegratedModel ? indexToTriangleID[ib] : il * elemColumnShift + elemLayerShift * (indexToTriangleID[ib]-1) + 1;
    int lId = il * lElemColumnShift + elemLayerShift * ib;

    double bf = 0;
    stk::mesh::Entity elem = meshStruct->bulkData->get_entity(ELEM_RANK, gId);
    if(!dissipationHeatOnPrisms.empty() && dissipationHeatField != nullptr) {
      const double* dissipationHeat = stk::mesh::field_data(*dissipationHeatField, elem);
      dissipationHeatOnPrisms[lId] += dissipationHeat[0];
    }

    if ((il==0) && (bodyForceField!=nullptr)) {
      const double* bodyForceVal = stk::mesh::field_data(*bodyForceField, elem);
      const double normSq = bodyForceVal[0]*bodyForceVal[0] + bodyForceVal[1]*bodyForceVal[1];
      bf += normSq;
    }
    if (!bodyForceMagnitudeOnBasalCell.empty() && (il==0)) {
      bodyForceMagnitudeOnBasalCell[ib] = std::sqrt(bf);
    }
  }
}

// ========================= Logically extruded mesh (Method "Extruded") ========================= //
//
// The basal mesh is the MPAS triangulation (MpasBasalSTKMeshStruct). The 3D fields passed by MPAS are
// stored on the basal mesh: the geometry and the stiffening factor as basal fields, extruded along the
// vertical; the temperature and the Dirichlet velocity as layered fields, on the MPAS layers and levels,
// interpolated on the mesh levels. The mesh levels are the MPAS levels, except with the depth-integrated
// model, where the mesh has one layer. See setExtrudedMeshParameters.

// Indices of the Albany dofs and elements corresponding to the MPAS entities. They are rebuilt at each
// solve, after the discretization is updated.
struct ExtrudedMeshMaps {
  int neq = 0;
  int numMeshLayers = 0;
  // Entry j*neq+eq: local id, in the overlapped (ovLid) and owned (ownedLid) solution vectors, of the dof
  // of equation eq at the MPAS 3D vertex j (see MpasFields::vertex3D), or -1 if there is no such dof
  // (with the depth-integrated model, only the bed and surface levels have dofs).
  std::vector<LO> ovLid, ownedLid;
  // Entry it*numMeshLayers+il: workset index of the 3D element of mesh layer il above triangle it
  std::vector<Albany::WsIdx> elemWsIdx;
};
ExtrudedMeshMaps extrudedMeshMaps;

// Node sets of the basal mesh with the Dirichlet nodes passed by MPAS (ids of the 3D nodes of the MPAS
// numbering, see velocity_solver_extrude_3d_grid):
//  - "dirichlet": vertices whose whole column is made of Dirichlet nodes;
//  - "dirichlet_bed": vertices where only the bed node is a Dirichlet node.
// These are the patterns supported by the logically extruded mesh (as node sets "extruded_dirichlet" and
// "basal_dirichlet_bed"). With any other pattern, this throws on all ranks.
std::map<std::string,std::vector<int>>
dirichletNodeSets (const std::vector<int>& dirichletNodesIds, const std::vector<int>& indexToVertexID,
                   const int nLayers, const int ordering, const int globalVerticesStride)
{
  const int nV = indexToVertexID.size();
  const int nLevels = nLayers+1;
  std::unordered_map<int,int> vertexIdToIndex;
  for (int ib=0; ib<nV; ++ib) {
    vertexIdToIndex[indexToVertexID[ib]] = ib;
  }

  std::vector<char> isDirichlet(static_cast<size_t>(nV)*nLevels, 0);
  int invalidId = -1;
  for (int id : dirichletNodesIds) {
    const int gid = id-1;
    const int vertexId = (ordering==0) ? gid%globalVerticesStride + 1 : gid/nLevels + 1;
    const int il = (ordering==0) ? gid/globalVerticesStride : gid%nLevels;
    const auto it = vertexIdToIndex.find(vertexId);
    if (gid<0 || il>=nLevels || it==vertexIdToIndex.end()) {
      invalidId = id;
      continue;
    }
    isDirichlet[static_cast<size_t>(it->second)*nLevels + il] = 1;
  }

  std::map<std::string,std::vector<int>> nodeSets = {{"dirichlet",{}}, {"dirichlet_bed",{}}};
  int unsupportedVertexId = -1;
  std::string unsupportedLevels;
  for (int ib=0; ib<nV; ++ib) {
    const char* levels = &isDirichlet[static_cast<size_t>(ib)*nLevels];
    const int count = std::count(levels, levels+nLevels, 1);
    if (count==0) {
      continue;
    } else if (count==nLevels) {
      nodeSets["dirichlet"].push_back(ib);
    } else if (count==1 && levels[0]==1) {
      nodeSets["dirichlet_bed"].push_back(ib);
    } else if (unsupportedVertexId<0) {
      unsupportedVertexId = indexToVertexID[ib];
      for (int il=0; il<nLevels; ++il) {
        if (levels[il]==1) unsupportedLevels += " " + std::to_string(il);
      }
    }
  }

  // Throw on all ranks
  int localErrors[2] = {invalidId>=0, unsupportedVertexId>=0};
  int globalErrors[2];
  Teuchos::reduceAll<int,int>(*mpiComm, Teuchos::REDUCE_MAX, 2, localErrors, globalErrors);
  TEUCHOS_TEST_FOR_EXCEPTION(globalErrors[0]!=0, std::runtime_error,
      "Error! Invalid Dirichlet node id passed by MPAS" <<
      (invalidId>=0 ? " (" + std::to_string(invalidId) + " on this rank)" : std::string("")) << ".\n");
  TEUCHOS_TEST_FOR_EXCEPTION(globalErrors[1]!=0, std::runtime_error,
      "Error! With the logically extruded mesh (Discretization Method 'Extruded', the default), the Dirichlet\n"
      "  nodes passed by MPAS (dirichletVelocityMask) at each vertex must be either the whole column, or the\n"
      "  bed node only.\n" <<
      (unsupportedVertexId>=0 ? "  Vertex " + std::to_string(unsupportedVertexId) + " (on this rank) has Dirichlet nodes at levels" +
                                unsupportedLevels + " (0 is the bed).\n" : std::string("")) <<
      "  To prescribe the velocity on other levels, set 'Method: STKExtruded' in the Discretization sublist\n"
      "  of albany_input.yaml, to use the 3D STK mesh.\n");

  return nodeSets;
}

// Settings of the logically extruded mesh. The 3D required fields (the same as with the 3D STK mesh)
// must be already set. Adds their basal counterparts, and lists them in "Extrude Basal Fields" and
// "Interpolate Basal Layered Fields".
void setExtrudedMeshParameters (Teuchos::ParameterList& discList, const int nLayers)
{
  const int numMeshLayers = depthIntegratedModel ? 1 : nLayers;
  if (discList.isParameter("NumLayers") && discList.get<int>("NumLayers")!=numMeshLayers) {
    std::cout<<"\nWARNING: Ignoring 'NumLayers' in the Discretization sublist of the Albany input file. The number of layers is set by MPAS ("
             << numMeshLayers << ").\n"<<std::endl;
  }
  discList.set<int>("NumLayers", numMeshLayers);
  discList.set<bool>("Columnwise Ordering", discList.get<bool>("Columnwise Ordering", true));
  discList.set<std::string>("Thickness Field Name", "ice_thickness");
  discList.set<std::string>("Surface Height Field Name", "surface_height");

  auto& rfi = discList.sublist("Required Fields Info");
  auto& basal_pl = discList.sublist("Side Set Discretizations").sublist("basalside");
  auto& basal_req = basal_pl.sublist("Required Fields Info");

  // The output of the logically extruded mesh is written by the basal mesh (3D fields are written as layered fields)
  if (discList.isParameter("Exodus Output File Name") && !basal_pl.isParameter("Exodus Output File Name")) {
    std::cout<<"\nWARNING: With the logically extruded mesh, the output is written on the basal mesh, with the 3D fields as layered fields. "
               "Using the 'Exodus Output File Name' of the Discretization sublist for the basal mesh.\n"<<std::endl;
    basal_pl.set("Exodus Output File Name", discList.get<std::string>("Exodus Output File Name"));
  }

  auto extrude = discList.get("Extrude Basal Fields", Teuchos::Array<std::string>());
  auto interpolate = discList.get("Interpolate Basal Layered Fields", Teuchos::Array<std::string>());

  // Basal counterparts of the 3D fields set by MPAS. Layered fields are defined on the MPAS layers
  // (temperature) or levels (Dirichlet velocity).
  struct BasalField {
    std::string name;
    std::string type;
    int numLayers;      // for layered fields
    bool interpolated;  // layered fields are interpolated, the others are extruded
  };
  const std::vector<BasalField> basalFields = {
    {"ice_thickness",         "Node Scalar",         0,         false},
    {"surface_height",        "Node Scalar",         0,         false},
    {"stiffening_factor_log", "Node Scalar",         0,         false},
    {"temperature",           "Elem Layered Scalar", nLayers,   true },
    {"dirichlet_field",       "Node Layered Vector", nLayers+1, true },
  };

  int bfp = basal_req.get<int>("Number Of Fields",0);
  for (const auto& bf : basalFields) {
    auto field3D = findRequiredField(rfi, bf.name);
    if (field3D==nullptr || field3D->get<std::string>("Field Usage","Input")=="Unused") continue;
    TEUCHOS_TEST_FOR_EXCEPTION(field3D->get<std::string>("Field Origin","Mesh")!="Mesh", std::runtime_error,
        "Error! With the logically extruded mesh (Discretization Method 'Extruded', the default), the 3D field '" << bf.name << "'\n"
        "  must have 'Field Origin: Mesh' in the 'Required Fields Info' of the Discretization sublist.\n"
        "  To load it from file, add it to the 'Required Fields Info' of the basalside discretization (as a basal field), or set\n"
        "  'Method: STKExtruded' in the Discretization sublist, to use the 3D STK mesh.\n");

    if (findRequiredField(basal_req, bf.name)==nullptr) {
      Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
      field.set<std::string>("Field Name", bf.name);
      field.set<std::string>("Field Type", bf.type);
      field.set<std::string>("Field Origin", "Mesh");
      if (bf.numLayers>0) {
        field.set<int>("Number Of Layers", bf.numLayers);
      }
      if (bf.type.find("Vector")!=std::string::npos) {
        field.set<int>("Vector Dim", 2);
      }
    }

    auto& names = bf.interpolated ? interpolate : extrude;
    if (std::find(names.begin(), names.end(), bf.name)==names.end()) {
      names.push_back(bf.name);
    }
  }
  basal_req.set<int>("Number Of Fields",bfp);

  discList.set("Extrude Basal Fields", extrude);
  discList.set("Interpolate Basal Layered Fields", interpolate);
}

// Translates the names of the node sets and side sets of the 3D STK mesh used in the Albany input file into
// the names of the logically extruded mesh, for backward compatibility of the input files:
//  - node set "dirichlet" -> "extruded_dirichlet" (whole columns) and "basal_dirichlet_bed" (bed only),
//    in the Dirichlet BCs;
//  - side set "ice_margin_side" -> "extruded_ice_margin_side", in the LandIce BCs.
void translateMeshPartNames (Teuchos::ParameterList& probParamList)
{
  if (probParamList.isSublist("Dirichlet BCs")) {
    auto& dbcs = probParamList.sublist("Dirichlet BCs");
    const std::string oldNS = " on NS dirichlet ";
    std::vector<std::string> oldNames;
    for (auto it = dbcs.begin(); it != dbcs.end(); ++it) {
      const auto& name = dbcs.name(it);
      if (name.find(oldNS)!=std::string::npos) oldNames.push_back(name);
    }
    for (const auto& name : oldNames) {
      const auto pos = name.find(oldNS);
      const Teuchos::ParameterEntry entry = dbcs.getEntry(name);
      for (const std::string ns : {"extruded_dirichlet", "basal_dirichlet_bed"}) {
        const auto newName = name.substr(0,pos) + " on NS " + ns + " " + name.substr(pos+oldNS.size());
        if (!dbcs.isParameter(newName)) {
          dbcs.setEntry(newName, entry);
        }
      }
      dbcs.remove(name);
      std::cout<<"\nWARNING: Dirichlet BC '" << name << "' refers to node set 'dirichlet' of the 3D STK mesh. "
                 "Applying it on node sets 'extruded_dirichlet' and 'basal_dirichlet_bed' of the logically extruded mesh.\n"<<std::endl;
    }
  }

  auto& landiceBcList = probParamList.sublist("LandIce BCs");
  for (int i=0; i<landiceBcList.get<int>("Number",0); ++i) {
    auto& bc = landiceBcList.sublist(util::strint("BC",i));
    if (bc.isParameter("Side Set Name") && bc.get<std::string>("Side Set Name")=="ice_margin_side") {
      bc.set<std::string>("Side Set Name", "extruded_ice_margin_side");
      std::cout<<"\nWARNING: LandIce BC " << i << " uses side set 'ice_margin_side' of the 3D STK mesh. "
                 "Using side set 'extruded_ice_margin_side' of the logically extruded mesh.\n"<<std::endl;
    }
  }
}

// Builds the indices of the Albany dofs and elements corresponding to the MPAS entities (see ExtrudedMeshMaps).
// The basal triangles have the MPAS vertices in the order of verticesOnTria, and the 3D elements have the
// nodes of the basal triangle at the bottom, followed by the same nodes at the top.
void buildExtrudedMeshMaps (const MpasFields& f)
{
  auto disc = albanyApp->getDiscretization();
  const auto& layers_data = disc->getMeshStruct()->layers_data;
  const int numMeshLayers = layers_data.cell.lid->numLayers;
  const auto basal_cell_indexer = disc->getSideSetDiscretizations().at("basalside")->getDOFManager()->cell_indexer();
  const auto sol_dof_mgr = disc->getDOFManager();
  const auto ov_indexer = sol_dof_mgr->ov_indexer();
  const auto indexer = sol_dof_mgr->indexer();
  const auto elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();
  const auto& elem_ws_idx = disc->get_elements_workset_idx();
  const int neq = sol_dof_mgr->getNumFields();
  constexpr int nb = 3; // nodes of the basal triangles

  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(mpasVerticesOnTria.size())!=3*f.numTriangles(), std::logic_error,
      "Error! The triangles passed to velocity_solver_solve_fo are not those of the FE grid.\n");

  // offsets[eq][k]: offset of the dof of equation eq at node k of the 3D element
  std::vector<std::array<int,2*nb>> offsets(neq);
  for (int eq=0; eq<neq; ++eq) {
    for (int k=0; k<2*nb; ++k) {
      offsets[eq][k] = sol_dof_mgr->getGIDFieldOffsetsSubcell(eq,0,k)[0];
    }
  }

  auto& maps = extrudedMeshMaps;
  maps.neq = neq;
  maps.numMeshLayers = numMeshLayers;
  maps.ovLid.assign(static_cast<size_t>(f.numVertices3D())*neq, -1);
  maps.ownedLid.assign(static_cast<size_t>(f.numVertices3D())*neq, -1);
  maps.elemWsIdx.resize(static_cast<size_t>(f.numTriangles())*numMeshLayers);
  for (int it=0; it<f.numTriangles(); ++it) {
    const LO basal_elem_lid = basal_cell_indexer->getLocalElement(GO(f.indexToTriangleID[it]-1));
    TEUCHOS_TEST_FOR_EXCEPTION(basal_elem_lid<0, std::logic_error,
        "Error! Triangle " << f.indexToTriangleID[it] << " is not in the basal mesh.\n");
    for (int il=0; il<numMeshLayers; ++il) {
      const LO elem_lid = layers_data.cell.lid->getId(basal_elem_lid,il);
      maps.elemWsIdx[it*numMeshLayers+il] = elem_ws_idx[elem_lid];
      for (int side : {0,1}) { // 0=elem-bottom, 1=elem-top
        const int level = il+side;
        const int mpasLevel = depthIntegratedModel ? level*f.nLayers : level;
        for (int k=0; k<nb; ++k) {
          const int j = f.vertex3D(mpasLevel, mpasVerticesOnTria[3*it+k]);
          for (int eq=0; eq<neq; ++eq) {
            const LO ov_lid = elem_dof_lids(elem_lid,offsets[eq][k+side*nb]);
            maps.ovLid[j*neq+eq] = ov_lid;
            maps.ownedLid[j*neq+eq] = ov_lid<0 ? -1 : indexer->getLocalElement(ov_indexer->getGlobalElement(ov_lid));
          }
        }
      }
    }
  }
}

// Imports the geometry, the stiffening factor, the temperature and the Dirichlet velocity into the basal mesh,
// and sets the levels of the mesh
void importFieldsExtruded (const MpasFields& f)
{
  const int nLevels = f.nLayers+1;
  const int numVertices3D = f.numVertices3D();
  const auto& levels = f.levelsNormalizedThickness;
  TEUCHOS_TEST_FOR_EXCEPTION(static_cast<int>(levels.size())!=nLevels, std::runtime_error,
      "Error! The normalized levels passed by MPAS should have nLayers+1 entries.\n");

  // Mesh levels: the MPAS levels, or the bed and the surface with the depth-integrated model (one layer)
  extrudedMesh->setLevelsNormalizedCoords(depthIntegratedModel ? std::vector<double>{0.0, 1.0} : levels);

  auto& meta = *basalMesh->metaData;
  auto& bulk = *basalMesh->bulkData;
  STKFieldType* surfaceHeightField = meta.get_field <double> (NODE_RANK, "surface_height");
  STKFieldType* thicknessField = meta.get_field <double> (NODE_RANK, "ice_thickness");
  STKFieldType* stiffeningFactorLogField = meta.get_field <double> (NODE_RANK, "stiffening_factor_log");
  STKFieldType* dirichletField = meta.get_field <double> (NODE_RANK, "dirichlet_field");
  STKFieldType* temperatureField = meta.get_field <double> (ELEM_RANK, "temperature");

  const bool update_stiffeningFactorLog = canUpdate(f.stiffeningFactorData, stiffeningFactorLogField, "stiffening_factor_log");
  const bool update_dirichlet = canUpdate(f.velocityOnVertices, dirichletField, "dirichlet_field");
  const bool update_temperature = canUpdate(f.temperatureDataOnPrisms, temperatureField, "temperature");

  for (int ib=0; ib<f.numVertices(); ++ib) {
    stk::mesh::Entity node = bulk.get_entity(NODE_RANK, f.indexToVertexID[ib]);
    stk::mesh::field_data(*thicknessField, node)[0] = f.thicknessData[ib];
    stk::mesh::field_data(*surfaceHeightField, node)[0] = f.elevationData[ib];
    if (update_stiffeningFactorLog) {
      stk::mesh::field_data(*stiffeningFactorLogField, node)[0] = std::log(f.stiffeningFactorData[ib]);
    }
    if (update_dirichlet) {
      // velocityOnVertices stores initial guess and dirichlet velocities.
      // Layered fields store level il of component icmp at icmp*nLevels+il
      double* dirichletVel = stk::mesh::field_data(*dirichletField, node);
      for (int icmp=0; icmp<2; ++icmp) {
        for (int il=0; il<nLevels; ++il) {
          dirichletVel[icmp*nLevels+il] = f.velocityOnVertices[f.vertex3D(il,ib) + icmp*numVertices3D];
        }
      }
    }
  }

  if (update_temperature) {
    for (int it=0; it<f.numTriangles(); ++it) {
      stk::mesh::Entity elem = bulk.get_entity(ELEM_RANK, f.indexToTriangleID[it]);
      double* temperature = stk::mesh::field_data(*temperatureField, elem);
      for (int il=0; il<f.nLayers; ++il) {
        temperature[il] = f.temperatureDataOnPrisms[f.prism(il,it)];
      }
    }
  }

  // Normalized coordinates of the data of the layered fields: the levels for the Dirichlet velocity,
  // and the layers midpoints for the temperature
  auto& vec_states = basalMesh->get_field_accessor()->getMeshVectorStates();
  if (update_dirichlet) {
    vec_states["dirichlet_field_NLC"] = levels;
  }
  if (update_temperature) {
    auto& nlc = vec_states["temperature_NLC"];
    nlc.resize(f.nLayers);
    for (int il=0; il<f.nLayers; ++il) {
      nlc[il] = 0.5*(levels[il]+levels[il+1]);
    }
  }
}

// Initial guess for the solver, from the velocity passed by MPAS (and the thickness, if it is a dof)
Teuchos::RCP<Thyra_Vector> extrudedInitialGuess (const MpasFields& f)
{
  const auto& maps = extrudedMeshMaps;
  const int numVertices3D = f.numVertices3D();
  auto x0 = Thyra::createMember(albanyApp->getDiscretization()->getVectorSpace());
  x0->assign(0.0);
  auto x0_data = Albany::getNonconstLocalData(*x0);
  for (int ib=0; ib<f.numVertices(); ++ib) {
    for (int il=0; il<=f.nLayers; ++il) {
      const int j = f.vertex3D(il,ib);
      for (int eq=0; eq<maps.neq; ++eq) {
        const LO lid = maps.ownedLid[j*maps.neq+eq];
        if (lid<0) continue;
        x0_data[lid] = (eq<2) ? f.velocityOnVertices[j + eq*numVertices3D] : f.thicknessData[ib];
      }
    }
  }
  return x0;
}

// Exports the velocity, the dissipation heat and the body force
void exportFieldsExtruded (MpasFields& f)
{
  const auto& maps = extrudedMeshMaps;
  const int neq = maps.neq;
  const int numVertices3D = f.numVertices3D();

  // After the solve, the observer stores the solution in the overlapped solution of the solution manager.
  // If the observer is not called, this holds the initial guess (see velocity_solver_solve_fo).
  const auto solVec = albanyApp->getAdaptSolMgr()->getOverlappedSolution()->col(0);
  const auto sol = Albany::getLocalData(*solVec);
  const auto& ovLid = maps.ovLid;
  if (depthIntegratedModel) {
    // Vertical profile of the depth-integrated velocity at the MPAS levels
    std::vector<double> phiBed, phiTop;
    depthIntegratedVerticalProfile(f.levelsNormalizedThickness, phiBed, phiTop);

    for (int ib=0; ib<f.numVertices(); ++ib) {
      const int jBed = f.vertex3D(0,ib), jTop = f.vertex3D(f.nLayers,ib);
      if (ovLid[jBed*neq]<0 || ovLid[jBed*neq+1]<0 || ovLid[jTop*neq]<0 || ovLid[jTop*neq+1]<0) continue;
      for (int icmp=0; icmp<2; ++icmp) {
        const double solBed = sol[ovLid[jBed*neq+icmp]];
        const double solTop = sol[ovLid[jTop*neq+icmp]];
        for (int il=0; il<=f.nLayers; ++il) {
          f.velocityOnVertices[f.vertex3D(il,ib) + icmp*numVertices3D] = phiBed[il]*solBed + phiTop[il]*solTop;
        }
      }
    }
  } else {
    for (int j=0; j<numVertices3D; ++j) {
      for (int icmp=0; icmp<2; ++icmp) {
        const LO lid = ovLid[j*neq+icmp];
        if (lid>=0) {
          f.velocityOnVertices[j + icmp*numVertices3D] = sol[lid];
        }
      }
    }
  }

  // The 3D elem states (stored, in the logically extruded mesh, by the field accessor)
  auto& states = albanyApp->getDiscretization()->getMeshStruct()->get_field_accessor()->getElemStates();
  if (!f.dissipationHeatOnPrisms.empty())
    std::fill(f.dissipationHeatOnPrisms.begin(), f.dissipationHeatOnPrisms.end(), 0.0);
  for (int it=0; it<f.numTriangles(); ++it) {
    for (int il=0; il<f.nLayers; ++il) {
      // With the depth-integrated model, every MPAS layer gets the value of the (only) mesh layer
      const int meshLayer = depthIntegratedModel ? 0 : il;
      const auto& wsIdx = maps.elemWsIdx[it*maps.numMeshLayers+meshLayer];
      auto& ws_states = states[wsIdx.ws];
      if (!f.dissipationHeatOnPrisms.empty()) {
        const auto st = ws_states.find("dissipation_heat");
        if (st!=ws_states.end()) {
          f.dissipationHeatOnPrisms[f.prism(il,it)] += st->second.host()(wsIdx.idx);
        }
      }
      if (il==0 && !f.bodyForceMagnitudeOnBasalCell.empty()) {
        double bf = 0;
        const auto st = ws_states.find("body_force");
        if (st!=ws_states.end()) {
          const auto bodyForce = st->second.host();
          bf = bodyForce(wsIdx.idx,0)*bodyForce(wsIdx.idx,0) + bodyForce(wsIdx.idx,1)*bodyForce(wsIdx.idx,1);
        }
        f.bodyForceMagnitudeOnBasalCell[it] = std::sqrt(bf);
      }
    }
  }
}

} // anonymous namespace

/***********************************************************/

// Note: betaData can be input (if prescribing basal friction)
//       or output (if using a sliding law)
void velocity_solver_solve_fo(int nLayers, int globalVerticesStride,
    int globalTrianglesStride, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double, // minBeta
    const std::vector<double>&, // regulThk,
    const std::vector<double>& levelsNormalizedThickness,
    const std::vector<double>& elevationData,
    const std::vector<double>& thicknessData,
          std::vector<double>& betaData,
    const std::vector<double>& bedTopographyData,
    const std::vector<double>&, // smbData,
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
    const double& deltat)
{
  InterfaceCallTimer callTimer;
  auto solveTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: SolveFO"));
  // No-op if velocity_solver_extrude_3d_grid already released it (FEM grid regenerated).
  releasePreviousSolver();
  PhaseTimer phase("Albany: Import Fields");

  MpasFields fields {nLayers, globalVerticesStride, globalTrianglesStride, ordering ? 1 : 0,
                     indexToVertexID, indexToTriangleID, levelsNormalizedThickness, elevationData, thicknessData,
                     betaData, bedTopographyData, stiffeningFactorData, effectivePressureData, muData,
                     bedRoughnessData, bulkFrictionData, basalDebrisData, temperatureDataOnPrisms,
                     bodyForceMagnitudeOnBasalCell, dissipationHeatOnPrisms, velocityOnVertices};

  *MPAS_dt =  deltat;

  auto& probParamList = paramList->sublist("Problem");

  TEUCHOS_TEST_FOR_EXCEPTION(noupdate_fields.find("surface_height") != noupdate_fields.end(), std::runtime_error,
          "Error! surface_height field needs to be set by MPAS.\n");
  TEUCHOS_TEST_FOR_EXCEPTION(noupdate_fields.find("ice_thickness") != noupdate_fields.end(), std::runtime_error,
          "Error! ice_thickness field needs to be set by MPAS.\n");

  // 3D fields (and, for the logically extruded mesh, the mesh levels)
  std::vector<stk::mesh::Entity> nodes3D;
  Teuchos::RCP<const Albany::AbstractSTKMeshStruct> basalSTKMesh;
  std::vector<int> basalNodeIds;
  if (useExtrudedMesh) {
    importFieldsExtruded(fields);
    basalSTKMesh = basalMesh;
    basalNodeIds = indexToVertexID;
  } else {
    nodes3D = importFieldsSTK3D(fields);
    basalSTKMesh = Teuchos::rcp_dynamic_cast<const Albany::AbstractSTKMeshStruct>(meshStruct->sideSetMeshStructs.at("basalside"),true);
    basalNodeIds = basalNodeIdsSTK3D(fields);
  }

  // basal fields
  const auto basalNodes = importBasalFields(*basalSTKMesh, basalNodeIds, fields);

  if (!useExtrudedMesh) {
    meshStruct->setHasRestartSolution(true);//!first_time_step);
  }

  if (!first_time_step) {
    const double restartDataTime = probParamList.get("Homotopy Restart Step", 1.);
    if (!useExtrudedMesh) {
      meshStruct->setRestartDataTime(restartDataTime);
    }
    double homotopy =
        probParamList.sublist("LandIce Viscosity").get(
            "Glen's Law Homotopy Parameter", 1.0);
    if (restartDataTime == homotopy) {
      probParamList.set("Solution Method", "Steady");
      paramList->sublist("Piro").set("Solver Type", "NOX");
    }
  }

  // updateMesh has its own timer ("STKDiscretization: updateMesh" or "ExtrudedDiscretization: updateMesh")
  phase.stop();
  albanyApp->getDiscretization()->updateMesh();

  phase.start("Albany: Final SetUp");
  // With the 3D STK mesh, the initial guess is read from the mesh (see setHasRestartSolution above)
  Teuchos::RCP<Thyra_Vector> initialGuess;
  if (useExtrudedMesh) {
    buildExtrudedMeshMaps(fields);
    initialGuess = extrudedInitialGuess(fields);
  }
  albanyApp->finalSetUp(paramList, initialGuess);
  if (useExtrudedMesh) {
    // The overlapped solution is updated by the observer after the solve. Set it to the initial guess,
    // so that, if the observer is not called (e.g., the solve fails), the initial guess is returned to
    // MPAS, as with the 3D STK mesh.
    albanyApp->getAdaptSolMgr()->updateAndReturnOverlapSolution(*initialGuess);
  }

  if (keptMesh) albanyApp->getPhxSetup()->reboot_memoizer();

  bool success = true;
  try {
    phase.start("Albany: Create Model And Solver");
    auto model = slvrfctry->createModel(albanyApp);
    solver = slvrfctry->createSolver(model, Teuchos::null, true);

    phase.start("Albany: Perform Solve");
    Teuchos::ParameterList solveParams;
    solveParams.set("Compute Sensitivities", false);

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<
    Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
    Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

    // Printing responses
    const unsigned int num_g = solver->Ng();
    for (unsigned int i=0; i<num_g-1; i++) {
      if (albanyApp->getResponse(i)->isScalarResponse()) {
        Thyra::ConstDetachedVectorView<double> g(thyraResponses[i]);
        std::cout << std::setprecision(15) << "\nResponse " << i << ": " << g[0] << std::endl;
      }
    }
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

  phase.start("Albany: Export Fields");
  error = !success || (albanyApp->getSolutionStatus() != Albany::Application::SolutionStatus::Converged);

  if (useExtrudedMesh) {
    exportFieldsExtruded(fields);
  } else {
    exportFieldsSTK3D(fields, nodes3D);
  }
  exportBasalFields(*basalSTKMesh, basalNodes, fields);

  keptMesh = true;
}

void velocity_solver_export_fo_velocity(MPI_Comm reducedComm) {
#ifdef ALBANY_SEACAS
  InterfaceCallTimer callTimer;
  auto exportTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Export FO Velocity"));
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(reducedComm));
  // With the logically extruded mesh, this writes the basal mesh, where the 3D fields (e.g., the solution)
  // are stored as layered fields
  if (useExtrudedMesh) {
    mesh_data->set_bulk_data(*basalMesh->bulkData);
  } else {
    mesh_data->set_bulk_data(*meshStruct->bulkData);
  }
  size_t idx = mesh_data->create_output_mesh("IceSheet.exo", stk::io::WRITE_RESULTS);
  mesh_data->process_output_request(idx, 0.0);
#endif
}

int velocity_solver_init_mpi(MPI_Comm comm) {
  mpiCommMPAS = Albany::createTeuchosCommFromMpiComm(comm);
  if(! Kokkos::is_initialized()) {
    Kokkos::initialize();
    kokkosInitializedByAlbany = true;
  }
  // The base timer is not started here: InterfaceCallTimer starts and stops it
  // at each call into Albany, so it does not include the time spent in MPAS.
  stackedTimer = Teuchos::rcp(new Teuchos::StackedTimer("Albany Velocity Solver", false));
  Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
  return 0;
}

void velocity_solver_finalize() {
  {
    InterfaceCallTimer callTimer;
    auto finalizeTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Finalize"));
    meshStruct = Teuchos::null;
    extrudedMesh = Teuchos::null;
    basalMesh = Teuchos::null;
    albanyApp = Teuchos::null;
    paramList = Teuchos::null;
    slvrfctry = Teuchos::null;
    MPAS_dt = Teuchos::null;
    solver = Teuchos::null;
    mpiComm = Teuchos::null;
  }

  // Print Teuchos timers into file
  std::ostream* os = &std::cout;
  std::ofstream ofs;
  if (mpiCommMPAS->getRank() == 0) {
    ofs.open("log.albany.timers.out", std::ofstream::out);
    os = &ofs;
  }
  // No need to stop the base timer: it only runs inside the interface calls (see InterfaceCallTimer).
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = true;
  options.output_minmax = true;
  stackedTimer->report(*os, mpiCommMPAS, options);
  stackedTimer = Teuchos::null;

  mpiCommMPAS = Teuchos::null;
  if(kokkosInitializedByAlbany)
    Kokkos::finalize();
}


//DEPRECATED
void velocity_solver_solve_fo(int nLayers, int globalVerticesStride,
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
    const double& deltat) {
      std::vector<double> effectivePressureDataNonConst(effectivePressureData);
      const std::vector<double> bedRoughnessData;
      const std::vector<double> bulkFrictionData;
      const std::vector<double> basalDebrisData;
      velocity_solver_solve_fo(nLayers, globalVerticesStride, globalTrianglesStride, ordering, first_time_step,
                               indexToVertexID,indexToTriangleID, minBeta, regulThk, levelsNormalizedThickness,
                               elevationData, thicknessData, betaData, bedTopographyData, smbData, stiffeningFactorData,
                               effectivePressureDataNonConst, muData, bedRoughnessData, bulkFrictionData, basalDebrisData,
                               temperatureDataOnPrisms, bodyForceMagnitudeOnBasalCell,dissipationHeatOnPrisms, velocityOnVertices, error, deltat );
}


/*duality:
 *
 *   mpas(F) |  albany
 *  ---------|---------
 *   cell    |  node
 *   vertex  |  triangle
 *   edge    |  edge
 *
 */

void velocity_solver_compute_2d_grid(MPI_Comm reducedComm) {
  InterfaceCallTimer callTimer;
  auto grid2DTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Compute 2D Grid"));
  keptMesh = false;
  mpiComm = Albany::createTeuchosCommFromMpiComm(reducedComm);
}

void velocity_solver_set_physical_parameters(double const& gravity, double const& ice_density, double const& ocean_density, double const& sea_level,
                                             double const& flowParamA, double const& flowLawExponent, double const& dynamic_thickness, bool const& use_GLP,
                                             double const& clausiusClapeyoronCoeff, double const& melting_temperature, double const& seconds_per_year, double const& gas_constant) {
  MPAS_gravity=gravity;
  MPAS_rho_ice = ice_density;
  MPAS_rho_seawater = ocean_density;
  MPAS_sea_level = sea_level;
  MPAS_flowParamA = flowParamA;
  MPAS_flowLawExponent = flowLawExponent;
  MPAS_dynamic_thickness = dynamic_thickness;
  MPAS_useGLP = use_GLP;
  MPAS_ClausiusClapeyoronCoeff = clausiusClapeyoronCoeff;
  MPAS_meltingTemperature = melting_temperature;
  MPAS_secondsPerYear = seconds_per_year;
  MPAS_gasConstant = gas_constant;
}

//DEPRECATED
void velocity_solver_set_physical_parameters(double const& gravity, double const& ice_density, double const& ocean_density, double const& sea_level, double const& flowParamA, double const& flowLawExponent, double const& dynamic_thickness, bool const& use_GLP, double const& clausiusClapeyoronCoeff) {
  velocity_solver_set_physical_parameters(gravity, ice_density, ocean_density, sea_level, flowParamA, flowLawExponent,
                                          dynamic_thickness, use_GLP, clausiusClapeyoronCoeff, 273.15, 3.1536e7, 8.314);
}

void velocity_solver_extrude_3d_grid(int nLayers, int globalTrianglesStride,
    int globalVerticesStride, int globalEdgesStride, int Ordering, MPI_Comm /* reducedComm */,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& vertexProcIDs,
    const std::vector<double>& verticesCoords,
    const std::vector<int>& verticesOnTria,
    const std::vector<std::vector<int>>  procsSharingVertices,
    const std::vector<bool>& isBoundaryEdge,
    const std::vector<int>& trianglesOnEdge,
    const std::vector<int>& verticesOnEdge,
    const std::vector<int>& indexToEdgeID,
    const std::vector<int>& indexToTriangleID,
    const std::vector<int>& dirichletNodesIds,
    const std::vector<int>& iceMarginEdgesIds) {

  InterfaceCallTimer callTimer;
  auto grid3DTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Extrude 3D Grid"));

  // The FEM grid is being regenerated: release the objects built on the previous grid before
  // building the new ones, so that their teardown is timed on its own and, on GPUs, their device
  // memory is freed before the new application, mesh and discretization are allocated.
  // The solver goes first, since it references the application, which references the mesh.
  releasePreviousSolver();
  if (Teuchos::nonnull(albanyApp) || Teuchos::nonnull(meshStruct) || Teuchos::nonnull(extrudedMesh)) {
    auto releaseTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Release Previous Application And Mesh"));
    albanyApp = Teuchos::null;
    meshStruct = Teuchos::null;
    extrudedMesh = Teuchos::null;
    basalMesh = Teuchos::null;
  }

  PhaseTimer phase("Albany: Setup Parameters");

  paramList = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromYamlFileAndBroadcast("albany_input.yaml", paramList.ptr(), *mpiComm);

  slvrfctry = Teuchos::rcp(new Albany::SolverFactory(paramList, mpiComm));
  //paramList = Teuchos::rcp(&slvrfctry->getParameters(), false);

  paramList->set("Overwrite Nominal Values With Final Point", true);
  Teuchos::ParameterList& probParamList = paramList->sublist("Problem");

  // Mesh: logically extruded mesh (default), or 3D STK mesh
  const std::string discMethod = paramList->sublist("Discretization").get<std::string>("Method", "Extruded");
  TEUCHOS_TEST_FOR_EXCEPTION(discMethod != "Extruded" && discMethod != "STKExtruded", std::runtime_error,
      "Error! Invalid Discretization Method '" << discMethod << "' in the Albany input file.\n"
      "  Valid methods are 'Extruded' (logically extruded mesh, the default) and 'STKExtruded' (3D STK mesh).\n");
  useExtrudedMesh = (discMethod == "Extruded");

  //Physical Parameters
  if(probParamList.isSublist("LandIce Physical Parameters")) {
    std::cout<<"\nWARNING: Using Physical Parameters (gravity, ice/ocean densities) provided in Albany input file. In order to use those provided by MPAS, remove \"LandIce Physical Parameters\" sublist from Albany input file.\n"<<std::endl;
  }

  depthIntegratedModel = probParamList.get("Depth Integrated Model",false);

  Teuchos::ParameterList& physParamList = probParamList.sublist("LandIce Physical Parameters");
  double rho_ice, rho_seawater;
  physParamList.set("Gravity Acceleration", physParamList.get("Gravity Acceleration", MPAS_gravity));
  physParamList.set("Ice Density", rho_ice = physParamList.get("Ice Density", MPAS_rho_ice));
  physParamList.set("Water Density", rho_seawater = physParamList.get("Water Density", MPAS_rho_seawater));
  physParamList.set("Clausius-Clapeyron Coefficient", physParamList.get("Clausius-Clapeyron Coefficient", MPAS_ClausiusClapeyoronCoeff));
  physParamList.set("Atmospheric Pressure Melting Temperature", physParamList.get("Atmospheric Pressure Melting Temperature", MPAS_meltingTemperature));
  physParamList.set("Seconds per Year", physParamList.get("Seconds per Year", MPAS_secondsPerYear));
  physParamList.set<bool>("Use GLP", physParamList.get("Use GLP", MPAS_useGLP)); //use GLP (Grounding line parametrization) unless actively disabled

  probParamList.set("Name", probParamList.get("Name", "LandIce Stokes First Order 3D"));

  MPAS_dt = Teuchos::rcp(new double(0.0));
  if (probParamList.get<std::string>("Name") == "LandIce Coupled FO H 3D") {
    probParamList.set("Time Step Ptr", MPAS_dt);
  }

  if(probParamList.isSublist("LandIce BCs"))
    std::cout<<"\nWARNING: Using LandIce BCs provided in Albany input file. In order to use boundary conditions provided by MPAS, remove \"LandIce BCs\" sublist from Albany input file.\n"<<std::endl;

  // ---- Setting Memoization ---- //
  probParamList.set<bool>("Use MDField Memoization", probParamList.get<bool>("Use MDField Memoization", true));

  // ---- Phalanx timers: field managers are rebuilt during a MALI run, so keep timer names
  //      free of the Phalanx index, otherwise each rebuild creates a new set of timers ---- //
  probParamList.set<bool>("Add Index To Phalanx Timer Names", probParamList.get<bool>("Add Index To Phalanx Timer Names", false));

  // ------ whether to use the flat bed approximation (that is, tangent velocity to the bed is the same as horizontal velocity at bed)
  probParamList.set<bool>("Flat Bed Approximation", probParamList.get<bool>("Flat Bed Approximation", true));

  // ---- Setting parameters for LandIce BCs ---- //
  Teuchos::ParameterList& landiceBcList = probParamList.sublist("LandIce BCs");
  landiceBcList.set<int>("Number",2);

  // Basal Friction BC
  auto& basalParams = landiceBcList.sublist("BC 0");
  auto& basalFrictionParams = basalParams.sublist("Basal Friction Coefficient");
  bool zeroEffectPressOnShelf = basalFrictionParams.get<bool>("Zero Effective Pressure On Floating Ice At Nodes", !physParamList.get<bool>("Use GLP"));
  bool zeroBetaOnShelf = basalFrictionParams.get<bool>("Zero Beta On Floating Ice", physParamList.get<bool>("Use GLP"));
  if(zeroEffectPressOnShelf && zeroBetaOnShelf)
    std::cout<<"\nWARNING: Both BC options \"Zero Effective Pressure On Floating Ice At Nodes\" and \"Zero Beta On Floating Ice\" are set to true. To avoid unexpected beahvior only one should be set to true.\n"<<std::endl;

  int basal_cub_degree = physParamList.get<bool>("Use GLP") ? 8 : 4;
  probParamList.set<int>("Basal Cubature Degree",probParamList.get<int>("Basal Cubature Degree", basal_cub_degree));
  basalParams.set("Side Set Name", basalParams.get("Side Set Name", "basalside"));
  probParamList.set("Basal Side Name", basalParams.get<std::string>("Side Set Name"));
  basalParams.set("Type", basalParams.get("Type", "Basal Friction"));
  auto betaType = util::upper_case(basalFrictionParams.get<std::string>("Type","Power Law"));
  basalFrictionParams.set("Type",betaType);
  basalFrictionParams.set("Power Exponent", basalFrictionParams.get("Power Exponent",1.0));
  basalFrictionParams.set("Mu Type",basalFrictionParams.get("Mu Type","Field"));
  basalFrictionParams.set("Mu Field Name",basalFrictionParams.get("Mu Field Name","mu"));
  basalFrictionParams.set("Effective Pressure Type",basalFrictionParams.get("Effective Pressure Type","Field"));
  std::string effectivePressureType = util::upper_case(basalFrictionParams.get<std::string>("Effective Pressure Type"));
  bool isEPressureOutput = effectivePressureType.find("HYDROSTATIC")!=std::string::npos;

  bool requireBedRoughness(false), requireBulkFriction(false), requireBasalDebris(false);
  if((betaType == "REGULARIZED COULOMB") || (betaType == "DEBRIS FRICTION")) {
    basalFrictionParams.set("Bed Roughness Type",basalFrictionParams.get("Bed Roughness Type","Field"));
    requireBedRoughness = true;
    if(betaType == "DEBRIS FRICTION") {
      basalFrictionParams.set("Bulk Friction Coefficient Type",basalFrictionParams.get("Bulk Friction Coefficient Type","Field"));
      basalFrictionParams.set("Basal Debris Factor Type", basalFrictionParams.get("Basal Debris Factor Type","Field"));
      requireBulkFriction = true;
      requireBasalDebris = true;
    }
  }

  basalFrictionParams.set<bool>("Zero Beta On Floating Ice", zeroBetaOnShelf);
  basalFrictionParams.set<bool>("Zero Effective Pressure On Floating Ice At Nodes", zeroEffectPressOnShelf);

  //Lateral floating ice BCs
  int lateral_cub_degree = 4;
  auto& lateralParams = landiceBcList.sublist("BC 1");
  lateralParams.set<int>("Cubature Degree",lateralParams.get<int>("Cubature Degree", lateral_cub_degree));
  //If the following option is not specified (recommended)
  // Albany will compute it based on the geometry
  if(lateralParams.isParameter("Immersed Ratio"))
    lateralParams.set<double>("Immersed Ratio",lateralParams.get<double>("Immersed Ratio"));
  // In the logically extruded mesh, the side sets of the basal mesh are extruded as extruded_<name>
  lateralParams.set("Side Set Name", lateralParams.get<std::string>("Side Set Name", useExtrudedMesh ? "extruded_ice_margin_side" : "ice_margin_side"));
  lateralParams.set("Type", lateralParams.get("Type", "Lateral"));

  //Dirichlet BCs
  if(!probParamList.isSublist("Dirichlet BCs")) {
    if (useExtrudedMesh) {
      // Whole columns and bed nodes (see dirichletNodeSets)
      for (const std::string ns : {"extruded_dirichlet", "basal_dirichlet_bed"}) {
        probParamList.sublist("Dirichlet BCs").set("SDBC on NS " + ns + " for DOF U0 prescribe Field", "dirichlet_field");
        probParamList.sublist("Dirichlet BCs").set("SDBC on NS " + ns + " for DOF U1 prescribe Field", "dirichlet_field");
      }
    } else {
      probParamList.sublist("Dirichlet BCs").set("SDBC on NS dirichlet for DOF U0 prescribe Field", "dirichlet_field");
      probParamList.sublist("Dirichlet BCs").set("SDBC on NS dirichlet for DOF U1 prescribe Field", "dirichlet_field");
    }
  }
  else {
    std::cout<<"\nWARNING: Using Dirichlet BCs options provided in Albany input file. In order to use those provided by MPAS, remove \"Dirichlet BCs\" sublist from Albany input file.\n"<<std::endl;
  }

  // Input files written for the 3D STK mesh can refer to its node sets and side sets
  if (useExtrudedMesh) {
    translateMeshPartNames(probParamList);
  }

  if(probParamList.isSublist("LandIce Field Norm") && probParamList.sublist("LandIce Field Norm").isSublist("sliding_velocity_basalside"))
    std::cout<<"\nWARNING: Using options for Velocity Norm provided in Albany input file. In order to use those provided by MPAS, remove \"LandIce Velocity Norm\" sublist from Albany input file.\n"<<std::endl;

  Teuchos::ParameterList& fieldNormList =  probParamList.sublist("LandIce Field Norm").sublist("sliding_velocity_basalside"); //empty list if LandIceViscosity not in input file.
  fieldNormList.set("Regularization Type", fieldNormList.get("Regularization Type", "Given Value"));
  double reg_value = 1e-6;
  fieldNormList.set("Regularization Value", fieldNormList.get("Regularization Value", reg_value));


  if(probParamList.isSublist("LandIce Viscosity"))
    std::cout<<"\nWARNING: Using Viscosity options provided in Albany input file. In order to use those provided by MPAS, remove \"LandIce Viscosity\" sublist from Albany input file.\n"<<std::endl;

  Teuchos::ParameterList& viscosityList =  probParamList.sublist("LandIce Viscosity"); //empty list if LandIceViscosity not in input file.

  viscosityList.set("Type", viscosityList.get("Type", "Glen's Law"));
  double homotopy_param = (probParamList.get("Solution Method", "Steady") == "Steady") ? 0.3 : 1.0;
  viscosityList.set("Glen's Law Homotopy Parameter", viscosityList.get("Glen's Law Homotopy Parameter", homotopy_param));
  // Convert MPAS value for A from [k^-1 kPa^-n yr^-1] to [Pa^-n s^-1]
  viscosityList.set("Glen's Law n", viscosityList.get("Glen's Law n",  MPAS_flowLawExponent));
  const double spy = physParamList.get<double>("Seconds per Year");
  const auto knp1 = std::pow(1000,MPAS_flowLawExponent+1);
  const auto A = MPAS_flowParamA / knp1 / spy;
  viscosityList.set("Glen's Law A", viscosityList.get("Glen's Law A", A));
  viscosityList.set("Flow Rate Type", viscosityList.get("Flow Rate Type", "Temperature Based"));
  Teuchos::ParameterList& arrheniusList = viscosityList.sublist("Arrhenius Relationship");
  arrheniusList.set("Gas Constant", arrheniusList.get("Gas Constant", MPAS_gasConstant));
  viscosityList.set("Use Stiffening Factor", viscosityList.get("Use Stiffening Factor", true));
  viscosityList.set("Extract Strain Rate Sq", viscosityList.get("Extract Strain Rate Sq", true)); //set true if not defined
  viscosityList.set("Use P0 Temperature", viscosityList.get("Use P0 Temperature", !depthIntegratedModel));
  TEUCHOS_TEST_FOR_EXCEPTION(viscosityList.get<bool>("Use P0 Temperature") == depthIntegratedModel, std::runtime_error,
      "Error! The provided value Use P0 Temperature is not consistent with the model (FO or MOLHO) used.\n");


  probParamList.sublist("Body Force").set("Type", "FO INTERP SURF GRAD");

  //! set Rigid body modes for prec. near null space
  if(!probParamList.isSublist("LandIce Rigid Body Modes For Preconditioner")) {
    Teuchos::ParameterList& rbmList = probParamList.sublist("LandIce Rigid Body Modes For Preconditioner");
    rbmList.set<bool>("Compute Constant Modes", true);
    rbmList.set<bool>("Compute Rotation Modes", true);
  }

  Teuchos::Array<int> defaultCubatureDegrees(2);
  defaultCubatureDegrees[0] = 4; defaultCubatureDegrees[1]= depthIntegratedModel ? 7 : 3;

  if(!probParamList.isParameter("Cubature Degree") || depthIntegratedModel)
    probParamList.set("Cubature Degrees (Horiz Vert)", probParamList.get("Cubature Degrees (Horiz Vert)", defaultCubatureDegrees));  //set Cubature Degrees if not defined


  auto discretizationList = Teuchos::sublist(paramList, "Discretization", true);

  discretizationList->set("Workset Size", discretizationList->get("Workset Size", -1));

  discretizationList->set("Method", discMethod);

  auto& rfi = discretizationList->sublist("Required Fields Info");
  int fp = rfi.get<int>("Number Of Fields",0);
  noupdate_fields.clear();
  std::set<std::string> existing_fields;
  //first process fields added in the input yaml file
  for (int i=0; i<fp; ++i) {
    auto field_name = rfi.sublist(util::strint("Field",i)).get<std::string>("Field Name");
    existing_fields.insert(field_name);
    //these fields are either read from file or outputs, they should not be overwritten with MPAS fields.
    if(rfi.sublist(util::strint("Field",i)).get("Field Origin", "Mesh") != "Mesh")
      noupdate_fields.insert(field_name);
  }

  //set temperature
  if (existing_fields.find("temperature") == existing_fields.end())
  {
    Teuchos::ParameterList& field = rfi.sublist(util::strint("Field",fp++));
    field.set<std::string>("Field Name", "temperature");
    if(depthIntegratedModel)
      field.set<std::string>("Field Type", "QuadPoint Scalar");
    else
      field.set<std::string>("Field Type", "Elem Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set ice thickness
  if (existing_fields.find("ice_thickness") == existing_fields.end())
  {
    Teuchos::ParameterList& field = rfi.sublist(util::strint("Field",fp++));
    field.set<std::string>("Field Name", "ice_thickness");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set surface_height
  if (existing_fields.find("surface_height") == existing_fields.end())
  {
    Teuchos::ParameterList& field = rfi.sublist(util::strint("Field",fp++));
    field.set<std::string>("Field Name", "surface_height");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set dirichlet field
  if (existing_fields.find("dirichlet_field") == existing_fields.end())
  {
    Teuchos::ParameterList& field = rfi.sublist(util::strint("Field",fp++));
    field.set<std::string>("Field Name", "dirichlet_field");
    field.set<std::string>("Field Type", "Node Vector");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set stiffening factor
  if (existing_fields.find("stiffening_factor_log") == existing_fields.end())
  {
    Teuchos::ParameterList& field = rfi.sublist(util::strint("Field",fp++));
    field.set<std::string>("Field Name", "stiffening_factor_log");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  // Outputs
  if (existing_fields.find("body_force") == existing_fields.end())
  {
    Teuchos::ParameterList& field = rfi.sublist(util::strint("Field",fp++));
    field.set<std::string>("Field Name", "body_force");
    field.set<std::string>("Field Type", "Elem Vector");
    field.set<std::string>("Field Usage", "Output");
  }

  // set number of requested fields
  rfi.set<int>("Number Of Fields",fp);

  // Side set outputs
  auto& ss_pl =discretizationList->sublist("Side Set Discretizations");
  Teuchos::Array<std::string> bsn (1,"basalside");
  ss_pl.set<Teuchos::Array<std::string>>("Side Sets", bsn);
  auto& basal_pl = ss_pl.sublist("basalside");
  // The basal mesh of the logically extruded mesh is built by the interface
  if (!useExtrudedMesh) {
    basal_pl.set<std::string>("Method","SideSetSTK");
  }
  auto& basal_req = basal_pl.sublist("Required Fields Info");

  int bfp = basal_req.get<int>("Number Of Fields",0);
  existing_fields.clear();
  for (int i=0; i<bfp; ++i) {
    auto field_name = basal_req.sublist(util::strint("Field",i)).get<std::string>("Field Name");
    existing_fields.insert(field_name);
    //these fields are either read from file or outputs, they should not be overwritten with MPAS fields.
    if(basal_req.sublist(util::strint("Field",i)).get("Field Origin", "Mesh") != "Mesh")
      noupdate_fields.insert(field_name);
  }


  //set bed topography
  if (existing_fields.find("bed_topography") == existing_fields.end())
  {
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "bed_topography");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set mu field
  if (existing_fields.find("mu") == existing_fields.end())
  {
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "mu");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set bed roughness
  if(requireBedRoughness && (existing_fields.find("bed_roughness") == existing_fields.end())) {
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "bed_roughness");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set basal debris factor
  if(requireBasalDebris && (existing_fields.find("basal_debris_factor") == existing_fields.end())){
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "basal_debris_factor");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set bulk friction
  if(requireBulkFriction && (existing_fields.find("bulk_friction") == existing_fields.end())) {
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "bulk_friction");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Origin", "Mesh");
  }

  //set effective pressure
  if (existing_fields.find("effective_pressure") == existing_fields.end())
  {
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "effective_pressure");
    field.set<std::string>("Field Type", "Node Scalar");
    if(isEPressureOutput) {
      field.set<std::string>("Field Usage", "Output");
      noupdate_fields.insert("effective_pressure");
    }
    else
      field.set<std::string>("Field Origin", "Mesh");
  }

  //set beta
  if (existing_fields.find("beta") == existing_fields.end())
  {
    Teuchos::ParameterList& field = basal_req.sublist(util::strint("Field",bfp++));
    field.set<std::string>("Field Name", "beta");
    field.set<std::string>("Field Type", "Node Scalar");
    field.set<std::string>("Field Usage", "Output");
  }

  basal_req.set<int>("Number Of Fields",bfp);

  // In the logically extruded mesh, the 3D fields are stored on the basal mesh
  if (useExtrudedMesh) {
    setExtrudedMeshParameters(*discretizationList, nLayers);
  }

  phase.start("Albany: Create Application");
  // Register LandIce problems
  auto& pb_factories = Albany::FactoriesContainer<Albany::ProblemFactory>::instance();
  pb_factories.add_factory(LandIce::LandIceProblemFactory::instance());

  // Create albany app
  albanyApp = Teuchos::rcp(new Albany::Application(mpiComm));
  albanyApp->initialSetUp(paramList);

  phase.start("Albany: Create Mesh");
  //Get number of params in problem - needed for MeshStruct constructor
  int num_params = Albany::CalculateNumberParams(Teuchos::sublist(paramList, "Problem", true));

  // MpasSTKMeshStruct stores references to its input arrays, and uses them in setBulkData (called by
  // createDiscretization, at the end of this function), so this array must live until then.
  //temporary fix, TODO: use GO for indexToTriangleID (need to synchronize with MPAS).
  std::vector<GO> indexToTriangleGOID;

  if (useExtrudedMesh) {
    // Basal mesh: the MPAS triangulation, with the Dirichlet nodes and the ice margin
    const auto nodeSets = dirichletNodeSets(dirichletNodesIds, indexToVertexID, nLayers, Ordering, globalVerticesStride);
    std::map<std::string,std::vector<int>> sideSets;
    auto& iceMarginEdges = sideSets["ice_margin_side"];
    std::unordered_map<int,int> edgeIdToIndex;
    for (int ie=0; ie<static_cast<int>(indexToEdgeID.size()); ++ie) {
      edgeIdToIndex[indexToEdgeID[ie]] = ie;
    }
    for (int id : iceMarginEdgesIds) {
      const auto it = edgeIdToIndex.find(id);
      TEUCHOS_TEST_FOR_EXCEPTION(it==edgeIdToIndex.end(), std::runtime_error,
          "Error! The ice margin edge " << id << " is not a local edge.\n");
      iceMarginEdges.push_back(it->second);
    }

    // Set the basal discretization parameters as the DiscretizationFactory does for Extruded meshes
    auto basalDiscParams = Teuchos::sublist(Teuchos::sublist(discretizationList, "Side Set Discretizations"), "basalside");
    if (!basalDiscParams->isParameter("Number Of Time Derivatives")) {
      basalDiscParams->set("Number Of Time Derivatives", discretizationList->get<int>("Number Of Time Derivatives"));
    }
    const int extruded_ws_size = discretizationList->get<int>("Workset Size");
    if (extruded_ws_size == -1) {
      basalDiscParams->set("Workset Size", -1);
    } else if (!basalDiscParams->isParameter("Workset Size")) {
      basalDiscParams->set("Workset Size", std::max(extruded_ws_size / discretizationList->get<int>("NumLayers"), 1));
    }

    basalMesh = Teuchos::rcp(
        new Albany::MpasBasalSTKMeshStruct(basalDiscParams, mpiComm, indexToVertexID, vertexProcIDs, verticesCoords,
            procsSharingVertices, indexToTriangleID, verticesOnTria, indexToEdgeID, isBoundaryEdge, trianglesOnEdge,
            verticesOnEdge, nodeSets, sideSets, num_params));
    extrudedMesh = Teuchos::rcp(new Albany::ExtrudedMesh(basalMesh, discretizationList, mpiComm));
    mpasVerticesOnTria = verticesOnTria;

    albanyApp->createMeshSpecs(extrudedMesh);
  } else {
    indexToTriangleGOID.assign(indexToTriangleID.begin(), indexToTriangleID.end());

    if(depthIntegratedModel && nLayers != 1) {
      int numLayers = 1;
      dirichletNodesIdsDepthInt.clear();
      dirichletNodesIdsDepthInt.reserve(2*dirichletNodesIds.size()/(nLayers+1));
      for(int i=0; i < static_cast<int>(dirichletNodesIds.size()); ++i) {
        int dnode = dirichletNodesIds[i]-1;
        int ib = (Ordering == 0)*(dnode%globalVerticesStride) + (Ordering == 1)*(dnode/(nLayers+1));
        int il = (Ordering == 0)*(dnode/globalVerticesStride) + (Ordering == 1)*(dnode%(nLayers+1));
        if((il == 0) || (il == nLayers)) {
          int layer = il/nLayers;
          dirichletNodesIdsDepthInt.push_back((Ordering == 0)*(ib+layer*globalVerticesStride) + (Ordering == 1)*(layer + ib*(numLayers+1))+1);
        }
      }
      meshStruct = Teuchos::rcp(
      new Albany::MpasSTKMeshStruct(discretizationList, mpiComm, indexToVertexID,
          vertexProcIDs, verticesCoords, globalVerticesStride,
          verticesOnTria, procsSharingVertices, isBoundaryEdge, trianglesOnEdge,
          verticesOnEdge, indexToEdgeID, globalEdgesStride, indexToTriangleGOID, globalTrianglesStride,
          dirichletNodesIdsDepthInt, iceMarginEdgesIds,
          numLayers, num_params, Ordering));
    } else {
      meshStruct = Teuchos::rcp(
          new Albany::MpasSTKMeshStruct(discretizationList, mpiComm, indexToVertexID,
              vertexProcIDs, verticesCoords, globalVerticesStride,
              verticesOnTria, procsSharingVertices, isBoundaryEdge, trianglesOnEdge,
              verticesOnEdge, indexToEdgeID, globalEdgesStride, indexToTriangleGOID, globalTrianglesStride,
              dirichletNodesIds, iceMarginEdgesIds,
              nLayers, num_params, Ordering));
    }

    // Create side meshes
    for (const auto& ss : ss_pl.get<Teuchos::Array<std::string>>("Side Sets")) {
      const auto ms = meshStruct->meshSpecs[0];
      auto params_ss = Teuchos::rcpFromRef(ss_pl.sublist(ss));
      params_ss->set<int>("Number Of Time Derivatives",discretizationList->get<int>("Number Of Time Derivatives"));

      auto ss_mesh = Teuchos::rcp(new Albany::SideSetSTKMeshStruct(*ms, params_ss, mpiComm, num_params));

      ss_mesh->setParentMeshInfo(*meshStruct, ss);

      meshStruct->sideSetMeshStructs[ss] = ss_mesh;
    }
    meshStruct->createSideMeshMaps();

    albanyApp->createMeshSpecs(meshStruct);
  }

  // buildProblem and createDiscretization have their own timers
  phase.stop();
  albanyApp->buildProblem();

  albanyApp->createDiscretization();
}
