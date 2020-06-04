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

#include <fstream>
#include <vector>
#include <mpi.h>
#include <list>
#include <iostream>
#include <limits>
#include <cmath>

#include "LandIce_ProblemFactory.hpp"

#include "Albany_MpasSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include "Teuchos_StandardCatchMacros.hpp"
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include "Piro_PerformSolve.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Thyra_DetachedVectorView.hpp"
#include "Teuchos_YamlParameterListHelpers.hpp"

#include "Teuchos_StackedTimer.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include "string.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <Ionit_Initializer.h>
#endif

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
bool MPAS_useGLP(true);

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;

bool keptMesh =false;

std::string elemShape;

typedef struct TET_ {
  int verts[4];
  int neighbours[4];
  char bound_type[4];
} TET;

bool use_sliding_law (const std::string& betaType) {
  if (betaType=="GIVEN FIELD" ||
      betaType=="EXPONENT OF GIVEN FIELD" ||
      betaType=="GALERKIN PROJECTION OF EXPONENT OF GIVEN FIELD") {
    return false;
  }

  return true;
}
/***********************************************************/

// Note: betaData can be input (if prescribing basal friction)
//       or output (if using a sliding law)
void velocity_solver_solve_fo(int nLayers, int globalVerticesStride,
    int globalTrianglesStride, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double minBeta,
    const std::vector<double>& /* regulThk */,
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
    const double& deltat)
{
  auto solveTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: SolveFO"));

  int numElemsInPrism = (elemShape=="Tetrahedron") ? 3 : 1;

  int numVertices3D = (nLayers + 1) * indexToVertexID.size();
  int numPrisms = nLayers * indexToTriangleID.size();
  int vertexColumnShift = (ordering == 1) ? 1 : globalVerticesStride;
  int lVertexColumnShift = (ordering == 1) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (ordering == 0) ? 1 : nLayers + 1;

  int elemColumnShift = (ordering == 1) ? 1 : globalTrianglesStride;
  int lElemColumnShift = (ordering == 1) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (ordering == 0) ? 1 : nLayers;

  int neq = meshStruct->neq;

  const bool interleavedOrdering = meshStruct->getInterleavedOrdering();

  *MPAS_dt =  deltat;

  Teuchos::ArrayRCP<double>& layerThicknessRatio = meshStruct->layered_mesh_numbering->layers_ratio;
  for (int i = 0; i < nLayers; i++) {
    layerThicknessRatio[i] = levelsNormalizedThickness[i+1]-levelsNormalizedThickness[i];
  }

  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;

  VectorFieldType* solutionField;

  if (interleavedOrdering) {
    solutionField = Teuchos::rcp_dynamic_cast<
        Albany::OrdinarySTKFieldContainer<true> >(
            meshStruct->getFieldContainer())->getSolutionField();
  } else {
    solutionField = Teuchos::rcp_dynamic_cast<
        Albany::OrdinarySTKFieldContainer<false> >(
            meshStruct->getFieldContainer())->getSolutionField();
  }

  ScalarFieldType* surfaceHeightField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_height");
  ScalarFieldType* thicknessField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "ice_thickness");
  ScalarFieldType* bedTopographyField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "bed_topography");
  ScalarFieldType* smbField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_mass_balance");
  VectorFieldType* dirichletField = meshStruct->metaData->get_field <VectorFieldType> (stk::topology::NODE_RANK, "dirichlet_field");
  ScalarFieldType* basalFrictionField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "basal_friction");
  ScalarFieldType* stiffeningFactorField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "stiffening_factor");
  ScalarFieldType* effectivePressureField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "effective_pressure");
  ScalarFieldType* betaField;

  const auto& landiceBcList = paramList->sublist("Problem").sublist("LandIce BCs");
  const auto& basalParams = landiceBcList.sublist("BC 0");
  const auto& basalFrictionParams = basalParams.sublist("Basal Friction Coefficient");
  const auto betaType = util::upper_case(basalFrictionParams.get<std::string>("Type"));
  std::string mu_name;
  Teuchos::RCP<Albany::AbstractSTKMeshStruct> ss_ms;
  if (betaType=="POWER LAW") {
    ss_ms = meshStruct->sideSetMeshStructs.at("basalside");
    betaField = ss_ms->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "beta");
    mu_name = "mu_power_law";
  } else if (betaType=="REGULARIZED COULOMB") {
    ss_ms = meshStruct->sideSetMeshStructs.at("basalside");
    betaField = ss_ms->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "beta");
    mu_name = "mu_coulomb";
  } else {
    mu_name = "mu";
  }
  ScalarFieldType* muField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, mu_name);

  for (int j = 0; j < numVertices3D; ++j) {
    int ib = (ordering == 0) * (j % lVertexColumnShift)
            + (ordering == 1) * (j / vertexLayerShift);
    int il = (ordering == 0) * (j / lVertexColumnShift)
            + (ordering == 1) * (j % vertexLayerShift);
    int gId = il * vertexColumnShift + vertexLayerShift * indexToVertexID[ib];
    stk::mesh::Entity node = meshStruct->bulkData->get_entity(stk::topology::NODE_RANK, gId + 1);
    double* coord = stk::mesh::field_data(*meshStruct->getCoordinatesField(), node);
    coord[2] = elevationData[ib] - levelsNormalizedThickness[nLayers - il] * thicknessData[ib];


    double* thickness = stk::mesh::field_data(*thicknessField, node);
    thickness[0] = thicknessData[ib];
    double* sHeight = stk::mesh::field_data(*surfaceHeightField, node);
    sHeight[0] = elevationData[ib];
    double* bedTopography = stk::mesh::field_data(*bedTopographyField, node);
    bedTopography[0] = bedTopographyData[ib];
    double* stiffeningFactor = stk::mesh::field_data(*stiffeningFactorField, node);
    stiffeningFactor[0] = std::log(stiffeningFactorData[ib]);

    if(!effectivePressureData.empty() && (effectivePressureField != nullptr)) {
      double* effectivePressure = stk::mesh::field_data(*effectivePressureField, node);
      effectivePressure[0] = effectivePressureData[ib];
    }

    if(smbField != NULL) {
      double* smb = stk::mesh::field_data(*smbField, node);
      smb[0] = smbData[ib];
    }
    double* sol = stk::mesh::field_data(*solutionField, node);
    sol[0] = velocityOnVertices[j];
    sol[1] = velocityOnVertices[j + numVertices3D];
    if(neq==3) {
      sol[2] = thicknessData[ib];
    }
    double* dirichletVel = stk::mesh::field_data(*dirichletField, node);
    dirichletVel[0]=velocityOnVertices[j]; //velocityOnVertices stores initial guess and dirichlet velocities.
    dirichletVel[1]=velocityOnVertices[j + numVertices3D];
    if (il == 0 && basalFrictionField!=nullptr) {
      double* beta = stk::mesh::field_data(*basalFrictionField, node);
      beta[0] = std::max(betaData[ib], minBeta);
    }

    if (!muData.empty() && (muField != nullptr)) {
      double* muVal = stk::mesh::field_data(*muField, node);
      muVal[0] = muData[ib];
    }
  }

  ScalarFieldType* temperature_field = meshStruct->metaData->get_field<ScalarFieldType>(stk::topology::ELEMENT_RANK, "temperature");

  for (int j = 0; j < numPrisms; ++j) {
    int ib = (ordering == 0) * (j % (lElemColumnShift))
            + (ordering == 1) * (j / (elemLayerShift));
    int il = (ordering == 0) * (j / (lElemColumnShift))
            + (ordering == 1) * (j % (elemLayerShift));
    int gId = numElemsInPrism * (il * elemColumnShift + elemLayerShift * indexToTriangleID[ib]);
    int lId = il * lElemColumnShift + elemLayerShift * ib;
    for (int iElem = 0; iElem < numElemsInPrism; iElem++) {
      stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, ++gId);
      double* temperature = stk::mesh::field_data(*temperature_field, elem);
      temperature[0] = temperatureDataOnPrisms[lId];
    }
  }

  meshStruct->setHasRestartSolution(true);//!first_time_step);

  if (!first_time_step) {
    meshStruct->setRestartDataTime(
        paramList->sublist("Problem").get("Homotopy Restart Step", 1.));
    double homotopy =
        paramList->sublist("Problem").sublist("LandIce Viscosity").get(
            "Glen's Law Homotopy Parameter", 1.0);
    if (meshStruct->restartDataTime() == homotopy) {
      paramList->sublist("Problem").set("Solution Method", "Steady");
      paramList->sublist("Piro").set("Solver Type", "NOX");
    }
  }

  if(!keptMesh) {
    albanyApp->createDiscretization();
  } else {
    auto abs_disc = albanyApp->getDiscretization();
    auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(abs_disc);
    stk_disc->updateMesh();
  }
  albanyApp->finalSetUp(paramList);

  bool success = true;
  Teuchos::ArrayRCP<const ST> solution_constView;
  try {
    auto model = slvrfctry->createModel(albanyApp);
    solver = slvrfctry->createSolver(model, mpiComm);

    Teuchos::ParameterList solveParams;
    solveParams.set("Compute Sensitivities", false);

    Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
    Teuchos::Array<
    Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
    Piro::PerformSolveBase(*solver, solveParams, thyraResponses, thyraSensitivities);

    // Printing responses
    const int num_g = solver->Ng();
    for (int i=0; i<num_g-1; i++) {
      if (albanyApp->getResponse(i)->isScalarResponse()) {
        Thyra::ConstDetachedVectorView<double> g(thyraResponses[i]);
        std::cout << std::setprecision(15) << "\nResponse " << i << ": " << g[0] << std::endl;
      }
    }

    auto overlapVS = albanyApp->getDiscretization()->getOverlapVectorSpace();
    auto disc = albanyApp->getDiscretization();
    auto cas_manager = Albany::createCombineAndScatterManager(disc->getVectorSpace(), disc->getOverlapVectorSpace());
    Teuchos::RCP<Thyra_Vector> solution = Thyra::createMember(disc->getOverlapVectorSpace());
    cas_manager->scatter(*disc->getSolutionField(), *solution, Albany::CombineMode::INSERT);
    solution_constView = Albany::getLocalData(solution.getConst());
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

  error = albanyApp->getSolutionStatus() != Albany::Application::SolutionStatus::Converged;

  auto overlapVS = albanyApp->getDiscretization()->getOverlapVectorSpace();

  auto indexer = Albany::createGlobalLocalIndexer(overlapVS);
  for (int j = 0; j < numVertices3D; ++j) {
    int ib = (ordering == 0) * (j % lVertexColumnShift)
            + (ordering == 1) * (j / vertexLayerShift);
    int il = (ordering == 0) * (j / lVertexColumnShift)
            + (ordering == 1) * (j % vertexLayerShift);
    int gId = il * vertexColumnShift + vertexLayerShift * indexToVertexID[ib];

    int lId0, lId1;

    if (interleavedOrdering) {
      lId0 = indexer->getLocalElement(neq * gId);
      lId1 = lId0 + 1;
    } else {
      lId0 = indexer->getLocalElement(gId);
      lId1 = lId0 + numVertices3D;
    }
    velocityOnVertices[j] = solution_constView[lId0];
    velocityOnVertices[j + numVertices3D] = solution_constView[lId1];

   if (Teuchos::nonnull(ss_ms) && !betaData.empty() && (betaField!=nullptr) && (il == 0)) {
      stk::mesh::Entity node = ss_ms->bulkData->get_entity(stk::topology::NODE_RANK, indexToVertexID[ib] + 1);
      const double* betaVal = stk::mesh::field_data(*betaField,node);
      betaData[ib] = betaVal[0];
    }
  }

  ScalarFieldType* dissipationHeatField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::ELEMENT_RANK, "dissipation_heat");
  VectorFieldType* bodyForceField  = meshStruct->metaData->get_field <VectorFieldType> (stk::topology::ELEMENT_RANK, "body_force");
  for (int j = 0; j < numPrisms; ++j) {
    int ib = (ordering == 0) * (j % (lElemColumnShift))
            + (ordering == 1) * (j / (elemLayerShift));
    int il = (ordering == 0) * (j / (lElemColumnShift))
            + (ordering == 1) * (j % (elemLayerShift));
    int gId = numElemsInPrism * (il * elemColumnShift + elemLayerShift * indexToTriangleID[ib]);
    int lId = il * lElemColumnShift + elemLayerShift * ib;

    if(!dissipationHeatOnPrisms.empty())
      dissipationHeatOnPrisms[elemLayerShift] = 0;
    double bf = 0;
    for (int iElem = 0; iElem < numElemsInPrism; iElem++) {
      stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, ++gId);
      if(!dissipationHeatOnPrisms.empty() && dissipationHeatField != nullptr) {
        const double* dissipationHeat = stk::mesh::field_data(*dissipationHeatField, elem);
        dissipationHeatOnPrisms[lId] += dissipationHeat[0]/numElemsInPrism;
      }

      if ((il==0) && (bodyForceField!=nullptr)) {
        const double* bodyForceVal = stk::mesh::field_data(*bodyForceField, elem);
        const double normSq = bodyForceVal[0]*bodyForceVal[0] + bodyForceVal[1]*bodyForceVal[1];
        bf += normSq;
      }
    }
    if (!bodyForceMagnitudeOnBasalCell.empty() && (il==0)) {
      bodyForceMagnitudeOnBasalCell[ib] = std::sqrt(bf)/numElemsInPrism;
    }
  }

  keptMesh = true;
}

void velocity_solver_export_fo_velocity(MPI_Comm reducedComm) {
#ifdef ALBANY_SEACAS
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(reducedComm));
  mesh_data->set_bulk_data(*meshStruct->bulkData);
  size_t idx = mesh_data->create_output_mesh("IceSheet.exo", stk::io::WRITE_RESULTS);
  mesh_data->process_output_request(idx, 0.0);
#endif
}

int velocity_solver_init_mpi(MPI_Comm comm) {
  mpiCommMPAS = Albany::createTeuchosCommFromMpiComm(comm);
  Kokkos::initialize();
  stackedTimer = Teuchos::rcp(new Teuchos::StackedTimer("Albany Velocity Solver"));
  Teuchos::TimeMonitor::setStackedTimer(stackedTimer);
  return 0;
}

void velocity_solver_finalize() {
  meshStruct = Teuchos::null;
  albanyApp = Teuchos::null;
  paramList = Teuchos::null;
  slvrfctry = Teuchos::null;
  MPAS_dt = Teuchos::null;
  solver = Teuchos::null;
  mpiComm = Teuchos::null;

  // Print Teuchos timers into file
  std::ostream* os = &std::cout;
  std::ofstream ofs;
  if (mpiCommMPAS->getRank() == 0) {
    ofs.open("log.albany.timers.out", std::ofstream::out);
    os = &ofs;
  }
  stackedTimer->stop("Albany Velocity Solver");
  Teuchos::StackedTimer::OutputOptions options;
  options.output_fraction = true;
  options.output_minmax = true;
  stackedTimer->report(*os, mpiCommMPAS, options);
  stackedTimer = Teuchos::null;

  mpiCommMPAS = Teuchos::null;
  Kokkos::finalize_all();
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
  auto grid2DTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Compute 2D Grid"));
  keptMesh = false;
  mpiComm = Albany::createTeuchosCommFromMpiComm(reducedComm);
}

void velocity_solver_set_physical_parameters(double const& gravity, double const& ice_density, double const& ocean_density, double const& sea_level, double const& flowParamA, double const& flowLawExponent, double const& dynamic_thickness, bool const& use_GLP, double const& clausiusClapeyoronCoeff) {
  MPAS_gravity=gravity;
  MPAS_rho_ice = ice_density;
  MPAS_rho_seawater = ocean_density;
  MPAS_sea_level = sea_level;
  MPAS_flowParamA = flowParamA;
  MPAS_flowLawExponent = flowLawExponent;
  MPAS_dynamic_thickness = dynamic_thickness;
  MPAS_useGLP = use_GLP;
  MPAS_ClausiusClapeyoronCoeff = clausiusClapeyoronCoeff;
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
    const std::vector<int>& floating2dEdgesIds) {

  auto grid3DTimer = Teuchos::TimeMonitor(*Teuchos::TimeMonitor::getNewTimer("Albany: Extrude 3D Grid"));

  paramList = Teuchos::createParameterList("Albany Parameters");
  Teuchos::updateParametersFromYamlFileAndBroadcast("albany_input.yaml", paramList.ptr(), *mpiComm);

  // Set build Type 
  auto bt = paramList->get<std::string>("Build Type", "NONE");
#ifdef ALBANY_EPETRA
  if(bt == "NONE") {
    bt = "Epetra";
    paramList->set("Build Type", bt);
  }
#else
  if(bt == "NONE") {
    bt = "Tpetra";
    paramList->set("Build Type",bt);
  }
  TEUCHOS_TEST_FOR_EXCEPTION(bt == "Epetra", Teuchos::Exceptions::InvalidArgument,
      "Error! ALBANY_EPETRA must be defined in order to perform an Epetra run.\n");
#endif

  if (bt=="Tpetra") {
    // Set the static variable that denotes this as a Tpetra run
    static_cast<void>(Albany::build_type(Albany::BuildType::Tpetra));
  } else if (bt=="Epetra") {
    // Set the static variable that denotes this as a Epetra run
    static_cast<void>(Albany::build_type(Albany::BuildType::Epetra));
  } else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, Teuchos::Exceptions::InvalidArgument,
        "Error! Invalid choice (" + bt + ") for 'Build Type'.\n"
        "       Valid choices are 'Epetra', 'Tpetra'.\n");
  }

  slvrfctry = Teuchos::rcp(new Albany::SolverFactory(paramList, mpiComm));
  //paramList = Teuchos::rcp(&slvrfctry->getParameters(), false);

  paramList->set("Overwrite Nominal Values With Final Point", true);

  Teuchos::Array<std::string> arrayRequiredFields(9);
  arrayRequiredFields[0]="temperature";  arrayRequiredFields[1]="ice_thickness"; arrayRequiredFields[2]="surface_height"; arrayRequiredFields[3]="bed_topography";
  arrayRequiredFields[4]="basal_friction";  arrayRequiredFields[5]="surface_mass_balance"; arrayRequiredFields[6]="dirichlet_field", arrayRequiredFields[7]="stiffening_factor", arrayRequiredFields[8]="effective_pressure";

  paramList->sublist("Problem").set("Required Fields", arrayRequiredFields);

  //Physical Parameters
  if(paramList->sublist("Problem").isSublist("LandIce Physical Parameters")) {
    std::cout<<"\nWARNING: Using Physical Parameters (gravity, ice/ocean densities) provided in Albany input file. In order to use those provided by MPAS, remove \"LandIce Physical Parameters\" sublist from Albany input file.\n"<<std::endl;
  }

  Teuchos::ParameterList& physParamList = paramList->sublist("Problem").sublist("LandIce Physical Parameters");

  double rho_ice, rho_seawater; 
  physParamList.set("Gravity Acceleration", physParamList.get("Gravity Acceleration", MPAS_gravity));
  physParamList.set("Ice Density", rho_ice = physParamList.get("Ice Density", MPAS_rho_ice));
  physParamList.set("Water Density", rho_seawater = physParamList.get("Water Density", MPAS_rho_seawater));
  physParamList.set("Clausius-Clapeyron Coefficient", physParamList.get("Clausius-Clapeyron Coefficient", MPAS_ClausiusClapeyoronCoeff));
  physParamList.set<bool>("Use GLP", physParamList.get("Use GLP", MPAS_useGLP)); //use GLP (Grounding line parametrization) unless actively disabled

  paramList->sublist("Problem").set("Name", paramList->sublist("Problem").get("Name", "LandIce Stokes First Order 3D"));

  MPAS_dt = Teuchos::rcp(new double(0.0));
  if (paramList->sublist("Problem").get<std::string>("Name") == "LandIce Coupled FO H 3D") {
    auto& arr = paramList->sublist("Problem").get<Teuchos::Array<std::string>>("Required Fields");
    arr.push_back("surface_mass_balance");
    // paramList->sublist("Problem").sublist("Parameter Fields").set("Register Surface Mass Balance", 1);
    *MPAS_dt = paramList->sublist("Problem").get("Time Step", 0.0);
    paramList->sublist("Problem").set("Time Step Ptr", MPAS_dt); //if it is not there set it to zero.
  }

  if(paramList->sublist("Problem").isSublist("LandIce BCs"))
    std::cout<<"\nWARNING: Using LandIce BCs provided in Albany input file. In order to use boundary conditions provided by MPAS, remove \"LandIce BCs\" sublist from Albany input file.\n"<<std::endl;

  // ---- Setting parameters for LandIce BCs ---- //
  Teuchos::ParameterList& landiceBcList = paramList->sublist("Problem").sublist("LandIce BCs");
  landiceBcList.set<int>("Number",2);

  // Basal Friction BC
  auto& basalParams = landiceBcList.sublist("BC 0");
  int basal_cub_degree = physParamList.get<bool>("Use GLP") ? 8 : 3;
  basalParams.set<int>("Cubature Degree",basalParams.get<int>("Cubature Degree", basal_cub_degree));
  basalParams.set("Side Set Name", basalParams.get("Side Set Name", "basalside"));
  basalParams.set("Type", basalParams.get("Type", "Basal Friction"));
  auto& basalFrictionParams = basalParams.sublist("Basal Friction Coefficient");
  auto betaType = util::upper_case(basalFrictionParams.get<std::string>("Type","Given Field"));
  basalFrictionParams.set("Type",betaType);
  basalFrictionParams.set("Given Field Variable Name",basalFrictionParams.get("Given Field Variable Name","basal_friction"));
  basalFrictionParams.set<bool>("Zero Beta On Floating Ice", basalFrictionParams.get<bool>("Zero Beta On Floating Ice", true));

  //Lateral floating ice BCs
  int lateral_cub_degree = 3;
  double immersed_ratio =  rho_ice/rho_seawater;
  auto& lateralParams = landiceBcList.sublist("BC 1");
  lateralParams.set<int>("Cubature Degree",lateralParams.get<int>("Cubature Degree", lateral_cub_degree));
  lateralParams.set<double>("Immersed Ratio",lateralParams.get<double>("Immersed Ratio",immersed_ratio));
  lateralParams.set("Side Set Name", lateralParams.get("Side Set Name", "floatinglateralside"));
  lateralParams.set("Type", lateralParams.get("Type", "Lateral"));

  //Dirichlet BCs
  if(!paramList->sublist("Problem").isSublist("Dirichlet BCs")) {
    paramList->sublist("Problem").sublist("Dirichlet BCs").set("DBC on NS dirichlet for DOF U0 prescribe Field", "dirichlet_field");
    paramList->sublist("Problem").sublist("Dirichlet BCs").set("DBC on NS dirichlet for DOF U1 prescribe Field", "dirichlet_field");
  }
  else {
    std::cout<<"\nWARNING: Using Dirichlet BCs options provided in Albany input file. In order to use those provided by MPAS, remove \"Dirichlet BCs\" sublist from Albany input file.\n"<<std::endl;
  }

  if(paramList->sublist("Problem").isSublist("LandIce Field Norm") && paramList->sublist("Problem").sublist("LandIce Field Norm").isSublist("sliding_velocity_basalside"))
    std::cout<<"\nWARNING: Using options for Velocity Norm provided in Albany input file. In order to use those provided by MPAS, remove \"LandIce Velocity Norm\" sublist from Albany input file.\n"<<std::endl;

  Teuchos::ParameterList& fieldNormList =  paramList->sublist("Problem").sublist("LandIce Field Norm").sublist("sliding_velocity_basalside"); //empty list if LandIceViscosity not in input file.
  fieldNormList.set("Regularization Type", fieldNormList.get("Regularization Type", "Given Value"));
  double reg_value = 1e-6;
  fieldNormList.set("Regularization Value", fieldNormList.get("Regularization Value", reg_value));
  fieldNormList.set("Regularization Parameter Name", fieldNormList.get("Regularization Parameter Name","Glen's Law Homotopy Parameter"));


  if(paramList->sublist("Problem").isSublist("LandIce Viscosity"))
    std::cout<<"\nWARNING: Using Viscosity options provided in Albany input file. In order to use those provided by MPAS, remove \"LandIce Viscosity\" sublist from Albany input file.\n"<<std::endl;

  Teuchos::ParameterList& viscosityList =  paramList->sublist("Problem").sublist("LandIce Viscosity"); //empty list if LandIceViscosity not in input file.

  viscosityList.set("Type", viscosityList.get("Type", "Glen's Law"));
  double homotopy_param = (paramList->sublist("Problem").get("Solution Method", "Steady") == "Steady") ? 0.3 : 1.0;
  viscosityList.set("Glen's Law Homotopy Parameter", viscosityList.get("Glen's Law Homotopy Parameter", homotopy_param));
  viscosityList.set("Glen's Law A", viscosityList.get("Glen's Law A", MPAS_flowParamA));
  viscosityList.set("Glen's Law n", viscosityList.get("Glen's Law n",  MPAS_flowLawExponent));
  viscosityList.set("Flow Rate Type", viscosityList.get("Flow Rate Type", "Temperature Based"));
  viscosityList.set("Use Stiffening Factor", viscosityList.get("Use Stiffening Factor", true));
  viscosityList.set("Extract Strain Rate Sq", viscosityList.get("Extract Strain Rate Sq", true)); //set true if not defined


  paramList->sublist("Problem").sublist("Body Force").set("Type", "FO INTERP SURF GRAD");

  auto discretizationList = Teuchos::sublist(paramList, "Discretization", true);
  discretizationList->set("Element Shape", discretizationList->get("Element Shape", "Tetrahedron")); //set to Extruded is not defined
  elemShape = discretizationList->get<std::string>("Element Shape");

  discretizationList->set("Method", discretizationList->get("Method", "Extruded")); //set to Extruded is not defined
  int cubatureDegree = (elemShape=="Tetrahedron") ? 1 : 4;
  discretizationList->set("Cubature Degree", discretizationList->get("Cubature Degree", cubatureDegree));  //set cubatureDegree if not defined
  discretizationList->set("Interleaved Ordering", discretizationList->get("Interleaved Ordering", true));  //set true if not define

  auto& rfi = discretizationList->sublist("Required Fields Info");
  int fp = rfi.get<int>("Number Of Fields",0);
  discretizationList->sublist("Required Fields Info").set<int>("Number Of Fields",fp+11);
  Teuchos::ParameterList& field0  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",0+fp));
  Teuchos::ParameterList& field1  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",1+fp));
  Teuchos::ParameterList& field2  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",2+fp));
  Teuchos::ParameterList& field3  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",3+fp));
  Teuchos::ParameterList& field4  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",4+fp));
  Teuchos::ParameterList& field5  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",5+fp));
  Teuchos::ParameterList& field6  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",6+fp));
  Teuchos::ParameterList& field7  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",7+fp));
  Teuchos::ParameterList& field8  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",8+fp));
  Teuchos::ParameterList& field9  = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",9+fp));
  Teuchos::ParameterList& field10 = discretizationList->sublist("Required Fields Info").sublist(Albany::strint("Field",10+fp));

  //set temperature
  field0.set<std::string>("Field Name", "temperature");
  field0.set<std::string>("Field Type", "Elem Scalar");
  field0.set<std::string>("Field Origin", "Mesh");

  //set ice thickness
  field1.set<std::string>("Field Name", "ice_thickness");
  field1.set<std::string>("Field Type", "Node Scalar");
  field1.set<std::string>("Field Origin", "Mesh");

  //set surface_height
  field2.set<std::string>("Field Name", "surface_height");
  field2.set<std::string>("Field Type", "Node Scalar");
  field2.set<std::string>("Field Origin", "Mesh");

  //set bed topography
  field3.set<std::string>("Field Name", "bed_topography");
  field3.set<std::string>("Field Type", "Node Scalar");
  field3.set<std::string>("Field Origin", "Mesh");

  //set basal friction
  field4.set<std::string>("Field Name", "basal_friction");
  field4.set<std::string>("Field Type", "Node Scalar");
  field4.set<std::string>("Field Origin", "Mesh");
  if (use_sliding_law(betaType)) {
    field4.set<std::string>("Field Usage", "Unused");
  }

  //set surface mass balance
  field5.set<std::string>("Field Name", "surface_mass_balance");
  field5.set<std::string>("Field Type", "Node Scalar");
  field5.set<std::string>("Field Origin", "Mesh");

  //set dirichlet field
  field6.set<std::string>("Field Name", "dirichlet_field");
  field6.set<std::string>("Field Type", "Node Vector");
  field6.set<std::string>("Field Origin", "Mesh");

  //set stiffening factor
  field7.set<std::string>("Field Name", "stiffening_factor");
  field7.set<std::string>("Field Type", "Node Scalar");
  field7.set<std::string>("Field Origin", "Mesh");

  //set effective pressure
  field8.set<std::string>("Field Name", "effective_pressure");
  field8.set<std::string>("Field Type", "Node Scalar");
  field8.set<std::string>("Field Origin", "Mesh");
  if (!use_sliding_law(betaType)) {
    field8.set<std::string>("Field Usage", "Unused");
  }

  //set mu power law
  std::string mu_name;
  if (betaType=="POWER LAW") {
    mu_name = "mu_power_law";
  } else if (betaType=="REGULARIZED COULOMB") {
    mu_name = "mu_coulomb";
  } else {
    mu_name = "mu";
  }
  field9.set<std::string>("Field Name", mu_name);
  field9.set<std::string>("Field Type", "Node Scalar");
  field9.set<std::string>("Field Origin", "Mesh");
  if (!use_sliding_law(betaType)) {
    field8.set<std::string>("Field Usage", "Unused");
  }

  // Outputs
  field10.set<std::string>("Field Name", "body_force");
  field10.set<std::string>("Field Type", "Elem Vector");
  field10.set<std::string>("Field Usage", "Output");

  // Side set outputs
  if (use_sliding_law(betaType)) {
    auto& ss_pl =discretizationList->sublist("Side Set Discretizations");
    Teuchos::Array<std::string> bsn (1,"basalside");
    ss_pl.set<Teuchos::Array<std::string>>("Side Sets", bsn);
    auto& basal_pl = ss_pl.sublist("basalside");
    basal_pl.set<std::string>("Method","SideSetSTK");
    auto& basal_req = basal_pl.sublist("Required Fields Info");
    basal_req.set<int>("Number Of Fields",1);

    auto& ss_field0 = basal_req.sublist("Field 0");
    ss_field0.set<std::string>("Field Name", "beta");
    ss_field0.set<std::string>("Field Type", "Node Scalar");
    ss_field0.set<std::string>("Field Usage", "Output");
  }

  Albany::AbstractFieldContainer::FieldContainerRequirements req;

  // Register LandIce problems
  auto& pb_factories = Albany::FactoriesContainer<Albany::ProblemFactory>::instance();
  pb_factories.add_factory(LandIce::LandIceProblemFactory::instance());

  // Create albany app
  albanyApp = Teuchos::rcp(new Albany::Application(mpiComm));
  albanyApp->initialSetUp(paramList);

  int neq = (paramList->sublist("Problem").get<std::string>("Name") == "LandIce Coupled FO H 3D") ? 3 : 2;

  //temporary fix, TODO: use GO for indexToTriangleID (need to synchronize with MPAS).
  std::vector<GO> indexToTriangleGOID;
  indexToTriangleGOID.assign(indexToTriangleID.begin(), indexToTriangleID.end());
  //Get number of params in problem - needed for MeshStruct constructor
  int num_params = Albany::CalculateNumberParams(Teuchos::sublist(paramList, "Problem", true)); 
  meshStruct = Teuchos::rcp(
      new Albany::MpasSTKMeshStruct(discretizationList, mpiComm, indexToTriangleGOID,
          globalTrianglesStride, nLayers, num_params, Ordering));
  albanyApp->createMeshSpecs(meshStruct);

  albanyApp->buildProblem();

  meshStruct->constructMesh(mpiComm, discretizationList, neq, req,
      albanyApp->getStateMgr(), indexToVertexID,
      vertexProcIDs, verticesCoords, globalVerticesStride,
      verticesOnTria, procsSharingVertices, isBoundaryEdge, trianglesOnEdge,
      verticesOnEdge, indexToEdgeID, globalEdgesStride, indexToTriangleGOID, globalTrianglesStride,
      dirichletNodesIds, floating2dEdgesIds,
      meshStruct->getMeshSpecs()[0]->worksetSize, nLayers, Ordering);
}
