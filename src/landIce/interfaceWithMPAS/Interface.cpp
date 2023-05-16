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

#include "Albany_StringUtils.hpp" // for 'upper_case'

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <Ionit_Initializer.h>
#endif

#include <fstream>
#include <vector>
#include <mpi.h>
#include <list>
#include <iostream>
#include <limits>
#include <cmath>

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

bool depthIntegratedModel(false);

std::vector<int> dirichletNodesIdsDepthInt;

Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;

bool keptMesh =false;
bool kokkosInitializedByAlbany = false;

typedef struct TET_ {
  int verts[4];
  int neighbours[4];
  char bound_type[4];
} TET;

/***********************************************************/

// Note: betaData can be input (if prescribing basal friction)
//       or output (if using a sliding law)
void velocity_solver_solve_fo(int nLayers, int globalVerticesStride,
    int globalTrianglesStride, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double /*minBeta*/,
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

  int numVertices3D = (nLayers + 1) * indexToVertexID.size();
  int numPrisms = nLayers * indexToTriangleID.size();
  int vertexColumnShift = (ordering == 1) ? 1 : globalVerticesStride;
  int lVertexColumnShift = (ordering == 1) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (ordering == 0) ? 1 : nLayers + 1;

  int elemColumnShift = (ordering == 1) ? 1 : globalTrianglesStride;
  int lElemColumnShift = (ordering == 1) ? 1 : indexToTriangleID.size();
  int elemLayerShift = (ordering == 0) ? 1 : nLayers;

  auto abs_disc = albanyApp->getDiscretization();
  auto stk_disc = Teuchos::rcp_dynamic_cast<Albany::STKDiscretization>(abs_disc);
  int neq = stk_disc->getNumberEquations();

  *MPAS_dt =  deltat;

  Teuchos::ArrayRCP<double>& layerThicknessRatio = meshStruct->mesh_layers_ratio;

  if(depthIntegratedModel)
    layerThicknessRatio[0] = 1.0;
  else {
    for (int i = 0; i < nLayers; i++)
      layerThicknessRatio[i] = levelsNormalizedThickness[i+1]-levelsNormalizedThickness[i];
  }

  using VectorFieldType = Albany::AbstractSTKFieldContainer::VectorFieldType;
  using ScalarFieldType = Albany::AbstractSTKFieldContainer::ScalarFieldType;
  using QPScalarFieldType = Albany::AbstractSTKFieldContainer::QPScalarFieldType;
  using SolFldContainerType = Albany::OrdinarySTKFieldContainer;

  auto fld_container = stk_disc->getSolutionFieldContainer();
  auto sol_fld_container = Teuchos::rcp_dynamic_cast<SolFldContainerType>(fld_container);
  auto solutionField = sol_fld_container->getSolutionField();

  ScalarFieldType* surfaceHeightField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_height");
  ScalarFieldType* thicknessField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "ice_thickness");
  ScalarFieldType* bedTopographyField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "bed_topography");
  ScalarFieldType* smbField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_mass_balance");
  VectorFieldType* dirichletField = meshStruct->metaData->get_field <VectorFieldType> (stk::topology::NODE_RANK, "dirichlet_field");
  ScalarFieldType* muField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "mu");
  ScalarFieldType* stiffeningFactorField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "stiffening_factor");
  ScalarFieldType* effectivePressureField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "effective_pressure");
  ScalarFieldType* betaField;

  auto& probParamList = paramList->sublist("Problem");
  const auto& landiceBcList = probParamList.sublist("LandIce BCs");
  const auto& basalParams = landiceBcList.sublist("BC 0");
  const auto& basalFrictionParams = basalParams.sublist("Basal Friction Coefficient");
  const auto betaType = util::upper_case(basalFrictionParams.get<std::string>("Type"));

  Teuchos::RCP<Albany::AbstractSTKMeshStruct> ss_ms;
  ss_ms = meshStruct->sideSetMeshStructs.at("basalside");
  betaField = ss_ms->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "beta");

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
    
    stk::mesh::Entity node = meshStruct->bulkData->get_entity(stk::topology::NODE_RANK, gId);
    double* coord = stk::mesh::field_data(*meshStruct->getCoordinatesField(), node);
    coord[2] = elevationData[ib] + (levelsNormalizedThickness[il]-1.0) * thicknessData[ib];


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

    if ((il == 0) && !muData.empty() && (muField != nullptr)) {
      double* muVal = stk::mesh::field_data(*muField, node);
      muVal[0] = muData[ib];
    }
  }

  //In the following we import the temperature field temperatureDataOnPrisms from MPAS, 
  //which is stored as a constant in each Prism (wedge), into a temperature filed in Albany mesh.
  if(depthIntegratedModel) {
    //In this case the Albany mesh only has one layer, but the MPAS mesh can still have multiple layers
    bool usePOTemp = probParamList.sublist("LandIce Viscosity").get<bool>("Use P0 Temperature");
    if(!usePOTemp) { 
      // In this case we compute the temperature at Albany quadrature points. 
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
      for(int qp=0; qp<numQPs; qp++) {
        int il=0; 
        auto z = (quadPointCoords(qp,2)+1.0)/2; //quad points are defined on [-1,1]
        while ((z > levelsNormalizedThickness[il+1]) && (il<nLayers)) il++;
        layerVec[qp] = il;
      }    
      
      // Populate the temperature mesh field, defined at quad points 
      QPScalarFieldType* temperature_field = meshStruct->metaData->get_field <QPScalarFieldType> (stk::topology::ELEMENT_RANK, "temperature");

      for(int ib=0; ib <indexToTriangleID.size(); ++ib ) {
        stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, indexToTriangleID[ib]);
        double* temperature = stk::mesh::field_data(*temperature_field, elem);
        for(int qp=0; qp<numQPs; qp++) {
          int lId = layerVec[qp] * lElemColumnShift + elemLayerShift * ib;        
          temperature[qp] = temperatureDataOnPrisms[lId];
        }
      }
    } else {  //P0 temperature
      // In this case we compute a column average of the MPAS temperature and save it as a P0 field in the 1-layer Albany mesh.
      ScalarFieldType* temperature_field = meshStruct->metaData->get_field<ScalarFieldType>(stk::topology::ELEMENT_RANK, "temperature");
      for(int ib=0; ib <indexToTriangleID.size(); ++ib ) {
        stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, indexToTriangleID[ib]);
        double* temperature = stk::mesh::field_data(*temperature_field, elem);
        for(int il=0; il<nLayers; il++) {
          int lId = il * lElemColumnShift + elemLayerShift * ib;  
          double tempFraction = temperatureDataOnPrisms[lId]*(levelsNormalizedThickness[il+1]-levelsNormalizedThickness[il]);
          if(il ==0)
            temperature[0] = tempFraction;
          else
            temperature[0] += tempFraction;
        }
      }
    }
  } else { // Here we copy the temperature on Prisms from MPAS into a P0 field in the Albany mesh.
    ScalarFieldType* temperature_field = meshStruct->metaData->get_field<ScalarFieldType>(stk::topology::ELEMENT_RANK, "temperature");
    for(int ib=0; ib <indexToTriangleID.size(); ++ib ) {
      for(int il=0; il<nLayers; il++) {
        int lId = il * lElemColumnShift + elemLayerShift * ib;
        int gId = il * elemColumnShift + elemLayerShift * (indexToTriangleID[ib]-1) + 1;
        stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, gId);
        double* temperature = stk::mesh::field_data(*temperature_field, elem);
        temperature[0] = temperatureDataOnPrisms[lId];
      }
    }
  }

  meshStruct->setHasRestartSolution(true);//!first_time_step);

  if (!first_time_step) {
    meshStruct->setRestartDataTime(
        probParamList.get("Homotopy Restart Step", 1.));
    double homotopy =
        probParamList.sublist("LandIce Viscosity").get(
            "Glen's Law Homotopy Parameter", 1.0);
    if (meshStruct->restartDataTime() == homotopy) {
      probParamList.set("Solution Method", "Steady");
      paramList->sublist("Piro").set("Solver Type", "NOX");
    }
  }

  stk_disc->updateMesh();
  albanyApp->finalSetUp(paramList);

  if (keptMesh) albanyApp->getPhxSetup()->reboot_memoizer();

  bool success = true;
  Teuchos::ArrayRCP<const ST> solution_constView;
  try {
    auto model = slvrfctry->createModel(albanyApp);
    solver = slvrfctry->createSolver(mpiComm, model);

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

    auto overlapVS = albanyApp->getDiscretization()->getOverlapVectorSpace();
    auto disc = albanyApp->getDiscretization();
    auto cas_manager = Albany::createCombineAndScatterManager(disc->getVectorSpace(), disc->getOverlapVectorSpace());
    Teuchos::RCP<Thyra_Vector> solution = Thyra::createMember(disc->getOverlapVectorSpace());
    cas_manager->scatter(*disc->getSolutionField(), *solution, Albany::CombineMode::INSERT);
    solution_constView = Albany::getLocalData(solution.getConst());
  }
  TEUCHOS_STANDARD_CATCH_STATEMENTS(true, std::cerr, success);

  error = !success || (albanyApp->getSolutionStatus() != Albany::Application::SolutionStatus::Converged);

  auto overlapVS = albanyApp->getDiscretization()->getOverlapVectorSpace();

  auto indexer = Albany::createGlobalLocalIndexer(overlapVS);

  if(depthIntegratedModel) {
    for(int ib = 0; ib < indexToVertexID.size(); ++ib) {
      int depthVertexLayerShift = (ordering == 0) ? 1 : 2;
      int gIdBed =  depthVertexLayerShift * (indexToVertexID[ib]-1) + 1;
      int gIdTop = gIdBed + vertexColumnShift;
      
      int lIdBed0 = indexer->getLocalElement(neq * (gIdBed-1));
      int lIdBed1 = lIdBed0 + 1;
      int lIdTop0 = indexer->getLocalElement(neq * (gIdTop-1));
      int lIdTop1 = lIdTop0 + 1;      
      double solBed0 = solution_constView[lIdBed0];
      double solBed1 = solution_constView[lIdBed1];
      double solTop0 = solution_constView[lIdTop0];
      double solTop1 = solution_constView[lIdTop1];

      for(int il=0; il<nLayers+1; ++il) {
        int j = il * lVertexColumnShift + vertexLayerShift * ib;
        double z = 1.0 - levelsNormalizedThickness[il];
        double fz = 1.0 - std::pow(z,4);
        velocityOnVertices[j] = solBed0 + fz * (solTop0-solBed0);
        velocityOnVertices[j + numVertices3D]  = solBed1 + fz * (solTop1-solBed1);
      }
    }
  } else {
    for (int j = 0; j < numVertices3D; ++j) {
      int ib = (ordering == 0) * (j % lVertexColumnShift)
              + (ordering == 1) * (j / vertexLayerShift);
      int il = (ordering == 0) * (j / lVertexColumnShift)
              + (ordering == 1) * (j % vertexLayerShift);
      int gId = il * vertexColumnShift + vertexLayerShift * (indexToVertexID[ib]-1) + 1;

      int lId0 = indexer->getLocalElement(neq * (gId-1));
      int lId1 = lId0 + 1;
      velocityOnVertices[j] = solution_constView[lId0];
      velocityOnVertices[j + numVertices3D] = solution_constView[lId1];
    }
  }

  if (Teuchos::nonnull(ss_ms) && !betaData.empty() && (betaField!=nullptr)) {
    for(int ib = 0; ib < indexToVertexID.size(); ++ib) {
      stk::mesh::Entity node = ss_ms->bulkData->get_entity(stk::topology::NODE_RANK, indexToVertexID[ib]);
      const double* betaVal = stk::mesh::field_data(*betaField,node);
      betaData[ib] = betaVal[0];
    }
  }

  ScalarFieldType* dissipationHeatField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::ELEMENT_RANK, "dissipation_heat");
  VectorFieldType* bodyForceField  = meshStruct->metaData->get_field <VectorFieldType> (stk::topology::ELEMENT_RANK, "body_force");
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
    stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, gId);
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
  if(! Kokkos::is_initialized()) {
    Kokkos::initialize();
    kokkosInitializedByAlbany = true;
  }
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
  if(kokkosInitializedByAlbany)
    Kokkos::finalize();
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

  //std::cout << "\nMPAS_gravity:" <<MPAS_gravity << " MPAS_rho_ice:" <<MPAS_rho_ice << " MPAS_rho_seawater:" << MPAS_rho_seawater <<
  //    " MPAS_sea_level:" << MPAS_sea_level << " MPAS_dynamic_thickness:" << MPAS_dynamic_thickness <<
  //    " MPAS_ClausiusClapeyoronCoeff:" << MPAS_ClausiusClapeyoronCoeff <<std::endl;
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
  Teuchos::ParameterList& probParamList = paramList->sublist("Problem");

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
  physParamList.set<bool>("Use GLP", physParamList.get("Use GLP", MPAS_useGLP)); //use GLP (Grounding line parametrization) unless actively disabled

  probParamList.set("Name", probParamList.get("Name", "LandIce Stokes First Order 3D"));

  MPAS_dt = Teuchos::rcp(new double(0.0));
  if (probParamList.get<std::string>("Name") == "LandIce Coupled FO H 3D") {
    // probParamList.sublist("Parameter Fields").set("Register Surface Mass Balance", 1);
    *MPAS_dt = probParamList.get("Time Step", 0.0);
    probParamList.set("Time Step Ptr", MPAS_dt); //if it is not there set it to zero.
  }

  if(probParamList.isSublist("LandIce BCs"))
    std::cout<<"\nWARNING: Using LandIce BCs provided in Albany input file. In order to use boundary conditions provided by MPAS, remove \"LandIce BCs\" sublist from Albany input file.\n"<<std::endl;
  
  // ---- Setting Memoization ---- //
  probParamList.set<bool>("Use MDField Memoization", probParamList.get<bool>("Use MDField Memoization", true));

  // ---- Setting parameters for LandIce BCs ---- //
  Teuchos::ParameterList& landiceBcList = probParamList.sublist("LandIce BCs");
  landiceBcList.set<int>("Number",2);

  // Basal Friction BC
  auto& basalParams = landiceBcList.sublist("BC 0");
  auto& basalFrictionParams = basalParams.sublist("Basal Friction Coefficient");
  bool zeroBetaOnShelf = basalFrictionParams.get<bool>("Zero Beta On Floating Ice", true);
  int basal_cub_degree = physParamList.get<bool>("Use GLP") ? 8 : (zeroBetaOnShelf ? 4 : 3);

  //TODO: remove this after fixing MPAS
  basal_cub_degree = basalParams.get<int>("Cubature Degree", basal_cub_degree);

  probParamList.set<int>("Basal Cubature Degree",probParamList.get<int>("Basal Cubature Degree", basal_cub_degree));
  basalParams.set("Side Set Name", basalParams.get("Side Set Name", "basalside"));
  basalParams.set("Type", basalParams.get("Type", "Basal Friction"));
  auto betaType = util::upper_case(basalFrictionParams.get<std::string>("Type","Power Law"));
  basalFrictionParams.set("Type",betaType);
  basalFrictionParams.set("Power Exponent", basalFrictionParams.get("Power Exponent",1.0));
  basalFrictionParams.set("Mu Field Name",basalFrictionParams.get("Mu Field Name","mu"));
  basalFrictionParams.set("Mu Type",basalFrictionParams.get("Mu Type","Field"));
  basalFrictionParams.set("Effective Pressure Type",basalFrictionParams.get("Effective Pressure Type","Field"));
  basalFrictionParams.set<bool>("Zero Beta On Floating Ice", zeroBetaOnShelf);

  //Lateral floating ice BCs
  int lateral_cub_degree = 3;
  auto& lateralParams = landiceBcList.sublist("BC 1");
  lateralParams.set<int>("Cubature Degree",lateralParams.get<int>("Cubature Degree", lateral_cub_degree));
  //If the following option is not specified (recommended) 
  // Albany will compute it based on the geometry
  if(lateralParams.isParameter("Immersed Ratio"))
    lateralParams.set<double>("Immersed Ratio",lateralParams.get<double>("Immersed Ratio"));
  lateralParams.set("Side Set Name", lateralParams.get("Side Set Name", "ice_margin_side"));
  lateralParams.set("Type", lateralParams.get("Type", "Lateral"));

  //Dirichlet BCs
  if(!probParamList.isSublist("Dirichlet BCs")) {
    probParamList.sublist("Dirichlet BCs").set("SDBC on NS dirichlet for DOF U0 prescribe Field", "dirichlet_field");
    probParamList.sublist("Dirichlet BCs").set("SDBC on NS dirichlet for DOF U1 prescribe Field", "dirichlet_field");
  }
  else {
    std::cout<<"\nWARNING: Using Dirichlet BCs options provided in Albany input file. In order to use those provided by MPAS, remove \"Dirichlet BCs\" sublist from Albany input file.\n"<<std::endl;
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
  const auto spy = 3600*24*365;
  const auto knp1 = std::pow(1000,MPAS_flowLawExponent+1);
  const auto A = MPAS_flowParamA / knp1 / spy;
  viscosityList.set("Glen's Law A", viscosityList.get("Glen's Law A", A));
  viscosityList.set("Flow Rate Type", viscosityList.get("Flow Rate Type", "Temperature Based"));
  viscosityList.set("Use Stiffening Factor", viscosityList.get("Use Stiffening Factor", true));
  viscosityList.set("Extract Strain Rate Sq", viscosityList.get("Extract Strain Rate Sq", true)); //set true if not defined
  viscosityList.set("Use P0 Temperature", viscosityList.get("Use P0 Temperature", !depthIntegratedModel)); 


  probParamList.sublist("Body Force").set("Type", "FO INTERP SURF GRAD");

  //! set Rigid body modes for prec. near null space
  if(!probParamList.isSublist("LandIce Rigid Body Modes For Preconditioner")) {
    Teuchos::ParameterList& rbmList = probParamList.sublist("LandIce Rigid Body Modes For Preconditioner");
    rbmList.set<bool>("Compute Constant Modes", true);
    rbmList.set<bool>("Compute Rotation Modes", true);
  }

  Teuchos::Array<int> defaultCubatureDegrees(2); 
  defaultCubatureDegrees[0] = 4; defaultCubatureDegrees[1]= depthIntegratedModel ? 6 : 3; 

  if(!probParamList.isParameter("Cubature Degree") || depthIntegratedModel)
    probParamList.set("Cubature Degrees (Horiz Vert)", probParamList.get("Cubature Degrees (Horiz Vert)", defaultCubatureDegrees));  //set Cubature Degrees if not defined


  auto discretizationList = Teuchos::sublist(paramList, "Discretization", true);

  discretizationList->set("Workset Size", discretizationList->get("Workset Size", -1));

  discretizationList->set("Method", discretizationList->get("Method", "Extruded")); //set to Extruded is not defined

  auto& rfi = discretizationList->sublist("Required Fields Info");
  int fp = rfi.get<int>("Number Of Fields",0);
  discretizationList->sublist("Required Fields Info").set<int>("Number Of Fields",fp+10);
  Teuchos::ParameterList& field0  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",0+fp));
  Teuchos::ParameterList& field1  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",1+fp));
  Teuchos::ParameterList& field2  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",2+fp));
  Teuchos::ParameterList& field3  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",3+fp));
  Teuchos::ParameterList& field4  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",4+fp));
  Teuchos::ParameterList& field5  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",5+fp));
  Teuchos::ParameterList& field6  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",6+fp));
  Teuchos::ParameterList& field7  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",7+fp));
  Teuchos::ParameterList& field8  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",8+fp));
  Teuchos::ParameterList& field9  = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",9+fp));
  //Teuchos::ParameterList& field10 = discretizationList->sublist("Required Fields Info").sublist(util::strint("Field",10+fp));

  //set temperature
  field0.set<std::string>("Field Name", "temperature");
  if(depthIntegratedModel)
    field0.set<std::string>("Field Type", "QuadPoint Scalar");
  else
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

  //set mu field
  field4.set<std::string>("Field Name", "mu");
  field4.set<std::string>("Field Type", "Node Scalar");
  field4.set<std::string>("Field Origin", "Mesh");

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

  // Outputs
  field9.set<std::string>("Field Name", "body_force");
  field9.set<std::string>("Field Type", "Elem Vector");
  field9.set<std::string>("Field Usage", "Output");

  // Side set outputs
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

  // Register LandIce problems
  auto& pb_factories = Albany::FactoriesContainer<Albany::ProblemFactory>::instance();
  pb_factories.add_factory(LandIce::LandIceProblemFactory::instance());

  // Create albany app
  albanyApp = Teuchos::rcp(new Albany::Application(mpiComm));
  albanyApp->initialSetUp(paramList);

  //temporary fix, TODO: use GO for indexToTriangleID (need to synchronize with MPAS).
  std::vector<GO> indexToTriangleGOID;
  indexToTriangleGOID.assign(indexToTriangleID.begin(), indexToTriangleID.end());
  //Get number of params in problem - needed for MeshStruct constructor
  int num_params = Albany::CalculateNumberParams(Teuchos::sublist(paramList, "Problem", true));

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

  albanyApp->createMeshSpecs(meshStruct);

  albanyApp->buildProblem();

  albanyApp->createDiscretization();
}
