// ===================================================
//! Includes
// ===================================================

#include "Interface.hpp"
#include "Albany_MpasSTKMeshStruct.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_RCP.hpp"
#include "Albany_Utils.hpp"
#include "Albany_SolverFactory.hpp"
#include "Teuchos_XMLParameterListHelpers.hpp"
#include <stk_mesh/base/FieldBase.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include "Piro_PerformSolve.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"

#ifdef ALBANY_SEACAS
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <Ionit_Initializer.h>
#endif

Teuchos::RCP<Albany::MpasSTKMeshStruct> meshStruct;
Teuchos::RCP<Albany::Application> albanyApp;
Teuchos::RCP<Teuchos::ParameterList> paramList;
Teuchos::RCP<const Teuchos_Comm> mpiCommT;
Teuchos::RCP<Teuchos::ParameterList> discParams;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
Teuchos::RCP<double> MPAS_dt;

double MPAS_gravity(9.8), MPAS_rho_ice(910), MPAS_rho_seawater(1028);

#ifdef MPAS_USE_EPETRA
  Teuchos::RCP<const Epetra_Comm> mpiComm;
  Teuchos::RCP<Thyra::ModelEvaluator<double> > solver;
  bool TpetraBuild = false;
#else
  Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;
  bool TpetraBuild = true;
#endif
bool keptMesh =false;

typedef struct TET_ {
  int verts[4];
  int neighbours[4];
  char bound_type[4];
} TET;

/***********************************************************/


void velocity_solver_solve_fo(int nLayers, int nGlobalVertices,
    int nGlobalTriangles, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double minBeta,
    const std::vector<double>& regulThk,
    const std::vector<double>& levelsNormalizedThickness,
    const std::vector<double>& elevationData,
    const std::vector<double>& thicknessData,
    const std::vector<double>& betaData,
    const std::vector<double>& smbData,
    const std::vector<double>& temperatureOnTetra,
    std::vector<double>& velocityOnVertices,
    const double& deltat) {

  int numVertices3D = (nLayers + 1) * indexToVertexID.size();
  int numPrisms = nLayers * indexToTriangleID.size();
  int vertexColumnShift = (ordering == 1) ? 1 : nGlobalVertices;
  int lVertexColumnShift = (ordering == 1) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (ordering == 0) ? 1 : nLayers + 1;

  int elemColumnShift = (ordering == 1) ? 3 : 3 * nGlobalTriangles;
  int lElemColumnShift = (ordering == 1) ? 3 : 3 * indexToTriangleID.size();
  int elemLayerShift = (ordering == 0) ? 3 : 3 * nLayers;

  int neq = meshStruct->neq;

  const bool interleavedOrdering = meshStruct->getInterleavedOrdering();

  *MPAS_dt =  deltat;

  Teuchos::ArrayRCP<double>& layerThicknessRatio = meshStruct->layered_mesh_numbering->layers_ratio;
  for (int i = 0; i < nLayers; i++) {
    layerThicknessRatio[i] = levelsNormalizedThickness[i+1]-levelsNormalizedThickness[i];
  }

  typedef Albany::AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef Albany::AbstractSTKFieldContainer::QPScalarFieldType ElemScalarFieldType;

  VectorFieldType* solutionField;

  if (interleavedOrdering)
    solutionField = Teuchos::rcp_dynamic_cast<
        Albany::OrdinarySTKFieldContainer<true> >(
        meshStruct->getFieldContainer())->getSolutionField();
  else
    solutionField = Teuchos::rcp_dynamic_cast<
        Albany::OrdinarySTKFieldContainer<false> >(
        meshStruct->getFieldContainer())->getSolutionField();

  ScalarFieldType* surfaceHeightField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_height");
  ScalarFieldType* thicknessField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "thickness");
  ScalarFieldType* smbField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_mass_balance");
  VectorFieldType* dirichletField = meshStruct->metaData->get_field <VectorFieldType> (stk::topology::NODE_RANK, "dirichlet_field");
  ScalarFieldType* basalFrictionField = meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "basal_friction");

  for (UInt j = 0; j < numVertices3D; ++j) {
    int ib = (ordering == 0) * (j % lVertexColumnShift)
        + (ordering == 1) * (j / vertexLayerShift);
    int il = (ordering == 0) * (j / lVertexColumnShift)
        + (ordering == 1) * (j % vertexLayerShift);
    int gId = il * vertexColumnShift + vertexLayerShift * indexToVertexID[ib];
    stk::mesh::Entity node = meshStruct->bulkData->get_entity(stk::topology::NODE_RANK, gId + 1);
    double* coord = stk::mesh::field_data(*meshStruct->getCoordinatesField(), node);
    coord[2] = elevationData[ib] - levelsNormalizedThickness[nLayers - il] * regulThk[ib];

    double* sHeight = stk::mesh::field_data(*surfaceHeightField, node);
    sHeight[0] = elevationData[ib];
    double* thickness = stk::mesh::field_data(*thicknessField, node);
    thickness[0] = thicknessData[ib];
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
    if (il == 0) {
      double* beta = stk::mesh::field_data(*basalFrictionField, node);
      beta[0] = std::max(betaData[ib], minBeta);
    }
  }

  ElemScalarFieldType* temperature_field = meshStruct->metaData->get_field<ElemScalarFieldType>(stk::topology::ELEMENT_RANK, "temperature");

  for (UInt j = 0; j < numPrisms; ++j) {
    int ib = (ordering == 0) * (j % (lElemColumnShift / 3))
        + (ordering == 1) * (j / (elemLayerShift / 3));
    int il = (ordering == 0) * (j / (lElemColumnShift / 3))
        + (ordering == 1) * (j % (elemLayerShift / 3));
    int gId = il * elemColumnShift + elemLayerShift * indexToTriangleID[ib];
    int lId = il * lElemColumnShift + elemLayerShift * ib;
    for (int iTetra = 0; iTetra < 3; iTetra++) {
      stk::mesh::Entity elem = meshStruct->bulkData->get_entity(stk::topology::ELEMENT_RANK, ++gId);
      double* temperature = stk::mesh::field_data(*temperature_field, elem);
      temperature[0] = temperatureOnTetra[lId++];
    }
  }

  meshStruct->setHasRestartSolution(true);//!first_time_step);

  if (!first_time_step) {
    meshStruct->setRestartDataTime(
        paramList->sublist("Problem").get("Homotopy Restart Step", 1.));
    double homotopy =
        paramList->sublist("Problem").sublist("FELIX Viscosity").get(
            "Glen's Law Homotopy Parameter", 1.0);
    if (meshStruct->restartDataTime() == homotopy) {
      paramList->sublist("Problem").set("Solution Method", "Steady");
      paramList->sublist("Piro").set("Solver Type", "NOX");
    }
  }



  if(!keptMesh) {
    albanyApp->createDiscretization();
    albanyApp->finalSetUp(paramList); //, albanyApp->getDiscretization()->getSolutionFieldT());
  }
  else
    albanyApp->getDiscretization()->updateMesh();

#ifdef MPAS_USE_EPETRA
  solver = slvrfctry->createThyraSolverAndGetAlbanyApp(albanyApp, mpiComm, mpiComm, Teuchos::null, false);
#else
   solver = slvrfctry->createAndGetAlbanyAppT(albanyApp, mpiCommT, mpiCommT, Teuchos::null, false);
#endif

  Teuchos::ParameterList solveParams;
  solveParams.set("Compute Sensitivities", false);

  Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
  Teuchos::Array<
      Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
  Piro::PerformSolveBase(*solver, solveParams, thyraResponses,
      thyraSensitivities);

  Teuchos::RCP<const Tpetra_Map> overlapMap = albanyApp->getDiscretization()->getOverlapMapT();
  Teuchos::RCP<Tpetra_Import> import = Teuchos::rcp(new Tpetra_Import(albanyApp->getDiscretization()->getMapT(), overlapMap));
  Teuchos::RCP<Tpetra_Vector> solution = Teuchos::rcp(new Tpetra_Vector(overlapMap));
  solution->doImport(*albanyApp->getDiscretization()->getSolutionFieldT(), *import, Tpetra::INSERT);
  Teuchos::ArrayRCP<const ST> solution_constView = solution->get1dView();


  for (UInt j = 0; j < numVertices3D; ++j) {
    int ib = (ordering == 0) * (j % lVertexColumnShift)
        + (ordering == 1) * (j / vertexLayerShift);
    int il = (ordering == 0) * (j / lVertexColumnShift)
        + (ordering == 1) * (j % vertexLayerShift);
    int gId = il * vertexColumnShift + vertexLayerShift * indexToVertexID[ib];

    int lId0, lId1;

    if (interleavedOrdering) {
      lId0 = overlapMap->getLocalElement(neq * gId);
      lId1 = lId0 + 1;
    } else {
      lId0 = overlapMap->getLocalElement(gId);
      lId1 = lId0 + numVertices3D;
    }
    velocityOnVertices[j] = solution_constView[lId0];
    velocityOnVertices[j + numVertices3D] = solution_constView[lId1];
  }

  keptMesh = true;

  //UInt componentGlobalLength = (nLayers+1)*nGlobalVertices; //mesh3DPtr->numGlobalVertices();
}

void velocity_solver_export_fo_velocity(MPI_Comm reducedComm) {
#ifdef ALBANY_SEACAS
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(reducedComm));
    mesh_data->set_bulk_data(*meshStruct->bulkData);
    size_t idx = mesh_data->create_output_mesh("IceSheet.exo", stk::io::WRITE_RESULTS);
    mesh_data->process_output_request(idx, 0.0);
#endif
}

void velocity_solver_finalize() {
}

/*duality:
 *
 *   mpas(F) |  lifev
 *  ---------|---------
 *   cell    |  vertex
 *   vertex  |  triangle
 *   edge    |  edge
 *
 */

void velocity_solver_compute_2d_grid(MPI_Comm reducedComm) {
  keptMesh = false;
#ifdef MPAS_USE_EPETRA
  mpiComm = Albany::createEpetraCommFromMpiComm(reducedComm);
#endif
  mpiCommT = Albany::createTeuchosCommFromMpiComm(reducedComm);
}

void velocity_solver_set_physical_parameters(double gravity_, double rho_ice_, double rho_seawater_) {
  MPAS_gravity=gravity_;
  MPAS_rho_ice = rho_ice_;
  MPAS_rho_seawater = rho_seawater_;
}

void velocity_solver_extrude_3d_grid(int nLayers, int nGlobalTriangles,
    int nGlobalVertices, int nGlobalEdges, int Ordering, MPI_Comm reducedComm,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& mpasIndexToVertexID,
    const std::vector<double>& verticesCoords,
    const std::vector<bool>& isVertexBoundary,
    const std::vector<int>& verticesOnTria,
    const std::vector<bool>& isBoundaryEdge,
    const std::vector<int>& trianglesOnEdge,
    const std::vector<int>& trianglesPositionsOnEdge,
    const std::vector<int>& verticesOnEdge,
    const std::vector<int>& indexToEdgeID,
    const std::vector<GO>& indexToTriangleID,
    const std::vector<int>& dirichletNodesIds,
    const std::vector<int>& floating2dEdgesIds) {

  std::cout<<std::endl;
  slvrfctry = Teuchos::rcp(
      new Albany::SolverFactory("albany_input.xml", mpiCommT));
  paramList = Teuchos::rcp(&slvrfctry->getParameters(), false);


  //Physical Parameters
  paramList->sublist("Problem").sublist("Physical Parameters").set("Gravity", MPAS_gravity);
  paramList->sublist("Problem").sublist("Physical Parameters").set("Ice Density", MPAS_rho_ice);
  paramList->sublist("Problem").sublist("Physical Parameters").set("Water Density", MPAS_rho_seawater);

  if (paramList->sublist("Problem").get<std::string>("Name") == "FELIX Coupled FO H 3D") {
    paramList->sublist("Problem").sublist("Parameter Fields").set("Register Surface Mass Balance", 1);
    MPAS_dt = Teuchos::rcp(new double(paramList->sublist("Problem").get("Time Step", 0.0)));
    paramList->sublist("Problem").set("Time Step Ptr", MPAS_dt); //if it is not there set it to zero.
  }

  Teuchos::RCP<Teuchos::Array<double> >inputArrayBasal = Teuchos::rcp(new Teuchos::Array<double> (1, 1.0));
  paramList->sublist("Problem").sublist("Neumann BCs").set("Cubature Degree", 3);
  paramList->sublist("Problem").sublist("Neumann BCs").set("NBC on SS basalside for DOF all set basal_scalar_field", *inputArrayBasal);

  //Lateral floating ice BCs
  Teuchos::RCP<Teuchos::Array<double> >inputArrayLateral = Teuchos::rcp(new Teuchos::Array<double> (1, MPAS_rho_ice/MPAS_rho_seawater));
  paramList->sublist("Problem").sublist("Neumann BCs").set("NBC on SS floatinglateralside for DOF all set lateral", *inputArrayLateral);

  //Dirichlet BCs
  paramList->sublist("Problem").sublist("Dirichlet BCs").set("DBC on NS dirichlet for DOF U0 prescribe Field", "dirichlet_field");
  paramList->sublist("Problem").sublist("Dirichlet BCs").set("DBC on NS dirichlet for DOF U1 prescribe Field", "dirichlet_field");
  paramList->sublist("Discretization").set("Method", "Extruded");

  discParams = Teuchos::sublist(paramList, "Discretization", true);

  Albany::AbstractFieldContainer::FieldContainerRequirements req;
  albanyApp = Teuchos::rcp(new Albany::Application(mpiCommT));
  albanyApp->initialSetUp(paramList);

  int neq = (paramList->sublist("Problem").get<std::string>("Name") == "FELIX Coupled FO H 3D") ? 3 : 2;

  meshStruct = Teuchos::rcp(
      new Albany::MpasSTKMeshStruct(discParams, mpiCommT, indexToTriangleID,
          nGlobalTriangles, nLayers, Ordering));
  albanyApp->createMeshSpecs(meshStruct);

  albanyApp->buildProblem();

  meshStruct->constructMesh(mpiCommT, discParams, neq, req,
      albanyApp->getStateMgr().getStateInfoStruct(), indexToVertexID,
      mpasIndexToVertexID, verticesCoords, isVertexBoundary, nGlobalVertices,
      verticesOnTria, isBoundaryEdge, trianglesOnEdge, trianglesPositionsOnEdge,
      verticesOnEdge, indexToEdgeID, nGlobalEdges, indexToTriangleID,
      dirichletNodesIds, floating2dEdgesIds,
      meshStruct->getMeshSpecs()[0]->worksetSize, nLayers, Ordering);
}
//}

