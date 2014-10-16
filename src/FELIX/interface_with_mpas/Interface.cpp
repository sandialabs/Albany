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
#include "Piro_PerformSolve.hpp"
#include <stk_io/IossBridge.hpp>
#include <stk_io/StkMeshIoBroker.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <Ionit_Initializer.h>
#include "Albany_OrdinarySTKFieldContainer.hpp"

// ===================================================
//! Namespaces
// ===================================================
//using namespace LifeV;

//typedef std::list<exchange> exchangeList_Type;

// ice_problem pointer
//ICEProblem *iceProblemPtr = 0;
Teuchos::RCP<Albany::MpasSTKMeshStruct> meshStruct2D;
Teuchos::RCP<Albany::MpasSTKMeshStruct> meshStruct;
Teuchos::RCP<Albany::Application> albanyApp;
Teuchos::RCP<Teuchos::ParameterList> paramList;
Teuchos::RCP<const Teuchos_Comm> mpiComm;
Teuchos::RCP<Teuchos::ParameterList> discParams;
Teuchos::RCP<Albany::SolverFactory> slvrfctry;
Teuchos::RCP<Thyra::ResponseOnlyModelEvaluatorBase<double> > solver;
bool keptMesh =false;

typedef struct TET_ {
  int verts[4];
  int neighbours[4];
  char bound_type[4];
} TET;


#ifdef ALBANY_EPETRA

/***********************************************************/
// epetra <-> thyra conversion utilities
Teuchos::RCP<const Epetra_Vector> epetraVectorFromThyra(
    const Teuchos::RCP<const Epetra_Comm> &comm,
    const Teuchos::RCP<const Thyra::VectorBase<double> > &thyra) {
  Teuchos::RCP<const Epetra_Vector> result;
  if (Teuchos::nonnull(thyra)) {
    const Teuchos::RCP<const Epetra_Map> epetra_map = Thyra::get_Epetra_Map(
        *thyra->space(), comm);
    result = Thyra::get_Epetra_Vector(*epetra_map, thyra);
  }
  return result;
}

Teuchos::RCP<const Epetra_MultiVector> epetraMultiVectorFromThyra(
    const Teuchos::RCP<const Epetra_Comm> &comm,
    const Teuchos::RCP<const Thyra::MultiVectorBase<double> > &thyra) {
  Teuchos::RCP<const Epetra_MultiVector> result;
  if (Teuchos::nonnull(thyra)) {
    const Teuchos::RCP<const Epetra_Map> epetra_map = Thyra::get_Epetra_Map(
        *thyra->range(), comm);
    result = Thyra::get_Epetra_MultiVector(*epetra_map, thyra);
  }
  return result;
}

void epetraFromThyra(const Teuchos::RCP<const Epetra_Comm> &comm,
    const Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > &thyraResponses,
    const Teuchos::Array<
        Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > &thyraSensitivities,
    Teuchos::Array<Teuchos::RCP<const Epetra_Vector> > &responses,
    Teuchos::Array<Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > > &sensitivities) {
  responses.clear();
  responses.reserve(thyraResponses.size());
  typedef Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > ThyraResponseArray;
  for (ThyraResponseArray::const_iterator it_begin = thyraResponses.begin(),
      it_end = thyraResponses.end(), it = it_begin; it != it_end; ++it) {
    responses.push_back(epetraVectorFromThyra(comm, *it));
  }

  sensitivities.clear();
  sensitivities.reserve(thyraSensitivities.size());
  typedef Teuchos::Array<
      Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > ThyraSensitivityArray;
  for (ThyraSensitivityArray::const_iterator it_begin =
      thyraSensitivities.begin(), it_end = thyraSensitivities.end(), it =
      it_begin; it != it_end; ++it) {
    ThyraSensitivityArray::const_reference sens_thyra = *it;
    Teuchos::Array<Teuchos::RCP<const Epetra_MultiVector> > sens;
    sens.reserve(sens_thyra.size());
    for (ThyraSensitivityArray::value_type::const_iterator jt =
        sens_thyra.begin(), jt_end = sens_thyra.end(); jt != jt_end; ++jt) {
      sens.push_back(epetraMultiVectorFromThyra(comm, *jt));
    }
    sensitivities.push_back(sens);
  }
}
#endif

/***********************************************************/

//extern "C" {
#ifdef HAVE_PHG
extern void phg_init_(int *fComm);
extern void *phgImportParallelGrid(void *old_grid,
    int nvert,
    int nelem,
    int nvert_global,
    int nelem_global,
    int *L2Gmap_vert,
    int *L2Gmap_elem,
    double *coord,
    TET *tet,
    MPI_Comm comm
);
extern void phgSolveIceStokes(void *g_ptr,
    int nlayer,
    const double *T,
    const double *beta,
    double *U);
#endif

// ===================================================
//! Interface functions
// ===================================================

#ifdef HAVE_PHG
static void
check_face (RegionMesh<LinearTetra>& mesh, ID i, ID j)
{
  ID faceId = mesh.localFaceId (i, j);

  if ( (ID) faceId != NotAnId)
  {
    RegionMesh<LinearTetra>::face_Type& face
    = mesh.face (faceId);

    if (0)
    fprintf (stdout, "  face: %5d %5d, ",
        faceId, face.id() );

    ID v0, v1, v2, gv0, gv1, gv2;
    v0 = LinearTetra::faceToPoint (j, 0);
    v1 = LinearTetra::faceToPoint (j, 1);
    v2 = LinearTetra::faceToPoint (j, 2);
    // fprintf(stdout, "[%5d %5d %5d], ", v0, v1, v2);

    gv0 = mesh.element (i).point (v0).id();
    gv1 = mesh.element (i).point (v1).id();
    gv2 = mesh.element (i).point (v2).id();
    fprintf (stdout, "[ %7d %7d %7d ], ", gv0, gv1, gv2);

    int mark = 0;
    if (face.flag() &
        EntityFlags::PHYSICAL_BOUNDARY)
    {
      mark = 1;
    }
    if (face.flag() &
        EntityFlags::SUBDOMAIN_INTERFACE)
    {
      mark += 2;
    }
    fprintf (stdout, "mark: %d, ", mark);

    ID vol0, vol1;
    ID pos0, pos1;
    vol0 = face.firstAdjacentElementIdentity();
    pos0 = face.firstAdjacentElementPosition();
    vol1 = face.secondAdjacentElementIdentity();
    pos1 = face.secondAdjacentElementPosition();

    if (vol0 == NotAnId)
    {
      vol0 = -1;
    }
    if (vol1 == NotAnId)
    {
      vol1 = -1;
    }
    if (pos0 == NotAnId)
    {
      pos0 = -1;
    }
    if (pos1 == NotAnId)
    {
      pos1 = -1;
    }

    fprintf (stdout, "first: %5d %d, second: %5d %d",
        vol0, pos0, vol1, pos1);

    fprintf (stdout, "\n");
  }
  else
  {
    fprintf (stdout, "Invalid face!!!\n");
  }
}
#endif

#ifdef HAVE_PHG
void velocity_solver_init_stokes (double const* levelsRatio_F)
{
  int verb = 1;

  // set velocity 0
  velocityOnVertices.resize (2 * nVertices * (nLayers + 1), 0.);
  velocityOnCells.resize (2 * nCells_F * (nLayers + 1), 0.);

  //
  // interface to phg
  //
  printf ("* Init full stokes?\n");
  if (isDomainEmpty)
  {
    return;
  }

  layersRatio.resize (nLayers);
  // !!Indexing of layers is reversed
  for (UInt i = 0; i < nLayers; i++)
  {
    layersRatio[i] = levelsRatio_F[nLayers - 1 - i];
  }
  //std::copy(levelsRatio_F, levelsRatio_F+nLayers, layersRatio.begin());

  mapCellsToVertices (velocityOnCells, velocityOnVertices, 2, nLayers, LayerWise);

  printf ("* Init full stokes.\n");
  //
  // Output 3D mesh
  //
  RegionMesh<LinearTetra>& mesh = * (iceProblemPtr->mesh3DPtr);
  unsigned int numVertices = mesh.numVertices();
  unsigned int numElements = mesh.numElements();

  std::vector<double > coord (3 * numVertices);

  TET* tet;
  tet = (TET*) calloc (numElements, sizeof (*tet) );

  // update faces
  mesh.updateElementFaces();
  //if (verb > 0) printf("has local face :%d\n", mesh.hasLocalFaces());

  // Num of verts (local, global)
  if (verb > 0) printf ("nvert:%5d, nvert_global:%5d\n",
      mesh.numVertices(), mesh.numGlobalVertices() );

  // Num of elements (local, global)
  if (verb > 0) printf ("nedge:%5d, nedge_global:%5d\n",
      mesh.numElements(), mesh.numGlobalElements() );

  if (verb > 0) printf ("nface:%5d, nface_global:%5d\n",
      mesh.numFaces(), mesh.numGlobalFaces() );

  if (verb > 0) printf ("nelem:%5d, nelem_global:%5d\n",
      mesh.numElements(), mesh.numGlobalElements() );

  //
  // Element:
  //
  // verts[4] (local)
  // bdry_type[4]
  // neigh[4]
  //
  for (unsigned int i = 0; i < numElements; i++)
  {
    if (verb > 1)
    {
      printf ("elem: %d\n", i);
    }
    for (unsigned int j = 0; j < 4; j++)
    {
      const double* c = &mesh.element (i).point (j).coordinates() [0];
      if (verb > 1) printf ("   vert: (%16.8f, %16.8f, %16.8f) %5d %5d\n",
          c[0], c[1], c[2],
          mesh.element (i).point (j).localId(),
          mesh.element (i).point (j).id() //global Id
      );
      int vert = mesh.element (i).point (j).localId();
      tet[i].verts[j] = vert;
      coord[vert * 3 ] = c[0];
      coord[vert * 3 + 1] = c[1];
      coord[vert * 3 + 2] = c[2];
    }

    for (unsigned int j = 0; j < 4; j++)
    {

      //  Life tet face (ElementShapes)
      //  0, 2, 1,
      //  0, 1, 3,
      //  1, 2, 3,
      //  0, 3, 2
      //
      //  PHG tet face (utils.c)
      //  1, 2, 3,
      //  0, 2, 3,
      //  0, 1, 3,
      //  0, 1, 2
      //
      //  map Life to PHG
      //  3, 2, 0, 1
      //  reverse
      //  2, 3, 1, 0
      //
      if (verb > 1)
      {
        check_face (mesh, i, j);
      }

      int faceId = mesh.localFaceId (i, j);
      if (verb > 1)
      {
        printf (" faceId %d\n", faceId);
      }
      assert ( (ID) faceId != NotAnId);
      static int map_[4] = {3, 2, 0, 1};

      RegionMesh<LinearTetra>::face_Type& face
      = mesh.face (faceId);

      ID vol0, vol1;
      ID pos0, pos1;
      vol0 = face.firstAdjacentElementIdentity();
      pos0 = face.firstAdjacentElementPosition();
      vol1 = face.secondAdjacentElementIdentity();
      pos1 = face.secondAdjacentElementPosition();

      //
      // bound_type: 1. interior(local), 2.interior(remote), [3|4|5].bdry
      //
      if (face.flag() & EntityFlags::PHYSICAL_BOUNDARY)
      {
        assert (vol0 == i);
        assert (pos0 == j);
        assert (vol1 == NotAnId);
        assert (pos1 == NotAnId);

        //
        // lower face: 1; upper face 2,
        //    pass as: 3(low), 4(up), 5(lateral, other)
        // Note: marker comes from mpas
        //
        unsigned int lowerSurfaceMarker = 1;
        unsigned int upperSurfaceMarker = 2;

        tet[i].neighbours[map_[j]] = -1;
        if (face.markerID() == lowerSurfaceMarker)
        {
          tet[i].bound_type[map_[j]] = 3;
        }
        else if (face.markerID() == upperSurfaceMarker)
        {
          tet[i].bound_type[map_[j]] = 4;
        }
        else
        {
          tet[i].bound_type[map_[j]] = 5;
        }

      }
      else if (face.flag() & EntityFlags::SUBDOMAIN_INTERFACE)
      {
        assert (vol0 == i);
        assert (pos0 == j);
        assert (vol1 == NotAnId);
        assert (pos1 == NotAnId);

        tet[i].neighbours[map_[j]] = -1;
        tet[i].bound_type[map_[j]] = 2;
      }
      else
      {
        if (vol0 == i)
        {
          assert (pos0 == j);
          tet[i].neighbours[map_[j]] = vol1;
          tet[i].bound_type[map_[j]] = 1;
        }
        else if (vol1 == i)
        {
          assert (pos1 == j);
          tet[i].neighbours[map_[j]] = vol0;
          tet[i].bound_type[map_[j]] = 1;
        }
        else
        {
          abort();
        }
      }
    }
  }

  // L2G map vert
  std::vector<int > L2Gmap_vert (numVertices);
  for (unsigned int i = 0; i < numVertices; i++)
  {
    L2Gmap_vert[i] = mesh.point (i).id();
    if (verb > 1) printf ("  L2G vert: %5d %5d\n", i,
        mesh.point (i).id() );
  }

  // L2G map elem
  std::vector<int > L2Gmap_elem (numElements);
  for (unsigned int i = 0; i < numElements; i++)
  {
    L2Gmap_elem[i] = mesh.element (i).id();
    if (verb > 1) printf ("  L2G elem: %5d %5d\n", i,
        mesh.element (i).id() );
  }

  // if (verb > 0) printf("check all face\n");
  // for (unsigned int i = 0; i < numFaces; i++) {
  //  check_face(mesh, i);
  // }

  printf ("  Call PHG\n");
  phgGrid = phgImportParallelGrid (phgGrid,// old grid
      numVertices,
      numElements,
      mesh.numGlobalVertices(),
      mesh.numGlobalElements(),
      &L2Gmap_vert[0],
      &L2Gmap_elem[0],
      &coord[0],
      tet,
      reducedComm
  );

  return;
}
#endif

#define GET_HERE printf("get %25s, line: %d\n", __FILE__, __LINE__);

#ifdef HAVE_PHG
void velocity_solver_solve_stokes (double const* lowerSurface_F, double const* thickness_F,
    double const* beta_F, double const* temperature_F,
    double* u_normal_F,
    double* /*heatIntegral_F*/, double* /*viscosity_F*/)
{
  std::fill (u_normal_F, u_normal_F + nEdges_F * nLayers, 0.);

  if (!isDomainEmpty)
  {

    RegionMesh<LinearTetra>& mesh3D = * (iceProblemPtr->mesh3DPtr);
    //  RegionMesh<LinearTriangle>& mesh2D =  *(iceProblemPtr->mesh2DPtr);

    // Full Stokes solution
    //   [nvert+nedge][u|v|w]
    std::vector<Real> Velocity_FS (3 * mesh3D.numVertices()
        + 3 * mesh3D.numEdges() );
    // P1:   [u|v][nvert] using velocityOnVertices

    // P2:   [u|v][nvert+nedge]
    std::vector<Real> Velocity_P2 (2 * mesh3D.numVertices()
        + 2 * mesh3D.numEdges() );

    importP0Temperature (temperature_F);// import temperature to
    // temperatureOnTetra[numElements]
    import2DFields (lowerSurface_F, thickness_F, beta_F);// import surf, thick and beta
    // to [numVertices]

    // Note:
    // Grid is supposed to be layerwise to import beta
    phgSolveIceStokes (phgGrid,
        nLayers,
        &temperatureOnTetra[0],
        &betaData[0],
        &Velocity_FS[0]);

    unsigned int nEdges3D = mesh3D.numEdges();
    unsigned int numElements = mesh3D.numElements();
    GET_HERE;
    unsigned int nVertices3D = mesh3D.numVertices();
    assert (nEdges3D > 0);

    GET_HERE;
    for (unsigned int i = 0; i < numElements; i++)
    {

      // Vert
      for (unsigned int j = 0; j < 4; j++)
      {
        int v = mesh3D.element (i).point (j).localId();
        velocityOnVertices[v] = Velocity_FS[3 * v];
        velocityOnVertices[nVertices3D + v] = Velocity_FS[3 * v + 1];
        Velocity_P2[v] = Velocity_FS[3 * v];
        Velocity_P2[nVertices3D + nEdges3D + v] = Velocity_FS[3 * v + 1];
      }

      //  Life tet edge (ElementShapes)
      //  0, 1,
      //  1, 2,
      //  2, 0,
      //  0, 3,
      //  1, 3,
      //  2, 3
      //
      //  PHG tet face (utils.c)
      //  0, 1
      //  0, 2
      //  0, 3
      //  1, 2
      //  1, 3
      //  2, 3
      //
      //  map Life to PHG
      //  0, 3, 1, 2, 4, 5
      //  reverse
      //  0, 2, 3, 1, 4, 5
      //

      // for (unsigned int j = 0; j < 4; j++) {
      //  int e = mesh3D.element(i).edge(j).localId();
      //  Velocity_P1[e] =  Velocity_FS[3*e];
      //  Velocity_P1[nVertices + v]  = Velocity_FS[3*e + 1];
      //  Velocity_P2[e] =  Velocity_FS[3*v];
      //  Velocity_P2[nVertices+nEdges + e]  = Velocity_FS[3*e + 1];
      // }
    }

    //save velocity to be used as an initial guess next iteration
    GET_HERE;
    std::vector<int> mpasIndexToVertexID (nVertices);
    for (int i = 0; i < nVertices; i++)
    {
      mpasIndexToVertexID[i] = indexToCellID_F[vertexToFCell[i]];
    }
    get_tetraP1_velocity_on_FEdges (u_normal_F, velocityOnVertices, edgeToFEdge, mpasIndexToVertexID);

  }

  GET_HERE;
  mapVerticesToCells (velocityOnVertices, &velocityOnCells[0], 2, nLayers, LayerWise);

  GET_HERE;
  allToAll (u_normal_F, &sendEdgesListReversed, &recvEdgesListReversed, nLayers);

  GET_HERE;
  allToAll (u_normal_F, sendEdgesList_F, recvEdgesList_F, nLayers);
  GET_HERE;

  return;
}
#endif

void velocity_solver_export_2d_data(MPI_Comm reducedComm,
    const std::vector<double>& elevationData,
    const std::vector<double>& thicknessData,
    const std::vector<double>& betaData,
    const std::vector<int>& indexToVertexID) {

  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(reducedComm));
  mesh_data->set_bulk_data(*meshStruct2D->bulkData);
  size_t idx = mesh_data->create_output_mesh("mesh2D.exo", stk::io::WRITE_RESULTS);
  mesh_data->process_output_request(idx, 0.0);
}



void velocity_solver_solve_fo(int nLayers, int nGlobalVertices,
    int nGlobalTriangles, bool ordering, bool first_time_step,
    const std::vector<int>& indexToVertexID,
    const std::vector<int>& indexToTriangleID, double minBeta,
    const std::vector<double>& regulThk,
    const std::vector<double>& levelsNormalizedThickness,
    const std::vector<double>& elevationData,
    const std::vector<double>& thicknessData,
    const std::vector<double>& betaData,
    const std::vector<double>& temperatureOnTetra,
    std::vector<double>& velocityOnVertices) {

  int numVertices3D = (nLayers + 1) * indexToVertexID.size();
  int numPrisms = nLayers * indexToTriangleID.size();
  int vertexColumnShift = (ordering == 1) ? 1 : nGlobalVertices;
  int lVertexColumnShift = (ordering == 1) ? 1 : indexToVertexID.size();
  int vertexLayerShift = (ordering == 0) ? 1 : nLayers + 1;

  int elemColumnShift = (ordering == 1) ? 3 : 3 * nGlobalTriangles;
  int lElemColumnShift = (ordering == 1) ? 3 : 3 * indexToTriangleID.size();
  int elemLayerShift = (ordering == 0) ? 3 : 3 * nLayers;

  const bool interleavedOrdering = meshStruct->getInterleavedOrdering();
  Albany::AbstractSTKFieldContainer::VectorFieldType* solutionField;

  if (interleavedOrdering)
    solutionField = Teuchos::rcp_dynamic_cast<
        Albany::OrdinarySTKFieldContainer<true> >(
        meshStruct->getFieldContainer())->getSolutionField();
  else
    solutionField = Teuchos::rcp_dynamic_cast<
        Albany::OrdinarySTKFieldContainer<false> >(
        meshStruct->getFieldContainer())->getSolutionField();

  typedef Albany::AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef Albany::AbstractSTKFieldContainer::QPScalarFieldType ElemScalarFieldType;



  for (UInt j = 0; j < numVertices3D; ++j) {
    int ib = (ordering == 0) * (j % lVertexColumnShift)
        + (ordering == 1) * (j / vertexLayerShift);
    int il = (ordering == 0) * (j / lVertexColumnShift)
        + (ordering == 1) * (j % vertexLayerShift);
    int gId = il * vertexColumnShift + vertexLayerShift * indexToVertexID[ib];
    stk::mesh::Entity node = meshStruct->bulkData->get_entity(stk::topology::NODE_RANK, gId + 1);
    double* coord = stk::mesh::field_data(*meshStruct->getCoordinatesField(), node);
    coord[2] = elevationData[ib] - levelsNormalizedThickness[nLayers - il] * regulThk[ib];

     double* sHeight = stk::mesh::field_data(*meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "surface_height"), node);
    sHeight[0] = elevationData[ib];
    double* thickness = stk::mesh::field_data(*meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "thickness"), node);
    thickness[0] = thicknessData[ib];
    double* sol = stk::mesh::field_data(*solutionField, node);
    sol[0] = velocityOnVertices[j];
    sol[1] = velocityOnVertices[j + numVertices3D];
    if (il == 0) {
      double* beta = stk::mesh::field_data(*meshStruct->metaData->get_field <ScalarFieldType> (stk::topology::NODE_RANK, "basal_friction"), node);
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

  meshStruct->setHasRestartSolution(!first_time_step);

  if (!first_time_step) {
    meshStruct->setRestartDataTime(
        paramList->sublist("Problem").get("Homotopy Restart Step", 1.));
    double homotopy =
        paramList->sublist("Problem").sublist("FELIX Viscosity").get(
            "Glen's Law Homotopy Parameter", 1.0);
    if (meshStruct->restartDataTime() == homotopy)
      paramList->sublist("Problem").set("Solution Method", "Steady");
  }



  if(!keptMesh) {
    albanyApp->createDiscretization();
    albanyApp->finalSetUp(paramList);
  }
  else
    albanyApp->getDiscretization()->updateMesh();



  //solver = slvrfctry->createThyraSolverAndGetAlbanyApp(albanyApp, mpiComm,
  //    mpiComm, Teuchos::null, false);
  solver = slvrfctry->createAndGetAlbanyAppT(albanyApp, mpiComm, mpiComm); 

  Teuchos::ParameterList solveParams;
  solveParams.set("Compute Sensitivities", false);

  Teuchos::Array<Teuchos::RCP<const Thyra::VectorBase<double> > > thyraResponses;
  Teuchos::Array<
      Teuchos::Array<Teuchos::RCP<const Thyra::MultiVectorBase<double> > > > thyraSensitivities;
  Piro::PerformSolveBase(*solver, solveParams, thyraResponses,
      thyraSensitivities);
  Teuchos::RCP<const Tpetra_Map> overlapMap = albanyApp->getDiscretization()->getOverlapMapT();
  Teuchos::RCP<Tpetra_Import> import = Teuchos::rcp(new Tpetra_Import(overlapMap, albanyApp->getDiscretization()->getMapT()));
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
      lId0 = overlapMap->getLocalElement(2 * gId);
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
  Teuchos::RCP<stk::io::StkMeshIoBroker> mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(reducedComm));
    mesh_data->set_bulk_data(*meshStruct->bulkData);
    size_t idx = mesh_data->create_output_mesh("IceSheet.exo", stk::io::WRITE_RESULTS);
    mesh_data->process_output_request(idx, 0.0);
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
  mpiComm = Albany::createTeuchosCommFromMpiComm(reducedComm);
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
    const std::vector<int>& indexToTriangleID) {

  slvrfctry = Teuchos::rcp(
      new Albany::SolverFactory("albany_input.xml", mpiComm));
  paramList = Teuchos::rcp(&slvrfctry->getParameters(), false);
  discParams = Teuchos::sublist(paramList, "Discretization", true);

  Albany::AbstractFieldContainer::FieldContainerRequirements req;
  albanyApp = Teuchos::rcp(new Albany::Application(mpiComm));
  albanyApp->initialSetUp(paramList);

  int neq = 2;

  meshStruct = Teuchos::rcp(
      new Albany::MpasSTKMeshStruct(discParams, mpiComm, indexToTriangleID,
          nGlobalTriangles, nLayers, Ordering));
  albanyApp->createMeshSpecs(meshStruct);

  albanyApp->buildProblem();

  meshStruct->constructMesh(mpiComm, discParams, neq, req,
      albanyApp->getStateMgr().getStateInfoStruct(), indexToVertexID,
      mpasIndexToVertexID, verticesCoords, isVertexBoundary, nGlobalVertices,
      verticesOnTria, isBoundaryEdge, trianglesOnEdge, trianglesPositionsOnEdge,
      verticesOnEdge, indexToEdgeID, nGlobalEdges, indexToTriangleID,
      meshStruct->getMeshSpecs()[0]->worksetSize, nLayers, Ordering);
}
//}

