//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_MeshAdapt.hpp"

//#include "AdaptUtil.h"
#include "Albany_SizeField.hpp"

#include "Teuchos_TimeMonitor.hpp"

Albany::MeshAdapt::
MeshAdapt(const Teuchos::RCP<Teuchos::ParameterList>& params_,
                     const Teuchos::RCP<ParamLib>& paramLib_,
                     Albany::StateManager& StateMgr_,
                     const Teuchos::RCP<const Epetra_Comm>& comm_) :
    Albany::AbstractAdapter(params_, paramLib_, StateMgr_, comm_),
    remeshFileIndex(1)
{

    disc = StateMgr.getDiscretization();

    fmdb_discretization = static_cast<Albany::FMDBDiscretization *>(disc.get());

    fmdbMeshStruct = fmdb_discretization->getFMDBMeshStruct();

    mesh = fmdbMeshStruct->getMesh();

//    this->sizeFieldFunc = &Albany::MeshAdapt::setSizeField;
//    this->sizeFieldFunc = &(this->setSizeField);

}

Albany::MeshAdapt::
~MeshAdapt()
{
}

bool
Albany::MeshAdapt::queryAdaptationCriteria(){

  int remesh_iter = params->get<int>("Remesh Step Number");

   if(iter == remesh_iter)
     return true;

  return false; 

}

bool
Albany::MeshAdapt::adaptMesh(){

  TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
    std::endl << "Error in Adaptation: calling Albany::MeshAdapt adaptMesh() without passing solution vector." << std::endl);

}

int 
setSizeField(pMesh mesh, pSField pSizeField, void *vp){

  Albany::SizeField *aSF = static_cast<Albany::SizeField*>(pSizeField);

  return aSF->computeSizeField();

}

int uniform(pPart part,  pSField field)
{
  pMeshEnt vtx;
  double h[3], dirs[3][3], xyz[3];

  pPartEntIter vtx_iter;
  FMDB_PartEntIter_Init(part, FMDB_VERTEX, FMDB_ALLTOPO, vtx_iter);
  while (FMDB_PartEntIter_GetNext(vtx_iter, vtx)==SCUtil_SUCCESS)
  {
    h[0] = .5;   
    h[1] = .5;
    h[2] = .5;

    dirs[0][0]=1.0;
    dirs[0][1]=0.;
    dirs[0][2]=0.;
    dirs[1][0]=0.;
    dirs[1][1]=1.0;
    dirs[1][2]=0.;
    dirs[2][0]=0.;
    dirs[2][1]=0.;
    dirs[2][2]=1.0;

    ((PWLsfield *)field)->setSize(vtx,dirs,h);
  }
  FMDB_PartEntIter_Del(vtx_iter);

  double beta[]={1.5,1.5,1.5};
  ((PWLsfield *)field)->anisoSmooth(beta);
  return 1;
}

int getCurrentSize(pMesh mesh, double& globMinSize, double& globMaxSize, double& globAvgSize) {
   EIter eit = M_edgeIter(mesh);
   pEdge edge;
   int numEdges = 0;
   double avgSize = 0.;
   double minSize = std::numeric_limits<double>::max();
   double maxSize = std::numeric_limits<double>::min();
   
   while (edge = EIter_next(eit)) {
      numEdges++;
      double len = sqrt(E_lengthSq(edge));
      avgSize += len;
      if ( len < minSize ) minSize = len;
      if ( len > maxSize ) maxSize = len;
   }
   EIter_delete(eit);
   avgSize /= numEdges; 

   MPI_Allreduce(&avgSize, &globAvgSize, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
   MPI_Allreduce(&maxSize, &globMaxSize, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
   MPI_Allreduce(&minSize, &globMinSize, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
   globAvgSize /= SCUTIL_CommSize();
}

int uniformRefSzFld(pMesh mesh, pSField field, void *vp) {
   pVertex vt;
   double h[3], dirs[3][3];
   
   static int numCalls = 0;
   static double initAvgEdgeLen = 0;
   static double currGlobMin = 0, currGlobMax = 0, currGlobAvg = 0;
   if (0 == numCalls) {
      getCurrentSize(mesh, currGlobMin, currGlobMax, initAvgEdgeLen);
   } else {
      getCurrentSize(mesh, currGlobMin, currGlobMax, currGlobAvg);
   }
   
  double minSize = std::numeric_limits<double>::max();
  double maxSize = std::numeric_limits<double>::min();
        
   VIter vit = M_vertexIter(mesh);
   while (vt = VIter_next(vit)) {
      const double sz = 0.5 * initAvgEdgeLen;
      if ( sz < minSize ) minSize = sz;
      if ( sz > maxSize ) maxSize = sz;   
      for (int i = 0; i < 3; i++) {
         h[i] = sz;
      }

      dirs[0][0] = 1.0;
      dirs[0][1] = 0.;
      dirs[0][2] = 0.;
      dirs[1][0] = 0.;
      dirs[1][1] = 1.0;
      dirs[1][2] = 0.;
      dirs[2][0] = 0.;
      dirs[2][1] = 0.;
      dirs[2][2] = 1.0;

      ((PWLsfield *) field)->setSize((pEntity) vt, dirs, h);
   }

   VIter_delete(vit);
//   double beta[] = {1.5, 1.5, 1.5};
//   ((PWLsfield *) field)->anisoSmooth(beta);
   

   double globMin = 0;
   double globMax = 0;
   MPI_Reduce(&minSize, &globMin, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
   MPI_Reduce(&maxSize, &globMax, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

   if (0 == SCUTIL_CommRank()) {
      if ( 0 == numCalls ) {
         printf("%s initial edgeLength avg %f min %f max %f\n", __FUNCTION__, initAvgEdgeLen, currGlobMin, currGlobMax);
         printf("%s target edgeLength min %f max %f\n", __FUNCTION__, globMin, globMax);
      } else {
         printf("%s current edgeLength avg %f min %f max %f\n", __FUNCTION__, currGlobAvg, currGlobMin, currGlobMax);      
      }
      fflush(stdout);      
   }   
   
   numCalls++;
   
   return 1;
}


int sizefield(pPart part, pSField field, void *)
{
  uniform (part, field);
  return 1;
}

bool
//Albany::MeshAdapt::adaptMesh(const Epetra_Vector& Solution, const Teuchos::RCP<Epetra_Import>& importer){
Albany::MeshAdapt::adaptMesh(const Epetra_Vector& sol, const Epetra_Vector& ovlp_sol){

  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  std::cout << "Adapting mesh using Albany::MeshAdapt method        " << std::endl;
  std::cout << "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~" << std::endl;
  // display # entities before adaptation
  FMDB_Mesh_DspNumEnt (mesh);

/*
The api to use displaced mesh for mesh adaptation will be
PUMI_Mesh_UseDisp (pMeshMdl mesh, pPag displacement_tag);

For our case, it will be PUMI_Mesh_UseDisp (*fmdbMeshStruct->getMesh(), *fmdbMeshStruct->solution_field);

In mesh adaptation, original coord+displacement will be used.
After mesh adaptation, the new displacement value will be available through solution_field. and new vertex coordinates will be new reference value.
*/
// FIXME: how about getting this as an input parameter
  int num_iteration = 1;

  // Do basic uniform refinement
  /** Type of the size field: 
      - Application - the size field will be provided by the application (default).
      - TagDriven - tag driven size field. 
      - Analytical - analytical size field.  */
  /** Type of model: 
      - 0 - no model (not snap), 1 - mesh model (always snap), 2 - solid model (always snap)
  */
  meshAdapt *rdr = new meshAdapt(mesh, /*size field type*/ Application, /*model type*/ 2 );

  /** void meshAdapt::run(int niter,    // specify the maximum number of iterations 
		    int flag,           // indicate if a size field function call is available
		    adaptSFunc sizefd)  // the size field function call  */
  rdr->run (num_iteration, 1, sizefield);

  // dump the adapted mesh for visualization
  FMDB_Mesh_WriteToFile (mesh, "adapted_mesh_out.vtu",  (SCUTIL_CommSize()>1?1:0));
  // display # entities after adaptation
  FMDB_Mesh_DspNumEnt (mesh);
  // check the validity of adapted mesh
  int isValid=0;
  FMDB_Mesh_Verify(mesh, &isValid);

  delete rdr;
  return true;
}

//! Transfer solution between meshes.
void
Albany::MeshAdapt::
solutionTransfer(const Epetra_Vector& oldSolution,
        Epetra_Vector& newSolution){

}

Teuchos::RCP<const Teuchos::ParameterList>
Albany::MeshAdapt::getValidAdapterParameters() const
{
  Teuchos::RCP<Teuchos::ParameterList> validPL =
    this->getGenericAdapterParams("ValidMeshAdaptParams");

/*
  if (numDim==1)
    validPL->set<bool>("Periodic BC", false, "Flag to indicate periodic BC for 1D problems");
  validPL->sublist("Thermal Conductivity", false, "");
  validPL->set("Convection Velocity", "{0,0,0}", "");
  validPL->set<bool>("Have Rho Cp", false, "Flag to indicate if rhoCp is used");
  validPL->set<string>("MaterialDB Filename","materials.xml","Filename of material database xml file");
*/

  validPL->set<int>("Remesh Step Number", 1, "Iteration step at which to remesh the problem");

  return validPL;
}


