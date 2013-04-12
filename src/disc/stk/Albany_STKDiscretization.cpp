//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//


#include <limits>
#include "Epetra_Export.h"

#include "Albany_Utils.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Petra_Converters.hpp"
#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>
#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/FieldData.hpp>
#include <stk_mesh/base/Selector.hpp>
//#include <stk_mesh/fem/CreateAdjacentEntities.hpp>

//#include <Intrepid_FieldContainer.hpp>
#include <PHAL_Dimension.hpp>

#include <stk_mesh/fem/FEMHelpers.hpp>

#ifdef ALBANY_SEACAS
#include <Ionit_Initializer.h>
#endif 

#include <algorithm>

const double pi = 3.1415926535897932385;

Albany::STKDiscretization::STKDiscretization(Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct_,
					     const Teuchos::RCP<const Epetra_Comm>& comm_) :
  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  metaData(*stkMeshStruct_->metaData),
  bulkData(*stkMeshStruct_->bulkData),
  comm(comm_),
  commT(Albany::createTeuchosCommFromMpiComm(Albany::getMpiCommFromEpetraComm(*comm_))),
  neq(stkMeshStruct_->neq),
  stkMeshStruct(stkMeshStruct_),
  interleavedOrdering(stkMeshStruct_->interleavedOrdering),
  allocated_xyz(false)
{
   //Ultimately Tpetra comm needs to be passed in to this constructor like Epetra comm...
   //Create the Kokkos Node instance to pass into Tpetra::Map constructors.
  Teuchos::ParameterList kokkosNodeParams;
  nodeT = Teuchos::rcp(new KokkosNode (kokkosNodeParams));
  Albany::STKDiscretization::updateMesh();
}

Albany::STKDiscretization::~STKDiscretization()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) delete mesh_data;
#endif
  if (allocated_xyz) { delete [] xx; delete [] yy; delete [] zz; delete [] rr; allocated_xyz=false;} 

  for (int i=0; i< toDelete.size(); i++) delete [] toDelete[i];
}

	    
Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getMap() const
{
  Teuchos::RCP<const Epetra_Map> map = Petra::TpetraMap_To_EpetraMap(mapT, comm); 
  return map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getMapT() const
{
  return mapT;
}


Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getOverlapMap() const
{
  Teuchos::RCP<const Epetra_Map> overlap_map = Petra::TpetraMap_To_EpetraMap(overlap_mapT, comm);
  return overlap_map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getOverlapMapT() const
{
  return overlap_mapT;
}


Teuchos::RCP<const Epetra_CrsGraph>
Albany::STKDiscretization::getJacobianGraph() const
{
  Teuchos::RCP<const Epetra_CrsGraph> graph= Petra::TpetraCrsGraph_To_EpetraCrsGraph(graphT, comm); 
  return graph;
}

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::STKDiscretization::getJacobianGraphT() const
{
  return graphT;
}

Teuchos::RCP<const Epetra_CrsGraph>
Albany::STKDiscretization::getOverlapJacobianGraph() const
{
  Teuchos::RCP<const Epetra_CrsGraph> overlap_graph= Petra::TpetraCrsGraph_To_EpetraCrsGraph(overlap_graphT, comm); 
  return overlap_graph;
}

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::STKDiscretization::getOverlapJacobianGraphT() const
{
  return overlap_graphT;
}


Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getNodeMap() const
{
  Teuchos::RCP<const Epetra_Map> node_map = Petra::TpetraMap_To_EpetraMap(node_mapT, comm); 
  return node_map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getNodeMapT() const
{
  return node_mapT;
}


const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<int> > > >&
Albany::STKDiscretization::getWsElNodeEqID() const
{
  return wsElNodeEqID;
}

const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >&
Albany::STKDiscretization::getCoords() const
{
  return coords;
}

const Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double> > >&
Albany::STKDiscretization::getSurfaceHeight() const
{
  return sHeight;
}

Teuchos::ArrayRCP<double>& 
Albany::STKDiscretization::getCoordinates() const
{
  // Coordinates are computed here, and not precomputed,
  // since the mesh can move in shape opt problems
  for (int i=0; i < numOverlapNodes; i++)  {
    int node_gid = gid(overlapnodes[i]);
    int node_lid = overlap_node_mapT->getLocalElement(node_gid);

    double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
    for (int dim=0; dim<stkMeshStruct->numDim; dim++)
      coordinates[3*node_lid + dim] = x[dim];

  }

  return coordinates;
}

//The function transformMesh() maps a unit cube domain by applying the transformation 
//x = L*x
//y = L*y
//z = s(x,y)*z + b(x,y)*(1-z)
//where b(x,y) and s(x,y) are curves specifying the bedrock and top surface 
//geometries respectively.   
//Currently this function is only needed for some FELIX problems.


void
Albany::STKDiscretization::transformMesh()
{
  std::string transformType = stkMeshStruct->transformType;
  if (transformType == "ISMIP-HOM Test A") {
    cout << "Test A!" << endl;
    double L = stkMeshStruct->felixL; 
    double alpha = stkMeshStruct->felixAlpha; 
    cout << "L: " << L << endl; 
    cout << "alpha degrees: " << alpha << endl; 
    alpha = alpha*pi/180; //convert alpha, read in from ParameterList, to radians
    cout << "alpha radians: " << alpha << endl;
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L; 
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = -x[0]*tan(alpha);
      double b = s - 1.0 + 0.5*sin(2*pi/L*x[0])*sin(2*pi/L*x[1]);
      x[2] = s*x[2] + b*(1-x[2]);
      localSHeight[i] = s;
     }
   }
  else if (transformType == "ISMIP-HOM Test B") {
    cout << "Test B!" << endl;
    double L = stkMeshStruct->felixL; 
    double alpha = stkMeshStruct->felixAlpha; 
    cout << "L: " << L << endl; 
    cout << "alpha degrees: " << alpha << endl; 
    alpha = alpha*pi/180; //convert alpha, read in from ParameterList, to radians
    cout << "alpha radians: " << alpha << endl;
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L; 
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = -x[0]*tan(alpha);
      double b = s - 1.0 + 0.5*sin(2*pi/L*x[0]);
      x[2] = s*x[2] + b*(1-x[2]);
      localSHeight[i] = s;
     }
   }
   else if ((transformType == "ISMIP-HOM Test C") || (transformType == "ISMIP-HOM Test D")) {
    cout << "Test C and D!" << endl;
    double L = stkMeshStruct->felixL; 
    double alpha = stkMeshStruct->felixAlpha; 
    cout << "L: " << L << endl; 
    cout << "alpha degrees: " << alpha << endl; 
    alpha = alpha*pi/180; //convert alpha, read in from ParameterList, to radians
    cout << "alpha radians: " << alpha << endl;
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L; 
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = -x[0]*tan(alpha);
      double b = s - 1.0;
      x[2] = s*x[2] + b*(1-x[2]);
      localSHeight[i] = s;
     }
   }
   else if (transformType == "Dome") { 
    cout << "Dome transform!" << endl; 
    double L = 0.7071*30; 
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L; 
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
      x[0] = L*x[0]; 
      x[1] = L*x[1]; 
      double s = 0.7071*sqrt(450.0 - x[0]*x[0] - x[1]*x[1])/sqrt(450.0);
      x[2] = s*x[2];
      localSHeight[i] = s;
    }
  }
   else if (transformType == "Confined Shelf") { 
    cout << "Confined shelf transform!" << endl; 
    double L = stkMeshStruct->felixL; 
    cout << "L: " << L << endl; 
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L; 
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
      x[0] = L*x[0]; 
      x[1] = L*x[1]; 
      double s = 0.06; //top surface is at z=0.06km=60m
      double b = -0.440; //basal surface is at z=-0.440km=-440m
      x[2] = s*x[2] + b*(1.0-x[2]);
      localSHeight[i] = s;
    }
  }
  else if (transformType == "Circular Shelf") { 
    cout << "Circular shelf transform!" << endl; 
    double L = stkMeshStruct->felixL; 
    cout << "L: " << L << endl; 
    double rhoIce = 910.0; //ice density, in kg/m^3
    double rhoOcean = 1028.0; //ocean density, in kg/m^3
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L; 
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
      x[0] = L*x[0]; 
      x[1] = L*x[1]; 
      double s = 1.0-rhoIce/rhoOcean; //top surface is at z=(1-rhoIce/rhoOcean) km
      double b = s - 1.0; //basal surface is at z=s-1 km
      x[2] = s*x[2] + b*(1.0-x[2]);
      localSHeight[i] = s;
    }
  }
}

void
Albany::STKDiscretization::getOwned_xyz(double** x, double** y, double** z,
                                        double **rbm, int& nNodes, int numPDEs, int numScalar,  int nullSpaceDim)
{
  // Function to return x,y,z at owned nodes as double*, specifically for ML
  int numDim = stkMeshStruct->numDim;
  nNodes = numOwnedNodes;

  if (allocated_xyz) { delete [] xx; delete [] yy; delete [] zz;} 
  xx = new double[numOwnedNodes];
  yy = new double[numOwnedNodes];
  zz = new double[numOwnedNodes];
  if (nullSpaceDim>0) rr = new double[(nullSpaceDim + numScalar)*numPDEs*nNodes];
  else                rr = new double[1]; // Just so there is something to delete in destructor
  allocated_xyz = true;

  for (int i=0; i < numOwnedNodes; i++)  {
    int node_gid = gid(ownednodes[i]);
    int node_lid = node_mapT->getLocalElement(node_gid);

    double* X = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *overlapnodes[i]);
    if (numDim > 0) xx[node_lid] = X[0];
    if (numDim > 1) yy[node_lid] = X[1];
    if (numDim > 2) zz[node_lid] = X[2];
  }

  // Leave unused dim as null pointers.
  if (numDim > 0) *x = xx;
  if (numDim > 1) *y = yy;
  if (numDim > 2) *z = zz;
  *rbm = rr;
}


const Teuchos::ArrayRCP<std::string>& 
Albany::STKDiscretization::getWsEBNames() const
{
  return wsEBNames;
}

const Teuchos::ArrayRCP<int>& 
Albany::STKDiscretization::getWsPhysIndex() const
{
  return wsPhysIndex;
}

void Albany::STKDiscretization::outputToExodus(const Epetra_Vector& soln, const double time, const bool overlapped)
{
  // Put solution as Epetra_Vector into STK Mesh
  if(!overlapped)
    setSolutionField(soln);

  // soln coming in is overlapped
  else
    setOvlpSolutionField(soln);


#ifdef ALBANY_SEACAS

  if (stkMeshStruct->exoOutput) {

    // Skip this write unless the proper interval has been reached
    if(outputInterval++ % stkMeshStruct->exoOutputInterval)

      return;

    double time_label = monotonicTimeLabel(time);

    int out_step = stk::io::process_output_request(*mesh_data, bulkData, time_label);

    if (mapT->getComm()->getRank()==0) {
      *out << "Albany::STKDiscretization::outputToExodus: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " <<out_step<<" in file "<<stkMeshStruct->exoOutFile<< endl;
    }
  }
#endif
}

double
Albany::STKDiscretization::monotonicTimeLabel(const double time) 
{
  // If increasing, then all is good
  if (time > previous_time_label) {
    previous_time_label = time;
    return time;
  }
// Try absolute value
  double time_label = fabs(time);
  if (time_label > previous_time_label) {
    previous_time_label = time_label;
    return time_label;
  }

  // Try adding 1.0 to time
  if (time_label+1.0 > previous_time_label) {
    previous_time_label = time_label+1.0;
    return time_label+1.0;
  }

  // Otherwise, just add 1.0 to previous
  previous_time_label += 1.0;
  return previous_time_label;
}

void 
Albany::STKDiscretization::setResidualField(const Epetra_Vector& residual) 
{
#ifdef ALBANY_LCM
  // Copy residual vector into residual field, one node at a time
  for (std::size_t i=0; i < ownednodes.size(); i++)  
  {
    double* res = stk::mesh::field_data(*stkMeshStruct->residual_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++)
      res[j] = residual[getOwnedDOF(i,j)];
  }
#endif
}

void
Albany::STKDiscretization::setResidualFieldT(const Tpetra_Vector& residualT)
{
#ifdef ALBANY_LCM
  Teuchos::Array<ST> valuesArray((residualT.getMap())->getGlobalNumElements());
  residualT.get1dCopy(valuesArray);
  // Copy residual vector into residual field, one node at a time
  for (std::size_t i=0; i < ownednodes.size(); i++)
  {
    double* res = stk::mesh::field_data(*stkMeshStruct->residual_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++){
      res[j] = valuesArray[getOwnedDOF(i,j)];
    }
  }
#endif
}


Teuchos::RCP<Epetra_Vector>
Albany::STKDiscretization::getSolutionField() const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::ArrayView<const int> indicesAV = mapT->getNodeElementList();
  int numElements = mapT->getNodeNumElements();
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(-1, numElements, indicesAV.getRawPtr(), 0, *comm));
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(new Epetra_Vector(*map));
  this->getSolutionField(*soln);
  return soln;
}

Teuchos::RCP<Epetra_MultiVector>
Albany::STKDiscretization::getSolutionFieldHistory() const
{
  const int stepCount = stkMeshStruct->solutionFieldHistoryDepth;
  return this->getSolutionFieldHistoryImpl(stepCount);
}

Teuchos::RCP<Epetra_MultiVector>
Albany::STKDiscretization::getSolutionFieldHistory(int maxStepCount) const
{
  const int stepCount = std::min(stkMeshStruct->solutionFieldHistoryDepth, maxStepCount);
  return this->getSolutionFieldHistoryImpl(stepCount);
}

Teuchos::RCP<Epetra_MultiVector>
Albany::STKDiscretization::getSolutionFieldHistoryImpl(int stepCount) const
{
  const int vectorCount = stepCount > 0 ? stepCount : 1; // A valid MultiVector has at least one vector
  Teuchos::ArrayView<const int> indicesAV = mapT->getNodeElementList();
  int numElements = mapT->getNodeNumElements();
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(-1, numElements, indicesAV.getRawPtr(), 0, *comm));
  const Teuchos::RCP<Epetra_MultiVector> result = Teuchos::rcp(new Epetra_MultiVector(*map, vectorCount));
  for (int i = 0; i < stepCount; ++i) {
    stkMeshStruct->loadSolutionFieldHistory(i);
    Epetra_Vector v(View, *result, i);
    this->getSolutionField(v);
  }
  return result;
}

void
Albany::STKDiscretization::getSolutionField(Epetra_Vector &result) const
{
  for (std::size_t i=0; i < ownednodes.size(); i++)  {
    const double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++) {
      result[getOwnedDOF(i,j)] = sol[j];
    }
  }
}

Teuchos::RCP<Tpetra_Vector>
Albany::STKDiscretization::getSolutionFieldT() const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Tpetra_Vector> solnT = Teuchos::rcp(new Tpetra_Vector(mapT));
  for (std::size_t i=0; i < ownednodes.size(); i++)  {
    const double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++){
      solnT->replaceLocalValue(getOwnedDOF(i,j), sol[j]);
    }
   }
  return solnT;
}



/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

void 
Albany::STKDiscretization::setSolutionField(const Epetra_Vector& soln) 
{
  // Copy soln vector into solution field, one node at a time
  // Note that soln coming in is the local (non overlapped) soln
  for (std::size_t i=0; i < ownednodes.size(); i++)  {
    double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *ownednodes[i]);
    for (std::size_t j=0; j<neq; j++)
      sol[j] = soln[getOwnedDOF(i,j)];
  }
}

void 
Albany::STKDiscretization::setOvlpSolutionField(const Epetra_Vector& soln) 
{
  // Copy soln vector into solution field, one node at a time
  // Note that soln coming in is the local+ghost (overlapped) soln
  for (std::size_t i=0; i < overlapnodes.size(); i++)  {
    double* sol = stk::mesh::field_data(*stkMeshStruct->solution_field, *overlapnodes[i]);
    for (std::size_t j=0; j<neq; j++)
      sol[j] = soln[getOwnedDOF(i,j)];
  }
}

inline int Albany::STKDiscretization::gid(const stk::mesh::Entity& node) const
{ return node.identifier()-1; }

inline int Albany::STKDiscretization::gid(const stk::mesh::Entity* node) const
{ return gid(*node); }

inline int Albany::STKDiscretization::getOwnedDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numOwnedNodes*eq;
}

inline int Albany::STKDiscretization::getOverlapDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numOverlapNodes*eq;
}

inline int Albany::STKDiscretization::getGlobalDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numGlobalNodes*eq;
}

int Albany::STKDiscretization::nonzeroesPerRow(const int neq) const
{
  int numDim = stkMeshStruct->numDim;
  int estNonzeroesPerRow;
  switch (numDim) {
  case 0: estNonzeroesPerRow=1*neq; break;
  case 1: estNonzeroesPerRow=3*neq; break;
  case 2: estNonzeroesPerRow=9*neq; break;
  case 3: estNonzeroesPerRow=27*neq; break;
  default: TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
			      "STKDiscretization:  Bad numDim"<< numDim);
  }
  return estNonzeroesPerRow;
}

void Albany::STKDiscretization::computeOwnedNodesAndUnknowns()
{
  // Loads member data:  ownednodes, numOwnedNodes, node_map, numGlobalNodes, map
  // maps for owned nodes and unknowns
  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
				    bulkData.buckets( metaData.node_rank() ) ,
				    ownednodes );

  numOwnedNodes = ownednodes.size();
  Teuchos::Array<int> indicesT(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; i++) indicesT[i] = gid(ownednodes[i]);
  // node_mapT = Teuchos::rcp(new Map(-1, indicesT(), 0, commT));
  node_mapT = Teuchos::null; // delete existing map happens here on remesh
  
  node_mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);


  numGlobalNodes = node_mapT->getMaxAllGlobalIndex() + 1;
  indicesT.resize(numOwnedNodes * neq);
  
  for (int i=0; i < numOwnedNodes; i++)
    for (std::size_t j=0; j < neq; j++)
      indicesT[getOwnedDOF(i,j)] = getGlobalDOF(gid(ownednodes[i]),j);
  
 //mapT = Teuchos::rcp(new Map(-1, indicesT(), 0, commT));
  mapT = Teuchos::null; // delete existing map happens here on remesh
 
  mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);
}

void Albany::STKDiscretization::computeOverlapNodesAndUnknowns()
{
  // Loads member data:  overlapodes, numOverlapodes, overlap_node_map, coordinates
  std::vector<int> indices;
  // maps for overlap unknowns
  stk::mesh::Selector select_overlap_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    ( stk::mesh::Selector( metaData.locally_owned_part() )
      | stk::mesh::Selector( metaData.globally_shared_part() ) );

  //  overlapnodes used for overlap map -- stored for changing coords
  stk::mesh::get_selected_entities( select_overlap_in_part ,
				    bulkData.buckets( metaData.node_rank() ) ,
				    overlapnodes );

  numOverlapNodes = overlapnodes.size();

  Teuchos::Array<int> indicesT(numOverlapNodes * neq);
  for (int i=0; i < numOverlapNodes; i++)
    for (std::size_t j=0; j < neq; j++)
      indicesT[getOverlapDOF(i,j)] = getGlobalDOF(gid(overlapnodes[i]),j);

  overlap_mapT = Teuchos::null; // delete existing map happens here on remesh
  
  overlap_mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);

  indicesT.resize(numOverlapNodes);
  for (int i=0; i < numOverlapNodes; i++)
    indicesT[i] = gid(overlapnodes[i]);

  overlap_node_mapT = Teuchos::null; // delete existing map happens here on remesh
  
  overlap_node_mapT = Tpetra::createNonContigMapWithNode<LO, GO, KokkosNode> (indicesT(), commT, nodeT);

  coordinates.resize(3*numOverlapNodes);
  localSHeight.resize(numOverlapNodes);
 
}


void Albany::STKDiscretization::computeGraphs()
{

  std::map<int, stk::mesh::Part*>::iterator pv = stkMeshStruct->partVec.begin();
  int nodes_per_element =  metaData.get_cell_topology(*(pv->second)).getNodeCount(); 
// int nodes_per_element_est =  metaData.get_cell_topology(*(stkMeshStruct->partVec[0])).getNodeCount();

  // Loads member data:  overlap_graph, numOverlapodes, overlap_node_map, coordinates, graphs

  overlap_graphT = Teuchos::null; // delete existing graph happens here on remesh
  
  overlap_graphT = Teuchos::rcp(new Tpetra_CrsGraph(overlap_mapT, neq*nodes_per_element));

  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
				    bulkData.buckets( metaData.element_rank() ) ,
				    cells );


  if (comm->MyPID()==0)
    *out << "STKDisc: " << cells.size() << " elements on Proc 0 " << endl;

  int row, col;
  Teuchos::ArrayView<int> colAV; 

  for (std::size_t i=0; i < cells.size(); i++) {
    stk::mesh::Entity& e = *cells[i];
    stk::mesh::PairIterRelation rel = e.relations(metaData.NODE_RANK);

    // loop over local nodes
    for (std::size_t j=0; j < rel.size(); j++) {
      stk::mesh::Entity& rowNode = * rel[j].entity();

      // loop over eqs
      for (std::size_t k=0; k < neq; k++) {
        row = getGlobalDOF(gid(rowNode), k);
        for (std::size_t l=0; l < rel.size(); l++) {
          stk::mesh::Entity& colNode = * rel[l].entity();
          for (std::size_t m=0; m < neq; m++) {
            col = getGlobalDOF(gid(colNode), m);
            colAV = Teuchos::arrayView(&col, 1);
            overlap_graphT->insertGlobalIndices(row, colAV);
          }
        }
      }
    }
  }
  overlap_graphT->fillComplete();

  // Create Owned graph by exporting overlap with known row map
  graphT = Teuchos::null; // delete existing graph happens here on remesh
  
  graphT = Teuchos::rcp(new Tpetra_CrsGraph(mapT, nonzeroesPerRow(neq)));
    
  // Create non-overlapped matrix using two maps and export object
  Teuchos::RCP<Tpetra_Export> exporterT = Teuchos::rcp(new Tpetra_Export(overlap_mapT, mapT));
  graphT->doExport(*overlap_graphT, *exporterT, Tpetra::INSERT);
  graphT->fillComplete();
}

void Albany::STKDiscretization::computeWorksetInfo()
{

  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  std::vector< stk::mesh::Bucket * > buckets ;
  stk::mesh::get_buckets( select_owned_in_part ,
                          bulkData.buckets( metaData.element_rank() ) ,
                          buckets);

  int numBuckets =  buckets.size();

  wsEBNames.resize(numBuckets);
  for (int i=0; i<numBuckets; i++) {
    std::vector< stk::mesh::Part * >  bpv;
    buckets[i]->supersets(bpv);
    for (std::size_t j=0; j<bpv.size(); j++) {
      if (bpv[j]->primary_entity_rank() == metaData.element_rank()) {
        if (bpv[j]->name()[0] != '{') {
	  // *out << "Bucket " << i << " is in Element Block:  " << bpv[j]->name() 
	  //      << "  and has " << buckets[i]->size() << " elements." << endl;
          wsEBNames[i]=bpv[j]->name();
        }
      }
    }
  }

  wsPhysIndex.resize(numBuckets);
  if (stkMeshStruct->allElementBlocksHaveSamePhysics)
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=0;
  else 
    for (int i=0; i<numBuckets; i++) wsPhysIndex[i]=stkMeshStruct->ebNameToIndex[wsEBNames[i]];

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID

  wsElNodeEqID.resize(numBuckets);
  coords.resize(numBuckets);
  sHeight.resize(numBuckets);

  // Clear map if remeshing
  if(!elemGIDws.empty()) elemGIDws.clear();

  for (int b=0; b < numBuckets; b++) {

    stk::mesh::Bucket& buck = *buckets[b];
    wsElNodeEqID[b].resize(buck.size());
    coords[b].resize(buck.size());
    sHeight[b].resize(buck.size());

    // i is the element index within bucket b

    for (std::size_t i=0; i < buck.size(); i++) {
  
      // Traverse all the elements in this bucket
      stk::mesh::Entity& element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[gid(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[gid(element)].LID = i;

      stk::mesh::PairIterRelation rel = element.relations(metaData.NODE_RANK);

      int nodes_per_element = rel.size();
      wsElNodeEqID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);
      sHeight[b][i].resize(nodes_per_element);
      // loop over local nodes
      for (int j=0; j < nodes_per_element; j++) {
        stk::mesh::Entity& rowNode = * rel[j].entity();
        int node_gid = gid(rowNode);
        int node_lid = overlap_node_mapT->getLocalElement(node_gid);
        
        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
			   "STK1D_Disc: node_lid out of range " << node_lid << endl);
        coords[b][i][j] = stk::mesh::field_data(*stkMeshStruct->coordinates_field, rowNode);
        sHeight[b][i][j] = localSHeight[node_lid];
        
        wsElNodeEqID[b][i][j].resize(neq);
        for (std::size_t eq=0; eq < neq; eq++) 
          wsElNodeEqID[b][i][j][eq] = getOverlapDOF(node_lid,eq);
      }
    }
  }

  for (int d=0; d<stkMeshStruct->numDim; d++) {
  if (stkMeshStruct->PBCStruct.periodic[d]) {
    for (int b=0; b < numBuckets; b++) {
      for (std::size_t i=0; i < buckets[b]->size(); i++) {
        int nodes_per_element = (*buckets[b])[i].relations(metaData.NODE_RANK).size();
        bool anyXeqZero=false;
        for (int j=0; j < nodes_per_element; j++)  if (coords[b][i][j][d]==0.0) anyXeqZero=true;
        if (anyXeqZero)  {
          bool flipZeroToScale=false;
          for (int j=0; j < nodes_per_element; j++) 
              if (coords[b][i][j][d] > stkMeshStruct->PBCStruct.scale[d]/1.9) flipZeroToScale=true;
          if (flipZeroToScale) {  
            for (int j=0; j < nodes_per_element; j++)  {
              if (coords[b][i][j][d] == 0.0) {
                double* xleak = new double [stkMeshStruct->numDim];
                for (int k=0; k < stkMeshStruct->numDim; k++) 
                  if (k==d) xleak[d]=stkMeshStruct->PBCStruct.scale[d];
                  else xleak[k] = coords[b][i][j][k];
                std::string transformType = stkMeshStruct->transformType;
                double alpha = stkMeshStruct->felixAlpha;
                alpha *= pi/180.; //convert alpha, read in from ParameterList, to radians
                if ((transformType=="ISMIP-HOM Test A" || transformType == "ISMIP-HOM Test B" || 
                     transformType=="ISMIP-HOM Test C" || transformType == "ISMIP-HOM Test D") && d==0) { 
                    xleak[2] -= stkMeshStruct->PBCStruct.scale[d]*tan(alpha);
                	sHeight[b][i][j] -= stkMeshStruct->PBCStruct.scale[d]*tan(alpha);
                }
                coords[b][i][j] = xleak; // replace ptr to coords
                toDelete.push_back(xleak);
              }
            }          
          }
        }
      }
    }
  }
  }

  // Pull out pointers to shards::Arrays for every bucket, for every state
  // Code is data-type dependent
  stateArrays.resize(numBuckets);
  for (std::size_t b=0; b < buckets.size(); b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    for (std::size_t i=0; i<stkMeshStruct->qpscalar_states.size(); i++) {
      stk::mesh::BucketArray<Albany::AbstractSTKMeshStruct::QPScalarFieldType> array(*stkMeshStruct->qpscalar_states[i], buck);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->qpscalar_states[i]->name()] = ar;
    }
    for (std::size_t i=0; i<stkMeshStruct->qpvector_states.size(); i++) {
      stk::mesh::BucketArray<Albany::AbstractSTKMeshStruct::QPVectorFieldType> array(*stkMeshStruct->qpvector_states[i], buck);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->qpvector_states[i]->name()] = ar;
    }
    for (std::size_t i=0; i<stkMeshStruct->qptensor_states.size(); i++) {
      stk::mesh::BucketArray<Albany::AbstractSTKMeshStruct::QPTensorFieldType> array(*stkMeshStruct->qptensor_states[i], buck);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->qptensor_states[i]->name()] = ar;
    }    
    for (std::size_t i=0; i<stkMeshStruct->scalarValue_states.size(); i++) {      
      const int size = 1;
      shards::Array<double, shards::NaturalOrder, Cell> array(&(stkMeshStruct->time), size);
      MDArray ar = array;
      stateArrays[b][stkMeshStruct->scalarValue_states[i]] = ar;
    }
  }
}

void Albany::STKDiscretization::computeSideSets(){

  // Clean up existing sideset structure if remeshing

  for(int i = 0; i < sideSets.size(); i++)
    sideSets[i].clear(); // empty the ith map

  const stk::mesh::EntityRank element_rank = metaData.element_rank();

  // iterator over all side_rank parts found in the mesh
  std::map<std::string, stk::mesh::Part*>::iterator ss = stkMeshStruct->ssPartVec.begin();

  int numBuckets = wsEBNames.size();

  sideSets.resize(numBuckets); // Need a sideset list per workset

  while ( ss != stkMeshStruct->ssPartVec.end() ) { 

    // Get all owned sides in this side set
    stk::mesh::Selector select_owned_in_sspart =

      // get only entities in the ss part (ss->second is the current sideset part)
      stk::mesh::Selector( *(ss->second) ) &
      // and only if the part is local
      stk::mesh::Selector( metaData.locally_owned_part() );

    std::vector< stk::mesh::Entity * > sides ;
    stk::mesh::get_selected_entities( select_owned_in_sspart , // sides local to this processor
				      bulkData.buckets( metaData.side_rank() ) ,
				      sides ); // store the result in "sides"

    *out << "STKDisc: sideset "<< ss->first <<" has size " << sides.size() << "  on Proc 0." << endl;

    // loop over the sides to see what they are, then fill in the data holder
    // for side set options, look at $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp

    for (std::size_t localSideID=0; localSideID < sides.size(); localSideID++) {

      stk::mesh::Entity &sidee = *sides[localSideID];

      const stk::mesh::PairIterRelation side_elems = sidee.relations(element_rank); // get the elements
            // containing the side. Note that if the side is internal, it will show up twice in the
            // element list, once for each element that contains it.

      TEUCHOS_TEST_FOR_EXCEPTION(side_elems.size() != 1, std::logic_error,
			   "STKDisc: cannot figure out side set topology for side set " << ss->first << endl);

      const stk::mesh::Entity & elem = *side_elems[0].entity();

      SideStruct sStruct;

      // Save elem id. This is the global element id
      sStruct.elem_GID = gid(elem);

      int workset = elemGIDws[sStruct.elem_GID].ws; // Get the ws that this element lives in

      // Save elem id. This is the local element id within the workset
      sStruct.elem_LID = elemGIDws[sStruct.elem_GID].LID;

      // Save the side identifier inside of the element. This starts at zero here.
      sStruct.side_local_id = determine_local_side_id(elem, sidee);

      // Save the index of the element block that this elem lives in
      sStruct.elem_ebIndex = stkMeshStruct->ebNameToIndex[wsEBNames[workset]];

      SideSetList& ssList = sideSets[workset];   // Get a ref to the side set map for this ws
      SideSetList::iterator it = ssList.find(ss->first); // Get an iterator to the correct sideset (if
                                                                // it exists)

      if(it != ssList.end()) // The sideset has already been created

        it->second.push_back(sStruct); // Save this side to the vector that belongs to the name ss->first

      else { // Add the key ss->first to the map, and the side vector to that map

        std::vector<SideStruct> tmpSSVec;
        tmpSSVec.push_back(sStruct);
        
        ssList.insert(SideSetList::value_type(ss->first, tmpSSVec));

      }

    }

    ss++;
  }
}

unsigned 
Albany::STKDiscretization::determine_local_side_id( const stk::mesh::Entity & elem , stk::mesh::Entity & side ) {

  using namespace stk;

  const CellTopologyData * const elem_top = mesh::fem::get_cell_topology( elem ).getCellTopologyData();

  const mesh::PairIterRelation elem_nodes = elem.relations( mesh::fem::FEMMetaData::NODE_RANK );
  const mesh::PairIterRelation side_nodes = side.relations( mesh::fem::FEMMetaData::NODE_RANK );

  int side_id = -1 ;

  if(elem_nodes.size() == 0 || side_nodes.size() == 0){ // Node relations are not present, look at elem->face

    int elem_rank = elem.entity_rank();
    const mesh::PairIterRelation elem_sides = elem.relations( elem_rank - 1);

    for ( unsigned i = 0 ; i < elem_sides.size() ; ++i ) {

      const stk::mesh::Entity & elem_side = *elem_sides[i].entity();

      if(elem_side.identifier() == side.identifier()){ // Found the local side in the element

         side_id = static_cast<int>(i);

         return side_id;

      }

    }

    if ( side_id < 0 ) {
      std::ostringstream msg ;
      msg << "determine_local_side_id( " ;
      msg << elem_top->name ;
      msg << " , Element[ " ;
      msg << elem.identifier();
      msg << " ]{" ;
      for ( unsigned i = 0 ; i < elem_sides.size() ; ++i ) {
        msg << " " << elem_sides[i].entity()->identifier();
      }
      msg << " } , Side[ " ;
      msg << side.identifier();
      msg << " ] ) FAILED" ;
      throw std::runtime_error( msg.str() );
    }

  }
  else { // Conventional elem->node - side->node connectivity present

    for ( unsigned i = 0 ; side_id == -1 && i < elem_top->side_count ; ++i ) {
      const CellTopologyData & side_top = * elem_top->side[i].topology ;
      const unsigned     * side_map =   elem_top->side[i].node ;
  
      if ( side_nodes.size() == side_top.node_count ) {
  
        side_id = i ;
  
        for ( unsigned j = 0 ;
              side_id == static_cast<int>(i) && j < side_top.node_count ; ++j ) {
  
          mesh::Entity * const elem_node = elem_nodes[ side_map[j] ].entity();
  
          bool found = false ;
  
          for ( unsigned k = 0 ; ! found && k < side_top.node_count ; ++k ) {
            found = elem_node == side_nodes[k].entity();
          }
  
          if ( ! found ) { side_id = -1 ; }
        }
      }
    }
  
    if ( side_id < 0 ) {
      std::ostringstream msg ;
      msg << "determine_local_side_id( " ;
      msg << elem_top->name ;
      msg << " , Element[ " ;
      msg << elem.identifier();
      msg << " ]{" ;
      for ( unsigned i = 0 ; i < elem_nodes.size() ; ++i ) {
        msg << " " << elem_nodes[i].entity()->identifier();
      }
      msg << " } , Side[ " ;
      msg << side.identifier();
      msg << " ]{" ;
      for ( unsigned i = 0 ; i < side_nodes.size() ; ++i ) {
        msg << " " << side_nodes[i].entity()->identifier();
      }
      msg << " } ) FAILED" ;
      throw std::runtime_error( msg.str() );
    }
  }

  return static_cast<unsigned>(side_id) ;
}

void Albany::STKDiscretization::computeNodeSets()
{

  std::map<std::string, stk::mesh::Part*>::iterator ns = stkMeshStruct->nsPartVec.begin();

  while ( ns != stkMeshStruct->nsPartVec.end() ) { // Iterate over Node Sets
    // Get all owned nodes in this node set
    stk::mesh::Selector select_owned_in_nspart =
      stk::mesh::Selector( *(ns->second) ) &
      stk::mesh::Selector( metaData.locally_owned_part() );

    std::vector< stk::mesh::Entity * > nodes ;
    stk::mesh::get_selected_entities( select_owned_in_nspart ,
				      bulkData.buckets( metaData.node_rank() ) ,
				      nodes );

    nodeSets[ns->first].resize(nodes.size());
    nodeSetCoords[ns->first].resize(nodes.size());
//    nodeSetIDs.push_back(ns->first); // Grab string ID
    cout << "STKDisc: nodeset "<< ns->first <<" has size " << nodes.size() << "  on Proc 0." << endl;
    for (std::size_t i=0; i < nodes.size(); i++) {
      int node_gid = gid(nodes[i]);
      int node_lid = node_mapT->getLocalElement(node_gid);
      nodeSets[ns->first][i].resize(neq);
      for (std::size_t eq=0; eq < neq; eq++)  nodeSets[ns->first][i][eq] = getOwnedDOF(node_lid,eq);
      nodeSetCoords[ns->first][i] = stk::mesh::field_data(*stkMeshStruct->coordinates_field, *nodes[i]);
    }
    ns++;
  }
}

void Albany::STKDiscretization::setupExodusOutput()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) {

    outputInterval = 0;

    std::string str = stkMeshStruct->exoOutFile;

    Ioss::Init::Initializer io;
    mesh_data = new stk::io::MeshData();
    stk::io::create_output_mesh(str,
		  Albany::getMpiCommFromEpetraComm(*comm),
		  bulkData, *mesh_data);

    stk::io::define_output_fields(*mesh_data, metaData);

  }
#else
  if (stkMeshStruct->exoOutput) 
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n" << endl;
  
#endif
}

void Albany::STKDiscretization::reNameExodusOutput(std::string& filename)
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput && mesh_data != NULL) {

   // Delete the mesh data object and recreate it
   delete mesh_data;

   stkMeshStruct->exoOutFile = filename;

   // reset reference value for monotonic time function call as we are writing to a new file
   previous_time_label = -1.0e32;

  }
#else
  if (stkMeshStruct->exoOutput) 
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n" << endl;
  
#endif
}

void
Albany::STKDiscretization::updateMesh()
{

  computeOwnedNodesAndUnknowns();

  computeOverlapNodesAndUnknowns();

  transformMesh(); 

  computeGraphs();

  computeWorksetInfo();

  computeNodeSets();

  computeSideSets();

  setupExodusOutput();

}
