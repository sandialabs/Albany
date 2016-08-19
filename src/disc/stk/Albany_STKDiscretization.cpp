//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include "Albany_Utils.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_NodalGraphUtils.hpp"
#include "Albany_STKNodeFieldContainer.hpp"
#include "Albany_BucketArray.hpp"

#include <string>
#include <iostream>
#include <fstream>

#include <Shards_BasicTopologies.hpp>

#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_Basis.hpp>
#include <Intrepid2_HGRAD_QUAD_Cn_FEM.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/Selector.hpp>

#include <PHAL_Dimension.hpp>

#include <stk_mesh/base/FEMHelpers.hpp>

#ifdef ALBANY_SEACAS
#include <Ionit_Initializer.h>
#include <netcdf.h>

#ifdef ALBANY_PAR_NETCDF
extern "C" {
#include <netcdf_par.h>
}
#endif
#endif

#include <algorithm>
#if defined(ALBANY_EPETRA)
#include "Epetra_Export.h"
#include "EpetraExt_MultiVectorOut.h"
#include "Petra_Converters.hpp"
#endif

#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
#include "PeridigmManager.hpp"
#endif
#endif

const double pi = 3.1415926535897932385;

const Tpetra::global_size_t INVALID = Teuchos::OrdinalTraits<Tpetra::global_size_t>::invalid ();

// Uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

Albany::STKDiscretization::
STKDiscretization(Teuchos::RCP<Albany::AbstractSTKMeshStruct> stkMeshStruct_,
                  const Teuchos::RCP<const Teuchos_Comm>& commT_,
                  const Teuchos::RCP<Albany::RigidBodyModes>& rigidBodyModes_,
                  const std::map<int,std::vector<std::string> >& sideSetEquations_) :

  out(Teuchos::VerboseObjectBase::getDefaultOStream()),
  previous_time_label(-1.0e32),
  metaData(*stkMeshStruct_->metaData),
  bulkData(*stkMeshStruct_->bulkData),
  commT(commT_),
  rigidBodyModes(rigidBodyModes_),
  neq(stkMeshStruct_->neq),
  stkMeshStruct(stkMeshStruct_),
  sideSetEquations(sideSetEquations_),
  interleavedOrdering(stkMeshStruct_->interleavedOrdering)
{
#if defined(ALBANY_EPETRA)
  comm = Albany::createEpetraCommFromTeuchosComm(commT_);
#endif
//  Albany::STKDiscretization::updateMesh();  //Mauro: cannot call virtual function in constructor

}

Albany::STKDiscretization::~STKDiscretization()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->cdfOutput)
      if (netCDFp)
    if (const int ierr = nc_close (netCDFp))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "close returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
#endif

  for (int i=0; i< toDelete.size(); i++) delete [] toDelete[i];
}



void
Albany::STKDiscretization::printConnectivity() const
{
  commT->barrier();
  for (int rank = 0; rank < commT->getSize(); ++rank)
  {
    commT->barrier();
    if (rank == commT->getRank())
    {
      std::cout << std::endl << "Process rank " << rank << std::endl;
      for (size_t ibuck = 0; ibuck < wsElNodeID.size(); ++ibuck)
      {
        std::cout << "  Bucket " << ibuck << std::endl;
        for (size_t ielem = 0; ielem < wsElNodeID[ibuck].size(); ++ielem)
        {
          int numNodes = wsElNodeID[ibuck][ielem].size();
          std::cout << "    Element " << ielem << ": Nodes = ";
          for (size_t inode = 0; inode < numNodes; ++inode)
            std::cout << wsElNodeID[ibuck][ielem][inode] << " ";
          std::cout << std::endl;
        }
      }
    }
    commT->barrier();
  }
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getMap() const
{
  return map;
}
#endif

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getMapT() const
{
  return mapT;
}


#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getOverlapMap() const
{
  return overlap_map;
}
#endif

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getOverlapMapT() const
{
  return overlap_mapT;
}

#if defined(ALBANY_EPETRA)

Teuchos::RCP<const Epetra_CrsGraph>
Albany::STKDiscretization::getJacobianGraph() const
{
  Teuchos::RCP<const Epetra_CrsGraph> graph= Petra::TpetraCrsGraph_To_EpetraCrsGraph(graphT, comm);
  return graph;
}
#endif


Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getMapT(const std::string& field_name) const {
  return nodalDOFsStructContainer.getDOFsStruct(field_name).map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getNodeMapT(const std::string& field_name) const {
  return nodalDOFsStructContainer.getDOFsStruct(field_name).node_map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getOverlapMapT(const std::string& field_name) const {
  return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_map;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getOverlapNodeMapT(const std::string& field_name) const {
  return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_node_map;
}

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::STKDiscretization::getJacobianGraphT() const
{
  return graphT;
}

#ifdef ALBANY_AERAS
Teuchos::RCP<const Tpetra_CrsGraph>
Albany::STKDiscretization::getImplicitJacobianGraphT() const
{
  return graphT;
}
#endif

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_CrsGraph>
Albany::STKDiscretization::getOverlapJacobianGraph() const
{
  Teuchos::RCP<const Epetra_CrsGraph> overlap_graph= Petra::TpetraCrsGraph_To_EpetraCrsGraph(overlap_graphT, comm);
  return overlap_graph;
}
#endif

Teuchos::RCP<const Tpetra_CrsGraph>
Albany::STKDiscretization::getOverlapJacobianGraphT() const
{
  return overlap_graphT;
}

#ifdef ALBANY_AERAS
Teuchos::RCP<const Tpetra_CrsGraph>
Albany::STKDiscretization::getImplicitOverlapJacobianGraphT() const
{
  return overlap_graphT;
}
#endif

#if defined(ALBANY_EPETRA)
Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getNodeMap() const
{
  return node_map;
}

Teuchos::RCP<const Epetra_Map>
Albany::STKDiscretization::getOverlapNodeMap() const
{
  return overlap_node_map;
}
#endif

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getNodeMapT() const
{
  return node_mapT;
}

Teuchos::RCP<const Tpetra_Map>
Albany::STKDiscretization::getOverlapNodeMapT() const
{
  return overlap_node_mapT;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<Teuchos::ArrayRCP<LO> > > >::type&
Albany::STKDiscretization::getWsElNodeEqID() const
{
  return wsElNodeEqID;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> > >::type&
Albany::STKDiscretization::getWsElNodeID() const
{
  return wsElNodeID;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type&
Albany::STKDiscretization::getCoords() const
{
  return coords;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<double> >::type&
Albany::STKDiscretization::getSphereVolume() const
{
  return sphereVolume;
}

const Albany::WorksetArray<Teuchos::ArrayRCP<double*> >::type&
Albany::STKDiscretization::getLatticeOrientation() const
{
  return latticeOrientation;
}

void
Albany::STKDiscretization::printCoords() const
{
  std::cout << "Processor " << bulkData.parallel_rank() << " has "
            << coords.size() << " worksets." << std::endl;
  for (int ws=0; ws<coords.size(); ws++)
  {
    for (int e=0; e<coords[ws].size(); e++)
    {
      for (int j=0; j<coords[ws][e].size(); j++)
      {
        std::cout << "Coord for workset: " << ws << " element: " << e
                  << " node: " << j << " x, y, z: "
                  << coords[ws][e][j][0] << ", " << coords[ws][e][j][1]
                  << ", " << coords[ws][e][j][2] << std::endl;
      }
    }
  }
}

const Teuchos::ArrayRCP<double>&
Albany::STKDiscretization::getCoordinates() const
{
  // Coordinates are computed here, and not precomputed,
  // since the mesh can move in shape opt problems

  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();

  for (int i=0; i < numOverlapNodes; i++)  {
    GO node_gid = gid(overlapnodes[i]);
    int node_lid = overlap_node_mapT->getLocalElement(node_gid);

    double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
    for (int dim=0; dim<stkMeshStruct->numDim; dim++)
      coordinates[3*node_lid + dim] = x[dim];

  }

  return coordinates;
}

// These methods were added to support mesh adaptation, which is currently
// limited to PUMIDiscretization.
void Albany::STKDiscretization::
setCoordinates(const Teuchos::ArrayRCP<const double>& c)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,
    "STKDiscretization::setCoordinates is not implemented.");
}
void Albany::STKDiscretization::
setReferenceConfigurationManager(const Teuchos::RCP<AAdapt::rc::Manager>& rcm)
{
  TEUCHOS_TEST_FOR_EXCEPTION(
    true, std::logic_error,
    "STKDiscretization::setReferenceConfigurationManager is not implemented.");
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
  using std::cout; using std::endl;
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();
  std::string transformType = stkMeshStruct->transformType;

  if (transformType == "None") {}
  else if (transformType == "Spherical") {
  //This form takes a mesh of a square / cube and transforms it into a mesh of a circle/sphere
#ifdef OUTPUT_TO_SCREEN
    *out << "Spherical!" << endl;
#endif
    const int numDim = stkMeshStruct->numDim;
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      double r = 0.0;
      for (int n=0; n<numDim; n++)
        r += x[n]*x[n];
      r = sqrt(r);
      for (int n=0; n<numDim; n++)
      //FIXME: there could be division by 0 here!
        x[n] = x[n]/r;
    }
  }
  else if (transformType == "ISMIP-HOM Test A") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test A!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha*pi/180; //convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = -x[0]*tan(alpha);
      double b = s - 1.0 + 0.5*sin(2*pi/L*x[0])*sin(2*pi/L*x[1]);
      x[2] = s*x[2] + b*(1-x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
     }
   }
  else if (transformType == "ISMIP-HOM Test B") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test B!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha*pi/180; //convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = -x[0]*tan(alpha);
      double b = s - 1.0 + 0.5*sin(2*pi/L*x[0]);
      x[2] = s*x[2] + b*(1-x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
     }
   }
   else if ((transformType == "ISMIP-HOM Test C") || (transformType == "ISMIP-HOM Test D")) {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test C and D!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha*pi/180; //convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = -x[0]*tan(alpha);
      double b = s - 1.0;
      x[2] = s*x[2] + b*(1-x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
     }
   }
   else if (transformType == "Dome") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Dome transform!" << endl;
#endif
    double L = 0.7071*30;
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = 0.7071*sqrt(450.0 - x[0]*x[0] - x[1]*x[1])/sqrt(450.0);
      x[2] = s*x[2];
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  }
   else if (transformType == "Confined Shelf") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Confined shelf transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    cout << "L: " << L << endl;
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = 0.06; //top surface is at z=0.06km=60m
      double b = -0.440; //basal surface is at z=-0.440km=-440m
      x[2] = s*x[2] + b*(1.0-x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  }
  else if (transformType == "Circular Shelf") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Circular shelf transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    double rhoIce = 910.0; //ice density, in kg/m^3
    double rhoOcean = 1028.0; //ocean density, in kg/m^3
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stkMeshStruct->PBCStruct.scale[1]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*x[0];
      x[1] = L*x[1];
      double s = 1.0-rhoIce/rhoOcean; //top surface is at z=(1-rhoIce/rhoOcean) km
      double b = s - 1.0; //basal surface is at z=s-1 km
      x[2] = s*x[2] + b*(1.0-x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  }
  else if (transformType == "FO XZ MMS") {
   //This test case assumes the domain read in from the input file is 0 < x < 2, 0 < y < 1, where y = z
#ifdef OUTPUT_TO_SCREEN
    *out << "FO XZ MMS transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    //hard-coding values of parameters...  make sure these are same as in the FOStokes body force evaluator!
    double alpha0 = 4e-5;
    double s0 = 2.0;
    double H = 1.0;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0]*=L;
    stk::mesh::Field<double>* surfaceHeight_field = metaData.get_field<stk::mesh::Field<double> >(stk::topology::NODE_RANK, "surface_height");
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L*(x[0]-1.0);  //test case assumes domain is from [-L, L], where unscaled domain is from [0, 2];
      double s = s0 - alpha0*x[0]*x[0];
      double b = s - H;
      //this transformation of y = [0,1] should give b(x) < y < s(x)
      x[1] = b*(1-x[1]) + s*x[1];
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
     }
  }
#ifdef ALBANY_AERAS
  else if (transformType == "Aeras Schar Mountain") {
    *out << "Aeras Schar Mountain transformation!" << endl;
    double rhoOcean = 1028.0; //ocean density, in kg/m^3
    for (int i=0; i < numOverlapNodes; i++)  {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = x[0];
      double hstar = 0.0, h;
      if (std::abs(x[0]-150.0) <= 25.0) hstar = 3.0* std::pow(cos(M_PI*(x[0]-150.0) / 50.0),2);
      h = hstar * std::pow(cos(M_PI*(x[0]-150.0) / 8.0),2);
      x[1] = x[1] + h*(25.0 - x[1])/25.0;
    }
  }
#endif
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
      "STKDiscretization::transformMesh() Unknown transform type :" << transformType << std::endl);
  }
}

void Albany::STKDiscretization::setupMLCoords()
{
  if (rigidBodyModes.is_null()) return;
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed()) return;

  const int numDim = stkMeshStruct->numDim;
  AbstractSTKFieldContainer::VectorFieldType*
    coordinates_field = stkMeshStruct->getCoordinatesField();
  coordMV = Teuchos::rcp(
      new Tpetra_MultiVector(node_mapT, numDim, false));


  for (int i = 0; i < numOwnedNodes; i++) {
    GO node_gid = gid(ownednodes[i]);
    int node_lid = node_mapT->getLocalElement(node_gid);
    double* X = stk::mesh::field_data(*coordinates_field, ownednodes[i]);
    for (int j = 0; j < numDim; j++)
      coordMV->replaceLocalValue(node_lid, j, X[j]);
  }

  rigidBodyModes->setCoordinatesAndNullspace(coordMV, mapT);

  // Some optional matrix-market output was tagged on here; keep that
  // functionality.
  writeCoordsToMatrixMarket();
}

void Albany::STKDiscretization::writeCoordsToMatrixMarket() const
{
  //if user wants to write the coordinates to matrix market file, write them to matrix market file
  if (rigidBodyModes->isMLUsed() && stkMeshStruct->writeCoordsToMMFile) {
    if (node_mapT->getComm()->getRank()==0) {std::cout << "Writing mesh coordinates to Matrix Market file." << std::endl;}
    int numMyElements = (node_mapT->getComm()->getRank() == 0) ? node_mapT->getGlobalNumElements() : 0;
    Teuchos::RCP<Tpetra_Import> importOperatorT;
    Teuchos::RCP<Tpetra_Map> serial_mapT;
    Teuchos::RCP<const Tpetra_Vector> xCoordsT = coordMV->getVector(0);
    //Writing of coordinates to MatrixMarket file for Ray
    if (node_mapT->getComm()->getSize() > 1) {
      serial_mapT = Teuchos::rcp(new Tpetra_Map(INVALID, numMyElements, 0, node_mapT->getComm()));
      //create importer from parallel map to serial map and populate serial solution xfinal_serial
      importOperatorT = Teuchos::rcp(new Tpetra_Import(node_mapT, serial_mapT));
      //Writing of coordinates to MatrixMarket file for Ray
      Teuchos::RCP<Tpetra_Vector> xCoords_serialT = Teuchos::rcp(new Tpetra_Vector(serial_mapT));
      xCoords_serialT->doImport(*xCoordsT, *importOperatorT, Tpetra::INSERT);
      Tpetra_MatrixMarket_Writer::writeDenseFile("xCoords.mm", xCoords_serialT);
    }
    else
      Tpetra_MatrixMarket_Writer::writeDenseFile("xCoords.mm", xCoordsT);
    if (coordMV->getNumVectors() > 1) {
      Teuchos::RCP<const Tpetra_Vector> yCoordsT =  coordMV->getVector(1);
      if (node_mapT->getComm()->getSize() > 1) {
        Teuchos::RCP<Tpetra_Vector> yCoords_serialT = Teuchos::rcp(new Tpetra_Vector(serial_mapT));
        yCoords_serialT->doImport(*yCoordsT, *importOperatorT, Tpetra::INSERT);
        Tpetra_MatrixMarket_Writer::writeDenseFile("yCoords.mm", yCoords_serialT);
      }
      else
        Tpetra_MatrixMarket_Writer::writeDenseFile("yCoords.mm", yCoordsT);
    }
    if (coordMV->getNumVectors() > 2){
      Teuchos::RCP<const Tpetra_Vector> zCoordsT =  coordMV->getVector(2);
      if (node_mapT->getComm()->getSize() > 1) {
        Teuchos::RCP<Tpetra_Vector> zCoords_serialT = Teuchos::rcp(new Tpetra_Vector(serial_mapT));
        zCoords_serialT->doImport(*zCoordsT, *importOperatorT, Tpetra::INSERT);
        Tpetra_MatrixMarket_Writer::writeDenseFile("zCoords.mm", zCoords_serialT);
      }
      else
        Tpetra_MatrixMarket_Writer::writeDenseFile("zCoords.mm", zCoordsT);
    }
  }
}

const Albany::WorksetArray<std::string>::type&
Albany::STKDiscretization::getWsEBNames() const
{
  return wsEBNames;
}

const Albany::WorksetArray<int>::type&
Albany::STKDiscretization::getWsPhysIndex() const
{
  return wsPhysIndex;
}

#if defined(ALBANY_EPETRA)
void Albany::STKDiscretization::writeSolution(const Epetra_Vector& soln, const double time, const bool overlapped){

  Teuchos::RCP<const Tpetra_Vector> solnT =
     Petra::EpetraVector_To_TpetraVectorConst(soln, commT);
  writeSolutionT(*solnT, time, overlapped);
}
#endif

void Albany::STKDiscretization::writeSolutionT(
  const Tpetra_Vector& solnT, const double time, const bool overlapped)
{
  writeSolutionToMeshDatabaseT(solnT, time, overlapped);
  writeSolutionToFileT(solnT, time, overlapped);
}

void Albany::STKDiscretization::writeSolutionMV(
  const Tpetra_MultiVector& solnT, const double time, const bool overlapped)
{
  writeSolutionMVToMeshDatabase(solnT, time, overlapped);
  writeSolutionMVToFile(solnT, time, overlapped);
}

void Albany::STKDiscretization::writeSolutionToMeshDatabaseT(
  const Tpetra_Vector& solnT, const double time, const bool overlapped)
{
  // Put solution as Epetra_Vector into STK Mesh
  if (!overlapped)
    setSolutionFieldT(solnT);
  // soln coming in is overlapped
  else
    setOvlpSolutionFieldT(solnT);
}

void Albany::STKDiscretization::writeSolutionMVToMeshDatabase(
  const Tpetra_MultiVector& solnT, const double time, const bool overlapped)
{
  // Put solution as Epetra_Vector into STK Mesh
  if (!overlapped)
    setSolutionFieldMV(solnT);
  // soln coming in is overlapped
  else
    setOvlpSolutionFieldMV(solnT);
}

void Albany::STKDiscretization::
writeSolutionToFileT(const Tpetra_Vector& solnT, const double time,
                     const bool overlapped)
{
#ifdef ALBANY_SEACAS

  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {

   Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

   container->transferSolutionToCoords();

   if (!mesh_data.is_null()) {
     // Mesh coordinates have changed. Rewrite output file by deleting the mesh data object and recreate it
     setupExodusOutput();
   }
  }

  // Skip this write unless the proper interval has been reached
  if (stkMeshStruct->exoOutput && !(outputInterval % stkMeshStruct->exoOutputInterval)) {

   double time_label = monotonicTimeLabel(time);

     mesh_data->begin_output_step(outputFileIdx, time_label);
     int out_step = mesh_data->write_defined_output_fields(outputFileIdx);
     // Writing mesh global variables
     for (auto& it : stkMeshStruct->getFieldContainer()->getMeshVectorStates())
     {
       mesh_data->write_global (outputFileIdx, it.first, it.second);
     }
     for (auto& it : stkMeshStruct->getFieldContainer()->getMeshScalarIntegerStates())
     {
       mesh_data->write_global (outputFileIdx, it.first, it.second);
     }
     mesh_data->end_output_step(outputFileIdx);

     if (mapT->getComm()->getRank()==0) {
       *out << "Albany::STKDiscretization::writeSolution: writing time " << time;
       if (time_label != time) *out << " with label " << time_label;
       *out << " to index " <<out_step<<" in file "<<stkMeshStruct->exoOutFile<< std::endl;
     }
   }
   if (stkMeshStruct->cdfOutput && !(outputInterval % stkMeshStruct->cdfOutputInterval)) {

     double time_label = monotonicTimeLabel(time);

     const int out_step = processNetCDFOutputRequestT(solnT);

     if (mapT->getComm()->getRank()==0) {
       *out << "Albany::STKDiscretization::writeSolution: writing time " << time;
       if (time_label != time) *out << " with label " << time_label;
       *out << " to index " <<out_step<<" in file "<<stkMeshStruct->cdfOutFile<< std::endl;
     }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations)
  {
    if (overlapped)
    {
      Tpetra_Vector ss_solnT (it.second->getOverlapMapT());
      const Tpetra_CrsMatrix& P = *ov_projectorsT.at(it.first);
      P.apply(solnT, ss_solnT);
      it.second->writeSolutionToFileT (ss_solnT, time, overlapped);
    }
    else
    {
      Tpetra_Vector ss_solnT (it.second->getMapT());
      const Tpetra_CrsMatrix& P = *projectorsT.at(it.first);
      P.apply(solnT, ss_solnT);
      it.second->writeSolutionToFileT (ss_solnT, time, overlapped);
    }
  }
#endif

}

void Albany::STKDiscretization::
writeSolutionMVToFile(const Tpetra_MultiVector& solnT, const double time,
                     const bool overlapped)
{
#ifdef ALBANY_SEACAS

  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {

   Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

   container->transferSolutionToCoords();

   if (!mesh_data.is_null()) {
     // Mesh coordinates have changed. Rewrite output file by deleting the mesh data object and recreate it
     setupExodusOutput();
   }
  }

  // Skip this write unless the proper interval has been reached
  if (stkMeshStruct->exoOutput && !(outputInterval % stkMeshStruct->exoOutputInterval)) {

   double time_label = monotonicTimeLabel(time);

     mesh_data->begin_output_step(outputFileIdx, time_label);
     int out_step = mesh_data->write_defined_output_fields(outputFileIdx);
     // Writing mesh global variables
     for (auto& it : stkMeshStruct->getFieldContainer()->getMeshVectorStates())
     {
       mesh_data->write_global (outputFileIdx, it.first, it.second);
     }
     for (auto& it : stkMeshStruct->getFieldContainer()->getMeshScalarIntegerStates())
     {
       mesh_data->write_global (outputFileIdx, it.first, it.second);
     }
     mesh_data->end_output_step(outputFileIdx);

     if (mapT->getComm()->getRank()==0) {
       *out << "Albany::STKDiscretization::writeSolution: writing time " << time;
       if (time_label != time) *out << " with label " << time_label;
       *out << " to index " <<out_step<<" in file "<<stkMeshStruct->exoOutFile<< std::endl;
     }
   }
   if (stkMeshStruct->cdfOutput && !(outputInterval % stkMeshStruct->cdfOutputInterval)) {

     double time_label = monotonicTimeLabel(time);

     const int out_step = processNetCDFOutputRequestMV(solnT);

     if (mapT->getComm()->getRank()==0) {
       *out << "Albany::STKDiscretization::writeSolution: writing time " << time;
       if (time_label != time) *out << " with label " << time_label;
       *out << " to index " <<out_step<<" in file "<<stkMeshStruct->cdfOutFile<< std::endl;
     }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations)
  {
    if (overlapped)
    {
      Tpetra_MultiVector ss_solnT (it.second->getOverlapMapT(),solnT.getNumVectors());
      const Tpetra_CrsMatrix& P = *ov_projectorsT.at(it.first);
      P.apply(solnT, ss_solnT);
      it.second->writeSolutionMVToFile (ss_solnT, time, overlapped);
    }
    else
    {
      Tpetra_MultiVector ss_solnT (it.second->getMapT(),solnT.getNumVectors());
      const Tpetra_CrsMatrix& P = *projectorsT.at(it.first);
      P.apply(solnT, ss_solnT);
      it.second->writeSolutionMVToFile (ss_solnT, time, overlapped);
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
Albany::STKDiscretization::setResidualFieldT(const Tpetra_Vector& residualT)
{
#if defined(ALBANY_LCM)
  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  if(container->hasResidualField()){

    // Write the overlapped data
    stk::mesh::Selector select_owned_or_shared = metaData.locally_owned_part() | metaData.globally_shared_part();
    container->saveResVectorT(residualT, select_owned_or_shared, overlap_node_mapT);
  }

  // Setting the residual on the side set meshes
  for (auto it : sideSetDiscretizations)
  {
    const Tpetra_CrsMatrix& P = *ov_projectorsT.at(it.first);
    Tpetra_Vector ss_residualT (it.second->getOverlapMapT());
    P.apply(residualT,ss_residualT);
    it.second->setResidualFieldT(ss_residualT);
  }
#endif
}


#if defined(ALBANY_EPETRA)
Teuchos::RCP<Epetra_Vector>
Albany::STKDiscretization::getSolutionField(bool overlapped) const
{
  /*
  // Copy soln vector into solution field, one node at a time
  Teuchos::ArrayView<const GO> indicesAV = mapT->getNodeElementList();
  int numElements = mapT->getNodeNumElements();
#ifdef ALBANY_64BIT_INT
  Teuchos::Array<int> i_indices(numElements);
  for(std::size_t k = 0; k < numElements; k++)
  i_indices[k] = Teuchos::as<int>(indicesAV[k]);
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(-1, numElements, i_indices.getRawPtr(), 0, *comm));
#else
  Teuchos::RCP<Epetra_Map> map = Teuchos::rcp(new Epetra_Map(-1, numElements, indicesAV.getRawPtr(), 0, *comm));
#endif */
  Teuchos::RCP<Epetra_Vector> soln = Teuchos::rcp(new Epetra_Vector(*map));
  this->getSolutionField(*soln, overlapped);
  return soln;
}
#endif

Teuchos::RCP<Tpetra_Vector>
Albany::STKDiscretization::getSolutionFieldT(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Tpetra_Vector> solnT = Teuchos::rcp(new Tpetra_Vector(mapT));
  this->getSolutionFieldT(*solnT, overlapped);
  return solnT;
}

Teuchos::RCP<Tpetra_MultiVector>
Albany::STKDiscretization::getSolutionMV(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  int num_time_deriv = stkMeshStruct->num_time_deriv;
  Teuchos::RCP<Tpetra_MultiVector> solnT = Teuchos::rcp(new Tpetra_MultiVector(mapT, num_time_deriv + 1, false));
  this->getSolutionMV(*solnT, overlapped);
  return solnT;
}

int
Albany::STKDiscretization::getSolutionFieldHistoryDepth() const
{
  return stkMeshStruct->getSolutionFieldHistoryDepth();
}

#if defined(ALBANY_EPETRA)
Teuchos::RCP<Epetra_MultiVector>
Albany::STKDiscretization::getSolutionFieldHistory() const
{
  const int stepCount = this->getSolutionFieldHistoryDepth();
  return this->getSolutionFieldHistoryImpl(stepCount);
}

Teuchos::RCP<Epetra_MultiVector>
Albany::STKDiscretization::getSolutionFieldHistory(int maxStepCount) const
{
  const int stepCount = std::min(this->getSolutionFieldHistoryDepth(), maxStepCount);
  return this->getSolutionFieldHistoryImpl(stepCount);
}

//IK, 10/28/13: this function should be converted to Tpetra...
void
Albany::STKDiscretization::getSolutionFieldHistory(Epetra_MultiVector &result) const
{
  TEUCHOS_TEST_FOR_EXCEPT(!map->SameAs(result.Map()));
  const int stepCount = std::min(this->getSolutionFieldHistoryDepth(), result.NumVectors());
  Epetra_MultiVector head(View, result, 0, stepCount);
  this->getSolutionFieldHistoryImpl(head);
}

Teuchos::RCP<Epetra_MultiVector>
Albany::STKDiscretization::getSolutionFieldHistoryImpl(int stepCount) const
{
  const int vectorCount = stepCount > 0 ? stepCount : 1; // A valid MultiVector has at least one vector
  const Teuchos::RCP<Epetra_MultiVector> result = Teuchos::rcp(new Epetra_MultiVector(*map, vectorCount));
  if (stepCount > 0) {
    this->getSolutionFieldHistoryImpl(*result);
  }
  return result;
}

void
Albany::STKDiscretization::getSolutionFieldHistoryImpl(Epetra_MultiVector &result) const
{
  const int stepCount = result.NumVectors();
  for (int i = 0; i < stepCount; ++i) {
    stkMeshStruct->loadSolutionFieldHistory(i);
    Epetra_Vector v(View, result, i);
    this->getSolutionField(v);
  }
}

#endif

#if defined(ALBANY_EPETRA)
void
Albany::STKDiscretization::getSolutionField(Epetra_Vector &result, const bool overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->fillSolnVector(result, locally_owned, node_map);
}
#endif

void
Albany::STKDiscretization::getFieldT(Tpetra_Vector &result, const std::string& name) const
{
  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  const std::string& part = nodalDOFsStructContainer.fieldToMap.find(name)->second->first.first;
  stk::mesh::Selector selector = metaData.locally_owned_part();
  if(part.size()) {
    std::map<std::string, stk::mesh::Part*>::const_iterator it = stkMeshStruct->nsPartVec.find(part);
    if(it != stkMeshStruct->nsPartVec.end())
      selector &= stk::mesh::Selector( *(it->second) );
  }

  const DOFsStruct& dofsStruct = nodalDOFsStructContainer.getDOFsStruct(name);

  container->fillVectorT(result, name, selector, dofsStruct.node_map,dofsStruct.dofManager);
}

void
Albany::STKDiscretization::getSolutionFieldT(Tpetra_Vector &resultT, const bool overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->fillSolnVectorT(resultT, locally_owned, node_mapT);
}

void
Albany::STKDiscretization::getSolutionMV(Tpetra_MultiVector &resultT, const bool overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->fillSolnMultiVector(resultT, locally_owned, node_mapT);
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

//Tpetra version of above
void
Albany::STKDiscretization::setFieldT(const Tpetra_Vector &result, const std::string& name, bool overlapped)
{
  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  const std::string& part = nodalDOFsStructContainer.fieldToMap.find(name)->second->first.first;

  stk::mesh::Selector selector = overlapped ?
      metaData.locally_owned_part() | metaData.globally_shared_part() :
      metaData.locally_owned_part();

  // Iterate over the on-processor nodes by getting node buckets and iterating over each bucket.
  if(part.size()) {
    std::map<std::string, stk::mesh::Part*>::const_iterator it = stkMeshStruct->nsPartVec.find(part);
    if(it != stkMeshStruct->nsPartVec.end())
      selector &= stk::mesh::Selector( *(it->second) );
  }

  const DOFsStruct& dofsStruct = nodalDOFsStructContainer.getDOFsStruct(name);

  if(overlapped)
   container->saveVectorT(result, name, selector, dofsStruct.overlap_node_map, dofsStruct.overlap_dofManager);
  else
    container->saveVectorT(result, name, selector, dofsStruct.node_map, dofsStruct.dofManager);
}

void
Albany::STKDiscretization::setSolutionFieldT(const Tpetra_Vector& solnT)
{

  // Copy soln vector into solution field, one node at a time
  // Note that soln coming in is the local (non overlapped) soln

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->saveSolnVectorT(solnT, locally_owned, node_mapT);
}

void
Albany::STKDiscretization::setSolutionFieldMV(const Tpetra_MultiVector& solnT)
{

  // Copy soln vector into solution field, one node at a time
  // Note that soln coming in is the local (non overlapped) soln

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the on-processor nodes
  stk::mesh::Selector locally_owned = metaData.locally_owned_part();

  container->saveSolnMultiVector(solnT, locally_owned, node_mapT);
}

void
Albany::STKDiscretization::setOvlpSolutionFieldT(const Tpetra_Vector& solnT)
{
  // Copy soln vector into solution field, one node at a time
  // Note that soln coming in is the local+ghost (overlapped) soln

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the processor-visible nodes
  stk::mesh::Selector select_owned_or_shared = metaData.locally_owned_part() | metaData.globally_shared_part();

  container->saveSolnVectorT(solnT, select_owned_or_shared, overlap_node_mapT);
}

void
Albany::STKDiscretization::setOvlpSolutionFieldMV(const Tpetra_MultiVector& solnT)
{
  // Copy soln vector into solution field, one node at a time
  // Note that soln coming in is the local+ghost (overlapped) soln

  Teuchos::RCP<AbstractSTKFieldContainer> container = stkMeshStruct->getFieldContainer();

  // Iterate over the processor-visible nodes
  stk::mesh::Selector select_owned_or_shared = metaData.locally_owned_part() | metaData.globally_shared_part();

  container->saveSolnMultiVector(solnT, select_owned_or_shared, overlap_node_mapT);

}

GO Albany::STKDiscretization::gid(const stk::mesh::Entity node) const
{ return bulkData.identifier(node)-1; }

int Albany::STKDiscretization::getOwnedDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numOwnedNodes*eq;
}

int Albany::STKDiscretization::getOverlapDOF(const int inode, const int eq) const
{
  if (interleavedOrdering) return inode*neq + eq;
  else  return inode + numOverlapNodes*eq;
}

GO Albany::STKDiscretization::getGlobalDOF(const GO inode, const int eq) const
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

void Albany::STKDiscretization::computeNodalMaps (bool overlapped)
{
  // Loads member data:  ownednodes, numOwnedNodes, node_map, numGlobalNodes, map
  // maps for owned nodes and unknowns

  stk::mesh::Selector map_type_selector = overlapped ?
           (metaData.locally_owned_part() | metaData.globally_shared_part()) :
           metaData.locally_owned_part();

  NodalDOFsStructContainer::MapOfDOFsStructs& mapOfDOFsStructs = nodalDOFsStructContainer.mapOfDOFsStructs;
  std::vector< stk::mesh::Entity> nodes;
  int numNodes(0);

  //compute NumGlobalNodes
  stk::mesh::get_selected_entities( map_type_selector ,
                  bulkData.buckets( stk::topology::NODE_RANK ) ,
                  nodes );

  GO maxID(0), maxGID(0);
  for (int i=0; i < nodes.size(); i++)
    maxID = std::max(maxID, gid(nodes[i]));
  Teuchos::reduceAll(*commT, Teuchos::REDUCE_MAX, 1, &maxID, &maxGID);
  numGlobalNodes = maxGID+1; //maxGID is the same for overlapped and unique maps

  // build maps
  for(auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end(); ++it ) {
    stk::mesh::Selector selector(map_type_selector);
    const std::string& part = it->first.first;
    int nComp = it->first.second;
    if(part.size())  {
      auto it3 = stkMeshStruct->nsPartVec.find(part);
      if(it3 != stkMeshStruct->nsPartVec.end())
        selector &= *(it3->second);
      else { //throw error
        std::ostringstream msg;
        msg << "Albany::STKDiscretization::computeNodalMaps(overlapped==" << overlapped <<
            "):\n    Part " << part << " is not in  stkMeshStruct->nsPartVec.\n";
        throw std::runtime_error(msg.str());
      }
    }

    stk::mesh::get_selected_entities( selector ,
                bulkData.buckets( stk::topology::NODE_RANK ) ,
                nodes );

    numNodes = nodes.size();

    Teuchos::Array<GO> indicesT(numNodes*nComp);
    NodalDOFManager* dofManager = (overlapped) ? &it->second.overlap_dofManager : &it->second.dofManager;
    dofManager->setup(nComp, numNodes, numGlobalNodes, interleavedOrdering);

    for (int i=0; i < numNodes; i++)
      for (int j=0; j < nComp; j++)
        indicesT[dofManager->getLocalDOF(i,j)] = dofManager->getGlobalDOF(bulkData.identifier(nodes[i])-1, j);


    Teuchos::RCP<const Tpetra_Map>& map = (overlapped) ? it->second.overlap_map : it->second.map;
    map = Teuchos::null;
    map = Tpetra::createNonContigMap<LO, GO>(indicesT(), commT);

    Teuchos::RCP<const Tpetra_Map>& node_map = (overlapped) ? it->second.overlap_node_map : it->second.node_map;
    node_map = Teuchos::null;

    auto it2=it;
    if((nComp==1) || ((it2=mapOfDOFsStructs.find(make_pair(part,1)))!=mapOfDOFsStructs.end())) {
      node_map = (overlapped) ? it2->second.overlap_map : it2->second.map;
    }
  }
}

void Albany::STKDiscretization::computeOwnedNodesAndUnknowns()
{
  // Loads member data:  ownednodes, numOwnedNodes, node_map, numGlobalNodes, map
  // maps for owned nodes and unknowns
  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::get_selected_entities( select_owned_in_part ,
            bulkData.buckets( stk::topology::NODE_RANK ) ,
            ownednodes );

  numOwnedNodes = ownednodes.size();
  node_mapT = nodalDOFsStructContainer.getDOFsStruct("mesh_nodes").map;
  mapT = nodalDOFsStructContainer.getDOFsStruct("ordinary_solution").map;

#ifdef ALBANY_EPETRA
  map = Petra::TpetraMap_To_EpetraMap(mapT, comm);
  node_map = Petra::TpetraMap_To_EpetraMap(node_mapT, comm);
#endif

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base))
    stkMeshStruct->nodal_data_base->resizeLocalMap(
      node_mapT->getNodeElementList(), commT);



/*
  Teuchos::Array<GO> indicesT(numOwnedNodes);
  for (int i=0; i < numOwnedNodes; i++) indicesT[i] = gid(ownednodes[i]);

  node_mapT = Teuchos::null; // delete existing map happens here on remesh
  node_mapT = Tpetra::createNonContigMap<LO, GO>(indicesT(), commT);

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base))
    stkMeshStruct->nodal_data_base->resizeLocalMap(indicesT, commT);

  numGlobalNodes = node_mapT->getMaxAllGlobalIndex() + 1;

  indicesT.resize(numOwnedNodes * neq);
  for (int i=0; i < numOwnedNodes; i++)
    for (std::size_t j=0; j < neq; j++)
      indicesT[getOwnedDOF(i,j)] = getGlobalDOF(gid(ownednodes[i]),j);

  mapT = Teuchos::null; // delete existing map happens here on remesh
  mapT = Tpetra::createNonContigMap<LO, GO>(indicesT(), commT);
*/
}

void Albany::STKDiscretization::computeOverlapNodesAndUnknowns()
{
  // maps for overlap unknowns
  stk::mesh::Selector select_overlap_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    ( stk::mesh::Selector( metaData.locally_owned_part() )
      | stk::mesh::Selector( metaData.globally_shared_part() ) );

  // overlapnodes used for overlap map; stored for changing coords
  stk::mesh::get_selected_entities( select_overlap_in_part ,
            bulkData.buckets( stk::topology::NODE_RANK ) ,
            overlapnodes );

  numOverlapNodes = overlapnodes.size();
  numOverlapNodes = overlapnodes.size();

  overlap_mapT = nodalDOFsStructContainer.getDOFsStruct("ordinary_solution").overlap_map;
  overlap_node_mapT = nodalDOFsStructContainer.getDOFsStruct("mesh_nodes").overlap_map;

#ifdef ALBANY_EPETRA
  overlap_map = Petra::TpetraMap_To_EpetraMap(overlap_mapT, comm);
  overlap_node_map = Petra::TpetraMap_To_EpetraMap(overlap_node_mapT, comm);
#endif

  if(Teuchos::nonnull(stkMeshStruct->nodal_data_base))
    stkMeshStruct->nodal_data_base->resizeOverlapMap(
      overlap_node_mapT->getNodeElementList(), commT);

/*
  Teuchos::Array<GO> indicesT(numOverlapNodes * neq);
  for (int i=0; i < numOverlapNodes; i++)
    for (std::size_t j=0; j < neq; j++)
      indicesT[getOverlapDOF(i,j)] = getGlobalDOF(gid(overlapnodes[i]),j);

  overlap_mapT = Teuchos::null; // delete existing map happens here on remesh
  overlap_mapT = Tpetra::createNonContigMap<LO, GO>(indicesT(), commT);

  indicesT.resize(numOverlapNodes);
  for (int i=0; i < numOverlapNodes; i++)
    indicesT[i] = gid(overlapnodes[i]);

  overlap_node_mapT = Teuchos::null; // delete existing map happens here on remesh
  overlap_node_mapT = Tpetra::createNonContigMap<LO, GO>(indicesT(), commT);

  if(Teuchos::nonnull(stkMeshStruct->nodal_data_base))
    stkMeshStruct->nodal_data_base->resizeOverlapMap(indicesT, commT);
*/
  coordinates.resize(3*numOverlapNodes);
}

void Albany::STKDiscretization::computeGraphs()
{
  computeGraphsUpToFillComplete();
  fillCompleteGraphs();
}

void Albany::STKDiscretization::computeGraphsUpToFillComplete()
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
            bulkData.buckets( stk::topology::ELEMENT_RANK ) ,
            cells );

  if (commT->getRank()==0)
    *out << "STKDisc: " << cells.size() << " elements on Proc 0 " << std::endl;

  GO row, col;
  Teuchos::ArrayView<GO> colAV;

  // determining the equations that are defined on the whole domain
  std::vector<int> globalEqns;
  for (int k(0); k<neq; ++k)
  {
    if (sideSetEquations.find(k)==sideSetEquations.end())
    {
      globalEqns.push_back(k);
    }
  }

  for (std::size_t i=0; i < cells.size(); i++) {
    stk::mesh::Entity e = cells[i];
    stk::mesh::Entity const* node_rels = bulkData.begin_nodes(e);
    const size_t num_nodes = bulkData.num_nodes(e);

    // loop over local nodes
    for (std::size_t j=0; j < num_nodes; j++) {
      stk::mesh::Entity rowNode = node_rels[j];

      // loop over eqs
      for (std::size_t k=0; k < globalEqns.size(); ++k)
      {
        row = getGlobalDOF(gid(rowNode), globalEqns[k]);
        for (std::size_t l=0; l < num_nodes; l++)
        {
          stk::mesh::Entity colNode = node_rels[l];
          for (std::size_t m=0; m < neq; m++) // Note: here we cycle through ALL the eqns (not just the global ones),
          {                                   //       since they could all be coupled with this eq
            col = getGlobalDOF(gid(colNode), m);
            colAV = Teuchos::arrayView(&col, 1);
            overlap_graphT->insertGlobalIndices(row, colAV);
          }
        }
      }
    }
  }

  if (sideSetEquations.size()>0)
  {
    // iterator over all sideSet-defined equations
    std::map<int,std::vector<std::string> >::iterator it;
    for (it=sideSetEquations.begin(); it!=sideSetEquations.end(); ++it)
    {
      // Get the eq number
      int eq = it->first;

      // In case we only have equations on side sets (no "volume" eqns),
      // there would be problem with linear solvers. To avoid this, we
      // put one diagonal entry for every side set equation.
      // NOTE: some nodes will be processed twice, but this is safe
      //       in Tpetra_CrsGraph: the redundant indices will be discarded
      for (std::size_t inode=0; inode < overlapnodes.size(); ++inode)
      {
        stk::mesh::Entity node = overlapnodes[inode];
        row = getGlobalDOF(gid(node), it->first);
        colAV = Teuchos::arrayView(&row, 1);
        overlap_graphT->insertGlobalIndices(row, colAV);
      }

      // Number of side sets this eq is defined on
      int numSideSets = it->second.size();
      for (int ss(0); ss<numSideSets; ++ss)
      {
        stk::mesh::Part& part = *stkMeshStruct->ssPartVec.find(it->second[ss])->second;

        // Get all owned sides in this side set
        stk::mesh::Selector select_owned_in_sspart = stk::mesh::Selector( part ) & stk::mesh::Selector( metaData.locally_owned_part() );

        std::vector< stk::mesh::Entity > sides;
        stk::mesh::get_selected_entities( select_owned_in_sspart, bulkData.buckets( metaData.side_rank() ), sides ); // store the result in "sides"

        // Loop on all the sides of this sideset
        for (std::size_t localSideID=0; localSideID < sides.size(); localSideID++)
        {
          stk::mesh::Entity sidee = sides[localSideID];
          stk::mesh::Entity const* node_rels = bulkData.begin_nodes(sidee);
          const size_t num_nodes = bulkData.num_nodes(sidee);

          // loop over local nodes of the side (row)
          for (std::size_t i=0; i < num_nodes; i++)
          {
            stk::mesh::Entity rowNode = node_rels[i];
            row = getGlobalDOF(gid(rowNode), eq);

            // loop over local nodes of the side (col)
            for (std::size_t j=0; j < num_nodes; j++)
            {
              stk::mesh::Entity colNode = node_rels[j];

             // loop on all the equations (the eq may be coupled with other eqns)
              for (std::size_t m=0; m < neq; m++)
              {
                col = getGlobalDOF(gid(colNode), m);
                colAV = Teuchos::arrayView(&col, 1);
                overlap_graphT->insertGlobalIndices(row, colAV);
              }
            }
          }
        }
      }
    }
  }
}

void Albany::STKDiscretization::fillCompleteGraphs()
{
  overlap_graphT->fillComplete();

  // Create Owned graph by exporting overlap with known row map
  graphT = Teuchos::null; // delete existing graph happens here on remesh

  graphT = Teuchos::rcp(new Tpetra_CrsGraph(mapT, nonzeroesPerRow(neq)));

  // Create non-overlapped matrix using two maps and export object
  Teuchos::RCP<Tpetra_Export> exporterT = Teuchos::rcp(new Tpetra_Export(overlap_mapT, mapT));
  graphT->doExport(*overlap_graphT, *exporterT, Tpetra::INSERT);
  graphT->fillComplete();
}

void Albany::STKDiscretization::insertPeridigmNonzerosIntoGraph()
{
#ifdef ALBANY_PERIDIGM
#if defined(ALBANY_EPETRA)
  if (Teuchos::nonnull(LCM::PeridigmManager::self()) && LCM::PeridigmManager::self()->hasTangentStiffnessMatrix()){

    // The Peridigm matrix is a subset of the Albany matrix.  The global ids are the same and the parallel
    // partitioning is the same.  fillComplete() has already been called for the Peridigm matrix.
    Teuchos::RCP<const Epetra_FECrsMatrix> peridigmMatrix = LCM::PeridigmManager::self()->getTangentStiffnessMatrix();

    // Allocate nonzeros for the standard FEM portion of the graph
    computeGraphsUpToFillComplete();

    // Allocate nonzeros for the peridynamic portion of the graph
    GO globalRow, globalCol;
    Teuchos::ArrayView<GO> globalColAV;
    int peridigmLocalRow;
    int numEntries;
    double* values;
    int* indices;
    for (std::size_t i=0; i < cells.size(); i++) {
      stk::mesh::Entity e = cells[i];
      stk::mesh::Entity const* node_rels = bulkData.begin_nodes(e);
      const size_t num_nodes = bulkData.num_nodes(e);
      // Search for sphere elements (they contain a single node)
      if(num_nodes == 1){
  stk::mesh::Entity rowNode = node_rels[0];
  for (std::size_t k=0; k < neq; k++) {
    globalRow = getGlobalDOF(gid(rowNode), k);
    peridigmLocalRow = peridigmMatrix->RowMap().LID(globalRow);
    peridigmMatrix->ExtractMyRowView(peridigmLocalRow, numEntries, values, indices);
    for(int i=0 ; i<numEntries ; ++i){
      globalCol = peridigmMatrix->ColMap().GID(indices[i]);
      globalColAV = Teuchos::arrayView(&globalCol, 1);
      overlap_graphT->insertGlobalIndices(globalRow, globalColAV);
    }
  }
      }
    }

    // Call fillComplete() for the overlap graph and create the non-overlap map
    fillCompleteGraphs();
  }
#endif
#endif
}

void Albany::STKDiscretization::computeWorksetInfo()
{

  stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  stk::mesh::BucketVector const& buckets = bulkData.get_buckets( stk::topology::ELEMENT_RANK, select_owned_in_part );

  const int numBuckets =  buckets.size();

  typedef AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef AbstractSTKFieldContainer::TensorFieldType TensorFieldType;
  typedef AbstractSTKFieldContainer::SphereVolumeFieldType SphereVolumeFieldType;

  VectorFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();

  SphereVolumeFieldType* sphereVolume_field;
  if(stkMeshStruct->getFieldContainer()->hasSphereVolumeField()){
    sphereVolume_field = stkMeshStruct->getFieldContainer()->getSphereVolumeField();
  }

  stk::mesh::FieldBase* latticeOrientation_field;
  if(stkMeshStruct->getFieldContainer()->hasLatticeOrientationField()){
    latticeOrientation_field = stkMeshStruct->getFieldContainer()->getLatticeOrientationField();
  }

  wsEBNames.resize(numBuckets);
  for (int i=0; i<numBuckets; i++) {
    stk::mesh::PartVector const& bpv = buckets[i]->supersets();

    for (std::size_t j=0; j<bpv.size(); j++) {
      if (bpv[j]->primary_entity_rank() == stk::topology::ELEMENT_RANK &&
          !stk::mesh::is_auto_declared_part(*bpv[j])) {
        // *out << "Bucket " << i << " is in Element Block:  " << bpv[j]->name()
        //      << "  and has " << buckets[i]->size() << " elements." << std::endl;
        wsEBNames[i]=bpv[j]->name();
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
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sphereVolume.resize(numBuckets);
  latticeOrientation.resize(numBuckets);

  nodesOnElemStateVec.resize(numBuckets);
  stateArrays.elemStateArrays.resize(numBuckets);
  const Albany::StateInfoStruct& nodal_states = stkMeshStruct->getFieldContainer()->getNodalSIS();

  // Clear map if remeshing
  if(!elemGIDws.empty()) elemGIDws.clear();

  typedef stk::mesh::Cartesian NodeTag;
  typedef stk::mesh::Cartesian ElemTag;
  typedef stk::mesh::Cartesian CompTag;

  NodalDOFsStructContainer::MapOfDOFsStructs& mapOfDOFsStructs = nodalDOFsStructContainer.mapOfDOFsStructs;
  for(auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end(); ++it) {
    it->second.wsElNodeEqID.resize(numBuckets);
    it->second.wsElNodeEqID_rawVec.resize(numBuckets);
    it->second.wsElNodeID.resize(numBuckets);
    it->second.wsElNodeID_rawVec.resize(numBuckets);
  }

  for (int b=0; b < numBuckets; b++) {

    stk::mesh::Bucket& buck = *buckets[b];
    wsElNodeEqID[b].resize(buck.size());
    //wsElNodeEqID_kokkos[b].resize(buck.size());
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());


    {  //nodalDataToElemNode.

      nodesOnElemStateVec[b].resize(nodal_states.size());

      for (int is=0; is< nodal_states.size(); ++is) {
        const std::string& name = nodal_states[is]->name;
        const Albany::StateStruct::FieldDims& dim = nodal_states[is]->dim;
        MDArray& array = stateArrays.elemStateArrays[b][name];
        std::vector<double>& stateVec = nodesOnElemStateVec[b][is];
        int dim0 = buck.size(); //may be different from dim[0];
        switch (dim.size()) {
        case 2:     //scalar
        {
          const ScalarFieldType& field = *metaData.get_field<ScalarFieldType>(stk::topology::NODE_RANK, name);
          stateVec.resize(dim0*dim[1]);
          array.assign<ElemTag, NodeTag>(stateVec.data(),dim0,dim[1]);
          for (int i=0; i < dim0; i++) {
            stk::mesh::Entity element = buck[i];
            stk::mesh::Entity const* rel = bulkData.begin_nodes(element);
            for (int j=0; j < dim[1]; j++) {
              stk::mesh::Entity rowNode = rel[j];
              array(i,j) = *stk::mesh::field_data(field, rowNode);
            }
          }
          break;
        }
        case 3:  //vector
        {
          const VectorFieldType& field = *metaData.get_field<VectorFieldType>(stk::topology::NODE_RANK, name);
          stateVec.resize(dim0*dim[1]*dim[2]);
          array.assign<ElemTag, NodeTag,CompTag>(stateVec.data(),dim0,dim[1],dim[2]);
          for (int i=0; i < dim0; i++) {
            stk::mesh::Entity element = buck[i];
            stk::mesh::Entity const* rel = bulkData.begin_nodes(element);
            for (int j=0; j < dim[1]; j++) {
              stk::mesh::Entity rowNode = rel[j];
              double* entry = stk::mesh::field_data(field, rowNode);
              for(int k=0; k<dim[2]; k++)
                array(i,j,k) = entry[k];
            }
          }
          break;
        }
        case 4: //tensor
        {
          const TensorFieldType& field = *metaData.get_field<TensorFieldType>(stk::topology::NODE_RANK, name);
          stateVec.resize(dim0*dim[1]*dim[2]*dim[3]);
          array.assign<ElemTag, NodeTag, CompTag, CompTag>(stateVec.data(),dim0,dim[1],dim[2],dim[3]);
          for (int i=0; i < dim0; i++) {
            stk::mesh::Entity element = buck[i];
            stk::mesh::Entity const* rel = bulkData.begin_nodes(element);
            for (int j=0; j < dim[1]; j++) {
              stk::mesh::Entity rowNode = rel[j];
              double* entry = stk::mesh::field_data(field, rowNode);
              for(int k=0; k<dim[2]; k++)
                for(int l=0; l<dim[3]; l++)
                  array(i,j,k,l) = entry[k*dim[3]+l]; //check this, is stride Correct?
            }
          }
          break;
        }
        }
      }
    }


#if defined(ALBANY_LCM)
    if(stkMeshStruct->getFieldContainer()->hasSphereVolumeField()) {
      sphereVolume[b].resize(buck.size());
    }
    if(stkMeshStruct->getFieldContainer()->hasLatticeOrientationField()) {
      latticeOrientation[b].resize(buck.size());
    }
#endif

    stk::mesh::Entity element = buck[0];
    int nodes_per_element = bulkData.num_nodes(element);
    for(auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end(); ++it) {
      int nComp = it->first.second;
      it->second.wsElNodeEqID_rawVec[b].resize(buck.size()*nodes_per_element*nComp);
      it->second.wsElNodeEqID[b].assign<ElemTag, NodeTag, CompTag>(
        it->second.wsElNodeEqID_rawVec[b].data(),(int)buck.size(),nodes_per_element,nComp);
      it->second.wsElNodeID_rawVec[b].resize(buck.size()*nodes_per_element);
      it->second.wsElNodeID[b].assign<ElemTag, NodeTag>(
        it->second.wsElNodeID_rawVec[b].data(),(int)buck.size(),nodes_per_element);
    }

    // i is the element index within bucket b
    for (std::size_t i=0; i < buck.size(); i++) {

      // Traverse all the elements in this bucket
      stk::mesh::Entity element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[gid(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[gid(element)].LID = i;

      stk::mesh::Entity const* node_rels = bulkData.begin_nodes(element);
      const int nodes_per_element = bulkData.num_nodes(element);

      wsElNodeEqID[b][i].resize(nodes_per_element);
      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);


      for(auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end(); ++it) {
        IDArray& wsElNodeEqID_array = it->second.wsElNodeEqID[b];
        GIDArray& wsElNodeID_array = it->second.wsElNodeID[b];
        int nComp = it->first.second;
        for (int j=0; j < nodes_per_element; j++) {
          stk::mesh::Entity node = node_rels[j];
          wsElNodeID_array((int)i,j) = gid(node);
          for (int k=0; k < nComp; k++) {
            const GO node_gid = it->second.overlap_dofManager.getGlobalDOF(bulkData.identifier(node)-1, k);
            const int node_lid = it->second.overlap_map->getLocalElement(node_gid);
            wsElNodeEqID_array((int)i,j,k) = node_lid;
          }
        }
      }

#if defined(ALBANY_LCM)
      if(stkMeshStruct->getFieldContainer()->hasSphereVolumeField() && nodes_per_element == 1){
	double* volumeTemp = stk::mesh::field_data(*sphereVolume_field, element);
	if(volumeTemp){
	  sphereVolume[b][i] = volumeTemp[0];
	}
      }
      if(stkMeshStruct->getFieldContainer()->hasLatticeOrientationField()){
        latticeOrientation[b][i] = static_cast<double*>( stk::mesh::field_data(*latticeOrientation_field, element) );
      }
#endif

      // loop over local nodes
      DOFsStruct& dofs_struct = mapOfDOFsStructs[make_pair(std::string(""),neq)];
      GIDArray& node_array = dofs_struct.wsElNodeID[b];
      IDArray& node_eq_array = dofs_struct.wsElNodeEqID[b];
      for (int j=0; j < nodes_per_element; j++) {
        const stk::mesh::Entity rowNode = node_rels[j];
        const GO node_gid = gid(rowNode);
        const LO node_lid = overlap_node_mapT->getLocalElement(node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
         "STK1D_Disc: node_lid out of range " << node_lid << std::endl);
        coords[b][i][j] = stk::mesh::field_data(*coordinates_field, rowNode);

        wsElNodeID[b][i][j] = node_array((int)i,j);

        wsElNodeEqID[b][i][j].resize(neq);
        for (int eq=0; eq < neq; eq++)
          wsElNodeEqID[b][i][j][eq] = node_eq_array((int)i,j,eq);
      }
/*
      for (int j=0; j < nodes_per_element; j++) {
        const stk::mesh::Entity rowNode = node_rels[j];
        const GO node_gid = gid(rowNode);
        const LO node_lid = overlap_node_mapT->getLocalElement(node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(node_lid<0, std::logic_error,
         "STK1D_Disc: node_lid out of range " << node_lid << std::endl);
        coords[b][i][j] = stk::mesh::field_data(*coordinates_field, rowNode);
        wsElNodeID[b][i][j] = node_gid;

        wsElNodeEqID[b][i][j].resize(neq);
        for (std::size_t eq=0; eq < neq; eq++)
          wsElNodeEqID[b][i][j][eq] = getOverlapDOF(node_lid,eq);
      }
*/
    }
  }
//Kopy workset to the Kokkos data
 //wsElNodeEqID_kokkos=Kokkos::View<int****, PHX::Device>("wsElNodeEqID_kokkos",numBuckets,wsElNodeEqID[0].size(),wsElNodeEqID[0][0].size(), neq);


 for (int d=0; d<stkMeshStruct->numDim; d++) {
  if (stkMeshStruct->PBCStruct.periodic[d]) {
    for (int b=0; b < numBuckets; b++) {
      for (std::size_t i=0; i < buckets[b]->size(); i++) {
        int nodes_per_element = buckets[b]->num_nodes(i);
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
                    StateArray::iterator sHeight = stateArrays.elemStateArrays[b].find("surface_height");
                    if(sHeight != stateArrays.elemStateArrays[b].end())
                      sHeight->second(int(i),j) -= stkMeshStruct->PBCStruct.scale[d]*tan(alpha);
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

  typedef Albany::AbstractSTKFieldContainer::ScalarValueState ScalarValueState;
  typedef Albany::AbstractSTKFieldContainer::QPScalarState QPScalarState;
  typedef Albany::AbstractSTKFieldContainer::QPVectorState QPVectorState;
  typedef Albany::AbstractSTKFieldContainer::QPTensorState QPTensorState;
  typedef Albany::AbstractSTKFieldContainer::QPTensor3State QPTensor3State;

  typedef Albany::AbstractSTKFieldContainer::ScalarState ScalarState;
  typedef Albany::AbstractSTKFieldContainer::VectorState VectorState;
  typedef Albany::AbstractSTKFieldContainer::TensorState TensorState;

  // Pull out pointers to shards::Arrays for every bucket, for every state
  // Code is data-type dependent

  Albany::AbstractSTKFieldContainer& container = *stkMeshStruct->getFieldContainer();

  ScalarValueState& scalarValue_states = container.getScalarValueStates();
  ScalarState& cell_scalar_states      = container.getCellScalarStates();
  VectorState& cell_vector_states      = container.getCellVectorStates();
  TensorState& cell_tensor_states      = container.getCellTensorStates();
  QPScalarState& qpscalar_states       = container.getQPScalarStates();
  QPVectorState& qpvector_states       = container.getQPVectorStates();
  QPTensorState& qptensor_states       = container.getQPTensorStates();
  QPTensor3State& qptensor3_states     = container.getQPTensor3States();
  std::map<std::string, double>& time  = container.getTime();

  for (std::size_t b=0; b < buckets.size(); b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    for (auto css = cell_scalar_states.begin(); css != cell_scalar_states.end(); ++css){
      BucketArray<Albany::AbstractSTKFieldContainer::ScalarFieldType> array(**css, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " SFT dim[1]: " << array.dimension(1) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*css)->name()] = ar;
    }
    for (auto cvs = cell_vector_states.begin(); cvs != cell_vector_states.end(); ++cvs){
      BucketArray<Albany::AbstractSTKFieldContainer::VectorFieldType> array(**cvs, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " VFT dim[2]: " << array.dimension(2) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*cvs)->name()] = ar;
    }
    for (auto cts = cell_tensor_states.begin(); cts != cell_tensor_states.end(); ++cts){
      BucketArray<Albany::AbstractSTKFieldContainer::TensorFieldType> array(**cts, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " TFT dim[3]: " << array.dimension(3) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*cts)->name()] = ar;
    }
    for (auto qpss = qpscalar_states.begin(); qpss != qpscalar_states.end(); ++qpss){
      BucketArray<Albany::AbstractSTKFieldContainer::QPScalarFieldType> array(**qpss, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " QPSFT dim[1]: " << array.dimension(1) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*qpss)->name()] = ar;
    }
    for (auto qpvs = qpvector_states.begin(); qpvs != qpvector_states.end(); ++qpvs){
      BucketArray<Albany::AbstractSTKFieldContainer::QPVectorFieldType> array(**qpvs, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " QPVFT dim[2]: " << array.dimension(2) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*qpvs)->name()] = ar;
    }
    for (auto qpts = qptensor_states.begin(); qpts != qptensor_states.end(); ++qpts){
      BucketArray<Albany::AbstractSTKFieldContainer::QPTensorFieldType> array(**qpts, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " QPTFT dim[3]: " << array.dimension(3) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*qpts)->name()] = ar;
    }
    for (auto qpts = qptensor3_states.begin(); qpts != qptensor3_states.end(); ++qpts){
      BucketArray<Albany::AbstractSTKFieldContainer::QPTensor3FieldType> array(**qpts, buck);
//Debug
//std::cout << "Buck.size(): " << buck.size() << " QPT3FT dim[4]: " << array.dimension(4) << std::endl;
      MDArray ar = array;
      stateArrays.elemStateArrays[b][(*qpts)->name()] = ar;
    }
//    for (ScalarValueState::iterator svs = scalarValue_states.begin();
//              svs != scalarValue_states.end(); ++svs){
    for (int i = 0; i < scalarValue_states.size(); i++){
      const int size = 1;
      shards::Array<double, shards::NaturalOrder, Cell> array(&time[*scalarValue_states[i]], size);
      MDArray ar = array;
//Debug
//std::cout << "Buck.size(): " << buck.size() << " SVState dim[0]: " << array.dimension(0) << std::endl;
//std::cout << "SV Name: " << *svs << " address : " << &array << std::endl;
      stateArrays.elemStateArrays[b][*scalarValue_states[i]] = ar;
    }
  }

// Process node data sets if present

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base) &&
      stkMeshStruct->nodal_data_base->isNodeDataPresent()) {
    Teuchos::RCP<Albany::NodeFieldContainer> node_states = stkMeshStruct->nodal_data_base->getNodeContainer();

    stk::mesh::BucketVector const& node_buckets = bulkData.get_buckets( stk::topology::NODE_RANK, select_owned_in_part );

    const size_t numNodeBuckets = node_buckets.size();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);
    for (std::size_t b=0; b < numNodeBuckets; b++) {
      stk::mesh::Bucket& buck = *node_buckets[b];
      for (Albany::NodeFieldContainer::iterator nfs = node_states->begin();
                nfs != node_states->end(); ++nfs){
        stateArrays.nodeStateArrays[b][(*nfs).first] =
             Teuchos::rcp_dynamic_cast<Albany::AbstractSTKNodeFieldContainer>((*nfs).second)->getMDA(buck);
      }
    }
  }
}

void Albany::STKDiscretization::computeSideSets(){

  // Clean up existing sideset structure if remeshing

  for(int i = 0; i < sideSets.size(); i++)
    sideSets[i].clear(); // empty the ith map

  const stk::mesh::EntityRank element_rank = stk::topology::ELEMENT_RANK;

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

    std::vector< stk::mesh::Entity > sides ;
    stk::mesh::get_selected_entities( select_owned_in_sspart , // sides local to this processor
              bulkData.buckets( metaData.side_rank() ) ,
              sides ); // store the result in "sides"

    *out << "STKDisc: sideset "<< ss->first <<" has size " << sides.size() << "  on Proc 0." << std::endl;

    // loop over the sides to see what they are, then fill in the data holder
    // for side set options, look at $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp

    for (std::size_t localSideID=0; localSideID < sides.size(); localSideID++) {

      stk::mesh::Entity sidee = sides[localSideID];

      TEUCHOS_TEST_FOR_EXCEPTION(bulkData.num_elements(sidee) != 1, std::logic_error,
                                 "STKDisc: cannot figure out side set topology for side set " << ss->first << std::endl);

      stk::mesh::Entity elem = bulkData.begin_elements(sidee)[0];

      // containing the side. Note that if the side is internal, it will show up twice in the
      // element list, once for each element that contains it.

      SideStruct sStruct;

      // Save side (global id)
      sStruct.side_GID = bulkData.identifier(sidee)-1;

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
Albany::STKDiscretization::determine_local_side_id( const stk::mesh::Entity elem , stk::mesh::Entity side ) {

  using namespace stk;

  stk::topology elem_top = bulkData.bucket(elem).topology();

  const unsigned num_elem_nodes = bulkData.num_nodes(elem);
  const unsigned num_side_nodes = bulkData.num_nodes(side);

  stk::mesh::Entity const* elem_nodes = bulkData.begin_nodes(elem);
  stk::mesh::Entity const* side_nodes = bulkData.begin_nodes(side);

  const stk::topology::rank_t side_rank = metaData.side_rank();

  int side_id = -1 ;

  if(num_elem_nodes == 0 || num_side_nodes == 0){ // Node relations are not present, look at elem->face

    const unsigned num_sides = bulkData.num_connectivity(elem, side_rank);
    stk::mesh::Entity const* elem_sides = bulkData.begin(elem, side_rank);

    for ( unsigned i = 0 ; i < num_sides ; ++i ) {

      const stk::mesh::Entity elem_side = elem_sides[i];

      if (bulkData.identifier(elem_side) == bulkData.identifier(side)){ // Found the local side in the element

         side_id = static_cast<int>(i);

         return side_id;
      }

    }

    if ( side_id < 0 ) {
      std::ostringstream msg ;
      msg << "determine_local_side_id( " ;
      msg << elem_top.name() ;
      msg << " , Element[ " ;
      msg << bulkData.identifier(elem);
      msg << " ]{" ;
      for ( unsigned i = 0 ; i < num_sides ; ++i ) {
        msg << " " << bulkData.identifier(elem_sides[i]);
      }
      msg << " } , Side[ " ;
      msg << bulkData.identifier(side);
      msg << " ] ) FAILED" ;
      throw std::runtime_error( msg.str() );
    }

  }
  else { // Conventional elem->node - side->node connectivity present

    std::vector<unsigned> side_map;
    for ( unsigned i = 0 ; side_id == -1 && i < elem_top.num_sides() ; ++i ) {
      stk::topology side_top    = elem_top.side_topology(i);
      side_map.clear();
      elem_top.side_node_ordinals(i, std::back_inserter(side_map));

      if ( num_side_nodes == side_top.num_nodes() ) {

        side_id = i ;

        for ( unsigned j = 0 ;
              side_id == static_cast<int>(i) && j < side_top.num_nodes() ; ++j ) {

          stk::mesh::Entity elem_node = elem_nodes[ side_map[j] ];

          bool found = false ;

          for ( unsigned k = 0 ; ! found && k < side_top.num_nodes() ; ++k ) {
            found = elem_node == side_nodes[k];
          }

          if ( ! found ) { side_id = -1 ; }
        }
      }
    }

    if ( side_id < 0 ) {
      std::ostringstream msg ;
      msg << "determine_local_side_id( " ;
      msg << elem_top.name() ;
      msg << " , Element[ " ;
      msg << bulkData.identifier(elem);
      msg << " ]{" ;
      for ( unsigned i = 0 ; i < num_elem_nodes ; ++i ) {
        msg << " " << bulkData.identifier(elem_nodes[i]);
      }
      msg << " } , Side[ " ;
      msg << bulkData.identifier(side);
      msg << " ]{" ;
      for ( unsigned i = 0 ; i < num_side_nodes ; ++i ) {
        msg << " " << bulkData.identifier(side_nodes[i]);
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
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();

  while ( ns != stkMeshStruct->nsPartVec.end() ) { // Iterate over Node Sets
    // Get all owned nodes in this node set
    stk::mesh::Selector select_owned_in_nspart =
      stk::mesh::Selector( *(ns->second) ) &
      stk::mesh::Selector( metaData.locally_owned_part() );

    std::vector< stk::mesh::Entity > nodes ;
    stk::mesh::get_selected_entities( select_owned_in_nspart ,
              bulkData.buckets( stk::topology::NODE_RANK ) ,
              nodes );

    nodeSets[ns->first].resize(nodes.size());
    nodeSetGIDs[ns->first].resize(nodes.size());
    nodeSetCoords[ns->first].resize(nodes.size());
//    nodeSetIDs.push_back(ns->first); // Grab string ID
    *out << "STKDisc: nodeset "<< ns->first <<" has size " << nodes.size() << "  on Proc 0." << std::endl;
    for (std::size_t i=0; i < nodes.size(); i++) {
      GO node_gid = gid(nodes[i]);
      int node_lid = node_mapT->getLocalElement(node_gid);
      nodeSetGIDs[ns->first][i] = node_gid;
      nodeSets[ns->first][i].resize(neq);
      for (std::size_t eq=0; eq < neq; eq++)  nodeSets[ns->first][i][eq] = getOwnedDOF(node_lid,eq);
      nodeSetCoords[ns->first][i] = stk::mesh::field_data(*coordinates_field, nodes[i]);
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

    mesh_data = Teuchos::rcp(new stk::io::StkMeshIoBroker(Albany::getMpiCommFromTeuchosComm(commT)));
    mesh_data->set_bulk_data(bulkData);
    outputFileIdx = mesh_data->create_output_mesh(str, stk::io::WRITE_RESULTS);

    // Adding mesh global variables
    for (auto& it : stkMeshStruct->getFieldContainer()->getMeshVectorStates())
    {
      boost::any mvs = it.second;
      mesh_data->add_global (outputFileIdx, it.first, mvs, stk::util::ParameterType::DOUBLEVECTOR);
    }
    for (auto& it : stkMeshStruct->getFieldContainer()->getMeshScalarIntegerStates())
    {
      boost::any mvs = it.second;
      mesh_data->add_global (outputFileIdx, it.first, mvs, stk::util::ParameterType::INTEGER);
    }

    const stk::mesh::FieldVector &fields = mesh_data->meta_data().get_fields();
    for (size_t i=0; i < fields.size(); i++) {
      // Hacky, but doesn't appear to be a way to query if a field is already
      // going to be output.
      try {
        mesh_data->add_field(outputFileIdx, *fields[i]);
      }
      catch (std::runtime_error const&) { }
    }
  }
#else
  if (stkMeshStruct->exoOutput)
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n" << std::endl;

#endif
}

namespace {
  const std::vector<double> spherical_to_cart(const std::pair<double, double> & sphere){
    const double radius_of_earth = 1;
    std::vector<double> cart(3);

    cart[0] = radius_of_earth*std::cos(sphere.first)*std::cos(sphere.second);
    cart[1] = radius_of_earth*std::cos(sphere.first)*std::sin(sphere.second);
    cart[2] = radius_of_earth*std::sin(sphere.first);

    return cart;
  }
  double distance (const double* x, const double* y) {
    const double d = std::sqrt((x[0]-y[0])*(x[0]-y[0]) +
                               (x[1]-y[1])*(x[1]-y[1]) +
                               (x[2]-y[2])*(x[2]-y[2]));
    return d;
  }
  double distance (const std::vector<double> &x, const std::vector<double> &y) {
    const double d = std::sqrt((x[0]-y[0])*(x[0]-y[0]) +
                               (x[1]-y[1])*(x[1]-y[1]) +
                               (x[2]-y[2])*(x[2]-y[2]));
    return d;
  }

  bool point_inside(const Teuchos::ArrayRCP<double*> &coords,
                    const std::vector<double>        &sphere_xyz) {
    // first check if point is near the element:
    const double  tol_inside = 1e-12;
    const double elem_diam = std::max(::distance(coords[0],coords[2]), ::distance(coords[1],coords[3]));
    std::vector<double> center(3,0);
    for (unsigned i=0; i<4; ++i)
      for (unsigned j=0; j<3; ++j) center[j] += coords[i][j];
    for (unsigned j=0; j<3; ++j) center[j] /= 4;
    bool inside = true;

    if ( ::distance(&center[0],&sphere_xyz[0]) > 1.0*elem_diam ) inside = false;

    unsigned j=3;
    for (unsigned i=0; i<4 && inside; ++i) {
      std::vector<double> cross(3);
      // outward normal to plane containing j->i edge:  corner(i) x corner(j)
      // sphere dot (corner(i) x corner(j) ) = negative if inside
      cross[0]=  coords[i][1]*coords[j][2] - coords[i][2]*coords[j][1];
      cross[1]=-(coords[i][0]*coords[j][2] - coords[i][2]*coords[j][0]);
      cross[2]=  coords[i][0]*coords[j][1] - coords[i][1]*coords[j][0];
      j = i;
      const double dotprod = cross[0]*sphere_xyz[0] +
                             cross[1]*sphere_xyz[1] +
                             cross[2]*sphere_xyz[2];

      // dot product is proportional to elem_diam. positive means outside,
      // but allow machine precision tolorence:
      if (tol_inside*elem_diam < dotprod) inside = false;
    }
    return inside;
  }


  const Teuchos::RCP<Intrepid2::Basis<double, Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> > >
  Basis(const int C)
  {
    // Static types
    typedef Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> Field_t;
    typedef Intrepid2::Basis< double, Field_t > Basis_t;
    static const Teuchos::RCP< Basis_t > HGRAD_Basis_4 =
      Teuchos::rcp( new Intrepid2::Basis_HGRAD_QUAD_C1_FEM< double, Field_t >() );
    static const Teuchos::RCP< Basis_t > HGRAD_Basis_9 =
      Teuchos::rcp( new Intrepid2::Basis_HGRAD_QUAD_C2_FEM< double, Field_t >() );

    // Check for valid value of C
    int deg = (int) std::sqrt((double)C);
    TEUCHOS_TEST_FOR_EXCEPTION(
      deg*deg != C || deg < 2,
      std::logic_error,
      " Albany_STKDiscretization Error Basis not perfect "
      "square > 1" << std::endl);

    // Quick return for linear or quad
    if (C == 4) return HGRAD_Basis_4;
    if (C == 9) return HGRAD_Basis_9;

    // Spectral bases
    return Teuchos::rcp(
      new Intrepid2::Basis_HGRAD_QUAD_Cn_FEM< double, Field_t >(
        deg, Intrepid2::POINTTYPE_SPECTRAL) );
  }

  double value(const std::vector<double> &soln,
               const std::pair<double, double> &ref) {

    const int C = soln.size();
    const Teuchos::RCP<Intrepid2::Basis<double, Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> > > HGRAD_Basis = Basis(C);

    const int numPoints        = 1;
    Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> basisVals (C, numPoints);
    Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> tempPoints(numPoints, 2);
    tempPoints(0,0) = ref.first;
    tempPoints(0,1) = ref.second;

    HGRAD_Basis->getValues(basisVals, tempPoints, Intrepid2::OPERATOR_VALUE);

    double x = 0;
    for (unsigned j=0; j<C; ++j) x += soln[j] * basisVals(j,0);
    return x;
  }

  void value(double x[3],
             const Teuchos::ArrayRCP<double*> &coords,
             const std::pair<double, double> &ref){

    const int C = coords.size();
    const Teuchos::RCP<Intrepid2::Basis<double, Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> > > HGRAD_Basis = Basis(C);

    const int numPoints        = 1;
    Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> basisVals (C, numPoints);
    Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> tempPoints(numPoints, 2);
    tempPoints(0,0) = ref.first;
    tempPoints(0,1) = ref.second;

    HGRAD_Basis->getValues(basisVals, tempPoints, Intrepid2::OPERATOR_VALUE);

    for (unsigned i=0; i<3; ++i) x[i] = 0;
    for (unsigned i=0; i<3; ++i)
      for (unsigned j=0; j<C; ++j)
        x[i] += coords[j][i] * basisVals(j,0);
  }

  void grad(double x[3][2],
             const Teuchos::ArrayRCP<double*> &coords,
             const std::pair<double, double> &ref){

    const int C = coords.size();
    const Teuchos::RCP<Intrepid2::Basis<double, Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> > > HGRAD_Basis = Basis(C);

    const int numPoints        = 1;
    Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> basisGrad (C, numPoints, 2);
    Intrepid2::FieldContainer_Kokkos<double, PHX::Layout, PHX::Device> tempPoints(numPoints, 2);
    tempPoints(0,0) = ref.first;
    tempPoints(0,1) = ref.second;

    HGRAD_Basis->getValues(basisGrad, tempPoints, Intrepid2::OPERATOR_GRAD);

    for (unsigned i=0; i<3; ++i) x[i][0] = x[i][1] = 0;
    for (unsigned i=0; i<3; ++i)
      for (unsigned j=0; j<C; ++j) {
        x[i][0] += coords[j][i] * basisGrad(j,0,0);
        x[i][1] += coords[j][i] * basisGrad(j,0,1);
      }
  }

  std::pair<double, double>  ref2sphere(const Teuchos::ArrayRCP<double*> &coords,
                                        const std::pair<double, double> &ref) {

    static const double DIST_THRESHOLD= 1.0e-9;

    double x[3];
    value(x,coords,ref);

    const double r = std::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);

    for (unsigned i=0; i<3; ++i) x[i] /= r;

    std::pair<double, double> sphere(std::asin(x[2]), std::atan2(x[1],x[0]));

    // ==========================================================
    // enforce three facts:
    //
    // 1) lon at poles is defined to be zero
    //
    // 2) Grid points must be separated by about .01 Meter (on earth)
    //   from pole to be considered "not the pole".
    //
    // 3) range of lon is { 0<= lon < 2*PI }
    //
    // ==========================================================

    if (std::abs(std::abs(sphere.first)-pi/2) < DIST_THRESHOLD) sphere.second = 0;
    else if (sphere.second < 0) sphere.second += 2*pi;

    return sphere;
  }

  void Dmap(const Teuchos::ArrayRCP<double*> &coords,
            const std::pair<double, double>  &sphere,
            const std::pair<double, double>  &ref,
            double D[][2]) {

    const double th     = sphere.first;
    const double lam    = sphere.second;
    const double sinlam = std::sin(lam);
    const double sinth  = std::sin(th);
    const double coslam = std::cos(lam);
    const double costh  = std::cos(th);

    const double D1[2][3] = {{-sinlam, coslam, 0},
                             {      0,      0, 1}};

    const double D2[3][3] = {{ sinlam*sinlam*costh*costh+sinth*sinth, -sinlam*coslam*costh*costh,             -coslam*sinth*costh},
                             {-sinlam*coslam*costh*costh,              coslam*coslam*costh*costh+sinth*sinth, -sinlam*sinth*costh},
                             {-coslam*sinth,                          -sinlam*sinth,                                        costh}};

    double D3[3][2] = {0};
    grad(D3,coords,ref);

    double D4[3][2] = {0};
    for (unsigned i=0; i<3; ++i)
      for (unsigned j=0; j<2; ++j)
        for (unsigned k=0; k<3; ++k)
           D4[i][j] += D2[i][k] * D3[k][j];

    for (unsigned i=0; i<2; ++i)
      for (unsigned j=0; j<2; ++j) D[i][j] = 0;

    for (unsigned i=0; i<2; ++i)
      for (unsigned j=0; j<2; ++j)
        for (unsigned k=0; k<3; ++k)
          D[i][j] += D1[i][k] * D4[k][j];

  }

  std::pair<double, double> parametric_coordinates(const Teuchos::ArrayRCP<double*> &coords,
                                                   const std::pair<double, double>  &sphere) {

    static const double tol_sq = 1e-26;
    static const unsigned MAX_NR_ITER = 10;
    double costh = std::cos(sphere.first);
    double D[2][2], Dinv[2][2];
    double resa = 1;
    double resb = 1;
    std::pair<double, double> ref(0,0); // initial guess is center of element.

    for (unsigned i=0; i<MAX_NR_ITER && tol_sq < (costh*resb*resb + resa*resa) ; ++i) {
      const std::pair<double, double> sph = ref2sphere(coords,ref);
      resa = sph.first  - sphere.first;
      resb = sph.second - sphere.second;

      if (resb >  pi) resb -= 2*pi;
      if (resb < -pi) resb += 2*pi;

      Dmap(coords, sph, ref, D);
      const double detD = D[0][0]*D[1][1] - D[0][1]*D[1][0];
      Dinv[0][0] =  D[1][1]/detD;
      Dinv[0][1] = -D[0][1]/detD;
      Dinv[1][0] = -D[1][0]/detD;
      Dinv[1][1] =  D[0][0]/detD;

      const std::pair<double, double> del( Dinv[0][0]*costh*resb + Dinv[0][1]*resa,
                                           Dinv[1][0]*costh*resb + Dinv[1][1]*resa);
      ref.first  -= del.first;
      ref.second -= del.second;
    }
    return ref;
  }

  const std::pair<bool,std::pair<unsigned, unsigned> >point_in_element(const std::pair<double, double> &sphere,
      const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& coords,
      std::pair<double, double> &parametric) {
    const std::vector<double> sphere_xyz = spherical_to_cart(sphere);
    std::pair<bool,std::pair<unsigned, unsigned> > element(false,std::pair<unsigned, unsigned>(0,0));
    for (unsigned i=0; i<coords.size() && !element.first; ++i) {
      for (unsigned j=0; j<coords[i].size() && !element.first; ++j) {
        const bool found =  point_inside(coords[i][j], sphere_xyz);
        if (found) {
          parametric = parametric_coordinates(coords[i][j], sphere);
          if (parametric.first  < -1) parametric.first  = -1;
          if (parametric.second < -1) parametric.second = -1;
          if (1 < parametric.first  ) parametric.first  =  1;
          if (1 < parametric.second ) parametric.second =  1;
          element.first         = true;
          element.second.first  = i;
          element.second.second = j;
        }
      }
    }
    return element;
  }

  void setup_latlon_interp(
    const unsigned nlat, const double nlon,
    const Albany::WorksetArray<Teuchos::ArrayRCP<Teuchos::ArrayRCP<double*> > >::type& coords,
    Albany::WorksetArray<Teuchos::ArrayRCP<std::vector<Albany::STKDiscretization::interp> > >::type& interpdata,
    const Teuchos::RCP<const Teuchos_Comm> commT) {

    double err=0;
    const long long unsigned rank = commT->getRank();
    std::vector<double> lat(nlat);
    std::vector<double> lon(nlon);

    unsigned count=0;
    for (unsigned i=0; i<nlat; ++i) lat[i] = -pi/2 + i*pi/(nlat-1);
    for (unsigned j=0; j<nlon; ++j) lon[j] =       2*j*pi/nlon;
    for (unsigned i=0; i<nlat; ++i) {
      for (unsigned j=0; j<nlon; ++j) {
        const std::pair<double, double> sphere(lat[i],lon[j]);
        std::pair<double, double> paramtric;
        const std::pair<bool,std::pair<unsigned, unsigned> >element = point_in_element(sphere, coords, paramtric);
        if (element.first) {
          // compute error: map 'cart' back to sphere and compare with original
          // interpolation point:
          const unsigned b = element.second.first ;
          const unsigned e = element.second.second;
          const std::vector<double> sphere2_xyz = spherical_to_cart(ref2sphere(coords[b][e], paramtric));
          const std::vector<double> sphere_xyz  = spherical_to_cart(sphere);
          err = std::max(err, ::distance(&sphere2_xyz[0],&sphere_xyz[0]));
          Albany::STKDiscretization::interp interp;
          interp.parametric_coords = paramtric;
          interp.latitude_longitude = std::pair<unsigned,unsigned>(i,j);
          interpdata[b][e].push_back(interp);
          ++count;
        }
      }
      if (!rank && (!(i%64) || i==nlat-1)) std::cout<< "Finished Latitude "<<i<<" of "<<nlat<<std::endl;
    }
    if (!rank) std::cout<<"Max interpolation point search error: "<<err<<std::endl;
  }
}

int Albany::STKDiscretization::processNetCDFOutputRequestT(const Tpetra_Vector& solution_fieldT) {
#ifdef ALBANY_SEACAS
//IK, 10/13/14: need to implement!
#endif
  return 0;
}

int Albany::STKDiscretization::processNetCDFOutputRequestMV(const Tpetra_MultiVector& solution_fieldT) {
#ifdef ALBANY_SEACAS
//IK, 10/13/14: need to implement!
#endif
  return 0;
}

void Albany::STKDiscretization::setupNetCDFOutput()
{
  const long long unsigned rank = commT->getRank();
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->cdfOutput) {
    outputInterval = 0;
    const unsigned nlat = stkMeshStruct->nLat;
    const unsigned nlon = stkMeshStruct->nLon;


    std::string str = stkMeshStruct->cdfOutFile;

    interpolateData.resize(coords.size());
    for (int b=0; b < coords.size(); b++) interpolateData[b].resize(coords[b].size());

    setup_latlon_interp(nlat, nlon, coords, interpolateData, commT);

    const std::string name = stkMeshStruct->cdfOutFile;
    netCDFp=0;
    netCDFOutputRequest=0;


#ifdef ALBANY_PAR_NETCDF
    MPI_Comm theMPIComm = Albany::getMpiCommFromTeuchosComm(commT);
    MPI_Info info;
    MPI_Info_create(&info);
    if (const int ierr = nc_create_par (name.c_str(), NC_NETCDF4 | NC_MPIIO | NC_CLOBBER | NC_64BIT_OFFSET, theMPIComm, info, &netCDFp))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_create_par returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
    MPI_Info_free(&info);
#else
    if (!rank)
    if (const int ierr = nc_create (name.c_str(), NC_CLOBBER | NC_SHARE | NC_64BIT_OFFSET | NC_CLASSIC_MODEL, &netCDFp))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_create returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
#endif

    const size_t nlev = 1;
    const char *dimnames[] = {"time","lev","lat","lon"};
    const size_t  dimlen[] = {NC_UNLIMITED, nlev, nlat, nlon};
    int dimID[4]={0,0,0,0};

    for (unsigned i=0; i<4; ++i) {
      if (netCDFp)
      if (const int ierr = nc_def_dim (netCDFp,  dimnames[i], dimlen[i], &dimID[i]))
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "nc_def_dim returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
    }
    varSolns.resize(neq,0);

    for (unsigned n=0; n<neq; ++n) {
      std::ostringstream var;
      var <<"variable_"<<n;
      const char *field_name = var.str().c_str();
      if (netCDFp)
      if (const int ierr = nc_def_var (netCDFp,  field_name, NC_DOUBLE, 4, dimID, &varSolns[n]))
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "nc_def_var "<<field_name<<" returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);

      const double fillVal = -9999.0;
      if (netCDFp)
      if (const int ierr = nc_put_att (netCDFp,  varSolns[n], "FillValue", NC_DOUBLE, 1, &fillVal))
        TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
          "nc_put_att FillValue returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
    }

    const char lat_name[] = "latitude";
    const char lat_unit[] = "degrees_north";
    const char lon_name[] = "longitude";
    const char lon_unit[] = "degrees_east";
    int latVarID=0;
      if (netCDFp)
    if (const int ierr = nc_def_var (netCDFp,  "lat", NC_DOUBLE, 1, &dimID[2], &latVarID))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_def_var lat returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
      if (netCDFp)
    if (const int ierr = nc_put_att_text (netCDFp,  latVarID, "long_name", sizeof(lat_name), lat_name))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_att_text "<<lat_name<<" returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
      if (netCDFp)
    if (const int ierr = nc_put_att_text (netCDFp,  latVarID, "units", sizeof(lat_unit), lat_unit))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_att_text "<<lat_unit<<" returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);

    int lonVarID=0;
      if (netCDFp)
    if (const int ierr = nc_def_var (netCDFp,  "lon", NC_DOUBLE, 1, &dimID[3], &lonVarID))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_def_var lon returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
      if (netCDFp)
    if (const int ierr = nc_put_att_text (netCDFp,  lonVarID, "long_name", sizeof(lon_name), lon_name))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_att_text "<<lon_name<<" returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
      if (netCDFp)
    if (const int ierr = nc_put_att_text (netCDFp,  lonVarID, "units", sizeof(lon_unit), lon_unit))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_att_text "<<lon_unit<<" returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);

    const char history[]="Created by Albany";
      if (netCDFp)
    if (const int ierr = nc_put_att_text (netCDFp,  NC_GLOBAL, "history", sizeof(history), history))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_att_text "<<history<<" returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);

      if (netCDFp)
    if (const int ierr = nc_enddef (netCDFp))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_enddef returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);

    std::vector<double> deglon(nlon);
    std::vector<double> deglat(nlat);
    for (unsigned i=0; i<nlon; ++i) deglon[i] =((      2*i*pi/nlon) *   (180/pi)) - 180;
    for (unsigned i=0; i<nlat; ++i) deglat[i] = (-pi/2 + i*pi/(nlat-1))*(180/pi);


      if (netCDFp)
    if (const int ierr = nc_put_var (netCDFp, lonVarID, &deglon[0]))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_var lon returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);
      if (netCDFp)
    if (const int ierr = nc_put_var (netCDFp, latVarID, &deglat[0]))
      TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
        "nc_put_var lat returned error code "<<ierr<<" - "<<nc_strerror(ierr)<<std::endl);

  }
#else
  if (stkMeshStruct->cdfOutput)
    *out << "\nWARNING: NetCDF output requested but SEACAS not compiled in:"
         << " disabling NetCDF output \n" << std::endl;
  stkMeshStruct->cdfOutput = false;

#endif
}

void Albany::STKDiscretization::reNameExodusOutput(std::string& filename)
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput && !mesh_data.is_null()) {
    // Delete the mesh data object and recreate it
    mesh_data = Teuchos::null;

    stkMeshStruct->exoOutFile = filename;

    // reset reference value for monotonic time function call as we are writing to a new file
    previous_time_label = -1.0e32;
  }
#else
  if (stkMeshStruct->exoOutput)
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n" << std::endl;

#endif
}

// Convert the stk mesh on this processor to a nodal graph.
//todo Dev/tested on linear elements only.
void Albany::STKDiscretization::meshToGraph () {
  if (Teuchos::is_null(stkMeshStruct->nodal_data_base)) return;
  if (!stkMeshStruct->nodal_data_base->isNodeDataPresent()) return;

  // Set up the CRS graph used for solution transfer and projection mass
  // matrices. Assume the Crs row size is 27, which is the maximum number
  // required for first-order hexahedral elements.
  nodalGraph = Teuchos::rcp(new Tpetra_CrsGraph(overlap_node_mapT, 27));

  // Elements that surround a given node, in the form of Entity's.
  std::vector<std::vector<stk::mesh::Entity> > sur_elem;
  // numOverlapNodes are the total # of nodes seen by this pe
  // numOwnedNodes are the total # of nodes owned by this pe
  sur_elem.resize(numOverlapNodes);

  // Get the elements owned by the current processor
  const stk::mesh::Selector select_owned_in_part =
    stk::mesh::Selector( metaData.universal_part() ) &
    stk::mesh::Selector( metaData.locally_owned_part() );

  const stk::mesh::BucketVector& buckets = bulkData.get_buckets(
    stk::topology::ELEMENT_RANK, select_owned_in_part);

  for (int b = 0; b < buckets.size(); ++b) {
    const stk::mesh::Bucket& cells = *buckets[b];
    // Find the surrounding elements for each node owned by this processor.
    for (std::size_t ecnt = 0; ecnt < cells.size(); ecnt++) {
      const stk::mesh::Entity e = cells[ecnt];
      const stk::mesh::Entity* node_rels = bulkData.begin_nodes(e);
      const size_t num_node_rels = bulkData.num_nodes(e);

      // Loop over nodes within the element.
      for (std::size_t ncnt = 0; ncnt < num_node_rels; ++ncnt) {
        const stk::mesh::Entity rowNode = node_rels[ncnt];
        GO nodeGID = gid(rowNode);
        int nodeLID = overlap_node_mapT->getLocalElement(nodeGID);
        // In the case of degenerate elements, where a node can be entered into
        // the connect table twice, need to check to make sure that this element
        // is not already listed as surrounding this node.
        if (sur_elem[nodeLID].empty() || entity_in_list(e, sur_elem[nodeLID]) < 0)
          sur_elem[nodeLID].push_back(e);
      }
    }
  }

  std::size_t max_nsur = 0;
  for (std::size_t ncnt = 0; ncnt < numOverlapNodes; ncnt++) {
    if (sur_elem[ncnt].empty()) {
      TEUCHOS_TEST_FOR_EXCEPTION(
        true, std::logic_error,
        "Node = " << ncnt+1 << " has no elements" << std::endl);
    } else {
      std::size_t nsur = sur_elem[ncnt].size();
      if (nsur > max_nsur) max_nsur = nsur;
    }
  }

  // end find_surrnd_elems

  // find_adjacency

  // Note that the center node of a subgraph must be owned by this pe, but we
  // want all nodes in the overlap graph to be covered in the nodal graph.

  // loop over all the nodes owned by this PE
  for(std::size_t ncnt = 0; ncnt < numOverlapNodes; ncnt++) {
    Teuchos::Array<GO> adjacency;
    GO globalrow = overlap_node_mapT->getGlobalElement(ncnt);
    // loop over the elements surrounding node ncnt
    for(std::size_t ecnt = 0; ecnt < sur_elem[ncnt].size(); ecnt++) {
      const stk::mesh::Entity elem  = sur_elem[ncnt][ecnt];
      const stk::mesh::Entity* node_rels = bulkData.begin_nodes(elem);
      const size_t num_node_rels = bulkData.num_nodes(elem);
      std::size_t ws = elemGIDws[gid(elem)].ws;
      // loop over the nodes in the surrounding element elem
      for (std::size_t lnode = 0; lnode < num_node_rels; ++lnode) {
        const stk::mesh::Entity node_a = node_rels[lnode];
        // entry is the GID of each node
        GO entry = gid(node_a);
        // Every node in an element adjacent to node 'globalrow' is in this
        // graph.
        if (in_list(entry, adjacency) < 0) adjacency.push_back(entry);
      }
    }
    nodalGraph->insertGlobalIndices(globalrow, adjacency());
  }

  // end find_adjacency

  nodalGraph->fillComplete();
  // Pass the graph RCP to the nodal data block
  stkMeshStruct->nodal_data_base->updateNodalGraph(nodalGraph);
}

void
Albany::STKDiscretization::printVertexConnectivity(){

  if(Teuchos::is_null(nodalGraph)) return;

  for(std::size_t i = 0; i < numOverlapNodes; i++){

    GO globalvert = overlap_node_mapT->getGlobalElement(i);

    std::cout << "Center vert is : " << globalvert + 1 << std::endl;

    Teuchos::ArrayView<const GO> adj;

    nodalGraph->getGlobalRowView(globalvert, adj);

    for(std::size_t j = 0; j < adj.size(); j++)

      std::cout << "                  " << adj[j] + 1 << std::endl;

   }
}

void Albany::STKDiscretization::buildSideSetProjectors()
{
  // Note: the Global index of a node should be the same in both this and the side discretizations
  //       since the underlying STK entities should have the same ID
  Teuchos::RCP<const Tpetra_Map> ss_ov_mapT, ss_mapT;
  Teuchos::RCP<Tpetra_CrsGraph> graphP, ov_graphP;
  Teuchos::RCP<Tpetra_CrsMatrix> P, ov_P;
#ifdef ALBANY_EPETRA
  Teuchos::RCP<Epetra_CrsMatrix> P_E;
#endif

  Teuchos::Array<GO> cols(1);
  Teuchos::Array<ST> vals(1);
  vals[0] = 1.0;

  LO num_entries;
  Teuchos::ArrayView<const GO> ss_indices;
  stk::mesh::EntityRank SIDE_RANK = stkMeshStruct->metaData->side_rank();
  for (auto it : sideSetDiscretizationsSTK)
  {
    // Extract the discretization
    const std::string& sideSetName = it.first;
    const Albany::STKDiscretization& disc = *it.second;
    const Albany::AbstractSTKMeshStruct& ss_mesh = *disc.stkMeshStruct;

    // Get the maps
    ss_ov_mapT = disc.getOverlapMapT();
    ss_mapT    = disc.getMapT();

    // Extract the sides
    stk::mesh::Part& part = *stkMeshStruct->ssPartVec.find(it.first)->second;
    stk::mesh::Selector selector = stk::mesh::Selector(part) & stk::mesh::Selector(stkMeshStruct->metaData->locally_owned_part());
    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(selector, stkMeshStruct->bulkData->buckets(SIDE_RANK), sides);

    // The projector: first the overlapped...
    ov_graphP = Teuchos::rcp(new Tpetra_CrsGraph(ss_ov_mapT,1,Tpetra::StaticProfile));
    num_entries = ss_ov_mapT->getNodeNumElements();
    ss_indices = ss_ov_mapT->getNodeElementList();

    const std::map<GO,GO>& side_cell_map = sideToSideSetCellMap.at(it.first);
    const std::map<GO,std::vector<int>>& node_numeration_map = sideNodeNumerationMap.at(it.first);
    std::set<GO> processed_node;
    GO node_gid, ss_node_gid, side_gid, ss_cell_gid, globalDOF, ss_globalDOF;
    std::pair<std::set<GO>::iterator,bool> check;
    stk::mesh::Entity ss_cell;
    for (auto side : sides)
    {
      side_gid = gid(side);
      ss_cell_gid = side_cell_map.at(side_gid);
      ss_cell = ss_mesh.bulkData->get_entity(stk::topology::ELEM_RANK, ss_cell_gid+1);

      int num_side_nodes = stkMeshStruct->bulkData->num_nodes(side);
      const stk::mesh::Entity* side_nodes = stkMeshStruct->bulkData->begin_nodes(side);
      const stk::mesh::Entity* ss_cell_nodes = ss_mesh.bulkData->begin_nodes(ss_cell);
      for (int i(0); i<num_side_nodes; ++i)
      {
        node_gid = gid(side_nodes[i]);
        check = processed_node.insert(node_gid);
        if (check.second)
        {
          // This node was not processed before. Let's do it.
          ss_node_gid = disc.gid(ss_cell_nodes[node_numeration_map.at(side_gid)[i]]);

          for (int eq(0); eq<neq; ++eq)
          {
            cols[0] = getGlobalDOF(node_gid,eq);
            ov_graphP->insertGlobalIndices(disc.getGlobalDOF(ss_node_gid,eq),cols());
          }
        }
      }
    }

    ov_graphP->fillComplete (overlap_mapT,ss_ov_mapT);
    ov_P = Teuchos::rcp(new Tpetra_CrsMatrix(ov_graphP)); // This constructor creates matrix with static profile
    ov_P->setAllToScalar (1.0);
    ov_P->fillComplete ();
    ov_projectorsT[sideSetName] = ov_P;

    // ...then the non-overlapped
    graphP = Teuchos::rcp(new Tpetra_CrsGraph(ss_mapT,1,Tpetra::StaticProfile));
    processed_node.clear();
    LO bad = Teuchos::OrdinalTraits<LO>::invalid();
    for (auto side : sides)
    {
      side_gid = gid(side);
      ss_cell_gid = side_cell_map.at(side_gid);
      ss_cell = ss_mesh.bulkData->get_entity(stk::topology::ELEM_RANK, ss_cell_gid+1);

      int num_side_nodes = stkMeshStruct->bulkData->num_nodes(side);
      const stk::mesh::Entity* side_nodes = stkMeshStruct->bulkData->begin_nodes(side);
      const stk::mesh::Entity* ss_cell_nodes = ss_mesh.bulkData->begin_nodes(ss_cell);
      for (int i(0); i<num_side_nodes; ++i)
      {
        node_gid = gid(side_nodes[i]);
        if (node_mapT->getLocalElement(node_gid)==bad)
        {
          // This node is not in the non-overlapped map
          continue;
        }

        check = processed_node.insert(node_gid);
        if (check.second)
        {
          // This node was not processed before. Let's do it.
          ss_node_gid = disc.gid(ss_cell_nodes[node_numeration_map.at(side_gid)[i]]);

          for (int eq(0); eq<neq; ++eq)
          {
            cols[0] = getGlobalDOF(node_gid,eq);
            graphP->insertGlobalIndices(disc.getGlobalDOF(ss_node_gid,eq),cols());
          }
        }
      }
    }

    graphP->fillComplete (mapT,ss_mapT);
    P = Teuchos::rcp(new Tpetra_CrsMatrix(graphP)); // This constructor creates matrix with static profile
    P->setAllToScalar (1.0);
    P->fillComplete ();
    projectorsT[sideSetName] = P;

#ifdef ALBANY_EPETRA
    P_E = Petra::TpetraCrsMatrix_To_EpetraCrsMatrix (ov_P,comm);
    ov_projectors[sideSetName] = P_E;

    P_E = Petra::TpetraCrsMatrix_To_EpetraCrsMatrix (P,comm);
    projectors[sideSetName] = P_E;
#endif
  }
}

void
Albany::STKDiscretization::updateMesh(bool /*shouldTransferIPData*/)
{
  const Albany::StateInfoStruct& nodal_param_states = stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
  nodalDOFsStructContainer.addEmptyDOFsStruct("ordinary_solution", "", neq);
  nodalDOFsStructContainer.addEmptyDOFsStruct("mesh_nodes", "", 1);
  for(int is=0; is<nodal_param_states.size(); is++) {
    const Albany::StateStruct& param_state = *nodal_param_states[is];
    const Albany::StateStruct::FieldDims& dim = param_state.dim;
    int numComps = 1;
    if (dim.size()==3) //vector
      numComps = dim[2];
    else if (dim.size()==4) //tensor
      numComps = dim[2]*dim[3];

    nodalDOFsStructContainer.addEmptyDOFsStruct(param_state.name, param_state.meshPart,numComps);
    }

  computeNodalMaps(false);


  computeOwnedNodesAndUnknowns();

#ifdef OUTPUT_TO_SCREEN
  //write owned maps to matrix market file for debug
  Tpetra_MatrixMarket_Writer::writeMapFile("mapT0.mm", *mapT);
  Tpetra_MatrixMarket_Writer::writeMapFile("node_mapT0.mm", *node_mapT);
#endif

  setupMLCoords();

  computeNodalMaps(true);

  computeOverlapNodesAndUnknowns();

  transformMesh();

  computeGraphs();

  computeWorksetInfo();
#ifdef OUTPUT_TO_SCREEN
  printConnectivity();
#endif

  computeNodeSets();

  computeSideSets();

  setupExodusOutput();

  // Build the node graph needed for the mass matrix for solution transfer and projection operations
  // FIXME this only needs to be called if we are using the L2 Projection response
  meshToGraph();
//  printVertexConnectivity();
  setupNetCDFOutput();
//meshToGraph();
//printVertexConnectivity();

#ifdef OUTPUT_TO_SCREEN
  printCoords();
#endif

  // If the mesh struct stores sideSet mesh structs, we update them
  if (stkMeshStruct->sideSetMeshStructs.size()>0)
  {
    for (auto it : stkMeshStruct->sideSetMeshStructs)
    {
      Teuchos::RCP<STKDiscretization> side_disc = Teuchos::rcp(new STKDiscretization(it.second,commT));
      side_disc->updateMesh();
      sideSetDiscretizations.insert(std::make_pair(it.first,side_disc));
      sideSetDiscretizationsSTK.insert(std::make_pair(it.first,side_disc));

      stkMeshStruct->buildCellSideNodeNumerationMap (it.first, sideToSideSetCellMap[it.first], sideNodeNumerationMap[it.first]);
    }

    buildSideSetProjectors();
  }
}
