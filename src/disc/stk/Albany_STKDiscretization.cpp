//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include <Albany_CommUtils.hpp>
#include <Albany_ThyraUtils.hpp>
#include "Albany_BucketArray.hpp"
#include "Albany_Macros.hpp"
#include "Albany_NodalGraphUtils.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_STKNodeFieldContainer.hpp"
#include "Albany_Utils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <Shards_BasicTopologies.hpp>

#include <Intrepid2_Basis.hpp>
#include <Intrepid2_CellTools.hpp>
#include <Intrepid2_HGRAD_QUAD_Cn_FEM.hpp>

#include <stk_util/parallel/Parallel.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>
#include <stk_mesh/base/GetBuckets.hpp>
#include <stk_mesh/base/GetEntities.hpp>
#include <stk_mesh/base/Selector.hpp>

#ifdef ALBANY_SEACAS
#include <Ionit_Initializer.h>
#include <netcdf.h>

#ifdef ALBANY_PAR_NETCDF
extern "C" {
#include <netcdf_par.h>
}
#endif
#endif  // ALBANY_SEACAS

#include <algorithm>
#include <math.h>
#include <PHAL_Dimension.hpp>

#include "Albany_MultiSTKFieldContainer.hpp"
#include "Albany_OrdinarySTKFieldContainer.hpp"

// Uncomment the following line if you want debug output to be printed to screen
// #define OUTPUT_TO_SCREEN

namespace Albany {

STKDiscretization::STKDiscretization(
    const Teuchos::RCP<Teuchos::ParameterList>&    discParams_,
    const int neq_,
    Teuchos::RCP<Albany::AbstractSTKMeshStruct>&   stkMeshStruct_,
    const Teuchos::RCP<const Teuchos_Comm>&        comm_,
    const Teuchos::RCP<Albany::RigidBodyModes>&    rigidBodyModes_,
    const std::map<int, std::vector<std::string>>& sideSetEquations_)
    : previous_time_label(-1.0e32),
      out(Teuchos::VerboseObjectBase::getDefaultOStream()),
      metaData(stkMeshStruct_->metaData),
      bulkData(stkMeshStruct_->bulkData),
      comm(comm_),
      neq(neq_),
      sideSetEquations(sideSetEquations_),
      rigidBodyModes(rigidBodyModes_),
      stkMeshStruct(stkMeshStruct_),
      discParams(discParams_),
      interleavedOrdering(stkMeshStruct_->interleavedOrdering)
{
  // nothing to do
}

STKDiscretization::~STKDiscretization()
{
  for (size_t i = 0; i < toDelete.size(); i++) delete[] toDelete[i];
}

void
STKDiscretization::printConnectivity() const
{
  comm->barrier();
  for (int rank = 0; rank < comm->getSize(); ++rank) {
    comm->barrier();
    if (rank == comm->getRank()) {
      std::cout << std::endl << "Process rank " << rank << std::endl;
      for (int ibuck = 0; ibuck < wsElNodeID.size(); ++ibuck) {
        std::cout << "  Bucket " << ibuck << std::endl;
        for (int ielem = 0; ielem < wsElNodeID[ibuck].size(); ++ielem) {
          int numNodes = wsElNodeID[ibuck][ielem].size();
          std::cout << "    Element " << ielem << ": Nodes = ";
          for (int inode = 0; inode < numNodes; ++inode)
            std::cout << wsElNodeID[ibuck][ielem][inode] << " ";
          std::cout << std::endl;
        }
      }
    }
    comm->barrier();
  }
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getVectorSpace(const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getNodeVectorSpace(const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).node_vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getOverlapVectorSpace(const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_vs;
}

Teuchos::RCP<const Thyra_VectorSpace>
STKDiscretization::getOverlapNodeVectorSpace(
    const std::string& field_name) const
{
  return nodalDOFsStructContainer.getDOFsStruct(field_name).overlap_node_vs;
}

void
STKDiscretization::printCoords() const
{
  std::cout << "Processor " << bulkData->parallel_rank() << " has "
            << coords.size() << " worksets.\n";

  const int numDim = stkMeshStruct->numDim;
  double xmin = std::numeric_limits<double>::max(), xmax = std::numeric_limits<double>::lowest(),
      ymin = xmin, ymax = xmax, zmin = xmin, zmax = xmax;
  for (int ws = 0; ws < coords.size(); ws++) {
    for (int e = 0; e < coords[ws].size(); e++) {
      for (int j = 0; j < coords[ws][e].size(); j++) {
        xmin = std::min(xmin, coords[ws][e][j][0]);
        xmax = std::max(xmax, coords[ws][e][j][0]);
        if (numDim > 1) {
          ymin = std::min(ymin, coords[ws][e][j][1]);
          ymax = std::max(ymax, coords[ws][e][j][1]);
        }
        if (numDim > 2) {
          zmin = std::min(zmin, coords[ws][e][j][2]);
          zmax = std::max(zmax, coords[ws][e][j][2]);
        }
        std::cout << "Coord for workset: " << ws << " element: " << e
                  << " node: " << j << " x, y, z: " << coords[ws][e][j][0]
                  << ", " << coords[ws][e][j][1] << ", " << coords[ws][e][j][2]
                  << std::endl;
      }
    }
  }

  std::cout << "Processor " << bulkData->parallel_rank() << " has the following x-range: ["
      << xmin << ", " << xmax << "]" << std::endl;
  if (numDim > 1)
    std::cout << "Processor " << bulkData->parallel_rank() << " has the following y-range: ["
        << ymin << ", " << ymax << "]" << std::endl;
  if (numDim > 2)
    std::cout << "Processor " << bulkData->parallel_rank() << " has the following z-range: ["
        << zmin << ", " << zmax << "]" << std::endl;
}

const Teuchos::ArrayRCP<double>&
STKDiscretization::getCoordinates() const
{
  // Coordinates are computed here, and not precomputed,
  // since the mesh can move in shape opt problems

  const auto& coordinates_field = *stkMeshStruct->getCoordinatesField();

  const auto& nodeDofStruct = nodalDOFsStructContainer.getDOFsStruct(nodes_dof_name());
  const auto& ov_node_indexer = nodeDofStruct.overlap_node_vs_indexer;
  const int numOverlapNodes = ov_node_indexer->getNumLocalElements();
  const int meshDim = stkMeshStruct->numDim;
  for (int node_lid = 0; node_lid < numOverlapNodes; ++node_lid) {
    GO node_gid = ov_node_indexer->getGlobalElement(node_lid);

    const auto ov_node = bulkData->get_entity(stk::topology::NODE_RANK, node_gid + 1);
    double* x = stk::mesh::field_data(coordinates_field, ov_node);
    for (int dim = 0; dim < meshDim; ++dim) {
      coordinates[meshDim * node_lid + dim] = x[dim];
    }
  }

  return coordinates;
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
void
STKDiscretization::transformMesh()
{
  using std::cout;
  using std::endl;
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();
  std::string transformType = stkMeshStruct->transformType;

  std::vector<stk::mesh::Entity> overlapnodes;
  stk::mesh::Selector selector(metaData->locally_owned_part());
  selector |= metaData->globally_shared_part();
  const auto& buckets = bulkData->buckets(stk::topology::NODE_RANK);
  stk::mesh::get_selected_entities(selector, buckets, overlapnodes);

  if (transformType == "None") {
  } else if (transformType == "Spherical") {
// This form takes a mesh of a square / cube and transforms it into a mesh of a
// circle/sphere
#ifdef OUTPUT_TO_SCREEN
    *out << "Spherical!" << endl;
#endif
    const int numDim = stkMeshStruct->numDim;
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      double  r = 0.0;
      for (int n = 0; n < numDim; n++) { r += x[n] * x[n]; }
      r = sqrt(r);
      for (int n = 0; n < numDim; n++) {
        // FIXME: there could be division by 0 here!
        x[n] = x[n] / r;
      }
    }
  } else if (transformType == "Shift") {
    //*out << "Shift!\n";
    double xshift = stkMeshStruct->xShift;
    double yshift = stkMeshStruct->yShift;
    double zshift = stkMeshStruct->zShift;
    //*out << "xshift, yshift, zshift = " << xshift << ", " << yshift << ", " <<
    // zshift << '\n';
    const int numDim = stkMeshStruct->numDim;
    //*out << "numDim = " << numDim << '\n';
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    if (numDim >= 0) {
      for (int i = 0; i < numOverlapNodes; i++) {
        double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
        x[0]      = xshift + x[0];
      }
    }
    if (numDim >= 1) {
      for (int i = 0; i < numOverlapNodes; i++) {
        double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
        x[1]      = yshift + x[1];
      }
    }
    if (numDim >= 1) {
      for (int i = 0; i < numOverlapNodes; i++) {
        double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
        x[2]      = zshift + x[2];
      }
    }
  } else if (transformType == "Tanh Boundary Layer") {
    //*out << "IKT Tanh Boundary Layer!\n";

    /* The way this transform type works is it takes a uniform STK mesh of [0,L]
   generated within Albany and applies the following transformation to it:

   x = L*(1.0 - tanh(beta*(L-x)))/tanh(beta*L))

   for a specified double beta (and similarly for x and y coordinates).  The
   result is a mesh that is finer near x = 0 and coarser near x = L.  The
   relative coarseness/fineness is controlled by the parameter beta: large beta
   => finer boundary layer near x = 0.  If beta = 0, no tranformation is
   applied.*/

    Teuchos::Array<double> betas  = stkMeshStruct->betas_BLtransform;
    const int              numDim = stkMeshStruct->numDim;
    ALBANY_ASSERT(
        betas.length() >= numDim,
        "\n Length of Betas BL Transform array (= "
            << betas.length() << ") cannot be "
            << " < numDim (= " << numDim << ")!\n");

    Teuchos::Array<double> scales = stkMeshStruct->scales;

    ALBANY_ASSERT(
        scales.length() == numDim,
        "\n Length of scales array (= "
            << scales.length() << ") must equal numDim (= " << numDim
            << ") to use transformType = Tanh Boundary Layer!\n");

    double beta;
    double scale;
    if (numDim >= 0) {
      beta  = betas[0];
      scale = scales[0];
      if (abs(beta) > 1.0e-12) {
        const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
        for (int i = 0; i < numOverlapNodes; i++) {
          double* x =
              stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
          x[0] =
              scale * (1.0 - tanh(beta * (scale - x[0])) / tanh(scale * beta));
        }
      }
    }
    if (numDim >= 1) {
      beta  = betas[1];
      scale = scales[1];
      if (abs(beta) > 1.0e-12) {
        const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
        for (int i = 0; i < numOverlapNodes; i++) {
          double* x =
              stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
          x[1] =
              scale * (1.0 - tanh(beta * (scale - x[1])) / tanh(scale * beta));
        }
      }
    }
    if (numDim >= 2) {
      beta  = betas[2];
      scale = scales[2];
      if (abs(beta) > 1.0e-12) {
        const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
        for (int i = 0; i < numOverlapNodes; i++) {
          double* x =
              stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
          x[2] =
              scale * (1.0 - tanh(beta * (scale - x[2])) / tanh(scale * beta));
        }
      }
    }

  } else if (transformType == "ISMIP-HOM Test A") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test A!" << endl;
#endif
    double L     = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha * M_PI /
            180;  // convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = -x[0] * tan(alpha);
      double b =
          s - 1.0 + 0.5 * sin(2 * M_PI / L * x[0]) * sin(2 * M_PI / L * x[1]);
      x[2] = s * x[2] + b * (1 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "ISMIP-HOM Test B") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test B!" << endl;
#endif
    double L     = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha * M_PI /
            180;  // convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = -x[0] * tan(alpha);
      double b  = s - 1.0 + 0.5 * sin(2 * M_PI / L * x[0]);
      x[2]      = s * x[2] + b * (1 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (
      (transformType == "ISMIP-HOM Test C") ||
      (transformType == "ISMIP-HOM Test D")) {
#ifdef OUTPUT_TO_SCREEN
    *out << "Test C and D!" << endl;
#endif
    double L     = stkMeshStruct->felixL;
    double alpha = stkMeshStruct->felixAlpha;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
    *out << "alpha degrees: " << alpha << endl;
#endif
    alpha = alpha * M_PI /
            180;  // convert alpha, read in from ParameterList, to radians
#ifdef OUTPUT_TO_SCREEN
    *out << "alpha radians: " << alpha << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = -x[0] * tan(alpha);
      double b  = s - 1.0;
      x[2]      = s * x[2] + b * (1 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "Dome") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Dome transform!" << endl;
#endif
    double L = 0.7071 * 30;
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s = 0.7071 * sqrt(450.0 - x[0] * x[0] - x[1] * x[1]) / sqrt(450.0);
      x[2]     = s * x[2];
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "Confined Shelf") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Confined shelf transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    cout << "L: " << L << endl;
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s  = 0.06;    // top surface is at z=0.06km=60m
      double b  = -0.440;  // basal surface is at z=-0.440km=-440m
      x[2]      = s * x[2] + b * (1.0 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "Circular Shelf") {
#ifdef OUTPUT_TO_SCREEN
    *out << "Circular shelf transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    double rhoIce   = 910.0;   // ice density, in kg/m^3
    double rhoOcean = 1028.0;  // ocean density, in kg/m^3
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0]      = L * x[0];
      x[1]      = L * x[1];
      double s =
          1.0 -
          rhoIce / rhoOcean;  // top surface is at z=(1-rhoIce/rhoOcean) km
      double b = s - 1.0;     // basal surface is at z=s-1 km
      x[2]     = s * x[2] + b * (1.0 - x[2]);
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  } else if (transformType == "FO XZ MMS") {
// This test case assumes the domain read in from the input file is 0 < x < 2, 0
// < y < 1, where y = z
#ifdef OUTPUT_TO_SCREEN
    *out << "FO XZ MMS transform!" << endl;
#endif
    double L = stkMeshStruct->felixL;
    // hard-coding values of parameters...  make sure these are same as in the
    // FOStokes body force evaluator!
    double alpha0 = 4e-5;
    double s0     = 2.0;
    double H      = 1.0;
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<stk::mesh::Field<double>>(
            stk::topology::NODE_RANK, "surface_height");
    const auto numOverlapNodes = getLocalSubdim(m_overlap_node_vs);
    for (int i = 0; i < numOverlapNodes; i++) {
      double* x = stk::mesh::field_data(*coordinates_field, overlapnodes[i]);
      x[0] = L * (x[0] - 1.0);  // test case assumes domain is from [-L, L],
                                // where unscaled domain is from [0, 2];
      double s = s0 - alpha0 * x[0] * x[0];
      double b = s - H;
      // this transformation of y = [0,1] should give b(x) < y < s(x)
      x[1] = b * (1 - x[1]) + s * x[1];
      *stk::mesh::field_data(*surfaceHeight_field, overlapnodes[i]) = s;
    }
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(
        true,
        std::logic_error,
        "STKDiscretization::transformMesh() Unknown transform type :"
            << transformType << std::endl);
  }
}

void
STKDiscretization::setupMLCoords()
{
  if (rigidBodyModes.is_null()) { return; }
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed() && !rigidBodyModes->isFROSchUsed()) { return; }

  const int                                   numDim = stkMeshStruct->numDim;
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();
  coordMV           = Thyra::createMembers(m_node_vs, numDim);
  auto coordMV_data = getNonconstLocalData(coordMV);

  auto node_indexer = createGlobalLocalIndexer(m_node_vs);

  std::vector<stk::mesh::Entity> ownedNodes;
  const auto& part    = metaData->locally_owned_part();
  const auto& buckets = bulkData->buckets(stk::topology::NODE_RANK);
  stk::mesh::get_selected_entities(part, buckets, ownedNodes);

  for (const auto node : ownedNodes) {
    GO      node_gid = stk_gid(node);
    int     node_lid = node_indexer->getLocalElement(node_gid);
    double* X        = stk::mesh::field_data(*coordinates_field, node);
    for (int j = 0; j < numDim; j++) {
      coordMV_data[j][node_lid] = X[j];
    }
  }
  rigidBodyModes->setCoordinatesAndComputeNullspace(coordMV, interleavedOrdering, m_vs, m_overlap_vs);
  writeCoordsToMatrixMarket();
}

void
STKDiscretization::writeCoordsToMatrixMarket() const
{
  // if user wants to write the coordinates to matrix market file, write them to
  // matrix market file
  if ((rigidBodyModes->isMLUsed() || rigidBodyModes->isMueLuUsed() || rigidBodyModes->isFROSchUsed()) &&
      stkMeshStruct->writeCoordsToMMFile) {
    if (comm->getRank() == 0) {
      std::cout << "Writing mesh coordinates to Matrix Market file."
                << std::endl;
    }
    writeMatrixMarket(coordMV->col(0), "xCoords");
    if (coordMV->domain()->dim() > 1) {
      writeMatrixMarket(coordMV->col(1), "yCoords");
    }
    if (coordMV->domain()->dim() > 2) {
      writeMatrixMarket(coordMV->col(2), "zCoords");
    }
  }
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double        time,
    const bool          overlapped)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dxdp?
  writeSolutionToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const double        time,
    const bool          overlapped)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dot and/or soln_dxdp?
  writeSolutionToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double        time,
    const bool          overlapped)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, soln_dotdot, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dot and soln_dotdot?
  writeSolutionToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolutionMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double             time,
    const bool               overlapped)
{
  writeSolutionMVToMeshDatabase(soln, soln_dxdp, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dxdp?
  writeSolutionMVToFile(soln, time, overlapped);
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionField(soln, soln_dxdp, overlapped);
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionField(soln, soln_dxdp, soln_dot, overlapped);
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionField(soln, soln_dxdp, soln_dot, soln_dotdot, overlapped);
}

void
STKDiscretization::writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double /* time */,
    const bool overlapped)
{
  // Put solution into STK Mesh
  setSolutionFieldMV(soln, soln_dxdp, overlapped);
}

void
STKDiscretization::writeSolutionToFile(
    const Thyra_Vector& soln,
    const double        time,
    const bool          overlapped)
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {
    solutionFieldContainer->transferSolutionToCoords();

    if (!mesh_data.is_null()) {
      // Mesh coordinates have changed. Rewrite output file by deleting the mesh
      // data object and recreate it
      setupExodusOutput();
    }
  }

  // Skip this write unless the proper interval has been reached
  if (stkMeshStruct->exoOutput &&
      !(outputInterval % stkMeshStruct->exoOutputInterval)) {
    double time_label = monotonicTimeLabel(time);

    mesh_data->begin_output_step(outputFileIdx, time_label);
    int out_step = mesh_data->write_defined_output_fields(outputFileIdx);
    // Writing mesh global variables
    auto fc = stkMeshStruct->getFieldContainer();
    for (auto& it : fc->getMeshVectorStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    for (const auto& it : fc->getMeshScalarIntegerStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    for (const auto& it : fc->getMeshScalarInteger64States()) {
      boost::any value;
      value = static_cast<int64_t>(it.second);
      mesh_data->write_global(outputFileIdx, it.first, value, stk::util::ParameterType::INT64);
    }
    mesh_data->end_output_step(outputFileIdx);

    if (comm->getRank() == 0) {
      *out << "STKDiscretization::writeSolution: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->exoOutFile << std::endl;
    }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations) {
    if (overlapped) {
      auto ss_soln = Thyra::createMember(it.second->getOverlapVectorSpace());
      const Thyra_LinearOp& P = *ov_projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionToFile(*ss_soln, time, overlapped);
    } else {
      auto ss_soln = Thyra::createMember(it.second->getVectorSpace());
      const Thyra_LinearOp& P = *projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionToFile(*ss_soln, time, overlapped);
    }
  }
#endif
}

void
STKDiscretization::writeSolutionMVToFile(
    const Thyra_MultiVector& soln,
    const double             time,
    const bool               overlapped)
{
#ifdef ALBANY_SEACAS

  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {
    solutionFieldContainer->transferSolutionToCoords();

    if (!mesh_data.is_null()) {
      // Mesh coordinates have changed. Rewrite output file by deleting the mesh
      // data object and recreate it
      setupExodusOutput();
    }
  }

  // Skip this write unless the proper interval has been reached
  if (stkMeshStruct->exoOutput &&
      !(outputInterval % stkMeshStruct->exoOutputInterval)) {
    double time_label = monotonicTimeLabel(time);

    mesh_data->begin_output_step(outputFileIdx, time_label);
    int out_step = mesh_data->write_defined_output_fields(outputFileIdx);
    // Writing mesh global variables
    auto fc = stkMeshStruct->getFieldContainer();
    for (auto& it : fc->getMeshVectorStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    for (const auto& it : fc->getMeshScalarIntegerStates()) {
      mesh_data->write_global(outputFileIdx, it.first, it.second);
    }
    for (const auto& it : fc->getMeshScalarInteger64States()) {
      boost::any value;
      value = static_cast<int64_t>(it.second);
      mesh_data->write_global(outputFileIdx, it.first, value, stk::util::ParameterType::INT64);
    }
    mesh_data->end_output_step(outputFileIdx);

    if (comm->getRank() == 0) {
      *out << "STKDiscretization::writeSolution: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->exoOutFile << std::endl;
    }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations) {
    if (overlapped) {
      auto ss_soln = Thyra::createMembers(
          it.second->getOverlapVectorSpace(), soln.domain()->dim());
      const Thyra_LinearOp& P = *ov_projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionMVToFile(*ss_soln, time, overlapped);
    } else {
      auto ss_soln = Thyra::createMembers(
          it.second->getVectorSpace(), soln.domain()->dim());
      const Thyra_LinearOp& P = *projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionMVToFile(*ss_soln, time, overlapped);
    }
  }
#endif
}

void STKDiscretization::addSolutionField(const std::string & fieldName,const std::string & blockId)
{
#if 0
   TEUCHOS_TEST_FOR_EXCEPTION(!validBlockId(blockId),ElementBlockException,
                      "Unknown element block \"" << blockId << "\"");
   std::pair<std::string,std::string> key = std::make_pair(fieldName,blockId);

   // add & declare field if not already added...currently assuming linears
   if(fieldNameToSolution_.find(key)==fieldNameToSolution_.end()) {
      SolutionFieldType * field = metaData_->get_field<SolutionFieldType>(stk::topology::NODE_RANK, fieldName);
      if(field==0)
         field = &metaData_->declare_field<SolutionFieldType>(stk::topology::NODE_RANK, fieldName);
      if ( initialized_ )  {
        metaData_->enable_late_fields();
        stk::mesh::FieldTraits<SolutionFieldType>::data_type* init_sol = nullptr;
        stk::mesh::put_field_on_mesh(*field, metaData_->universal_part(),init_sol );
      }
      fieldNameToSolution_[key] = field;
   }
#endif
}

void STKDiscretization::addCellField(const std::string & fieldName,const std::string & blockId)
{
#if 0
   TEUCHOS_TEST_FOR_EXCEPTION(!validBlockId(blockId),ElementBlockException,
                      "Unknown element block \"" << blockId << "\"");
   std::pair<std::string,std::string> key = std::make_pair(fieldName,blockId);

   // add & declare field if not already added...currently assuming linears
   if(fieldNameToCellField_.find(key)==fieldNameToCellField_.end()) {
      SolutionFieldType * field = metaData_->get_field<SolutionFieldType>(stk::topology::ELEMENT_RANK, fieldName);
      if(field==0)
         field = &metaData_->declare_field<SolutionFieldType>(stk::topology::ELEMENT_RANK, fieldName);

      if ( initialized_ )  {
        metaData_->enable_late_fields();
        stk::mesh::FieldTraits<SolutionFieldType>::data_type* init_sol = nullptr;
        stk::mesh::put_field_on_mesh(*field, metaData_->universal_part(),init_sol );
      }
      fieldNameToCellField_[key] = field;
   }
#endif
}

double
STKDiscretization::monotonicTimeLabel(const double time)
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
  if (time_label + 1.0 > previous_time_label) {
    previous_time_label = time_label + 1.0;
    return time_label + 1.0;
  }

  // Otherwise, just add 1.0 to previous
  previous_time_label += 1.0;
  return previous_time_label;
}

Teuchos::RCP<Thyra_Vector>
STKDiscretization::getSolutionField(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  Teuchos::RCP<Thyra_Vector> soln = Thyra::createMember(m_vs);
  this->getSolutionField(*soln, overlapped);
  return soln;
}

Teuchos::RCP<Thyra_MultiVector>
STKDiscretization::getSolutionMV(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  int num_time_deriv = stkMeshStruct->num_time_deriv;
  Teuchos::RCP<Thyra_MultiVector> soln =
      Thyra::createMembers(m_vs, num_time_deriv + 1);
  this->getSolutionMV(*soln, overlapped);
  return soln;
}

void
STKDiscretization::getField(Thyra_Vector& result, const std::string& name) const
{
  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  const std::string& part =
      nodalDOFsStructContainer.fieldToMap.find(name)->second->first.first;
  stk::mesh::Selector selector = metaData->locally_owned_part();
  if (part.size()) {
    std::map<std::string, stk::mesh::Part*>::const_iterator it =
        stkMeshStruct->nsPartVec.find(part);
    if (it != stkMeshStruct->nsPartVec.end())
      selector &= stk::mesh::Selector(*(it->second));
  }

  const DOFsStruct& dofsStruct = nodalDOFsStructContainer.getDOFsStruct(name);

  solutionFieldContainer->fillVector(
      result, name, selector, dofsStruct.node_vs, dofsStruct.dofManager);
}

void
STKDiscretization::getSolutionField(Thyra_Vector& result, const bool overlapped)
    const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::Selector locally_owned = metaData->locally_owned_part();

  solutionFieldContainer->fillSolnVector(result, locally_owned, m_node_vs);
}

void
STKDiscretization::getSolutionMV(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  stk::mesh::Selector locally_owned = metaData->locally_owned_part();

  solutionFieldContainer->fillSolnMultiVector(result, locally_owned, m_node_vs);
}

/*****************************************************************/
/*** Private functions follow. These are just used in above code */
/*****************************************************************/

void
STKDiscretization::setField(
    const Thyra_Vector& result,
    const std::string&  name,
    bool                overlapped)
{
  const std::string& part =
      nodalDOFsStructContainer.fieldToMap.find(name)->second->first.first;

  stk::mesh::Selector selector =
      overlapped ?
          metaData->locally_owned_part() | metaData->globally_shared_part() :
          metaData->locally_owned_part();

  // Iterate over the on-processor nodes by getting node buckets and iterating
  // over each bucket.
  if (part.size()) {
    std::map<std::string, stk::mesh::Part*>::const_iterator it =
        stkMeshStruct->nsPartVec.find(part);
    if (it != stkMeshStruct->nsPartVec.end()) {
      selector &= stk::mesh::Selector(*(it->second));
    }
  }

  const DOFsStruct& dofsStruct = nodalDOFsStructContainer.getDOFsStruct(name);

  if (overlapped) {
    solutionFieldContainer->saveVector(
        result,
        name,
        selector,
        dofsStruct.overlap_node_vs,
        dofsStruct.overlap_dofManager);
  } else {
    solutionFieldContainer->saveVector(
        result, name, selector, dofsStruct.node_vs, dofsStruct.dofManager);
  }
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool          overlapped)
{
  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData->locally_owned_part();
  if (overlapped) { part |= metaData->globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, part, node_vs);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const bool          overlapped)
{
  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData->locally_owned_part();
  if (overlapped) { part |= metaData->globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, part, node_vs);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const bool          overlapped)
{
  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData->locally_owned_part();
  if (overlapped) { part |= metaData->globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, soln_dotdot, part, node_vs);
}

void
STKDiscretization::setSolutionFieldMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool               overlapped)
{
  // Select the proper mesh part and node vector space
  stk::mesh::Selector part = metaData->locally_owned_part();
  if (overlapped) { part |= metaData->globally_shared_part(); }
  auto node_vs = overlapped ? m_overlap_node_vs : m_node_vs;

  solutionFieldContainer->saveSolnMultiVector(soln, soln_dxdp, part, node_vs);
}

void STKDiscretization::computeVectorSpaces()
{
  // Loads member data:  ownednodes, numOwnedNodes, node_map, maxGlobalNodeGID,
  // map
  // maps for owned nodes and unknowns

  const auto& owned_part = metaData->locally_owned_part();
  const auto& ov_part    = metaData->globally_shared_part();

  std::vector<stk::mesh::Entity> nodes, ghosted_nodes;

  const auto& buckets = bulkData->buckets(stk::topology::NODE_RANK);
  stk::mesh::get_selected_entities(owned_part, buckets, nodes);

  // Compute NumGlobalNodes (the same for both unique and overlapped maps)
  GO maxID = -1;
  for (const auto& node : nodes) {
    maxID = std::max(maxID, stk_gid(node));
  }
  Teuchos::reduceAll(*comm, Teuchos::REDUCE_MAX, 1, &maxID, &maxGlobalNodeGID);

  // Use a different container for the dofs struct, just for the purposes of
  // this method. We do it in order to easily recycle vector spaces, since:
  //  1) same part dof structs can use the same node_vs
  //  2) scalar dof structs can use the same vs for node and vs

  // map[part_name][num_components] = dofs_struct;
  auto& mapOfDOFsStructs = nodalDOFsStructContainer.mapOfDOFsStructs;
  std::map<std::string, std::map<int, DOFsStruct*>> tmp_map;
  for (auto& it : mapOfDOFsStructs) {
    tmp_map[it.first.first][it.first.second] = &it.second;
  }

  // Build vector spaces. First owned, then shared.
  int numNodes, numGhostedNodes;
  for (auto& it1 : tmp_map) {
    stk::mesh::Selector selector(owned_part);
    stk::mesh::Selector ghosted_selector(ov_part);
    ghosted_selector &= !selector;
    const std::string&  name = it1.first;
    if (name.size()) {
      auto it2 = stkMeshStruct->nsPartVec.find(name);
      TEUCHOS_TEST_FOR_EXCEPTION (it2==stkMeshStruct->nsPartVec.end(), std::runtime_error,
        "STKDiscretization::computeNodalMaps():\n  Part " + name + " is not in  stkMeshStruct->nsPartVec.\n");
      selector &= *(it2->second);
      ghosted_selector &= *(it2->second);
    }

    stk::mesh::get_selected_entities(selector,    buckets, nodes);
    stk::mesh::get_selected_entities(ghosted_selector, buckets, ghosted_nodes);
    numNodes = nodes.size();
    numGhostedNodes = ghosted_nodes.size();

    // First, compute the nodal vs. We compute them once, for all dofs on this part
    Teuchos::Array<GO> indices;
    indices.reserve(numNodes+numGhostedNodes);

    // Owned
    for (const auto& node : nodes) {
      // STK ids start from 1. Subtract 1 to get 0-based indexing.
      const GO nodeId = stk_gid(node);
      indices.push_back(nodeId);
    }
    auto part_node_vs = createVectorSpace(comm, indices());

    // Overlapped.
    // IMPORTANT: make sure the ghosted nodes come *after* the owned ones
    for (const auto& node : ghosted_nodes) {
      // STK ids start from 1. Subtract 1 to get 0-based indexing.
      const GO nodeId = stk_gid(node);
      indices.push_back(nodeId);
    }
    auto ov_part_node_vs = createVectorSpace(comm, indices());

    // Now that the node vs are created, we can loop over the dofs struct on this part
    for (auto& it2 : it1.second) {
      const int   numComponents = it2.first;
      DOFsStruct* dofs   = it2.second;

      // Set nodal vs and indexers right away
      dofs->overlap_node_vs_indexer = createGlobalLocalIndexer(ov_part_node_vs);
      dofs->node_vs_indexer = createGlobalLocalIndexer(part_node_vs);
      dofs->overlap_node_vs = ov_part_node_vs;
      dofs->node_vs = part_node_vs;

      if (numComponents == 1) {
        // Life is easy: copy node_vs into the dofs's dof vs
        dofs->overlap_vs = dofs->overlap_node_vs;
        dofs->vs         = dofs->node_vs;
        dofs->overlap_vs_indexer = dofs->overlap_node_vs_indexer;
        dofs->vs_indexer         = dofs->node_vs_indexer;
      } else {
        // Create dof vs by replicating the nodal one (possibly interleaved)
        dofs->vs         = createVectorSpace(dofs->node_vs,numComponents,interleavedOrdering);
        dofs->overlap_vs = createVectorSpace(dofs->overlap_node_vs,numComponents,interleavedOrdering);
        dofs->overlap_vs_indexer = createGlobalLocalIndexer(dofs->overlap_vs);
        dofs->vs_indexer         = createGlobalLocalIndexer(dofs->vs);
      }

      dofs->dofManager.setup(numComponents,
                             numNodes,
                             maxGlobalNodeGID,
                             interleavedOrdering);

      dofs->overlap_dofManager.setup(numComponents,
                             numNodes+numGhostedNodes,
                             maxGlobalNodeGID,
                             interleavedOrdering);
    }
  }

  const auto& solDOF  = nodalDOFsStructContainer.getDOFsStruct(solution_dof_name());
  const auto& meshDOF = nodalDOFsStructContainer.getDOFsStruct(nodes_dof_name());

  m_node_vs = meshDOF.vs;
  m_vs      = solDOF.vs;
  m_overlap_node_vs = meshDOF.overlap_vs;
  m_overlap_vs      = solDOF.overlap_vs;

  auto& ndb = stkMeshStruct->nodal_data_base;
  if (!ndb.is_null()) {
    ndb->replaceOwnedVectorSpace(m_node_vs);
    ndb->replaceOverlapVectorSpace(m_overlap_node_vs);
  }

  coordinates.resize(3 * getLocalSubdim(m_overlap_node_vs));
}

void
STKDiscretization::computeGraphs()
{
  // Loads member data:  overlap_graph, numOverlapodes, overlap_node_map,
  // coordinates, graphs

  m_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(
      m_vs, m_vs, m_overlap_vs, m_overlap_vs));

  stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData->universal_part()) &
      stk::mesh::Selector(metaData->locally_owned_part());

  std::vector<stk::mesh::Entity> cells;
  stk::mesh::get_selected_entities(
      select_owned_in_part,
      bulkData->buckets(stk::topology::ELEMENT_RANK),
      cells);

  if (comm->getRank() == 0)
    *out << "STKDisc: " << cells.size() << " elements on Proc 0 " << std::endl;

  GO                     row, col;
  Teuchos::ArrayView<GO> colAV;

  // determining the equations that are defined on the whole domain
  std::vector<int> globalEqns;
  for (unsigned int k(0); k < neq; ++k) {
    if (sideSetEquations.find(k) == sideSetEquations.end()) {
      globalEqns.push_back(k);
    }
  }

  // The global solution dof manager, to get the correct dof id (interleaved vs blocked)
  const auto dofMgr = getOverlapDOFManager(solution_dof_name());
  for (const auto& e : cells) {
    stk::mesh::Entity const* node_rels = bulkData->begin_nodes(e);
    const size_t             num_nodes = bulkData->num_nodes(e);

    // loop over local nodes
    for (std::size_t j = 0; j < num_nodes; j++) {
      stk::mesh::Entity rowNode = node_rels[j];

      // loop over global eqs
      for (std::size_t k = 0; k < globalEqns.size(); ++k) {
        row = dofMgr.getGlobalDOF(stk_gid(rowNode), globalEqns[k]);
        for (std::size_t l = 0; l < num_nodes; l++) {
          stk::mesh::Entity colNode = node_rels[l];
          for (std::size_t m = 0; m < globalEqns.size(); ++m) {
            col   = dofMgr.getGlobalDOF(stk_gid(colNode), globalEqns[m]);
            colAV = Teuchos::arrayView(&col, 1);
            m_jac_factory->insertGlobalIndices(row, colAV);
          }
        }
      }
      // For sideset equations, we set a diagonal jacobian outside the side set.
      // Namely, we will set res=solution outside the side set (not res=0, otherwise
      // jac is singular).
      // Note: if this node happens to be on the side set, we will add the entry
      //       again in the next loop. But that's fine, cause ThyraCrsMatrixFactory
      //       is storing GIDs of each row in a std::set (until fill complete time).
      for (const auto& it : sideSetEquations) {
        int eq = it.first;
        row = dofMgr.getGlobalDOF(stk_gid(rowNode), eq);
        colAV = Teuchos::arrayView(&row, 1);
        m_jac_factory->insertGlobalIndices(row, colAV);
      }
    }
  }

  if (sideSetEquations.size() > 0) {
    const auto lmn = getLayeredMeshNumbering();
    const auto& nodeDofStruct = nodalDOFsStructContainer.getDOFsStruct(nodes_dof_name());
    const auto& ov_node_indexer = nodeDofStruct.overlap_node_vs_indexer;
    const int numOverlapNodes = ov_node_indexer->getNumLocalElements();

    // iterator over all sideSet-defined equations
    for (const auto& it : sideSetEquations) {
      // Get the eq number
      int eq = it.first;

      // In case we only have equations on side sets (no "volume" eqns),
      // there would be problem with linear solvers. To avoid this, we
      // put one diagonal entry for every side set equation.
      // NOTE: some nodes will be processed twice, but this is safe:
      //       the redundant indices will be discarded
      for (int inode=0; inode<numOverlapNodes; ++inode) {
        const GO node_gid = ov_node_indexer->getGlobalElement(inode);
        row                    = dofMgr.getGlobalDOF(node_gid, eq);
        colAV                  = Teuchos::arrayView(&row, 1);
        m_jac_factory->insertGlobalIndices(row, colAV);
      }

      // Do a first loop on all sideset, to establish whether column couplling is allowed.
      // We store the sides while we're at it, to avoid redoing it later
      // Note: column coupling means that 1) the mesh is layered, and 2) the ss eqn is
      //       defined ONLY on side sets at the top or bottom.
      bool allowColumnCoupling = !lmn.is_null();
      std::map<std::string,std::vector<stk::mesh::Entity>> all_sides;
      GO baseId, iLayer;
      for (const auto& ss_name : it.second) {
        stk::mesh::Part& part =
            *stkMeshStruct->ssPartVec.find(ss_name)->second;

        // Get all owned sides in this side set
        stk::mesh::Selector select_owned_in_sspart =
            stk::mesh::Selector(part) &
            stk::mesh::Selector(metaData->locally_owned_part());

        auto& sides = all_sides[ss_name];
        stk::mesh::get_selected_entities(
            select_owned_in_sspart,
            bulkData->buckets(metaData->side_rank()),
            sides);  // store the result in "sides"

        if (allowColumnCoupling && sides.size()>0) {
          const auto& side = sides[0];
          const auto& node = bulkData->begin_nodes(side)[0];
          lmn->getIndices(stk_gid(node),baseId,iLayer);
          allowColumnCoupling = (iLayer==0 || iLayer==lmn->numLayers);
        }
      }

      for (const auto& ss_name : it.second) {
        const auto& sides = all_sides[ss_name];

        // Loop on all the sides of this sideset
        for (const auto& sidee : sides) {
          stk::mesh::Entity const* node_rels = bulkData->begin_nodes(sidee);
          const size_t             num_nodes = bulkData->num_nodes(sidee);

          // loop over local nodes of the side (row)
          for (std::size_t i = 0; i < num_nodes; i++) {
            stk::mesh::Entity rowNode = node_rels[i];
            row                       = dofMgr.getGlobalDOF(stk_gid(rowNode), eq);

            // loop over local nodes of the side (col)
            for (std::size_t j = 0; j < num_nodes; j++) {
              stk::mesh::Entity colNode = node_rels[j];

              // TODO: this is to accommodate the scenario where the side equation is coupled with
              //       the volume equations over a whole column of a layered mesh. However, this
              //       introduces pointless nonzeros if such coupling is not needed.
              //       The only way to fix this would be to access more information from the problem.
              //       Until then, couple with *all* equations, over the whole column.
              if (allowColumnCoupling) {
                // It's a layered mesh. Assume the worst, and add coupling of the whole column
                // with all the equations.
                lmn->getIndices(stk_gid(colNode),baseId,iLayer);
                for (int il=0; il<=lmn->numLayers; ++il) {
                  const GO node3d = lmn->getId(baseId,il);
                  for (unsigned int m=0; m<neq; ++m) {
                    col = dofMgr.getGlobalDOF(node3d, m);
                    m_jac_factory->insertGlobalIndices(
                        row, Teuchos::arrayView(&col, 1));
                    m_jac_factory->insertGlobalIndices(
                        col, Teuchos::arrayView(&row, 1));
                  }
                }
              } else {
                // Not a layered mesh, or the eqn is not defined on top/bottom.
                // Couple locally with volume eqn and the other ss eqn on this sideSet
                for (auto m : globalEqns) {
                  col = dofMgr.getGlobalDOF(stk_gid(colNode), m);
                  m_jac_factory->insertGlobalIndices(
                      row, Teuchos::arrayView(&col, 1));
                  m_jac_factory->insertGlobalIndices(
                      col, Teuchos::arrayView(&row, 1));
                }
                for (auto ssEqIt : sideSetEquations) {
                  for (const auto& ssEq_ss_name : ssEqIt.second) {
                    if (ssEq_ss_name == ss_name) {
                      col = dofMgr.getGlobalDOF(stk_gid(colNode),ssEqIt.first);
                      m_jac_factory->insertGlobalIndices(
                          row, Teuchos::arrayView(&col, 1));
                      m_jac_factory->insertGlobalIndices(
                          col, Teuchos::arrayView(&row, 1));
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  m_jac_factory->fillComplete();
}

void
STKDiscretization::computeWorksetInfo()
{
  stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData->universal_part()) &
      stk::mesh::Selector(metaData->locally_owned_part());

  const stk::mesh::BucketVector& buckets =
      bulkData->get_buckets(stk::topology::ELEMENT_RANK, select_owned_in_part);

  const int numBuckets = buckets.size();

  typedef AbstractSTKFieldContainer::ScalarFieldType ScalarFieldType;
  typedef AbstractSTKFieldContainer::VectorFieldType VectorFieldType;
  typedef AbstractSTKFieldContainer::TensorFieldType TensorFieldType;

  VectorFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();

  wsEBNames.resize(numBuckets);
  for (int i = 0; i < numBuckets; i++) {
    stk::mesh::PartVector const& bpv = buckets[i]->supersets();

    for (std::size_t j = 0; j < bpv.size(); j++) {
      if (bpv[j]->primary_entity_rank() == stk::topology::ELEMENT_RANK &&
          !stk::mesh::is_auto_declared_part(*bpv[j])) {
        // *out << "Bucket " << i << " is in Element Block:  " << bpv[j]->name()
        //      << "  and has " << buckets[i]->size() << " elements." <<
        //      std::endl;
        wsEBNames[i] = bpv[j]->name();
      }
    }
  }

  wsPhysIndex.resize(numBuckets);
  if (stkMeshStruct->allElementBlocksHaveSamePhysics) {
    for (int i = 0; i < numBuckets; ++i) { wsPhysIndex[i] = 0; }
  } else {
    for (int i = 0; i < numBuckets; ++i) {
      wsPhysIndex[i] =
          stkMeshStruct->getMeshSpecs()[0]->ebNameToIndex[wsEBNames[i]];
    }
  }

  // Fill  wsElNodeEqID(workset, el_LID, local node, Eq) => unk_LID
  wsElNodeEqID.resize(numBuckets);
  wsElNodeID.resize(numBuckets);
  coords.resize(numBuckets);
  sphereVolume.resize(numBuckets);
  latticeOrientation.resize(numBuckets);

  nodesOnElemStateVec.resize(numBuckets);
  stateArrays.elemStateArrays.resize(numBuckets);
  const StateInfoStruct& nodal_states =
      stkMeshStruct->getFieldContainer()->getNodalSIS();

  // Clear map if remeshing
  if (!elemGIDws.empty()) { elemGIDws.clear(); }

  typedef stk::mesh::Cartesian NodeTag;
  typedef stk::mesh::Cartesian ElemTag;
  typedef stk::mesh::Cartesian CompTag;

  NodalDOFsStructContainer::MapOfDOFsStructs& mapOfDOFsStructs =
      nodalDOFsStructContainer.mapOfDOFsStructs;
  for (auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end(); ++it) {
    it->second.wsElNodeEqID.resize(numBuckets);
    it->second.wsElNodeEqID_rawVec.resize(numBuckets);
    it->second.wsElNodeID.resize(numBuckets);
    it->second.wsElNodeID_rawVec.resize(numBuckets);
  }

  auto ov_node_indexer = createGlobalLocalIndexer(m_overlap_node_vs);
  for (int b = 0; b < numBuckets; b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    wsElNodeID[b].resize(buck.size());
    coords[b].resize(buck.size());

    // Set size of Kokkos views
    // Note: Assumes nodes_per_element is the same across all elements in a
    // workset
    {
      const int         buckSize          = buck.size();
      stk::mesh::Entity element           = buck[0];
      const int         nodes_per_element = bulkData->num_nodes(element);
      wsElNodeEqID[b] =
          WorksetConn("wsElNodeEqID", buckSize, nodes_per_element, neq);
    }

    {  // nodalDataToElemNode.

      nodesOnElemStateVec[b].resize(nodal_states.size());

      for (size_t is = 0; is < nodal_states.size(); ++is) {
        const std::string&            name = nodal_states[is]->name;
        const StateStruct::FieldDims& dim  = nodal_states[is]->dim;
        MDArray&             array    = stateArrays.elemStateArrays[b][name];
        std::vector<double>& stateVec = nodesOnElemStateVec[b][is];
        int dim0 = buck.size();  // may be different from dim[0];
        switch (dim.size()) {
          case 2:  // scalar
          {
            const ScalarFieldType& field = *metaData->get_field<ScalarFieldType>(
                stk::topology::NODE_RANK, name);
            stateVec.resize(dim0 * dim[1]);
            array.assign<ElemTag, NodeTag>(stateVec.data(), dim0, dim[1]);
            for (int i = 0; i < dim0; i++) {
              stk::mesh::Entity        element = buck[i];
              stk::mesh::Entity const* rel     = bulkData->begin_nodes(element);
              for (int j = 0; j < static_cast<int>(dim[1]); j++) {
                stk::mesh::Entity rowNode = rel[j];
                array(i, j) = *stk::mesh::field_data(field, rowNode);
              }
            }
            break;
          }
          case 3:  // vector
          {
            const VectorFieldType& field = *metaData->get_field<VectorFieldType>(
                stk::topology::NODE_RANK, name);
            stateVec.resize(dim0 * dim[1] * dim[2]);
            array.assign<ElemTag, NodeTag, CompTag>(
                stateVec.data(), dim0, dim[1], dim[2]);
            for (int i = 0; i < dim0; i++) {
              stk::mesh::Entity        element = buck[i];
              stk::mesh::Entity const* rel     = bulkData->begin_nodes(element);
              for (int j = 0; j < static_cast<int>(dim[1]); j++) {
                stk::mesh::Entity rowNode = rel[j];
                double*           entry = stk::mesh::field_data(field, rowNode);
                for (int k = 0; k < static_cast<int>(dim[2]); k++) {
                  array(i, j, k) = entry[k];
                }
              }
            }
            break;
          }
          case 4:  // tensor
          {
            const TensorFieldType& field = *metaData->get_field<TensorFieldType>(
                stk::topology::NODE_RANK, name);
            stateVec.resize(dim0 * dim[1] * dim[2] * dim[3]);
            array.assign<ElemTag, NodeTag, CompTag, CompTag>(
                stateVec.data(), dim0, dim[1], dim[2], dim[3]);
            for (int i = 0; i < dim0; i++) {
              stk::mesh::Entity        element = buck[i];
              stk::mesh::Entity const* rel     = bulkData->begin_nodes(element);
              for (int j = 0; j < static_cast<int>(dim[1]); j++) {
                stk::mesh::Entity rowNode = rel[j];
                double*           entry = stk::mesh::field_data(field, rowNode);
                for (int k = 0; k < static_cast<int>(dim[2]); k++) {
                  for (int l = 0; l < static_cast<int>(dim[3]); l++) {
                    array(i, j, k, l) = entry[k * dim[3] + l];  // check this,
                                                                // is stride
                                                                // Correct?
                  }
                }
              }
            }
            break;
          }
        }
      }
    }

    stk::mesh::Entity element           = buck[0];
    int               nodes_per_element = bulkData->num_nodes(element);
    for (auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end();
         ++it) {
      int nComp = it->first.second;
      it->second.wsElNodeEqID_rawVec[b].resize(
          buck.size() * nodes_per_element * nComp);
      it->second.wsElNodeEqID[b].assign<ElemTag, NodeTag, CompTag>(
          it->second.wsElNodeEqID_rawVec[b].data(),
          (int)buck.size(),
          nodes_per_element,
          nComp);
      it->second.wsElNodeID_rawVec[b].resize(buck.size() * nodes_per_element);
      it->second.wsElNodeID[b].assign<ElemTag, NodeTag>(
          it->second.wsElNodeID_rawVec[b].data(),
          (int)buck.size(),
          nodes_per_element);
    }

    // i is the element index within bucket b
    for (std::size_t i = 0; i < buck.size(); i++) {
      // Traverse all the elements in this bucket
      element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[stk_gid(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[stk_gid(element)].LID = i;

      stk::mesh::Entity const* node_rels = bulkData->begin_nodes(element);
      nodes_per_element                  = bulkData->num_nodes(element);

      wsElNodeID[b][i].resize(nodes_per_element);
      coords[b][i].resize(nodes_per_element);

      for (auto it = mapOfDOFsStructs.begin(); it != mapOfDOFsStructs.end();
           ++it) {
        const auto& ov_indexer = it->second.overlap_vs_indexer;
        IDArray&  wsElNodeEqID_array = it->second.wsElNodeEqID[b];
        GIDArray& wsElNodeID_array   = it->second.wsElNodeID[b];
        int       nComp              = it->first.second;
        for (int j = 0; j < nodes_per_element; j++) {
          stk::mesh::Entity node      = node_rels[j];
          wsElNodeID_array((int)i, j) = stk_gid(node);
          for (int k = 0; k < nComp; k++) {
            const GO node_gid = it->second.overlap_dofManager.getGlobalDOF(
                stk_gid(node), k);
            const int node_lid = ov_indexer->getLocalElement(node_gid);
            wsElNodeEqID_array((int)i, j, k) = node_lid;
          }
        }
      }

      // loop over local nodes
      DOFsStruct& dofs_struct =
          mapOfDOFsStructs[make_pair(std::string(""), neq)];
      GIDArray& node_array    = dofs_struct.wsElNodeID[b];
      IDArray&  node_eq_array = dofs_struct.wsElNodeEqID[b];
      for (int j = 0; j < nodes_per_element; j++) {
        const stk::mesh::Entity rowNode  = node_rels[j];
        const GO                node_gid = stk_gid(rowNode);
        const LO node_lid = ov_node_indexer->getLocalElement(node_gid);

        TEUCHOS_TEST_FOR_EXCEPTION(
            node_lid < 0,
            std::logic_error,
            "STK1D_Disc: node_lid out of range " << node_lid << std::endl);
        coords[b][i][j] = stk::mesh::field_data(*coordinates_field, rowNode);

        wsElNodeID[b][i][j] = node_array((int)i, j);

        for (int eq = 0; eq < static_cast<int>(neq); ++eq)
          wsElNodeEqID[b](i, j, eq) = node_eq_array((int)i, j, eq);
      }
    }
  }

  for (int d = 0; d < stkMeshStruct->numDim; d++) {
    if (stkMeshStruct->PBCStruct.periodic[d]) {
      for (int b = 0; b < numBuckets; b++) {
        for (std::size_t i = 0; i < buckets[b]->size(); i++) {
          int  nodes_per_element = buckets[b]->num_nodes(i);
          bool anyXeqZero        = false;
          for (int j = 0; j < nodes_per_element; j++)
            if (coords[b][i][j][d] == 0.0) anyXeqZero = true;
          if (anyXeqZero) {
            bool flipZeroToScale = false;
            for (int j = 0; j < nodes_per_element; j++)
              if (coords[b][i][j][d] > stkMeshStruct->PBCStruct.scale[d] / 1.9)
                flipZeroToScale = true;
            if (flipZeroToScale) {
              for (int j = 0; j < nodes_per_element; j++) {
                if (coords[b][i][j][d] == 0.0) {
                  double* xleak = new double[stkMeshStruct->numDim];
                  for (int k = 0; k < stkMeshStruct->numDim; k++)
                    if (k == d)
                      xleak[d] = stkMeshStruct->PBCStruct.scale[d];
                    else
                      xleak[k] = coords[b][i][j][k];
                  std::string transformType = stkMeshStruct->transformType;
                  double      alpha         = stkMeshStruct->felixAlpha;
                  alpha *= M_PI / 180.;  // convert alpha, read in from
                                       // ParameterList, to radians
                  if ((transformType == "ISMIP-HOM Test A" ||
                       transformType == "ISMIP-HOM Test B" ||
                       transformType == "ISMIP-HOM Test C" ||
                       transformType == "ISMIP-HOM Test D") &&
                      d == 0) {
                    xleak[2] -= stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                    StateArray::iterator sHeight =
                        stateArrays.elemStateArrays[b].find("surface_height");
                    if (sHeight != stateArrays.elemStateArrays[b].end())
                      sHeight->second(int(i), j) -=
                          stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                  }
                  coords[b][i][j] = xleak;  // replace ptr to coords
                  toDelete.push_back(xleak);
                }
              }
            }
          }
        }
      }
    }
  }

  typedef AbstractSTKFieldContainer::ScalarValueState ScalarValueState;
  typedef AbstractSTKFieldContainer::QPScalarState    QPScalarState;
  typedef AbstractSTKFieldContainer::QPVectorState    QPVectorState;
  typedef AbstractSTKFieldContainer::QPTensorState    QPTensorState;

  typedef AbstractSTKFieldContainer::ScalarState ScalarState;
  typedef AbstractSTKFieldContainer::VectorState VectorState;
  typedef AbstractSTKFieldContainer::TensorState TensorState;

  // Pull out pointers to shards::Arrays for every bucket, for every state
  // Code is data-type dependent

  AbstractSTKFieldContainer& container = *stkMeshStruct->getFieldContainer();

  ScalarValueState& scalarValue_states = container.getScalarValueStates();
  ScalarState&      cell_scalar_states = container.getCellScalarStates();
  VectorState&      cell_vector_states = container.getCellVectorStates();
  TensorState&      cell_tensor_states = container.getCellTensorStates();
  QPScalarState&    qpscalar_states    = container.getQPScalarStates();
  QPVectorState&    qpvector_states    = container.getQPVectorStates();
  QPTensorState&    qptensor_states    = container.getQPTensorStates();
  std::map<std::string, double>& time  = container.getTime();

  for (std::size_t b = 0; b < buckets.size(); b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    for (auto css = cell_scalar_states.begin(); css != cell_scalar_states.end();
         ++css) {
      BucketArray<AbstractSTKFieldContainer::ScalarFieldType> array(
          **css, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " SFT dim[1]: " <<
      // array.extent(1) << std::endl;
      MDArray ar                                     = array;
      stateArrays.elemStateArrays[b][(*css)->name()] = ar;
    }
    for (auto cvs = cell_vector_states.begin(); cvs != cell_vector_states.end();
         ++cvs) {
      BucketArray<AbstractSTKFieldContainer::VectorFieldType> array(
          **cvs, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " VFT dim[2]: " <<
      // array.extent(2) << std::endl;
      MDArray ar                                     = array;
      stateArrays.elemStateArrays[b][(*cvs)->name()] = ar;
    }
    for (auto cts = cell_tensor_states.begin(); cts != cell_tensor_states.end();
         ++cts) {
      BucketArray<AbstractSTKFieldContainer::TensorFieldType> array(
          **cts, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " TFT dim[3]: " <<
      // array.extent(3) << std::endl;
      MDArray ar                                     = array;
      stateArrays.elemStateArrays[b][(*cts)->name()] = ar;
    }
    for (auto qpss = qpscalar_states.begin(); qpss != qpscalar_states.end();
         ++qpss) {
      BucketArray<AbstractSTKFieldContainer::QPScalarFieldType> array(
          **qpss, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " QPSFT dim[1]: " <<
      // array.extent(1) << std::endl;
      MDArray ar                                      = array;
      stateArrays.elemStateArrays[b][(*qpss)->name()] = ar;
    }
    for (auto qpvs = qpvector_states.begin(); qpvs != qpvector_states.end();
         ++qpvs) {
      BucketArray<AbstractSTKFieldContainer::QPVectorFieldType> array(
          **qpvs, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " QPVFT dim[2]: " <<
      // array.extent(2) << std::endl;
      MDArray ar                                      = array;
      stateArrays.elemStateArrays[b][(*qpvs)->name()] = ar;
    }
    for (auto qpts = qptensor_states.begin(); qpts != qptensor_states.end();
         ++qpts) {
      BucketArray<AbstractSTKFieldContainer::QPTensorFieldType> array(
          **qpts, buck);
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " QPTFT dim[3]: " <<
      // array.extent(3) << std::endl;
      MDArray ar                                      = array;
      stateArrays.elemStateArrays[b][(*qpts)->name()] = ar;
    }
    //    for (ScalarValueState::iterator svs = scalarValue_states.begin();
    //              svs != scalarValue_states.end(); ++svs){
    for (size_t i = 0; i < scalarValue_states.size(); i++) {
      const int                                         size = 1;
      shards::Array<double, shards::NaturalOrder, Cell> array(
          &time[*scalarValue_states[i]], size);
      MDArray ar = array;
      // Debug
      // std::cout << "Buck.size(): " << buck.size() << " SVState dim[0]: " <<
      // array.extent(0) << std::endl;
      // std::cout << "SV Name: " << *svs << " address : " << &array <<
      // std::endl;
      stateArrays.elemStateArrays[b][*scalarValue_states[i]] = ar;
    }
  }

  // Process node data sets if present

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base) &&
      stkMeshStruct->nodal_data_base->isNodeDataPresent()) {
    Teuchos::RCP<NodeFieldContainer> node_states =
        stkMeshStruct->nodal_data_base->getNodeContainer();

    stk::mesh::BucketVector const& node_buckets =
        bulkData->get_buckets(stk::topology::NODE_RANK, select_owned_in_part);

    const size_t numNodeBuckets = node_buckets.size();

    stateArrays.nodeStateArrays.resize(numNodeBuckets);
    for (std::size_t b = 0; b < numNodeBuckets; b++) {
      stk::mesh::Bucket& buck = *node_buckets[b];
      for (NodeFieldContainer::iterator nfs = node_states->begin();
           nfs != node_states->end();
           ++nfs) {
        stateArrays.nodeStateArrays[b][(*nfs).first] =
            Teuchos::rcp_dynamic_cast<AbstractSTKNodeFieldContainer>(
                (*nfs).second)
                ->getMDA(buck);
      }
    }
  }
}

void
STKDiscretization::computeSideSets()
{
  // Clean up existing sideset structure if remeshing
  for (size_t i = 0; i < sideSets.size(); ++i) {
    sideSets[i].clear();  // empty the ith map
  }

  // iterator over all side_rank parts found in the mesh
  std::map<std::string, stk::mesh::Part*>::iterator ss =
      stkMeshStruct->ssPartVec.begin();

  int numBuckets = wsEBNames.size();

  sideSets.resize(numBuckets);  // Need a sideset list per workset

  while (ss != stkMeshStruct->ssPartVec.end()) {
    // Get all owned sides in this side set
    stk::mesh::Selector select_owned_in_sspart =
        stk::mesh::Selector(*(ss->second)) &
        stk::mesh::Selector(metaData->locally_owned_part());

    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        select_owned_in_sspart,  // sides local to this processor
        bulkData->buckets(metaData->side_rank()),
        sides);

    *out << "STKDisc: sideset " << ss->first << " has size " << sides.size()
         << "  on Proc 0." << std::endl;

    // If the sideSet has mesh specs, then we need to check if slim allocation is valid. There
    //   are some cases where mesh specs for a sideset haven't been created (see Albany_GenericSTKMeshStruct.cpp:482)
    //   but we don't need to check slim allocation for these because if they are used in an evaluator,
    //   other exceptions will be thrown.
    if (stkMeshStruct->getMeshSpecs()[0]->sideSetMeshSpecs[ss->first].size() > 0) {
      int ssWorksetSize = stkMeshStruct->getMeshSpecs()[0]->sideSetMeshSpecs[ss->first][0]->worksetSize;
      bool ssSingleWorksetSizeAllocation = stkMeshStruct->getMeshSpecs()[0]->sideSetMeshSpecs[ss->first][0]->singleWorksetSizeAllocation;
      if (ssSingleWorksetSizeAllocation)
        *out << "STKDisc: sideset " << ss->first << " set to single workset size allocation." << std::endl;

      // Slim sideset alloction is automatically activated when using a single workset and Ioss,
      //  therefore we need to make sure that the meshspecs for each sideset have a large enough
      //  workset size to avoid writing or reading out of bounds.
      TEUCHOS_TEST_FOR_EXCEPTION(
        ssSingleWorksetSizeAllocation && ssWorksetSize != (int) sides.size(),
        std::logic_error,
        "STKDisc: MeshSpec workset size should be the same as sideset size for slim sideset allocation on sideset "
          << ss->first << ", (sideSetMeshStructs[" << ss->first << "] = " << ssWorksetSize 
          << ", ssSingleWorksetSizeAllocation = " << ssSingleWorksetSizeAllocation << ")" << std::endl);
    }

    // loop over the sides to see what they are, then fill in the data holder
    // for side set options, look at
    // $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp

    for (std::size_t localSideID = 0; localSideID < sides.size();
         localSideID++) {
      stk::mesh::Entity sidee = sides[localSideID];

      TEUCHOS_TEST_FOR_EXCEPTION(
          bulkData->num_elements(sidee) != 1,
          std::logic_error,
          "STKDisc: cannot figure out side set topology for side set "
              << ss->first << std::endl);

      stk::mesh::Entity elem = bulkData->begin_elements(sidee)[0];

      // containing the side. Note that if the side is internal, it will show up
      // twice in the
      // element list, once for each element that contains it.

      SideStruct sStruct;

      // Save side (global id)
      sStruct.side_GID = bulkData->identifier(sidee) - 1;

      // Save elem id. This is the global element id
      sStruct.elem_GID = bulkData->identifier(elem) - 1;

      int workset = elemGIDws[sStruct.elem_GID]
                        .ws;  // Get the ws that this element lives in

      // Save elem id. This is the local element id within the workset
      sStruct.elem_LID = elemGIDws[sStruct.elem_GID].LID;

      // Save the side identifier inside of the element. This starts at zero
      // here.
      sStruct.side_local_id = determine_local_side_id(elem, sidee);

      // Save the index of the element block that this elem lives in
      sStruct.elem_ebIndex =
          stkMeshStruct->getMeshSpecs()[0]->ebNameToIndex[wsEBNames[workset]];

      SideSetList& ssList =
          sideSets[workset];  // Get a ref to the side set map for this ws
      SideSetList::iterator it = ssList.find(
          ss->first);  // Get an iterator to the correct sideset (if any)

      if (it != ssList.end()) {
        // The sideset has already been created
        it->second.push_back(sStruct);  // Save this side to the vector that
                                        // belongs to the name ss->first
      } else {
        // Add the key ss->first to the map, and the side vector to that map
        std::vector<SideStruct> tmpSSVec;
        tmpSSVec.push_back(sStruct);

        ssList.insert(SideSetList::value_type(ss->first, tmpSSVec));
      }
    }

    ss++;
  }

  // =============================================================
  // (Kokkos Refactor) Convert sideSets to sideSetViews

  // 1) Compute view extents (num_local_worksets, max_sideset_length, max_sides) and local workset counter (current_local_index)
  std::map<std::string, int> num_local_worksets;
  std::map<std::string, int> max_sideset_length;
  std::map<std::string, int> max_sides;
  std::map<std::string, int> current_local_index;
  for (size_t i = 0; i < sideSets.size(); ++i) {
    SideSetList& ssList = sideSets[i];
    std::map<std::string, std::vector<SideStruct>>::iterator ss_it = ssList.begin();

    while (ss_it != ssList.end()) {
      std::string             ss_key = ss_it->first;
      std::vector<SideStruct> ss_val = ss_it->second;

      // Initialize values if this is the first time seeing a sideset key
      if (num_local_worksets.find(ss_key) == num_local_worksets.end())
        num_local_worksets[ss_key] = 0;
      if (max_sideset_length.find(ss_key) == max_sideset_length.end())
        max_sideset_length[ss_key] = 0;
      if (max_sides.find(ss_key) == max_sides.end())
        max_sides[ss_key] = 0;
      if (current_local_index.find(ss_key) == current_local_index.end())
        current_local_index[ss_key] = 0;

      // Update extents for given workset/sideset
      num_local_worksets[ss_key]++;
      max_sideset_length[ss_key] = std::max(max_sideset_length[ss_key], (int) ss_val.size());
      for (size_t j = 0; j < ss_val.size(); ++j)
        max_sides[ss_key] = std::max(max_sides[ss_key], (int) ss_val[j].side_local_id);

      ss_it++;
    }
  }

  // 2) Construct GlobalSideSetList (map of GlobalSideSetInfo)
  std::map<std::string, int>::iterator ss_it = num_local_worksets.begin();
  while (ss_it != num_local_worksets.end()) {
    std::string             ss_key = ss_it->first;

    max_sides[ss_key]++; // max sides is the largest local ID + 1 and needs to be incremented once for each key here

    globalSideSetViews[ss_key].num_local_worksets = num_local_worksets[ss_key];
    globalSideSetViews[ss_key].max_sideset_length = max_sideset_length[ss_key];
    globalSideSetViews[ss_key].side_GID         = Kokkos::View<GO**,       Kokkos::LayoutRight>("side_GID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_GID         = Kokkos::View<GO**,       Kokkos::LayoutRight>("elem_GID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_LID         = Kokkos::View<int**,      Kokkos::LayoutRight>("elem_LID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_ebIndex     = Kokkos::View<int**,      Kokkos::LayoutRight>("elem_ebIndex", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].side_local_id    = Kokkos::View<unsigned**, Kokkos::LayoutRight>("side_local_id", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].max_sides        = max_sides[ss_key];
    globalSideSetViews[ss_key].numCellsOnSide   = Kokkos::View<int**,      Kokkos::LayoutRight>("numCellsOnSide", num_local_worksets[ss_key], max_sides[ss_key]);
    globalSideSetViews[ss_key].cellsOnSide      = Kokkos::View<int***,     Kokkos::LayoutRight>("cellsOnSide", num_local_worksets[ss_key], max_sides[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].sideSetIdxOnSide = Kokkos::View<int***,     Kokkos::LayoutRight>("sideSetIdxOnSide", num_local_worksets[ss_key], max_sides[ss_key], max_sideset_length[ss_key]);

    ss_it++;
  }

  // 3) Populate global views
  for (size_t i = 0; i < sideSets.size(); ++i) {
    SideSetList& ssList = sideSets[i];
    std::map<std::string, std::vector<SideStruct>>::iterator ss_it = ssList.begin();

    while (ss_it != ssList.end()) {
      std::string             ss_key = ss_it->first;
      std::vector<SideStruct> ss_val = ss_it->second;

      int current_index = current_local_index[ss_key];
      int numSides = max_sides[ss_key];

      int max_cells_on_side = 0;
      std::vector<int> numCellsOnSide(numSides);
      std::vector<std::vector<int>> cellsOnSide(numSides);
      std::vector<std::vector<int>> sideSetIdxOnSide(numSides);
      for (size_t j = 0; j < ss_val.size(); ++j) {
        int cell = ss_val[j].elem_LID;
        int side = ss_val[j].side_local_id;
        cellsOnSide[side].push_back(cell);
        sideSetIdxOnSide[side].push_back(j);
      }
      for (int side = 0; side < numSides; ++side) {
        numCellsOnSide[side] = cellsOnSide[side].size();
        max_cells_on_side = std::max(max_cells_on_side, numCellsOnSide[side]);
      }

      for (int side = 0; side < numSides; ++side) {
        globalSideSetViews[ss_key].numCellsOnSide(current_index, side) = numCellsOnSide[side];
        for (int j = 0; j < numCellsOnSide[side]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide(current_index, side, j) = cellsOnSide[side][j];
          globalSideSetViews[ss_key].sideSetIdxOnSide(current_index, side, j) = sideSetIdxOnSide[side][j];
        }
        for (int j = numCellsOnSide[side]; j < max_sideset_length[ss_key]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide(current_index, side, j) = -1;
          globalSideSetViews[ss_key].sideSetIdxOnSide(current_index, side, j) = -1;
        }
      }

      for (size_t j = 0; j < ss_val.size(); ++j) {
        globalSideSetViews[ss_key].side_GID(current_index, j)      = ss_val[j].side_GID;
        globalSideSetViews[ss_key].elem_GID(current_index, j)      = ss_val[j].elem_GID;
        globalSideSetViews[ss_key].elem_LID(current_index, j)      = ss_val[j].elem_LID;
        globalSideSetViews[ss_key].elem_ebIndex(current_index, j)  = ss_val[j].elem_ebIndex;
        globalSideSetViews[ss_key].side_local_id(current_index, j) = ss_val[j].side_local_id;
      }

      current_local_index[ss_key]++;

      ss_it++;
    }
  }

  // 4) Reset current_local_index
  std::map<std::string, int>::iterator counter_it = current_local_index.begin();
  while (counter_it != current_local_index.end()) {
    std::string counter_key = counter_it->first;
    current_local_index[counter_key] = 0;
    counter_it++;
  }

  // 5) Populate map of LocalSideSetInfos
  for (size_t i = 0; i < sideSets.size(); ++i) {
    SideSetList& ssList = sideSets[i];
    LocalSideSetInfoList& lssList = sideSetViews[i];
    std::map<std::string, std::vector<SideStruct>>::iterator ss_it = ssList.begin();

    while (ss_it != ssList.end()) {
      std::string             ss_key = ss_it->first;
      std::vector<SideStruct> ss_val = ss_it->second;

      int current_index = current_local_index[ss_key];
      std::pair<int,int> range(0, ss_val.size());

      lssList[ss_key].size           = ss_val.size();
      lssList[ss_key].side_GID       = Kokkos::subview(globalSideSetViews[ss_key].side_GID, current_index, range );
      lssList[ss_key].elem_GID       = Kokkos::subview(globalSideSetViews[ss_key].elem_GID, current_index, range );
      lssList[ss_key].elem_LID       = Kokkos::subview(globalSideSetViews[ss_key].elem_LID, current_index, range );
      lssList[ss_key].elem_ebIndex   = Kokkos::subview(globalSideSetViews[ss_key].elem_ebIndex,  current_index, range );
      lssList[ss_key].side_local_id  = Kokkos::subview(globalSideSetViews[ss_key].side_local_id, current_index, range );
      lssList[ss_key].numSides       = globalSideSetViews[ss_key].max_sides;
      lssList[ss_key].numCellsOnSide = Kokkos::subview(globalSideSetViews[ss_key].numCellsOnSide, current_index, Kokkos::ALL() );
      lssList[ss_key].cellsOnSide    = Kokkos::subview(globalSideSetViews[ss_key].cellsOnSide,    current_index, Kokkos::ALL(), Kokkos::ALL() );
      lssList[ss_key].sideSetIdxOnSide    = Kokkos::subview(globalSideSetViews[ss_key].sideSetIdxOnSide,    current_index, Kokkos::ALL(), Kokkos::ALL() );

      current_local_index[ss_key]++;

      ss_it++;
    }
  }

  // 6) Determine size of global DOFView structure and allocate
  std::map<std::string, int> total_sideset_idx;
  std::map<std::string, int> sideset_idx_offset;
  unsigned int maxSideNodes = 0;
  if (!stkMeshStruct->layered_mesh_numbering.is_null()) {

    const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *(stkMeshStruct->layered_mesh_numbering);
    const Teuchos::RCP<const CellTopologyData> cell_topo = Teuchos::rcp(new CellTopologyData(stkMeshStruct->getMeshSpecs()[0]->ctd));
    const Albany::NodalDOFManager& solDOFManager = nodalDOFsStructContainer.getDOFsStruct("ordinary_solution").overlap_dofManager;
    const unsigned int numLayers = layeredMeshNumbering.numLayers;
    const unsigned int numComps = solDOFManager.numComponents();

    // Determine maximum number of side nodes
    for (unsigned int elem_side = 0; elem_side < cell_topo->side_count; ++elem_side) {
      const CellTopologyData_Subcell& side =  cell_topo->side[elem_side];
      const unsigned int numSideNodes = side.topology->node_count;
      maxSideNodes = std::max(maxSideNodes, numSideNodes);
    }

    // Determine total number of sideset indices per each sideset name
    for (unsigned int ws = 0; ws < sideSets.size(); ++ws) {
      SideSetList& ssList = sideSets[ws];
      std::map<std::string, std::vector<SideStruct>>::iterator ss_it = ssList.begin();
      while (ss_it != ssList.end()) {
        std::string             ss_key = ss_it->first;
        std::vector<SideStruct> ss_val = ss_it->second;

        if (sideset_idx_offset.find(ss_key) == sideset_idx_offset.end())
          sideset_idx_offset[ss_key] = 0;
        if (total_sideset_idx.find(ss_key) == total_sideset_idx.end())
          total_sideset_idx[ss_key] = 0;

        total_sideset_idx[ss_key] += ss_val.size();

        ss_it++;
      }
    }

    // Allocate total localDOFView for each sideset name
    ss_it = num_local_worksets.begin();
    while (ss_it != num_local_worksets.end()) {
      std::string ss_key = ss_it->first;
      allLocalDOFViews[ss_key] = Kokkos::View<LO****, PHX::Device>(ss_key + " localDOFView", total_sideset_idx[ss_key], maxSideNodes, numLayers+1, numComps);

      ss_it++;
    }

  }

  // 7) Populate localDOFViews for GatherVerticallyContractedSolution
  for (unsigned int i = 0; i < sideSets.size(); ++i) {

    // Need to look at localDOFViews for each i so that there is a view available for each workset even if it is empty
    std::map<std::string, Kokkos::View<LO****, PHX::Device>>& wsldofViews = wsLocalDOFViews[i];

    // Not all mesh structs that come through here are extruded mesh structs. This is to check if
    //   the mesh struct is an extruded one. If it isn't extruded, it won't need to do any of the following work.
    if (!stkMeshStruct->layered_mesh_numbering.is_null()) {

      const Albany::LayeredMeshNumbering<GO>& layeredMeshNumbering = *(stkMeshStruct->layered_mesh_numbering);
      const Teuchos::RCP<const CellTopologyData> cell_topo = Teuchos::rcp(new CellTopologyData(stkMeshStruct->getMeshSpecs()[0]->ctd));
      const Albany::NodalDOFManager& solDOFManager = nodalDOFsStructContainer.getDOFsStruct("ordinary_solution").overlap_dofManager;
      const auto& ov_node_indexer = *(getOverlapGlobalLocalIndexer(nodes_dof_name()));
      const unsigned int numLayers = layeredMeshNumbering.numLayers;
      const unsigned int numComps = solDOFManager.numComponents();

      // Loop over the sides that form the boundary condition
      const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID_i = wsElNodeID[i];

      SideSetList& ssList = sideSets[i];
      std::map<std::string, std::vector<SideStruct>>::iterator ss_it = ssList.begin();

      while (ss_it != ssList.end()) {
        std::string             ss_key = ss_it->first;
        std::vector<SideStruct> ss_val = ss_it->second;
        
        Kokkos::View<LO****, PHX::Device>& globalDOFView = allLocalDOFViews[ss_key];

        for (unsigned int sideSet_idx = 0; sideSet_idx < ss_val.size(); ++sideSet_idx) {
          // Get the data that corresponds to the side
          const unsigned int elem_LID = ss_val[sideSet_idx].elem_LID;
          const unsigned int elem_side = ss_val[sideSet_idx].side_local_id;
          const CellTopologyData_Subcell& side =  cell_topo->side[elem_side];
          const unsigned int numSideNodes = side.topology->node_count;

          const Teuchos::ArrayRCP<GO>& elNodeID = wsElNodeID_i[elem_LID];

          //we only consider elements on the top.
          GO baseId;
          for (unsigned int j = 0; j < numSideNodes; ++j) {
            const std::size_t node = side.node[j];
            baseId = layeredMeshNumbering.getColumnId(elNodeID[node]);
            for (unsigned int il = 0; il < numLayers+1; ++il) {
              const GO gnode = layeredMeshNumbering.getId(baseId, il);
              const LO inode = ov_node_indexer.getLocalElement(gnode);
              for (unsigned int comp = 0; comp < numComps; ++comp) {
                globalDOFView(sideSet_idx + sideset_idx_offset[ss_key], j, il, comp) = solDOFManager.getLocalDOF(inode, comp);
              }
            }
          }
        }

        // Set workset-local sub-view
        std::pair<int,int> range(sideset_idx_offset[ss_key], sideset_idx_offset[ss_key]+ss_val.size());
        wsldofViews[ss_key] = Kokkos::subview(globalDOFView, range, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

        sideset_idx_offset[ss_key] += ss_val.size();

        ss_it++;
      }
    }
  }
  
}

unsigned
STKDiscretization::determine_local_side_id(
    const stk::mesh::Entity elem,
    stk::mesh::Entity       side)
{
  using namespace stk;

  stk::topology elem_top = bulkData->bucket(elem).topology();

  const unsigned num_elem_nodes = bulkData->num_nodes(elem);
  const unsigned num_side_nodes = bulkData->num_nodes(side);

  stk::mesh::Entity const* elem_nodes = bulkData->begin_nodes(elem);
  stk::mesh::Entity const* side_nodes = bulkData->begin_nodes(side);

  const stk::topology::rank_t side_rank = metaData->side_rank();

  int side_id = -1;

  if (num_elem_nodes == 0 || num_side_nodes == 0) {
    // Node relations are not present, look at elem->face

    const unsigned num_sides = bulkData->num_connectivity(elem, side_rank);
    stk::mesh::Entity const* elem_sides = bulkData->begin(elem, side_rank);

    for (unsigned i = 0; i < num_sides; ++i) {
      const stk::mesh::Entity elem_side = elem_sides[i];

      if (bulkData->identifier(elem_side) == bulkData->identifier(side)) {
        // Found the local side in the element
        side_id = static_cast<int>(i);
        return side_id;
      }
    }

    if (side_id < 0) {
      std::ostringstream msg;
      msg << "determine_local_side_id( ";
      msg << elem_top.name();
      msg << " , Element[ ";
      msg << bulkData->identifier(elem);
      msg << " ]{";
      for (unsigned i = 0; i < num_sides; ++i) {
        msg << " " << bulkData->identifier(elem_sides[i]);
      }
      msg << " } , Side[ ";
      msg << bulkData->identifier(side);
      msg << " ] ) FAILED";
      throw std::runtime_error(msg.str());
    }
  } else {  // Conventional elem->node - side->node connectivity present

    std::vector<unsigned> side_map;
    for (unsigned i = 0; side_id == -1 && i < elem_top.num_sides(); ++i) {
      stk::topology side_top = elem_top.side_topology(i);
      side_map.clear();
      elem_top.side_node_ordinals(i, std::back_inserter(side_map));

      if (num_side_nodes == side_top.num_nodes()) {
        side_id = i;

        for (unsigned j = 0;
             side_id == static_cast<int>(i) && j < side_top.num_nodes();
             ++j) {
          stk::mesh::Entity elem_node = elem_nodes[side_map[j]];

          bool found = false;

          for (unsigned k = 0; !found && k < side_top.num_nodes(); ++k) {
            found = elem_node == side_nodes[k];
          }

          if (!found) { side_id = -1; }
        }
      }
    }

    if (side_id < 0) {
      std::ostringstream msg;
      msg << "determine_local_side_id( ";
      msg << elem_top.name();
      msg << " , Element[ ";
      msg << bulkData->identifier(elem);
      msg << " ]{";
      for (unsigned i = 0; i < num_elem_nodes; ++i) {
        msg << " " << bulkData->identifier(elem_nodes[i]);
      }
      msg << " } , Side[ ";
      msg << bulkData->identifier(side);
      msg << " ]{";
      for (unsigned i = 0; i < num_side_nodes; ++i) {
        msg << " " << bulkData->identifier(side_nodes[i]);
      }
      msg << " } ) FAILED";
      throw std::runtime_error(msg.str());
    }
  }

  return static_cast<unsigned>(side_id);
}

void
STKDiscretization::computeNodeSets()
{
  std::map<std::string, stk::mesh::Part*>::iterator ns =
      stkMeshStruct->nsPartVec.begin();
  AbstractSTKFieldContainer::VectorFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();

  auto node_indexer = createGlobalLocalIndexer(m_node_vs);
  // A dof manager, to get the correct numbering (interleaved vs blocked)
  const auto dofMgr = getDOFManager(solution_dof_name());
  while (ns != stkMeshStruct->nsPartVec.end()) {  // Iterate over Node Sets
    // Get all owned nodes in this node set
    stk::mesh::Selector select_owned_in_nspart =
        stk::mesh::Selector(*(ns->second)) &
        stk::mesh::Selector(metaData->locally_owned_part());

    std::vector<stk::mesh::Entity> nodes;
    stk::mesh::get_selected_entities(
        select_owned_in_nspart,
        bulkData->buckets(stk::topology::NODE_RANK),
        nodes);

    nodeSets[ns->first].resize(nodes.size());
    nodeSetGIDs[ns->first].resize(nodes.size());
    nodeSetCoords[ns->first].resize(nodes.size());
    //    nodeSetIDs.push_back(ns->first); // Grab string ID
    *out << "STKDisc: nodeset " << ns->first << " has size " << nodes.size()
         << "  on Proc 0." << std::endl;
    for (std::size_t i = 0; i < nodes.size(); i++) {
      GO  node_gid              = bulkData->identifier(nodes[i]) - 1;
      int node_lid              = node_indexer->getLocalElement(node_gid);
      nodeSetGIDs[ns->first][i] = node_gid;
      nodeSets[ns->first][i].resize(neq);
      for (std::size_t eq = 0; eq < neq; ++eq) {
        nodeSets[ns->first][i][eq] = dofMgr.getLocalDOF(node_lid, eq);
      }
      nodeSetCoords[ns->first][i] =
          stk::mesh::field_data(*coordinates_field, nodes[i]);
    }
    ns++;
  }
}

void
STKDiscretization::setupExodusOutput()
{
#ifdef ALBANY_SEACAS
  if (stkMeshStruct->exoOutput) {
    outputInterval = 0;

    std::string str = stkMeshStruct->exoOutFile;

    Ioss::Init::Initializer io;

    mesh_data = Teuchos::rcp(
        new stk::io::StkMeshIoBroker(getMpiCommFromTeuchosComm(comm)));
    mesh_data->set_bulk_data(bulkData);
    //IKT, 8/16/19: The following is needed to get correct output file for Schwarz problems
    //Please see: https://github.com/trilinos/Trilinos/issues/5479
    mesh_data->property_add(Ioss::Property("FLUSH_INTERVAL", 1));
    outputFileIdx = mesh_data->create_output_mesh(str, stk::io::WRITE_RESULTS);

    // Adding mesh global variables
    /*
     * for (auto& it : field_container->getMeshVectorStates()) {
     *   const auto DV_Type = stk::util::ParameterType::DOUBLEVECTOR;
     *   boost::any mvs     = it.second;
     *   mesh_data->add_global(outputFileIdx, it.first, mvs, DV_Type);
     * }
     * for (auto& it : field_container->getMeshScalarIntegerStates()) {
     *   const auto INT_Type = stk::util::ParameterType::INTEGER;
     *   boost::any mvs      = it.second;
     *   mesh_data->add_global(outputFileIdx, it.first, mvs, INT_Type);
     * }
     */
    auto fc = stkMeshStruct->getFieldContainer();
    for (auto& it : fc->getMeshVectorStates()) {
      const auto DV_Type = stk::util::ParameterType::DOUBLEVECTOR;
      boost::any mvs     = it.second;
      mesh_data->add_global(outputFileIdx, it.first, mvs, DV_Type);
    }
    for (const auto& it : fc->getMeshScalarIntegerStates()) {
     const auto INT_Type = stk::util::ParameterType::INTEGER;
     boost::any ms      = it.second;
     mesh_data->add_global(outputFileIdx, it.first, ms, INT_Type);
    }
    for (const auto& it : fc->getMeshScalarInteger64States()) {
     const auto INT64_Type = stk::util::ParameterType::INT64;
     boost::any ms      = it.second;
     mesh_data->add_global(outputFileIdx, it.first, ms, INT64_Type);
    }

    // STK and Ioss/Exodus only allow TRANSIENT fields to be exported.
    // *Some* fields with MESH role are also allowed, but only if they
    // have a predefined name (e.g., "coordinates", "ids", "connectivity",...).
    // Therefore, we *ignore* all fields not marked as TRANSIENT.
    const stk::mesh::FieldVector& fields = mesh_data->meta_data().get_fields();
    for (size_t i = 0; i < fields.size(); i++) {
      auto attr = fields[i]->attribute<Ioss::Field::RoleType>();
      if (attr != nullptr && *attr == Ioss::Field::TRANSIENT) {
        mesh_data->add_field(outputFileIdx, *fields[i]);
      }
    }
  }

#else
  if (stkMeshStruct->exoOutput) {
    *out << "\nWARNING: exodus output requested but SEACAS not compiled in:"
         << " disabling exodus output \n";
  }
#endif
}

void
STKDiscretization::buildSideSetProjectors()
{
  // Note: the Global index of a node should be the same in both this and the
  // side discretizations
  //       since the underlying STK entities should have the same ID
  Teuchos::RCP<ThyraCrsMatrixFactory>   ov_graphP, graphP;
  Teuchos::RCP<Thyra_LinearOp>          P, ov_P;

  Teuchos::Array<GO> cols(1);
  Teuchos::Array<ST> vals(1);
  vals[0] = 1.0;

  // The global solution dof manager, to get the correct dof id (interleaved vs blocked)
  const auto dofMgr = getDOFManager(solution_dof_name());

  Teuchos::ArrayView<const GO> ss_indices;
  stk::mesh::EntityRank        SIDE_RANK = stkMeshStruct->metaData->side_rank();
  auto vs = getVectorSpace();
  auto ov_vs = getOverlapVectorSpace();
  for (auto it : sideSetDiscretizationsSTK) {
    // Extract the discretization
    const std::string&           sideSetName = it.first;
    const STKDiscretization&     disc        = *it.second;
    const AbstractSTKMeshStruct& ss_mesh     = *disc.stkMeshStruct;

    // Get the vector spaces
    auto ss_ov_vs   = disc.getOverlapVectorSpace();
    auto ss_vs      = disc.getVectorSpace();
    auto ss_node_vs = disc.getNodeVectorSpace();

    auto ss_indexer = createGlobalLocalIndexer(ss_vs);

    // A dof manager, to figure out interleaved vs blocked numbering
    const auto ss_dofMgr = disc.getDOFManager(solution_dof_name());

    // Extract the sides
    stk::mesh::Part&    part = *stkMeshStruct->ssPartVec.find(it.first)->second;
    stk::mesh::Selector selector =
        stk::mesh::Selector(part) &
        stk::mesh::Selector(stkMeshStruct->metaData->locally_owned_part());
    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        selector, stkMeshStruct->bulkData->buckets(SIDE_RANK), sides);

    // The projector: build both overlapped and non-overlapped range vs
    graphP = Teuchos::rcp(new ThyraCrsMatrixFactory(vs, ss_vs));
    ov_graphP = Teuchos::rcp(new ThyraCrsMatrixFactory(ov_vs, ss_ov_vs));

    const std::map<GO, GO>& side_cell_map = sideToSideSetCellMap.at(it.first);
    const std::map<GO, std::vector<int>>& node_numeration_map =
        sideNodeNumerationMap.at(it.first);
    std::set<GO> processed_node;
    GO           node_gid, ss_node_gid, side_gid, ss_cell_gid;
    std::pair<std::set<GO>::iterator, bool> check;
    stk::mesh::Entity                       ss_cell;
    for (auto side : sides) {
      side_gid    = stk_gid(side);
      ss_cell_gid = side_cell_map.at(side_gid);
      ss_cell     = ss_mesh.bulkData->get_entity(
          stk::topology::ELEM_RANK, ss_cell_gid + 1);

      int num_side_nodes = stkMeshStruct->bulkData->num_nodes(side);
      const stk::mesh::Entity* side_nodes =
          stkMeshStruct->bulkData->begin_nodes(side);
      const stk::mesh::Entity* ss_cell_nodes =
          ss_mesh.bulkData->begin_nodes(ss_cell);
      for (int i(0); i < num_side_nodes; ++i) {
        node_gid = stk_gid(side_nodes[i]);
        check    = processed_node.insert(node_gid);
        if (check.second) {
          // This node was not processed before. Let's do it.
          ss_node_gid =
              disc.stk_gid(ss_cell_nodes[node_numeration_map.at(side_gid)[i]]);

          for (int eq(0); eq < static_cast<int>(neq); ++eq) {
            cols[0] = dofMgr.getGlobalDOF(node_gid, eq);
            const GO row = ss_dofMgr.getGlobalDOF(ss_node_gid, eq);
            ov_graphP->insertGlobalIndices(row, cols());
            if (ss_indexer->isLocallyOwnedElement(row)) {
              graphP->insertGlobalIndices(row,cols());
            }
          }
        }
      }
    }

    // Fill graphs
    ov_graphP->fillComplete();
    graphP->fillComplete();

    ov_P = ov_graphP->createOp(true);
    assign(ov_P, 1.0);
    ov_projectors[sideSetName] = ov_P;

    P = graphP->createOp(true);
    assign(P, 1.0);
    projectors[sideSetName] = P;
  }
}

void
STKDiscretization::updateMesh()
{
  bulkData = stkMeshStruct->bulkData;

  const StateInfoStruct& nodal_param_states =
      stkMeshStruct->getFieldContainer()->getNodalParameterSIS();
  nodalDOFsStructContainer.addEmptyDOFsStruct(solution_dof_name(), "", neq);
  nodalDOFsStructContainer.addEmptyDOFsStruct(nodes_dof_name(), "", 1);
  for (size_t is = 0; is < nodal_param_states.size(); is++) {
    const StateStruct&            param_state = *nodal_param_states[is];
    const StateStruct::FieldDims& dim         = param_state.dim;
    int                           numComps    = 1;
    if (dim.size() == 3) {  // vector
      numComps = dim[2];
    } else if (dim.size() == 4) {  // tensor
      numComps = dim[2] * dim[3];
    }

    nodalDOFsStructContainer.addEmptyDOFsStruct(
        param_state.name, param_state.meshPart, numComps);
  }

  computeVectorSpaces();

#ifdef OUTPUT_TO_SCREEN
  // write owned maps to matrix market file for debug
  writeMatrixMarket(m_vs, "dof_vs");
  writeMatrixMarket(m_node_vs, "node_vs");
#endif
    
  setupMLCoords();

  transformMesh();

  computeGraphs();

  computeWorksetInfo();
#ifdef OUTPUT_TO_SCREEN
  printConnectivity();
#endif

  computeNodeSets();

  computeSideSets();

  setupExodusOutput();


#ifdef OUTPUT_TO_SCREEN
  printCoords();
#endif

  // If the mesh struct stores sideSet mesh structs, we update them
  if (stkMeshStruct->sideSetMeshStructs.size() > 0) {
    for (auto it : stkMeshStruct->sideSetMeshStructs) {
      Teuchos::RCP<STKDiscretization> side_disc =
          Teuchos::rcp(new STKDiscretization(discParams, neq, it.second, comm));
      side_disc->updateMesh();
      sideSetDiscretizations.insert(std::make_pair(it.first, side_disc));
      sideSetDiscretizationsSTK.insert(std::make_pair(it.first, side_disc));

      stkMeshStruct->buildCellSideNodeNumerationMap(
          it.first,
          sideToSideSetCellMap[it.first],
          sideNodeNumerationMap[it.first]);
    }

    buildSideSetProjectors();
  }
}

void
STKDiscretization::setFieldData(
      const AbstractFieldContainer::FieldContainerRequirements& req,
      const Teuchos::RCP<StateInfoStruct>& sis)
{
  Teuchos::RCP<AbstractSTKFieldContainer> fieldContainer = stkMeshStruct->getFieldContainer();

  Teuchos::RCP<MultiSTKFieldContainer<DiscType::Interleaved>> mISTKFieldContainer =
    Teuchos::rcp_dynamic_cast<MultiSTKFieldContainer<DiscType::Interleaved>>(fieldContainer,false);
  Teuchos::RCP<MultiSTKFieldContainer<DiscType::BlockedMono>> mBSTKFieldContainer =
    Teuchos::rcp_dynamic_cast<MultiSTKFieldContainer<DiscType::BlockedMono>>(fieldContainer,false);

  Teuchos::RCP<OrdinarySTKFieldContainer<DiscType::Interleaved>> oISTKFieldContainer =
    Teuchos::rcp_dynamic_cast<OrdinarySTKFieldContainer<DiscType::Interleaved>>(fieldContainer,false);
  Teuchos::RCP<OrdinarySTKFieldContainer<DiscType::BlockedMono>> oBSTKFieldContainer =
    Teuchos::rcp_dynamic_cast<OrdinarySTKFieldContainer<DiscType::BlockedMono>>(fieldContainer,false);

  Teuchos::RCP<GenericSTKFieldContainer<DiscType::Interleaved>> gISTKFieldContainer =
    Teuchos::rcp_dynamic_cast<GenericSTKFieldContainer<DiscType::Interleaved>>(fieldContainer,false);
  Teuchos::RCP<GenericSTKFieldContainer<DiscType::BlockedMono>> gBSTKFieldContainer =
    Teuchos::rcp_dynamic_cast<GenericSTKFieldContainer<DiscType::BlockedMono>>(fieldContainer,false);

  int num_time_deriv, numDim, num_params;
  Teuchos::RCP<Teuchos::ParameterList> params;
  
  if(Teuchos::nonnull(gISTKFieldContainer))
  {
    params = gISTKFieldContainer->getParams();
    numDim = gISTKFieldContainer->getNumDim();
    num_params = gISTKFieldContainer->getNumParams();
  }
  if(Teuchos::nonnull(gBSTKFieldContainer))
  {
    params = gBSTKFieldContainer->getParams();
    numDim = gBSTKFieldContainer->getNumDim();
    num_params = gBSTKFieldContainer->getNumParams();
  }

  num_time_deriv = params->get<int>("Number Of Time Derivatives");

  Teuchos::Array<std::string> default_solution_vector; // Empty
  Teuchos::Array<Teuchos::Array<std::string> > solution_vector;
  solution_vector.resize(num_time_deriv + 1);
  solution_vector[0] =
    params->get<Teuchos::Array<std::string> >("Solution Vector Components", default_solution_vector);


  if(num_time_deriv >= 1){
    solution_vector[1] =
      params->get<Teuchos::Array<std::string> >("SolutionDot Vector Components", default_solution_vector);
  }

  if(num_time_deriv >= 2){
    solution_vector[2] =
      params->get<Teuchos::Array<std::string> >("SolutionDotDot Vector Components", default_solution_vector);
  }


  if(Teuchos::nonnull(mISTKFieldContainer))
  {
    solutionFieldContainer = Teuchos::rcp(new MultiSTKFieldContainer<DiscType::Interleaved>(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, neq, numDim, sis, solution_vector, num_params));
  }
  if(Teuchos::nonnull(mBSTKFieldContainer))
  {
    solutionFieldContainer = Teuchos::rcp(new MultiSTKFieldContainer<DiscType::BlockedMono>(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, neq, numDim, sis, solution_vector, num_params));
  }
  if(Teuchos::nonnull(oISTKFieldContainer))
  {
    solutionFieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer<DiscType::Interleaved>(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, neq, req, numDim, sis, num_params));
  }
  if(Teuchos::nonnull(oBSTKFieldContainer))
  {
    solutionFieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer<DiscType::BlockedMono>(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, neq, req, numDim, sis, num_params));
  }
}

}  // namespace Albany
