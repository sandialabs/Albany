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
#include "STKConnManager.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <Shards_BasicTopologies.hpp>

#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_ElemFieldPattern.hpp>

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

#include <Teuchos_RCPStdSharedPtrConversions.hpp>

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
      discParams(discParams_)
{
  if (stkMeshStruct->sideSetMeshStructs.size() > 0) {
    for (auto it : stkMeshStruct->sideSetMeshStructs) {
      auto side_disc = Teuchos::rcp(new STKDiscretization(discParams, neq, it.second, comm));
      sideSetDiscretizations.insert(std::make_pair(it.first, side_disc));
      sideSetDiscretizationsSTK.insert(std::make_pair(it.first, side_disc));
    }
  }
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
      const auto& elem_lids = m_workset_elements.host();
      const auto& elem_gids = getDOFManager()->getAlbanyConnManager()->getElementsInBlock();

      std::ostringstream ss;

      ss << std::endl << "Process rank " << rank << std::endl;
      for (int ws=0; ws<elem_lids.extent_int(0); ++ws) {
        ss << "  Bucket (aka workset) " << ws << std::endl;
        for (int ielem=0; ielem<elem_lids.extent_int(1); ++ielem) {
          const int elem_LID = elem_lids(ws,ielem);
          const GO elem_GID = elem_gids[elem_LID];
          const auto e = bulkData->get_entity(stk::topology::ELEM_RANK,elem_GID+1);
          const auto nodes = bulkData->begin_nodes(e);
          const int num_nodes = bulkData->num_nodes(e);
          ss << "    Element " << ielem << ": Nodes =";
          for (int i=0; i<num_nodes; ++i) {
            ss << " " << bulkData->identifier(nodes[i])-1;
          }
          ss << std::endl;
        }
      }
      std::cout << ss.str();
    }
    comm->barrier();
  }
}

void
STKDiscretization::printCoords() const
{
  std::cout << "Processor " << bulkData->parallel_rank() << " has "
            << m_ws_elem_coords.size() << " worksets.\n";

  const int numDim = stkMeshStruct->numDim;
  double xmin = std::numeric_limits<double>::max(), xmax = std::numeric_limits<double>::lowest(),
      ymin = xmin, ymax = xmax, zmin = xmin, zmax = xmax;
  const auto& coords = m_ws_elem_coords;
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
  constexpr auto ELEM_RANK = stk::topology::ELEM_RANK;

  const auto& coordinates_field = *stkMeshStruct->getCoordinatesField();
  const int meshDim = stkMeshStruct->numDim;

  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& elem_lids = node_dof_mgr->elem_dof_lids().host();
  const int num_nodes = elem_lids.extent_int(1);
  const auto& elems = node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int num_elems = elems.size();
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto elem = bulkData->get_entity(ELEM_RANK,elems[ielem]+1);
    const auto nodes = bulkData->begin_nodes(elem);
    for (int node=0; node<num_nodes; ++node) {
      double* x = stk::mesh::field_data(coordinates_field, nodes[node]);
      for (int dim=0; dim<meshDim; ++dim) {
        coordinates[meshDim*elem_lids(ielem,node) + dim] = x[dim];
      }
    }
  }

  return coordinates;
}

// The function transformMesh() maps a unit cube domain by applying a
// transformation to the mesh.
void
STKDiscretization::transformMesh()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: transformMesh");
  using std::cout;
  using std::endl;
  AbstractSTKFieldContainer::STKFieldType* coordinates_field =
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
   => finer boundary layer near x = 0.  If beta = 0, no transformation is
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
        const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
        const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
        const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
    const auto numOverlapNodes = getLocalSubdim(getOverlapNodeVectorSpace());
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
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: setupMLCoords");
  if (rigidBodyModes.is_null()) { return; }
  if (!rigidBodyModes->isMLUsed() && !rigidBodyModes->isMueLuUsed() && !rigidBodyModes->isFROSchUsed()) { return; }

  const int                                   numDim = stkMeshStruct->numDim;
  AbstractSTKFieldContainer::STKFieldType* coordinates_field =
      stkMeshStruct->getCoordinatesField();
  coordMV           = Thyra::createMembers(getNodeVectorSpace(), numDim);
  auto coordMV_data = getNonconstLocalData(coordMV);

  // NOTE: you cannot use DOFManager dof gids as entity ID in stk, and viceversa.
  // All you can do is loop over dofs/nodes in an element, since you have the following guarantees:
  //  - elem GIDs are the same in DOFManager and stk mesh
  //  - nodes ordering is the same in DOFManager and stk mesh
  // We'll loop over certain nodes more than once, but this is a setup method, so it's fine
  const auto& node_dof_mgr = getNodeDOFManager();
  const auto& elems = node_dof_mgr->getAlbanyConnManager()->getElementsInBlock();
  const int   num_elems = elems.size();
  for (int ielem=0; ielem<num_elems; ++ielem) {
    const auto& node_dofs = node_dof_mgr->getElementGIDs(ielem);
    const auto e = bulkData->get_entity(stk::topology::ELEM_RANK,elems[ielem]+1);
    const auto nodes = bulkData->begin_nodes (e);
    const int num_nodes = bulkData->num_nodes(e);
    for (int i=0; i<num_nodes; ++i) {
      LO node_lid = node_dof_mgr->indexer()->getLocalElement(node_dofs[i]);
      if (node_lid>=0) {
        double* X = stk::mesh::field_data(*coordinates_field,nodes[i]);
        for (int j=0; j<numDim; ++j) {
          coordMV_data[j][node_lid] = X[j];
        }
      }
    }
  }

  rigidBodyModes->setCoordinatesAndComputeNullspace(
      coordMV,
      getVectorSpace(),
      getOverlapVectorSpace());

  writeCoordsToMatrixMarket();
}

void
STKDiscretization::writeCoordsToMatrixMarket() const
{
#ifdef ALBANY_DISABLE_OUTPUT_MESH
  *out << "[STKDiscretization::writeCoordsToMatrixMarket] ALBANY_DISABLE_OUTPUT_MESH=TRUE. Skip.\n";
#else
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
#endif
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double        time,
    const bool          overlapped,
    const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dxdp?
  writeSolutionToFile(soln, time, overlapped, force_write_solution);
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const double        time,
    const bool          overlapped,
    const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dot and/or soln_dxdp?
  writeSolutionToFile(soln, time, overlapped, force_write_solution);
}

void
STKDiscretization::writeSolution(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const double        time,
    const bool          overlapped,
    const bool          force_write_solution)
{
  writeSolutionToMeshDatabase(soln, soln_dxdp, soln_dot, soln_dotdot, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dot and soln_dotdot?
  writeSolutionToFile(soln, time, overlapped, force_write_solution);
}

void
STKDiscretization::writeSolutionMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const double             time,
    const bool               overlapped,
    const bool               force_write_solution)
{
  writeSolutionMVToMeshDatabase(soln, soln_dxdp, time, overlapped);
  // IKT, FIXME? extend writeSolutionToFile to take in soln_dxdp?
  writeSolutionMVToFile(soln, time, overlapped, force_write_solution);
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
    const bool          overlapped,
    const bool          force_write_solution)
{
#ifdef ALBANY_DISABLE_OUTPUT_MESH
  *out << "[STKDiscretization::writeSolutionToFile] ALBANY_DISABLE_OUTPUT_MESH=TRUE. Skip.\n";
  (void) soln;
  (void) time;
  (void) overlapped;
  (void) force_write_solution;
#else
#ifdef ALBANY_SEACAS
  TEUCHOS_FUNC_TIME_MONITOR("Albany: write solution to file");
  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {
    solutionFieldContainer->transferSolutionToCoords();

    if (!mesh_data.is_null()) {
      // Mesh coordinates have changed. Rewrite output file by deleting the mesh
      // data object and recreate it
      setupExodusOutput();
    }
  }

  // Skip this write unless the proper interval has been reached
  if ((stkMeshStruct->exoOutput &&
      !(outputInterval % stkMeshStruct->exoOutputInterval)) ||
      (force_write_solution == true) ) {
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
      mesh_data->write_global(outputFileIdx, it.first, static_cast<int64_t>(it.second), stk::util::ParameterType::INT64);
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
      it.second->writeSolutionToFile(*ss_soln, time, overlapped, force_write_solution);
    } else {
      auto ss_soln = Thyra::createMember(it.second->getVectorSpace());
      const Thyra_LinearOp& P = *projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionToFile(*ss_soln, time, overlapped, force_write_solution);
    }
  }
#endif
#endif
}

void
STKDiscretization::writeSolutionMVToFile(
    const Thyra_MultiVector& soln,
    const double             time,
    const bool               overlapped,
    const bool               force_write_solution)
{
#ifdef ALBANY_DISABLE_OUTPUT_MESH
  *out << "[STKDiscretization::writeSolutionMVToFile] ALBANY_DISABLE_OUTPUT_MESH=TRUE. Skip.\n";
  (void) soln;
  (void) time;
  (void) overlapped;
  (void) force_write_solution;
#else
#ifdef ALBANY_SEACAS
  TEUCHOS_FUNC_TIME_MONITOR("Albany: write solution MV to file");

  if (stkMeshStruct->exoOutput && stkMeshStruct->transferSolutionToCoords) {
    solutionFieldContainer->transferSolutionToCoords();

    if (!mesh_data.is_null()) {
      // Mesh coordinates have changed. Rewrite output file by deleting the mesh
      // data object and recreate it
      setupExodusOutput();
    }
  }

  // Skip this write unless the proper interval has been reached
  if ((stkMeshStruct->exoOutput &&
      !(outputInterval % stkMeshStruct->exoOutputInterval)) ||
      (force_write_solution == true) ) {
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
      mesh_data->write_global(outputFileIdx, it.first, static_cast<int64_t>(it.second), stk::util::ParameterType::INT64);
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
      it.second->writeSolutionMVToFile(*ss_soln, time, overlapped, force_write_solution);
    } else {
      auto ss_soln = Thyra::createMembers(
          it.second->getVectorSpace(), soln.domain()->dim());
      const Thyra_LinearOp& P = *projectors.at(it.first);
      P.apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
      it.second->writeSolutionMVToFile(*ss_soln, time, overlapped, force_write_solution);
    }
  }
#endif
#endif
}

void STKDiscretization::
addSolutionField (const std::string & /* fieldName */,
                  const std::string & /* blockId */)
{
#if 0
   TEUCHOS_TEST_FOR_EXCEPTION(!validBlockId(blockId),ElementBlockException,
                      "Unknown element block \"" << blockId << "\"");
   std::pair<std::string,std::string> key = std::make_pair(fieldName,blockId);

   // add & declare field if not already added...currently assuming linears
   if(fieldNameToSolution_.find(key)==fieldNameToSolution_.end()) {
      SolutionFieldType * field = metaData_->get_field<double>(stk::topology::NODE_RANK, fieldName);
      if(field==0)
         field = &metaData_->declare_field<double>(stk::topology::NODE_RANK, fieldName);
      if ( initialized_ )  {
        metaData_->enable_late_fields();
        stk::mesh::FieldTraits<SolutionFieldType>::data_type* init_sol = nullptr;
        stk::mesh::put_field_on_mesh(*field, metaData_->universal_part(),init_sol );
      }
      fieldNameToSolution_[key] = field;
   }
#endif
}

void STKDiscretization::
addCellField (const std::string & /* fieldName */,
              const std::string & /* blockId */)
{
#if 0
   TEUCHOS_TEST_FOR_EXCEPTION(!validBlockId(blockId),ElementBlockException,
                      "Unknown element block \"" << blockId << "\"");
   std::pair<std::string,std::string> key = std::make_pair(fieldName,blockId);

   // add & declare field if not already added...currently assuming linears
   if(fieldNameToCellField_.find(key)==fieldNameToCellField_.end()) {
      SolutionFieldType * field = metaData_->get_field<double>(stk::topology::ELEMENT_RANK, fieldName);
      if(field==0)
         field = &metaData_->declare_field<double>(stk::topology::ELEMENT_RANK, fieldName);

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
  Teuchos::RCP<Thyra_Vector> soln = Thyra::createMember(getVectorSpace());
  this->getSolutionField(*soln, overlapped);
  return soln;
}

Teuchos::RCP<Thyra_MultiVector>
STKDiscretization::getSolutionMV(bool overlapped) const
{
  // Copy soln vector into solution field, one node at a time
  int num_time_deriv = stkMeshStruct->num_time_deriv;
  Teuchos::RCP<Thyra_MultiVector> soln =
      Thyra::createMembers(getVectorSpace(), num_time_deriv + 1);
  this->getSolutionMV(*soln, overlapped);
  return soln;
}

void
STKDiscretization::getField(Thyra_Vector& result, const std::string& name) const
{
  auto dof_mgr = getDOFManager(name);
  solutionFieldContainer->fillVector(result, name, dof_mgr, false);
}

void
STKDiscretization::getSolutionField(Thyra_Vector& result, const bool overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  solutionFieldContainer->fillSolnVector(result, getDOFManager(), overlapped);
}

void
STKDiscretization::getSolutionMV(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  TEUCHOS_TEST_FOR_EXCEPTION(overlapped, std::logic_error, "Not implemented.");

  solutionFieldContainer->fillSolnMultiVector(result, getDOFManager(), overlapped);
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
  const auto dof_mgr = getDOFManager(name);
  solutionFieldContainer->saveVector(result,name,dof_mgr,overlapped);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool          overlapped)
{
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, dof_mgr, overlapped);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const bool          overlapped)
{
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, dof_mgr, overlapped);
}

void
STKDiscretization::setSolutionField(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const bool          overlapped)
{
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, soln_dotdot, dof_mgr, overlapped);
}

void
STKDiscretization::setSolutionFieldMV(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool               overlapped)
{
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnMultiVector(soln, soln_dxdp, dof_mgr, overlapped);
}

void STKDiscretization::computeVectorSpaces()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeVectorSpaces");
  // NOTE: in Albany we use the mesh part name "" to refer to the whole mesh.
  //       That's not the name that stk uses for the whole mesh. So if the
  //       dof part name is "", we get the part stored in the stk mesh struct
  //       for the element block, where we REQUIRE that there is only ONE element block.
  TEUCHOS_TEST_FOR_EXCEPTION (stkMeshStruct->ebNames_.size() > 1,std::logic_error,
      "Error! We currently only support meshes with 1 element block.\n");
  TEUCHOS_TEST_FOR_EXCEPTION (stkMeshStruct->ebNames_.size() < 1,std::logic_error,
      "Error! Albany problems must have at least 1 element block.  Your mesh has 0 element blocks.\n");

  strmap_t<std::pair<std::string,int>> name_to_partAndDim;
  name_to_partAndDim[solution_dof_name()] = std::make_pair("",neq);
  name_to_partAndDim[nodes_dof_name()] = std::make_pair("",1);
  for (const auto& sis : stkMeshStruct->getFieldContainer()->getNodalParameterSIS()) {
    const auto& dims = sis->dim;
    int dof_dim = -1;
    switch (dims.size()) {
      case 2: dof_dim = 1;               break;
      case 3: dof_dim = dims[2];         break;
      case 4: dof_dim = dims[2]*dims[3]; break;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Error! Unsupported layout for nodal parameter '" + sis->name + ".\n");
    }

    name_to_partAndDim[sis->name] = std::make_pair(sis->meshPart,dof_dim);
  }

  for (const auto& it : name_to_partAndDim) {
    const auto& field_name = it.first;
    const auto& part_name  = it.second.first;
    const auto& dof_dim    = it.second.second;

    // NOTE: for now we hard code P1. In the future, we must be able to
    //       store this info somewhere and retrieve it here.
    auto dof_mgr = create_dof_mgr(part_name,field_name,FE_Type::HGRAD,1,dof_dim);
    m_dof_managers[field_name][part_name] = dof_mgr;
    m_node_dof_managers[part_name] = Teuchos::null;
  }

  // For each part, also make a Node dof manager
  for (auto& it : m_node_dof_managers) {
    const auto& part_name = it.first;
    it.second = create_dof_mgr(part_name, "node", FE_Type::HGRAD,1,1);
  }

  coordinates.resize(3 * getLocalSubdim(getOverlapNodeVectorSpace()));
}

void
STKDiscretization::computeGraphs()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeGraphs");
  const auto vs = getVectorSpace();
  const auto ov_vs = getOverlapVectorSpace();
  m_jac_factory = Teuchos::rcp(new ThyraCrsMatrixFactory(vs, vs, ov_vs, ov_vs));

  // Determine which equations are defined on the whole domain,
  // as well as what eqn are on each sideset
  std::vector<int> volumeEqns;
  std::map<std::string,std::vector<int>> ss_to_eqns;
  for (int k=0; k < neq; ++k) {
    if (sideSetEquations.find(k) == sideSetEquations.end()) {
      volumeEqns.push_back(k);
    }
  }
  const int numVolumeEqns = volumeEqns.size();

  // The global solution dof manager
  const auto sol_dof_mgr = getDOFManager();
  const int num_elems = sol_dof_mgr->cell_indexer()->getNumLocalElements();

  // Handle the simple case, and return immediately
  if (numVolumeEqns==neq) {
    // This is the easy case: couple everything with everything
    for (int icell=0; icell<num_elems; ++icell) {
      const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);
      m_jac_factory->insertGlobalIndices(elem_gids,elem_gids,true);
    }
    m_jac_factory->fillComplete();
    return;
  }

  // Ok, if we're here there is at least 1 side equation
  Teuchos::Array<GO> rows,cols;

  // First, couple global eqn (row) coupled with global eqn (col)
  for (int icell=0; icell<num_elems; ++icell) {
    const auto& elem_gids = sol_dof_mgr->getElementGIDs(icell);

    for (int ieq=0; ieq<numVolumeEqns; ++ieq) {

      // Couple eqn=ieq with itself
      const auto& row_gids_offsets = sol_dof_mgr->getGIDFieldOffsets(volumeEqns[ieq]);
      const int num_row_gids = row_gids_offsets.size();
      rows.resize(num_row_gids);
      for (int idof=0; idof<num_row_gids; ++idof) {
        rows[idof] = elem_gids[row_gids_offsets[idof]];
      }
      m_jac_factory->insertGlobalIndices(rows(),rows(),false);

      // Couple eqn=ieq with eqn=jeq!=ieq
      for (int jeq=0; jeq<numVolumeEqns; ++jeq) {
        const auto& col_gids_offsets = sol_dof_mgr->getGIDFieldOffsets(jeq);
        const int num_col_gids = col_gids_offsets.size();
        cols.resize(num_col_gids);
        for (int jdof=0; jdof<num_col_gids; ++jdof) {
          cols[jdof] = elem_gids[col_gids_offsets[jdof]];
        }
        m_jac_factory->insertGlobalIndices(rows(),cols(),true);
      }
    }

    // While at it, for side set equations, set the diag entry, so that jac pattern
    // is for sure non-singular in the volume.
    for (const auto& it : sideSetEquations) {
      int eq = it.first;
      const auto& eq_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
      for (auto o : eq_offsets) {
        GO row = elem_gids[o];
        m_jac_factory->insertGlobalIndices(row,row,false);
      }
    }
  }

  // Now, process rows/cols corresponding to ss equations
  const auto& cell_layers_data_lid = stkMeshStruct->local_cell_layers_data;
  const auto& cell_layers_data_gid = stkMeshStruct->global_cell_layers_data;
  const auto SIDE_RANK = metaData->side_rank();
  for (const auto& it : sideSetEquations) {
    const int side_eq = it.first;

    // If the side eqn is column-coupled, it needs special treatment.
    // A side eqn can be coupled to the whole column if
    //   1) the mesh is layered, AND
    //   2) all sidesets where it's defined are on the top or bottom
    int allowColumnCoupling = not cell_layers_data_lid.is_null();
    if (not cell_layers_data_lid.is_null()) {
      for (const auto& ss_name : it.second) {
        std::vector<stk::mesh::Entity> sides;
        stk::mesh::Selector sel (*stkMeshStruct->ssPartVec.at(ss_name));
        stk::mesh::get_selected_entities(sel,bulkData()->buckets(SIDE_RANK),sides);
        if (sides.size()==0) {
          // This rank owns 0 sides on this sideset
          continue;
        }

        // Grab any side of this side set and check layerId and pos within element
        const auto& s = sides[0];
        const auto& e = bulkData->begin_elements(s)[0];
        const auto pos = determine_entity_pos(e,s);
        const auto layer = cell_layers_data_gid->getLayerId(stk_gid(e));

        if (layer==(cell_layers_data_lid->numLayers-1)) {
          allowColumnCoupling = pos==cell_layers_data_lid->top_side_pos;
        } else if (layer==0) {
          allowColumnCoupling = pos==cell_layers_data_lid->bot_side_pos;
        } else {
          // The mesh is layered, but this sideset is niether top nor bottom
          allowColumnCoupling = false;
        }
      }
    }
    // NOTE: Teuchos::reduceAll does not accept bool Packet, despite offerint REDUCE_AND as reduction op, so use int
    int globalAllowColumnCoupling = allowColumnCoupling;
    Teuchos::reduceAll(*comm(),Teuchos::REDUCE_AND,1,&allowColumnCoupling,&globalAllowColumnCoupling);

    // Loop over all side sets where this eqn is defined
    for (const auto& ss_name : it.second) {
      for (int ws=0; ws<getNumWorksets(); ++ws) {
        const auto& elem_lids = getElementLIDs_host(ws);
        const auto& ss = sideSets[ws].at(ss_name);

        // Loop over all sides in this side set
        for (const auto& side : ss) {
          const LO ws_elem_idx = side.ws_elem_idx;
          const LO elem_LID = elem_lids(ws_elem_idx);
          const auto& side_elem_gids = sol_dof_mgr->getElementGIDs(elem_LID);
          const int side_pos = side.side_pos;
          const auto& side_eq_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(side_eq,side_pos);

          // Compute row GIDs
          const int num_row_gids = side_eq_offsets.size();
          rows.resize(num_row_gids);
          for (int idof=0; idof<num_row_gids; ++idof) {
            rows[idof] = side_elem_gids[side_eq_offsets[idof]];
          }

          if (globalAllowColumnCoupling) {
            // Assume the worst, and couple with all eqns over the whole column
            const int numLayers = cell_layers_data_lid->numLayers;
            const LO basal_elem_LID = cell_layers_data_lid->getColumnId(elem_LID);
            for (int eq=0; eq<neq; ++eq) {
              const auto& eq_offsets = sol_dof_mgr->getGIDFieldOffsets(eq);
              const int num_col_gids = eq_offsets.size();
              cols.resize(num_col_gids);
              for (int il=0; il<numLayers; ++il) {
                const LO layer_elem_lid = cell_layers_data_lid->getId(basal_elem_LID,il);
                const auto& elem_gids = sol_dof_mgr->getElementGIDs(layer_elem_lid);

                for (int jdof=0; jdof<num_col_gids; ++jdof) {
                  cols[jdof] = elem_gids[eq_offsets[jdof]];
                }
                m_jac_factory->insertGlobalIndices(rows(),cols(),true);
              }
            }
          } else {

            // Add local coupling (on this side) with all eqns
            // NOTE: we could be fancier, and couple only with volume eqn or side eqn that are defined
            //       on this side set. However, if a sideset is a subset of another, we might miss the
            //       coupling since the side sets have different names. We'd have to inspect if a ss is
            //       contained in the other, but that starts to get too involved. Given that it's not
            //       a common scenario (need 2+ ss eqn defined on 2 different sidesets), and that we
            //       might have to redo this when we assemble by blocks, we just don't bother.
            for (int col_eq=0; col_eq<neq; ++col_eq) {
              const auto& col_eq_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(col_eq,side_pos);
              const int num_col_gids = col_eq_offsets.size();
              cols.resize(num_col_gids);
              for (int jdof=0; jdof<num_col_gids; ++jdof) {
                cols[jdof] = side_elem_gids[col_eq_offsets[jdof]];
              }

              m_jac_factory->insertGlobalIndices(rows(),cols(),true);
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
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeWorksetInfo");
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;

  stk::mesh::Selector select_owned_in_part =
      stk::mesh::Selector(metaData->universal_part()) &
      stk::mesh::Selector(metaData->locally_owned_part());

  const stk::mesh::BucketVector& buckets =
      bulkData->get_buckets(stk::topology::ELEMENT_RANK, select_owned_in_part);

  const int numBuckets = buckets.size();

  m_workset_sizes.resize(numBuckets);
  stk::mesh::Bucket::size_type max_ws_size = 0;
  for (int b=0; b<numBuckets; ++b) {
    max_ws_size = std::max(max_ws_size,buckets[b]->size());
  }
  m_workset_elements = DualView<int**>("ws_elem",numBuckets,max_ws_size);
  for (int b=0,lid=0; b<numBuckets; ++b) {
    const auto& bucket = *buckets[b];
    m_workset_sizes[b] = bucket.size();
    // NOTE: STKConnManager is built in such a way that the elements
    //       in the conn mgr are ordered by bucket, and, if in the same
    //       bucket, they are ordered in the same way.
    //       That means that, if elem E1 is in bucket 3 and elem E2 is
    //       in bucket 4, then elemLid(E1)<elemLid(E2). If they are in
    //       the same bucket, then elemLid(E1)<elemLid(E2) IIF E1 is
    //       listed first in the bucket.
    for (unsigned ie=0; ie<bucket.size(); ++ie, ++lid) {
      m_workset_elements.host()(b,ie) = lid;
    }
    // Fill the remainder (if any) with very invalid numbers
    for (unsigned ie=bucket.size(); ie<m_workset_elements.host().extent(1); ++ie) {
      m_workset_elements.host()(b,ie) = -1;
    }
  }
  m_workset_elements.sync_to_dev();

  using STKFieldType = AbstractSTKFieldContainer::STKFieldType;

  STKFieldType* coordinates_field = stkMeshStruct->getCoordinatesField();

  m_wsEBNames.resize(numBuckets);
  for (int i = 0; i < numBuckets; i++) {
    stk::mesh::PartVector const& bpv = buckets[i]->supersets();

    for (std::size_t j = 0; j < bpv.size(); j++) {
      if (bpv[j]->primary_entity_rank() == stk::topology::ELEMENT_RANK &&
          !stk::mesh::is_auto_declared_part(*bpv[j])) {
        // *out << "Bucket " << i << " is in Element Block:  " << bpv[j]->name()
        //      << "  and has " << buckets[i]->size() << " elements." <<
        //      std::endl;
        m_wsEBNames[i] = bpv[j]->name();
      }
    }
  }

  m_wsPhysIndex.resize(numBuckets);
  if (stkMeshStruct->allElementBlocksHaveSamePhysics) {
    for (int i = 0; i < numBuckets; ++i) { m_wsPhysIndex[i] = 0; }
  } else {
    for (int i = 0; i < numBuckets; ++i) {
      m_wsPhysIndex[i] =
          stkMeshStruct->getMeshSpecs()[0]->ebNameToIndex[m_wsEBNames[i]];
    }
  }

  m_ws_elem_coords.resize(numBuckets);

  nodesOnElemStateVec.resize(numBuckets);
  m_stateArrays.elemStateArrays.resize(numBuckets);
  const StateInfoStruct& nodal_states =
      stkMeshStruct->getFieldContainer()->getNodalSIS();

  // Clear map if remeshing
  if (!elemGIDws.empty()) { elemGIDws.clear(); }

  typedef stk::mesh::Cartesian NodeTag;
  typedef stk::mesh::Cartesian ElemTag;
  typedef stk::mesh::Cartesian CompTag;

  for (int b = 0; b < numBuckets; b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    m_ws_elem_coords[b].resize(buck.size());

    nodesOnElemStateVec[b].resize(nodal_states.size());

    for (size_t is = 0; is < nodal_states.size(); ++is) {
      const std::string&            name = nodal_states[is]->name;
      const StateStruct::FieldDims& dim  = nodal_states[is]->dim;
      auto& state = m_stateArrays.elemStateArrays[b][name];
      std::vector<double>& stateVec = nodesOnElemStateVec[b][is];
      int dim0 = buck.size();  // may be different from dim[0];
      const auto& field = *metaData->get_field<double>(NODE_RANK, name);
      switch (dim.size()) {
        case 2:  // scalar
        {          
          stateVec.resize(dim0 * dim[1]);
          state.reset_from_host_ptr(stateVec.data(),dim0,dim[1]);
          auto state_h = state.host();
          for (int i = 0; i < dim0; i++) {
            stk::mesh::Entity        element = buck[i];
            stk::mesh::Entity const* rel     = bulkData->begin_nodes(element);
            for (int j = 0; j < static_cast<int>(dim[1]); j++) {
              stk::mesh::Entity rowNode = rel[j];
              state_h(i, j) = *stk::mesh::field_data(field, rowNode);
            }
          }
          state.sync_to_dev();
          break;
        }
        case 3:  // vector
        {
          stateVec.resize(dim0 * dim[1] * dim[2]);
          state.reset_from_host_ptr(stateVec.data(),dim0,dim[1],dim[2]);
          auto state_h = state.host();
          for (int i = 0; i < dim0; i++) {
            stk::mesh::Entity        element = buck[i];
            stk::mesh::Entity const* rel     = bulkData->begin_nodes(element);
            for (int j = 0; j < static_cast<int>(dim[1]); j++) {
              stk::mesh::Entity rowNode = rel[j];
              double*           entry = stk::mesh::field_data(field, rowNode);
              for (int k = 0; k < static_cast<int>(dim[2]); k++) {
                state_h(i, j, k) = entry[k];
              }
            }
          }
          state.sync_to_dev();
          break;
        }
        case 4:  // tensor
        {
          stateVec.resize(dim0 * dim[1] * dim[2] * dim[3]);
          state.reset_from_host_ptr(stateVec.data(),dim0,dim[1],dim[2],dim[3]);
          auto state_h = state.host();
          for (int i = 0; i < dim0; i++) {
            stk::mesh::Entity        element = buck[i];
            stk::mesh::Entity const* rel     = bulkData->begin_nodes(element);
            for (int j = 0; j < static_cast<int>(dim[1]); j++) {
              stk::mesh::Entity rowNode = rel[j];
              double*           entry = stk::mesh::field_data(field, rowNode);
              for (int k = 0; k < static_cast<int>(dim[2]); k++) {
                for (int l = 0; l < static_cast<int>(dim[3]); l++) {
                  state_h(i, j, k, l) = entry[k * dim[3] + l];  // check this,
                                                              // is stride
                                                              // Correct?
                }
              }
            }
          }
          state.sync_to_dev();
          break;
        }
      }
    }
    
    stk::mesh::Entity element = buck[0];

    // i is the element index within bucket b
    for (std::size_t i = 0; i < buck.size(); i++) {
      // Traverse all the elements in this bucket
      element = buck[i];

      // Now, save a map from element GID to workset on this PE
      elemGIDws[stk_gid(element)].ws = b;

      // Now, save a map from element GID to local id on this workset on this PE
      elemGIDws[stk_gid(element)].LID = i;

      // Set coords at nodes
      const auto* nodes = bulkData->begin_nodes(element);
      const int num_nodes = bulkData->num_nodes(element);
      m_ws_elem_coords[b][i].resize(num_nodes);
      for (int j=0; j<num_nodes; ++j) {
        m_ws_elem_coords[b][i][j] = stk::mesh::field_data(*coordinates_field, nodes[j]);
      }
    }
  }

  for (int d = 0; d < stkMeshStruct->numDim; d++) {
    if (stkMeshStruct->PBCStruct.periodic[d]) {
      for (int b = 0; b < numBuckets; b++) {
        auto has_sheight = m_stateArrays.elemStateArrays[b].count("surface_height")==1;
        DualDynRankView<double> sHeight;
        if (has_sheight) {
          sHeight = m_stateArrays.elemStateArrays[b]["surface_height"];
        }
        for (std::size_t i = 0; i < buckets[b]->size(); i++) {
          int  nodes_per_element = buckets[b]->num_nodes(i);
          bool anyXeqZero        = false;
          for (int j = 0; j < nodes_per_element; j++)
            if (m_ws_elem_coords[b][i][j][d] == 0.0) anyXeqZero = true;
          if (anyXeqZero) {
            bool flipZeroToScale = false;
            for (int j = 0; j < nodes_per_element; j++)
              if (m_ws_elem_coords[b][i][j][d] > stkMeshStruct->PBCStruct.scale[d] / 1.9)
                flipZeroToScale = true;
            if (flipZeroToScale) {
              for (int j = 0; j < nodes_per_element; j++) {
                if (m_ws_elem_coords[b][i][j][d] == 0.0) {
                  double* xleak = new double[stkMeshStruct->numDim];
                  for (int k = 0; k < stkMeshStruct->numDim; k++)
                    if (k == d)
                      xleak[d] = stkMeshStruct->PBCStruct.scale[d];
                    else
                      xleak[k] = m_ws_elem_coords[b][i][j][k];
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
                    if (has_sheight) {
                      sHeight.host()(i, j) -=
                          stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                    }
                  }
                  m_ws_elem_coords[b][i][j] = xleak;  // replace ptr to coords
                  toDelete.push_back(xleak);
        }}}}}
        if (has_sheight) {
          sHeight.sync_to_dev();
        }
      }
    }
  }

  using ValueState = AbstractSTKFieldContainer::ValueState;
  using STKState = AbstractSTKFieldContainer::STKState;

  // Pull out pointers to shards::Arrays for every bucket, for every state
  // Code is data-type dependent

  AbstractSTKFieldContainer& container = *stkMeshStruct->getFieldContainer();

  ValueState& scalarValue_states = container.getScalarValueStates();
  STKState&   cell_scalar_states = container.getCellScalarStates();
  STKState&   cell_vector_states = container.getCellVectorStates();
  STKState&   cell_tensor_states = container.getCellTensorStates();
  STKState&   qpscalar_states    = container.getQPScalarStates();
  STKState&   qpvector_states    = container.getQPVectorStates();
  STKState&   qptensor_states    = container.getQPTensorStates();
  std::map<std::string, double>& time  = container.getTime();

  // Setup state arrays DualDynRankView's, so that host view stores a pointer
  // to the stk field data
  for (std::size_t b = 0; b < buckets.size(); b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    for (auto& css : cell_scalar_states) {
      auto data = reinterpret_cast<double*>(buck.field_data_location(*css));

      auto& state = m_stateArrays.elemStateArrays[b][css->name()];
      state.reset_from_host_ptr(data,buck.size());
    }
    for (auto& cvs : cell_vector_states) {
      auto data = reinterpret_cast<double*>(buck.field_data_location(*cvs));
      auto vec_dim = stk::mesh::field_scalars_per_entity(*cvs,buck);

      auto& state = m_stateArrays.elemStateArrays[b][cvs->name()];
      state.reset_from_host_ptr(data,buck.size(),vec_dim);
    }
    for (auto& cts : cell_tensor_states) {
      auto data = reinterpret_cast<double*>(buck.field_data_location(*cts));
      auto dim0 = stk::mesh::field_extent0_per_entity(*cts,buck);
      auto dim1 = stk::mesh::field_extent1_per_entity(*cts,buck);

      auto& state = m_stateArrays.elemStateArrays[b][cts->name()];
      state.reset_from_host_ptr(data,buck.size(),dim0,dim1);
    }
    for (auto& qpss : qpscalar_states) {
      auto data = reinterpret_cast<double*>(buck.field_data_location(*qpss));
      auto num_qps = stk::mesh::field_scalars_per_entity(*qpss,buck);

      auto& state = m_stateArrays.elemStateArrays[b][qpss->name()];
      state.reset_from_host_ptr(data,buck.size(),num_qps);
    }
    for (auto& qpvs : qpvector_states) {
      auto data = reinterpret_cast<double*>(buck.field_data_location(*qpvs));
      auto num_qps = stk::mesh::field_extent0_per_entity(*qpvs,buck);
      auto vec_dim = stk::mesh::field_extent1_per_entity(*qpvs,buck);

      auto& state = m_stateArrays.elemStateArrays[b][qpvs->name()];
      state.reset_from_host_ptr(data,buck.size(),num_qps,vec_dim);
    }
    for (auto& qpts : qptensor_states) {
      auto data = reinterpret_cast<double*>(buck.field_data_location(*qpts));
      auto num_qps = stk::mesh::field_extent0_per_entity(*qpts,buck);
      auto dimSq   = stk::mesh::field_extent1_per_entity(*qpts,buck);

      auto dim = static_cast<int>(std::floor(std::sqrt(double(dimSq))));
      TEUCHOS_TEST_FOR_EXCEPTION (dimSq!=(dim*dim), std::logic_error,
          "Error! QP Tensor states only supported for square tensors.\n"
          " - state name: " + qpts->name() + "\n"
          " - num scalars per qp: " + std::to_string(dimSq) + "\n");

      auto& state = m_stateArrays.elemStateArrays[b][qpts->name()];
      state.reset_from_host_ptr(data,buck.size(),num_qps,dim,dim);
    }
    for (auto& svs : scalarValue_states) {
      auto& state = m_stateArrays.elemStateArrays[b][*svs];
      state.reset_from_host_ptr(&time[*svs],1);
    }
  }

  // Process node data sets if present

  if (Teuchos::nonnull(stkMeshStruct->nodal_data_base) &&
      stkMeshStruct->nodal_data_base->isNodeDataPresent()) {
    auto node_states = stkMeshStruct->nodal_data_base->getNodeContainer();
    const auto& node_buckets = bulkData->get_buckets(NODE_RANK, select_owned_in_part);
    const size_t numNodeBuckets = node_buckets.size();

    m_stateArrays.nodeStateArrays.resize(numNodeBuckets);
    for (std::size_t b = 0; b < numNodeBuckets; b++) {
      stk::mesh::Bucket& buck = *node_buckets[b];
      for (auto it : *node_states) {
        auto stk_node_state = Teuchos::rcp_dynamic_cast<AbstractSTKNodeFieldContainer>(it.second);
        m_stateArrays.nodeStateArrays[b][it.first] = stk_node_state->getMDA(buck);
      }
    }
  }
}

void
STKDiscretization::computeSideSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeSideSets");
  // Clean up existing sideset structure if remeshing
  for (size_t i = 0; i < sideSets.size(); ++i) {
    sideSets[i].clear();  // empty the ith map
  }

  int numBuckets = m_wsEBNames.size();

  sideSets.resize(numBuckets);  // Need a sideset list per workset

  for (const auto& ss : stkMeshStruct->ssPartVec) {
    // Make sure the sideset exist even if no sides are owned
    for (int i=0; i<numBuckets; ++i) {
      sideSets[i][ss.first].resize(0);
    }

    // Get all owned sides in this side set
    stk::mesh::Selector select_owned_in_sspart =
        stk::mesh::Selector(*(ss.second)) &
        stk::mesh::Selector(metaData->locally_owned_part());

    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        select_owned_in_sspart,  // sides local to this processor
        bulkData->buckets(metaData->side_rank()),
        sides);

    *out << "STKDisc: sideset " << ss.first << " has size " << sides.size()
         << "  on Proc 0." << std::endl;

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
              << ss.first << std::endl);

      stk::mesh::Entity elem = bulkData->begin_elements(sidee)[0];

      // containing the side. Note that if the side is internal, it will show up
      // twice in the
      // element list, once for each element that contains it.

      SideStruct sStruct;

      // Save side stk GID.
      sStruct.side_GID = bulkData->identifier(sidee) - 1;

      // Save elem GID and LID. Here, LID is the local id *within* the workset
      sStruct.elem_GID = bulkData->identifier(elem) - 1;
      sStruct.ws_elem_idx = elemGIDws[sStruct.elem_GID].LID;

      // Get the ws that this element lives in
      int workset = elemGIDws[sStruct.elem_GID].ws;

      // Save the position of the side within element (0-based).
      sStruct.side_pos = determine_entity_pos(elem, sidee);

      // Save the index of the element block that this elem lives in
      sStruct.elem_ebIndex =
          stkMeshStruct->getMeshSpecs()[0]->ebNameToIndex[m_wsEBNames[workset]];

      // Get or create the vector of side structs for this side set on this workset
      auto& ss_vec = sideSets[workset][ss.first];
      ss_vec.push_back(sStruct);
    }
  }

  // =============================================================
  // (Kokkos Refactor) Convert sideSets to sideSetViews

  // 1) Compute view extents (num_local_worksets, max_sideset_length, max_sides) and local workset counter (current_local_index)
  std::map<std::string, int> num_local_worksets;
  std::map<std::string, int> max_sideset_length;
  std::map<std::string, int> max_sides;
  std::map<std::string, int> current_local_index;
  for (size_t i = 0; i < sideSets.size(); ++i) {
    for (const auto& ss_it : sideSets[i]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

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
        max_sides[ss_key] = std::max(max_sides[ss_key], (int) ss_val[j].side_pos);
    }
  }

  // 2) Construct GlobalSideSetList (map of GlobalSideSetInfo)
  for (const auto& ss_it : num_local_worksets) {
    std::string             ss_key = ss_it.first;

    max_sides[ss_key]++; // max sides is the largest local ID + 1 and needs to be incremented once for each key here

    globalSideSetViews[ss_key].num_local_worksets = num_local_worksets[ss_key];
    globalSideSetViews[ss_key].max_sideset_length = max_sideset_length[ss_key];
    globalSideSetViews[ss_key].side_GID         = Kokkos::DualView<GO**,   Kokkos::LayoutRight, PHX::Device>("side_GID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_GID         = Kokkos::DualView<GO**,   Kokkos::LayoutRight, PHX::Device>("elem_GID", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].ws_elem_idx      = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("ws_elem_idx", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].elem_ebIndex     = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("elem_ebIndex", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].side_pos         = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("side_pos", num_local_worksets[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].max_sides        = max_sides[ss_key];
    globalSideSetViews[ss_key].numCellsOnSide   = Kokkos::DualView<int**,  Kokkos::LayoutRight, PHX::Device>("numCellsOnSide", num_local_worksets[ss_key], max_sides[ss_key]);
    globalSideSetViews[ss_key].cellsOnSide      = Kokkos::DualView<int***, Kokkos::LayoutRight, PHX::Device>("cellsOnSide", num_local_worksets[ss_key], max_sides[ss_key], max_sideset_length[ss_key]);
    globalSideSetViews[ss_key].sideSetIdxOnSide = Kokkos::DualView<int***, Kokkos::LayoutRight, PHX::Device>("sideSetIdxOnSide", num_local_worksets[ss_key], max_sides[ss_key], max_sideset_length[ss_key]);
  }

  // 3) Populate global views
  for (size_t i = 0; i < sideSets.size(); ++i) {
    for (const auto& ss_it : sideSets[i]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

      int current_index = current_local_index[ss_key];
      int numSides = max_sides[ss_key];

      int max_cells_on_side = 0;
      std::vector<int> numCellsOnSide(numSides);
      std::vector<std::vector<int>> cellsOnSide(numSides);
      std::vector<std::vector<int>> sideSetIdxOnSide(numSides);
      for (size_t j = 0; j < ss_val.size(); ++j) {
        int cell = ss_val[j].ws_elem_idx;
        int side = ss_val[j].side_pos;

        cellsOnSide[side].push_back(cell);
        sideSetIdxOnSide[side].push_back(j);
      }
      for (int side = 0; side < numSides; ++side) {
        numCellsOnSide[side] = cellsOnSide[side].size();
        max_cells_on_side = std::max(max_cells_on_side, numCellsOnSide[side]);
      }

      for (int side = 0; side < numSides; ++side) {
        globalSideSetViews[ss_key].numCellsOnSide.h_view(current_index, side) = numCellsOnSide[side];
        for (int j = 0; j < numCellsOnSide[side]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide.h_view(current_index, side, j) = cellsOnSide[side][j];
          globalSideSetViews[ss_key].sideSetIdxOnSide.h_view(current_index, side, j) = sideSetIdxOnSide[side][j];
        }
        for (int j = numCellsOnSide[side]; j < max_sideset_length[ss_key]; ++j) {
          globalSideSetViews[ss_key].cellsOnSide.h_view(current_index, side, j) = -1;
          globalSideSetViews[ss_key].sideSetIdxOnSide.h_view(current_index, side, j) = -1;
        }
      }

      for (size_t j = 0; j < ss_val.size(); ++j) {
        globalSideSetViews[ss_key].side_GID.h_view(current_index, j)      = ss_val[j].side_GID;
        globalSideSetViews[ss_key].elem_GID.h_view(current_index, j)      = ss_val[j].elem_GID;
        globalSideSetViews[ss_key].ws_elem_idx.h_view(current_index, j)   = ss_val[j].ws_elem_idx;
        globalSideSetViews[ss_key].elem_ebIndex.h_view(current_index, j)  = ss_val[j].elem_ebIndex;
        globalSideSetViews[ss_key].side_pos.h_view(current_index, j) = ss_val[j].side_pos;
      }

      globalSideSetViews[ss_key].side_GID.modify_host();
      globalSideSetViews[ss_key].elem_GID.modify_host();
      globalSideSetViews[ss_key].ws_elem_idx.modify_host();
      globalSideSetViews[ss_key].elem_ebIndex.modify_host();
      globalSideSetViews[ss_key].side_pos.modify_host();
      globalSideSetViews[ss_key].numCellsOnSide.modify_host();
      globalSideSetViews[ss_key].cellsOnSide.modify_host();
      globalSideSetViews[ss_key].sideSetIdxOnSide.modify_host();

      globalSideSetViews[ss_key].side_GID.sync_device();
      globalSideSetViews[ss_key].elem_GID.sync_device();
      globalSideSetViews[ss_key].ws_elem_idx.sync_device();
      globalSideSetViews[ss_key].elem_ebIndex.sync_device();
      globalSideSetViews[ss_key].side_pos.sync_device();
      globalSideSetViews[ss_key].numCellsOnSide.sync_device();
      globalSideSetViews[ss_key].cellsOnSide.sync_device();
      globalSideSetViews[ss_key].sideSetIdxOnSide.sync_device();

      current_local_index[ss_key]++;
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
    LocalSideSetInfoList& lssList = sideSetViews[i];

    for (const auto& ss_it : sideSets[i]) {
      std::string             ss_key = ss_it.first;
      std::vector<SideStruct> ss_val = ss_it.second;

      int current_index = current_local_index[ss_key];
      std::pair<int,int> range(0, ss_val.size());

      lssList[ss_key].size           = ss_val.size();
      lssList[ss_key].side_GID       = Kokkos::subview(globalSideSetViews[ss_key].side_GID, current_index, range );
      lssList[ss_key].elem_GID       = Kokkos::subview(globalSideSetViews[ss_key].elem_GID, current_index, range );
      lssList[ss_key].ws_elem_idx    = Kokkos::subview(globalSideSetViews[ss_key].ws_elem_idx, current_index, range );
      lssList[ss_key].elem_ebIndex   = Kokkos::subview(globalSideSetViews[ss_key].elem_ebIndex,  current_index, range );
      lssList[ss_key].side_pos  = Kokkos::subview(globalSideSetViews[ss_key].side_pos, current_index, range );
      lssList[ss_key].numSides       = globalSideSetViews[ss_key].max_sides;
      lssList[ss_key].numCellsOnSide = Kokkos::subview(globalSideSetViews[ss_key].numCellsOnSide, current_index, Kokkos::ALL() );
      lssList[ss_key].cellsOnSide    = Kokkos::subview(globalSideSetViews[ss_key].cellsOnSide,    current_index, Kokkos::ALL(), Kokkos::ALL() );
      lssList[ss_key].sideSetIdxOnSide    = Kokkos::subview(globalSideSetViews[ss_key].sideSetIdxOnSide,    current_index, Kokkos::ALL(), Kokkos::ALL() );

      current_local_index[ss_key]++;
    }
  }

  // 6) Determine size of global DOFView structure and allocate
  std::map<std::string, int> total_sideset_idx;
  std::map<std::string, int> sideset_idx_offset;
  unsigned int maxSideNodes = 0;
  const auto& cell_layers_data = stkMeshStruct->local_cell_layers_data;
  if (!cell_layers_data.is_null()) {
    const Teuchos::RCP<const CellTopologyData> cell_topo = Teuchos::rcp(new CellTopologyData(stkMeshStruct->getMeshSpecs()[0]->ctd));
    const int numLayers = cell_layers_data->numLayers;
    const int numComps = getDOFManager()->getNumFields();

    // Determine maximum number of side nodes
    for (unsigned int elem_side = 0; elem_side < cell_topo->side_count; ++elem_side) {
      const CellTopologyData_Subcell& side =  cell_topo->side[elem_side];
      const unsigned int numSideNodes = side.topology->node_count;
      maxSideNodes = std::max(maxSideNodes, numSideNodes);
    }

    // Determine total number of sideset indices per each sideset name
    for (auto& ssList : sideSets) {
      for (auto& ss_it : ssList) {
        std::string             ss_key = ss_it.first;
        std::vector<SideStruct> ss_val = ss_it.second;

        if (sideset_idx_offset.find(ss_key) == sideset_idx_offset.end())
          sideset_idx_offset[ss_key] = 0;
        if (total_sideset_idx.find(ss_key) == total_sideset_idx.end())
          total_sideset_idx[ss_key] = 0;

        total_sideset_idx[ss_key] += ss_val.size();
      }
    }

    // Allocate total localDOFView for each sideset name
    for (auto& ss_it : num_local_worksets) {
      std::string ss_key = ss_it.first;
      allLocalDOFViews[ss_key] = Kokkos::DualView<LO****, PHX::Device>(ss_key + " localDOFView", total_sideset_idx[ss_key], maxSideNodes, numLayers+1, numComps);
    }
  }

  // Not all mesh structs that come through here are extruded mesh structs.
  // If the mesh isn't extruded, we won't need to do any of the following work.
  if (not cell_layers_data.is_null()) {
    // Get topo data
    auto ctd = stkMeshStruct->getMeshSpecs()[0]->ctd;

    // Ensure we have ONE cell per layer.
    const auto topo_hexa  = shards::getCellTopologyData<shards::Hexahedron<8>>();
    const auto topo_wedge = shards::getCellTopologyData<shards::Wedge<6>>();
    TEUCHOS_TEST_FOR_EXCEPTION (
        ctd.name!=topo_hexa->name &&
        ctd.name!=topo_wedge->name, std::runtime_error,
        "Extruded meshses only allowed if there is one element per layer (hexa or wedges).\n"
        "  - current topology name: " << ctd.name << "\n");

    const auto& sol_dof_mgr = getDOFManager();
    const auto& elem_dof_lids = sol_dof_mgr->elem_dof_lids().host();

    // Build a LayeredMeshNumbering for cells, so we can get the LIDs of elems over the column
    const auto numLayers = cell_layers_data->numLayers;
    const int top = cell_layers_data->top_side_pos;
    const int bot = cell_layers_data->bot_side_pos;

    // 7) Populate localDOFViews for GatherVerticallyContractedSolution
    for (int ws=0; ws<getNumWorksets(); ++ws) {

      // Need to look at localDOFViews for each i so that there is a view available for each workset even if it is empty
      std::map<std::string, Kokkos::DualView<LO****, PHX::Device>>& wsldofViews = wsLocalDOFViews[ws];

      const auto& elem_lids = getElementLIDs_host(ws);

      // Loop over the sides that form the boundary condition
      // const Teuchos::ArrayRCP<Teuchos::ArrayRCP<GO> >& wsElNodeID_i = wsElNodeID[i];
      for (auto& ss_it : sideSets[ws]) {
        std::string             ss_key = ss_it.first;
        std::vector<SideStruct> ss_val = ss_it.second;

        Kokkos::DualView<LO****, PHX::Device>& globalDOFView = allLocalDOFViews[ss_key];

        for (unsigned int sideSet_idx = 0; sideSet_idx < ss_val.size(); ++sideSet_idx) {
          const auto& side = ss_val[sideSet_idx];

          // Get the data that corresponds to the side
          const int ws_elem_idx = side.ws_elem_idx;
          const int side_pos    = side.side_pos;

          // Check if this sideset is the top or bot of the mesh. If not, the data structure
          // for coupling vertical dofs is not needed.
          if (side_pos!=top && side_pos!=bot)
            break;

          const int elem_LID = elem_lids(ws_elem_idx);
          const int basal_elem_LID = cell_layers_data->getColumnId(elem_LID);

          for (int eq=0; eq<neq; ++eq) {
            const auto& sol_top_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(eq,top,side_pos);
            const auto& sol_bot_offsets = sol_dof_mgr->getGIDFieldOffsetsSide(eq,bot,side_pos);
            const int numSideNodes = sol_top_offsets.size();

            for (int j=0; j<numSideNodes; ++j) {
              for (int il=0; il<numLayers; ++il) {
                const LO layer_elem_LID = cell_layers_data->getId(basal_elem_LID,il);
                globalDOFView.h_view(sideSet_idx + sideset_idx_offset[ss_key], j, il, eq) =
                  elem_dof_lids(layer_elem_LID,sol_bot_offsets[j]);
              }

              // Add top side in last layer
              const int il = numLayers-1;
              const LO layer_elem_LID = cell_layers_data->getId(basal_elem_LID,il);
              globalDOFView.h_view(sideSet_idx + sideset_idx_offset[ss_key], j, il+1, eq) =
                elem_dof_lids(layer_elem_LID,sol_top_offsets[j]);
            }
          }
        }

        globalDOFView.modify_host();
        globalDOFView.sync_device();

        // Set workset-local sub-view
        std::pair<int,int> range(sideset_idx_offset[ss_key], sideset_idx_offset[ss_key]+ss_val.size());
        wsldofViews[ss_key] = Kokkos::subview(globalDOFView, range, Kokkos::ALL(), Kokkos::ALL(), Kokkos::ALL());

        sideset_idx_offset[ss_key] += ss_val.size();
      }
    }
  } else {
    // We still need this view to be present (even if of size 0), so create them
    std::map<std::string, Kokkos::DualView<LO****, PHX::Device>> dummy;
    for (int ws=0; ws<getNumWorksets(); ++ws) {
      wsLocalDOFViews.emplace(std::make_pair(ws,dummy));
    }
  }
}

int
STKDiscretization::determine_entity_pos(
    const stk::mesh::Entity parent,
    const stk::mesh::Entity child) const
{
  using namespace stk;

  const auto rank = bulkData->entity_rank(child);
#ifdef ALBANY_DEBUG
  TEUCHOS_TEST_FOR_EXCEPTION (rank>bulkData->entity_rank(parent), std::logic_error,
      "Error! Child entity has rank greater than parent one.\n"
      "  - parent rank: " << bulkData->entity_rank(parent) << "\n"
      "  - child rank : " << rank << "\n"
      "  - parent ID  : " << bulkData->identifier(parent) << "\n"
      "  - child ID   : " << bulkData->identifier(child) << "\n");
#endif
  const auto node_child = rank==stk::topology::NODE_RANK;

  const auto& topo = bulkData->bucket(parent).topology();

  const auto num_child_nodes  = bulkData->num_nodes(child);
  const auto num_parent_nodes = bulkData->num_nodes(parent);

  const auto parent_nodes = bulkData->begin_nodes(parent);
  const auto child_nodes = bulkData->begin_nodes(child);

  if (node_child) {
    TEUCHOS_TEST_FOR_EXCEPTION (
        num_parent_nodes!=topo.num_sub_topology(0), std::runtime_error,
        "Error! Cannot locate node position in parent. Missing some parent nodes.\n"
        " - parent gid: " << stk_gid(parent) << "\n"
        " - child gid: " << stk_gid(child) << "\n"
        " - child rank: " << rank << "\n"
        " - num parent nodes: " << num_parent_nodes << "\n"
        " - expected nodes: " << topo.num_sub_topology(0) << "\n");

    // We're looking for a single node.
    // Simply loop over the parent nodes, to look for the given node
    for (unsigned i=0; i<num_parent_nodes; ++i) {
      if (parent_nodes[i]==child) {
        return i;
      }
    }
  } else if (num_child_nodes>0 && num_parent_nodes>0) {
    // Look for a subentity of dimension>=1, and we have all the nodes,
    // for both parent and child. Compare nodes of child and parent subentities,
    // to find the correct subcell id

    const int num_sub_topo = topo.num_sub_topology(rank);
    for (int isub=0; isub<num_sub_topo; ++isub) {
      // Check if this sub entity has the right topology. It may not be true
      // if, e.g., parent is a Wedge, and child is a face, since not all
      // Wedge's faces have the same number of nodes.
      if (topo.sub_topology(rank,isub).num_nodes()==num_child_nodes) {
        std::vector<stk::mesh::Entity> sub_nodes(num_child_nodes);
        topo.sub_topology_nodes(parent_nodes,rank,isub,sub_nodes.begin());

        bool sub_found = true;
        for (unsigned inode=0; inode<num_child_nodes; ++inode) {
          bool inode_found = false;
          for (unsigned jnode=0; jnode<num_child_nodes; ++jnode) {
            if (child_nodes[inode]==sub_nodes[jnode]) {
              inode_found = true;
              break;
            }
          }
          if (not inode_found) {
            sub_found = false;
            break;
          }
        }
        if (sub_found) {
          return isub;
        }
      }
    }
  } else {
    const auto num_children = bulkData->num_connectivity(parent,rank);
    const auto children = bulkData->begin(parent,rank);
    for (unsigned isub=0; isub<num_children; ++isub) {
      if (bulkData->relation_exist(parent,rank,isub) && children[isub]==child)
        return isub;
    }
  }

  std::ostringstream msg;
  msg << " ERROR! Could not determine sub-entity position in parent entity.\n"
      << "  parent:\n"
      << "    topo: " << bulkData->bucket(parent).topology().name() << "\n"
      << "    ID: " << bulkData->identifier(parent) << "\n"
      << "    nodes:";
  for (unsigned i=0; i<bulkData->num_nodes(parent); ++i) {
    msg << " " << bulkData->identifier(bulkData->begin_nodes(parent)[i]);
  }
  msg << "\n"
      << " child:\n"
      << "    topo: " << bulkData->bucket(child).topology().name() << "\n"
      << "    ID: " << bulkData->identifier(child) << "\n"
      << "    nodes:";
  for (unsigned i=0; i<bulkData->num_nodes(child); ++i) {
    msg << " " << bulkData->identifier(bulkData->begin_nodes(child)[i]);
  }
  msg << "\n";

  throw std::runtime_error(msg.str());

  return -1;
}

void
STKDiscretization::computeNodeSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeNodeSets");
  auto coordinates_field = stkMeshStruct->getCoordinatesField();

  const auto& sol_dof_mgr = getDOFManager();
  const auto& cell_indexer = sol_dof_mgr->cell_indexer();

  std::vector<std::vector<int>> offsets (sol_dof_mgr->getNumFields());
  for (int eq=0; eq<neq; ++eq) {
    offsets[eq] = sol_dof_mgr->getGIDFieldOffsets(eq);
  }

  // Loop over all node sets
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  for (const auto& ns : stkMeshStruct->nsPartVec) {
    auto& ns_gids     = nodeSetGIDs[ns.first];
    auto& ns_elem_pos = nodeSets[ns.first];
    auto& ns_coords   = nodeSetCoords[ns.first];

    // Grab all nodes on this nodeset.
    // NOTE: we take the globally shared part, since the way Tpetra/Epetra resolved
    //       sharing might be different from the way STK did. So we loop on ALL
    //       owned+shared nodes, and pick the ones that Tpetra/Epetra marked as owned
    stk::mesh::Selector ns_selector = metaData->globally_shared_part();
    ns_selector |= metaData->locally_owned_part();
    ns_selector &= *ns.second;
    std::vector<stk::mesh::Entity> nodes;
    stk::mesh::get_selected_entities(ns_selector, bulkData->buckets(NODE_RANK), nodes);

    // Remove nodes that are not owned according to Tpetra/Epetra
    std::vector<stk::mesh::Entity>::iterator it = nodes.begin();
    auto dof_mgr = getNodeDOFManager();
    while(it != nodes.end()){
      auto gid = stk_gid(*it);
      if (not dof_mgr->indexer()->isLocallyOwnedElement(gid)){
        it = nodes.erase(it);
      }else{
        it++;
      }
    }

    const int num_nodes = nodes.size();
    *out << "[STKDisc] nodeset " << ns.first << " has size " << num_nodes
         << "  on Proc " << comm->getRank() << std::endl;

    ns_gids.resize(num_nodes);
    ns_elem_pos.resize(num_nodes);
    ns_coords.resize(num_nodes);

    // Grab node GIDs, node coords, and dof LIDs at nodes
    for (int i=0; i<num_nodes; ++i) {
      const auto& n = nodes[i];
      ns_gids[i] = stk_gid(n);

      const auto e = *bulkData->begin_elements(n);
      ns_elem_pos[i] = std::make_pair(cell_indexer->getLocalElement(stk_gid(e)),determine_entity_pos(e,n));
      ns_coords[i] = stk::mesh::field_data(*coordinates_field, n);
    }
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
    mesh_data->set_bulk_data(Teuchos::get_shared_ptr(bulkData));
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
      mesh_data->add_global(outputFileIdx, it.first, it.second, DV_Type);
    }
    for (const auto& it : fc->getMeshScalarIntegerStates()) {
     const auto INT_Type = stk::util::ParameterType::INTEGER;
     mesh_data->add_global(outputFileIdx, it.first, it.second, INT_Type);
    }
    for (const auto& it : fc->getMeshScalarInteger64States()) {
     const auto INT64_Type = stk::util::ParameterType::INT64;
     mesh_data->add_global(outputFileIdx, it.first, it.second, INT64_Type);
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
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: buildSideSetProjectors");
  Teuchos::RCP<ThyraCrsMatrixFactory>   ov_graphP, graphP;
  Teuchos::RCP<Thyra_LinearOp>          P, ov_P;

  Teuchos::Array<GO> cols(1);
  Teuchos::Array<ST> vals(1);
  vals[0] = 1.0;

  const auto dofMgr = getDOFManager();
  const int sideDim = getNumDim()-1;

  const auto SIDE_RANK = stkMeshStruct->metaData->side_rank();
  const auto vs = getVectorSpace();
  const auto ov_vs = getOverlapVectorSpace();
  const auto cell_indexer = getDOFManager()->cell_indexer();
  for (auto it : sideSetDiscretizationsSTK) {
    // Extract the discretization
    const std::string&           sideSetName = it.first;
    const STKDiscretization&     ss_disc     = *it.second;

    // A dof manager defined exclusively on the side
    const auto ss_dofMgr = ss_disc.getDOFManager();
    const auto ss_ov_vs  = ss_dofMgr->ov_vs();
    const auto ss_vs     = ss_dofMgr->vs();

    // Extract the sides
    stk::mesh::Part&    part = *stkMeshStruct->ssPartVec.at(it.first);
    stk::mesh::Selector selector =
        stk::mesh::Selector(part) &
        stk::mesh::Selector(stkMeshStruct->metaData->locally_owned_part());
    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        selector, stkMeshStruct->bulkData->buckets(SIDE_RANK), sides);

#ifdef ALBANY_DEBUG
    const auto ss_cells = ss_dofMgr->getAlbanyConnManager()->getElementsInBlock();
    TEUCHOS_TEST_FOR_EXCEPTION (sides.size()!=ss_cells.size(), std::runtime_error,
        "Error! Conflicting data between sideset sides and sideset-dofMgr data.\n"
        "  - num sides in sideset: " << std::to_string(sides.size()) << "\n"
        "  - num elems on sideset dof mgr: " << ss_cells.size() << "\n");
#endif

    // The projector: build both overlapped and non-overlapped range vs
    graphP = Teuchos::rcp(new ThyraCrsMatrixFactory(vs, ss_vs));
    ov_graphP = Teuchos::rcp(new ThyraCrsMatrixFactory(ov_vs, ss_ov_vs));

    // Recall, if node_map(i,j)=k, then on side_gid=i, the j-th side node corresponds
    // to the k-th node in the side set cell numeration.
    const auto& node_numeration_map = sideNodeNumerationMap.at(it.first);
    const auto& side_cell_gid_map   = sideToSideSetCellMap.at(it.first);
    for (auto side : sides) {
      TEUCHOS_TEST_FOR_EXCEPTION (bulkData->num_elements(side)!=1, std::logic_error,
          "Error! Found a side with not exactly one element attached.\n"
          "  - side stk ID: " << bulkData->identifier(side) << "\n"
          "  - num elems  : " << bulkData->num_elements(side) << "\n");

      const auto side_gid    = stk_gid(side);
      const int num_side_nodes = bulkData->num_nodes(side);
      const auto ss_elem_gid = side_cell_gid_map.at(side_gid);
      const auto ss_elem_lid = ss_dofMgr->cell_indexer()->getLocalElement(ss_elem_gid);

      const auto elem = bulkData->begin_elements(side)[0];

      const auto pos = determine_entity_pos(elem,side);
      const auto ielem = cell_indexer->getLocalElement(stk_gid(elem));
      const auto elem_dof_gids = dofMgr->getElementGIDs(ielem);
      const auto ss_elem_dof_gids = ss_dofMgr->getElementGIDs(ss_elem_lid);

      const auto& permutation = node_numeration_map.at(side_gid);
      for (int eq=0; eq<neq; ++eq) {
        const auto& offsets    = dofMgr->getGIDFieldOffsetsSubcell(eq,sideDim,pos);
        const auto& ss_offsets = ss_dofMgr->getGIDFieldOffsets(eq);
        for (int i=0; i<num_side_nodes; ++i) {
          cols[0] = elem_dof_gids[offsets[i]];
          const GO row = ss_elem_dof_gids[ss_offsets[permutation[i]]];
          ov_graphP->insertGlobalIndices(row, cols());
          if (ss_dofMgr->indexer()->isLocallyOwnedElement(row)) {
            graphP->insertGlobalIndices(row, cols());
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
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: updateMesh");
  bulkData = stkMeshStruct->bulkData;

  computeVectorSpaces();

#ifdef OUTPUT_TO_SCREEN
  // write owned maps to matrix market file for debug
  writeMatrixMarket(getVectorSpace(), "dof_vs");
  writeMatrixMarket(getNodeVectorSpace(), "node_vs");
#endif

  setupMLCoords();

  transformMesh();

  computeWorksetInfo();
#ifdef OUTPUT_TO_SCREEN
  printConnectivity();
#endif

  computeNodeSets();

  computeSideSets();

  computeGraphs();

#ifndef ALBANY_DISABLE_OUTPUT_MESH
  setupExodusOutput();
#endif

#ifdef OUTPUT_TO_SCREEN
  printCoords();
#endif

  // Update sideset discretizations (if any)
  for (auto it : sideSetDiscretizationsSTK) {
    it.second->updateMesh();

    stkMeshStruct->buildCellSideNodeNumerationMap(
        it.first,
        sideToSideSetCellMap[it.first],
        sideNodeNumerationMap[it.first]);
  }

  if (sideSetDiscretizations.size()>0) {
    buildSideSetProjectors();
  }
}

void STKDiscretization::
setFieldData(const Teuchos::RCP<StateInfoStruct>& sis)
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: setFieldData");
  Teuchos::RCP<AbstractSTKFieldContainer> fieldContainer = stkMeshStruct->getFieldContainer();

  auto mSTKFieldContainer = Teuchos::rcp_dynamic_cast<MultiSTKFieldContainer>(fieldContainer,false);
  auto oSTKFieldContainer = Teuchos::rcp_dynamic_cast<OrdinarySTKFieldContainer>(fieldContainer,false);

  int num_time_deriv, numDim, num_params;
  Teuchos::RCP<Teuchos::ParameterList> params;

  auto gSTKFieldContainer = Teuchos::rcp_dynamic_cast<GenericSTKFieldContainer>(fieldContainer,false);
  params = gSTKFieldContainer->getParams();
  numDim = gSTKFieldContainer->getNumDim();
  num_params = gSTKFieldContainer->getNumParams();

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

  if (Teuchos::nonnull(mSTKFieldContainer)) {
    solutionFieldContainer = Teuchos::rcp(new MultiSTKFieldContainer(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, neq, numDim, sis, solution_vector, num_params));
  } else if (Teuchos::nonnull(oSTKFieldContainer)) {
    solutionFieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, neq, numDim, sis, num_params));
  } else {
    ALBANY_ABORT ("Error! Failed to cast the AbstractSTKFieldContainer to a concrete type.\n");
  }
}

Teuchos::RCP<DOFManager>
STKDiscretization::
create_dof_mgr (const std::string& part_name,
                const std::string& field_name,
                const FE_Type fe_type,
                const int order,
                const int dof_dim) const
{
  // Figure out which element blocks this part belongs to
  std::vector<std::string> elem_blocks;
  const auto& ebn = stkMeshStruct->ebNames_;
  if (part_name=="") {
    // Not specifying a mesh part is considered the same as the whole mesh
    elem_blocks = ebn;
  } else if (std::find(ebn.begin(),ebn.end(),part_name)!=ebn.end()) {
    // The part name is just one of the element blocks.
    elem_blocks = {part_name};
  } else {
    // This part is not an element block.
    // We need to consider any element block that has non-empty interesection
    // with this part. Note: the interesection is at the level of entities
    // with rank equal to the specified part (e.g., it may be just nodes).
    const auto& part = metaData->get_part(part_name);

    for (const auto& eb : ebn) {
      const auto& ebp = metaData->get_part(eb);
      stk::mesh::Selector sel (*part);
      sel |= *ebp;
      const auto& buckets = bulkData->buckets(part->primary_entity_rank());
      if (stk::mesh::count_selected_entities(sel,buckets)>0) {
        elem_blocks.push_back(eb);
      }
    }
  }

  // Ensure that all topologies are the same
  for (unsigned i=1; i<elem_blocks.size(); ++i) {
    const auto& topo0 = stkMeshStruct->elementBlockCT_.at(elem_blocks[0]);
    const auto& topo  = stkMeshStruct->elementBlockCT_.at(elem_blocks[i]);
    TEUCHOS_TEST_FOR_EXCEPTION (topo0.getName()!=topo.getName(), std::runtime_error,
        "Error! DOFManager requires all element blocks to have the same topology.\n");
  }

  // Create conn and dof managers
  auto conn_mgr = Teuchos::rcp(new STKConnManager(metaData,bulkData,elem_blocks));
  auto dof_mgr  = Teuchos::rcp(new DOFManager(conn_mgr,comm,part_name));

  const auto& topo = stkMeshStruct->elementBlockCT_.at(elem_blocks[0]);
  Teuchos::RCP<panzer::FieldPattern> fp;
  if (topo.getName()==std::string("Particle")) {
    // ODE equations are defined on a Particle geometry, where Intrepid2 doesn't work.
    fp = Teuchos::rcp(new panzer::ElemFieldPattern(shards::CellTopology(topo)));
  } else {
    // For space-dependent equations, we rely on Intrepid2 for patterns
    const auto basis = getIntrepid2Basis(*topo.getBaseCellTopologyData(),fe_type,order);
    fp = Teuchos::rcp(new panzer::Intrepid2FieldPattern(basis));
  }
  // NOTE: we add $dof_dim copies of the field pattern to the dof mgr,
  //       and call the fields ${field_name}_n, n=0,..,$dof_dim-1
  for (int i=0; i<dof_dim; ++i) {
    dof_mgr->addField(field_name + "_" + std::to_string(i),fp);
  }

  dof_mgr->build();

  return dof_mgr;
}

}  // namespace Albany
