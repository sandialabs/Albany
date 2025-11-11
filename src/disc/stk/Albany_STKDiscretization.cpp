//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include <limits>

#include <Albany_CommUtils.hpp>
#include <Albany_ThyraUtils.hpp>
#include "Albany_Macros.hpp"
#include "Albany_STKDiscretization.hpp"
#include "Albany_Utils.hpp"
#include "Albany_GlobalLocalIndexer.hpp"
#include "STKConnManager.hpp"
#include "Albany_TmplSTKMeshStruct.hpp"

#include <fstream>
#include <iostream>
#include <string>

#include <Shards_BasicTopologies.hpp>

#include <Panzer_IntrepidFieldPattern.hpp>
#include <Panzer_ElemFieldPattern.hpp>

#include <stk_mesh/base/Entity.hpp>
#include <stk_mesh/base/FEMHelpers.hpp>

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
      sideSetEquations(sideSetEquations_),
      rigidBodyModes(rigidBodyModes_),
      stkMeshStruct(stkMeshStruct_),
      discParams(discParams_)
{
  m_neq = neq_;

  if (stkMeshStruct->sideSetMeshStructs.size() > 0) {
    for (auto it : stkMeshStruct->sideSetMeshStructs) {
      auto stk_mesh = Teuchos::rcp_dynamic_cast<AbstractSTKMeshStruct>(it.second,true);
      auto side_disc = Teuchos::rcp(new STKDiscretization(discParams, m_neq, stk_mesh, comm));
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
  auto transformType = discParams->get<std::string>("Transform Type", "None");

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
    double xshift = discParams->get("x-shift", 0.0);
    double yshift = discParams->get("y-shift", 0.0);
    double zshift = discParams->get("z-shift", 0.0);
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

    auto betas = discParams->get<Teuchos::Array<double> >("Betas BL Transform",  Teuchos::tuple<double>(0.0, 0.0, 0.0));
    const int  numDim = stkMeshStruct->numDim;
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
    double alpha = discParams->get("LandIce alpha", 0.0);
    double L     = discParams->get("LandIce L", 1.0);
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
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
    double alpha = discParams->get("LandIce alpha", 0.0);
    double L     = discParams->get("LandIce L", 1.0);
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
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
    double alpha = discParams->get("LandIce alpha", 0.0);
    double L     = discParams->get("LandIce L", 1.0);
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
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
    double L = discParams->get("LandIce L", 1.0);
    cout << "L: " << L << endl;
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
    double alpha = discParams->get("LandIce alpha", 0.0);
    double L     = discParams->get("LandIce L", 1.0);
#ifdef OUTPUT_TO_SCREEN
    *out << "L: " << L << endl;
#endif
    double rhoIce   = 910.0;   // ice density, in kg/m^3
    double rhoOcean = 1028.0;  // ocean density, in kg/m^3
    stkMeshStruct->PBCStruct.scale[0] *= L;
    stkMeshStruct->PBCStruct.scale[1] *= L;
    stk::mesh::Field<double>* surfaceHeight_field =
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
    double L = discParams->get("LandIce L", 1.0);
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
        metaData->get_field<double>(stk::topology::NODE_RANK, "surface_height");
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
  if (!rigidBodyModes->isTekoUsed() && !rigidBodyModes->isMueLuUsed() && !rigidBodyModes->isFROSchUsed()) { return; }

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
  if ((rigidBodyModes->isTekoUsed() || rigidBodyModes->isMueLuUsed() || rigidBodyModes->isFROSchUsed()) &&
      stkMeshStruct->writeCoordsToMMFile) {
    *out << "Writing mesh coordinates to Matrix Market file." << std::endl;
    writeMatrixMarket(coordMV, "coords");
  }
#endif
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool overlapped)
{
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, dof_mgr, overlapped);

  // If any, process side discs as well
  for (auto it : sideSetDiscretizations) {
    auto disc = it.second;
    auto vs = overlapped ? disc->getOverlapVectorSpace() : disc->getVectorSpace();
    auto ss_soln = Thyra::createMember(vs);
    auto P = overlapped ? ov_projectors.at(it.first) : projectors.at(it.first);
    P->apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
    Teuchos::RCP<Thyra_MultiVector> ss_soln_dxdp;
    if (not soln_dxdp.is_null()) {
      auto dim = soln_dxdp->domain()->dim();
      ss_soln_dxdp = Thyra::createMembers(vs,dim);
      P->apply(Thyra::NOTRANS, *soln_dxdp, ss_soln_dxdp.ptr(), 1.0, 0.0);
    }
    disc->writeSolutionToMeshDatabase(*ss_soln,ss_soln_dxdp,overlapped);
  }
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const bool overlapped)
{
  // Put solution into STK Mesh
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, dof_mgr, overlapped);

  // If any, process side discs as well
  for (auto it : sideSetDiscretizations) {
    auto disc = it.second;
    auto vs = overlapped ? disc->getOverlapVectorSpace() : disc->getVectorSpace();
    auto ss_soln = Thyra::createMember(vs);
    auto ss_soln_dot = Thyra::createMember(vs);
    auto P = overlapped ? ov_projectors.at(it.first) : projectors.at(it.first);
    P->apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
    P->apply(Thyra::NOTRANS, soln_dot, ss_soln_dot.ptr(), 1.0, 0.0);
    Teuchos::RCP<Thyra_MultiVector> ss_soln_dxdp;
    if (not soln_dxdp.is_null()) {
      auto dim = soln_dxdp->domain()->dim();
      ss_soln_dxdp = Thyra::createMembers(vs,dim);
      P->apply(Thyra::NOTRANS, *soln_dxdp, ss_soln_dxdp.ptr(), 1.0, 0.0);
    }
    disc->writeSolutionToMeshDatabase(*ss_soln,ss_soln_dxdp,*ss_soln_dot,overlapped);
  }
}

void
STKDiscretization::writeSolutionToMeshDatabase(
    const Thyra_Vector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const Thyra_Vector& soln_dot,
    const Thyra_Vector& soln_dotdot,
    const bool overlapped)
{
  // Put solution into STK Mesh
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnVector(soln, soln_dxdp, soln_dot, soln_dotdot, dof_mgr, overlapped);

  // If any, process side discs as well
  for (auto it : sideSetDiscretizations) {
    auto disc = it.second;
    auto vs = overlapped ? disc->getOverlapVectorSpace() : disc->getVectorSpace();
    auto ss_soln = Thyra::createMember(vs);
    auto ss_soln_dot = Thyra::createMember(vs);
    auto ss_soln_dotdot = Thyra::createMember(vs);
    auto P = overlapped ? ov_projectors.at(it.first) : projectors.at(it.first);
    P->apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
    P->apply(Thyra::NOTRANS, soln_dot, ss_soln_dot.ptr(), 1.0, 0.0);
    P->apply(Thyra::NOTRANS, soln_dotdot, ss_soln_dot.ptr(), 1.0, 0.0);
    Teuchos::RCP<Thyra_MultiVector> ss_soln_dxdp;
    if (not soln_dxdp.is_null()) {
      auto dim = soln_dxdp->domain()->dim();
      ss_soln_dxdp = Thyra::createMembers(vs,dim);
      P->apply(Thyra::NOTRANS, *soln_dxdp, ss_soln_dxdp.ptr(), 1.0, 0.0);
    }
    disc->writeSolutionToMeshDatabase(*ss_soln,ss_soln_dxdp,*ss_soln_dot,*ss_soln_dotdot,overlapped);
  }
}

void
STKDiscretization::writeSolutionMVToMeshDatabase(
    const Thyra_MultiVector& soln,
    const Teuchos::RCP<const Thyra_MultiVector>& soln_dxdp,
    const bool overlapped)
{
  // Put solution into STK Mesh
  const auto& dof_mgr = getDOFManager();
  solutionFieldContainer->saveSolnMultiVector(soln, soln_dxdp, dof_mgr, overlapped);

  // If any, process side discs as well
  for (auto it : sideSetDiscretizations) {
    auto disc = it.second;
    auto vs = overlapped ? disc->getOverlapVectorSpace() : disc->getVectorSpace();
    auto ss_soln = Thyra::createMembers(vs,soln.domain()->dim());
    auto P = overlapped ? ov_projectors.at(it.first) : projectors.at(it.first);
    P->apply(Thyra::NOTRANS, soln, ss_soln.ptr(), 1.0, 0.0);
    Teuchos::RCP<Thyra_MultiVector> ss_soln_dxdp;
    if (not soln_dxdp.is_null()) {
      auto dim = soln_dxdp->domain()->dim();
      ss_soln_dxdp = Thyra::createMembers(vs,dim);
      P->apply(Thyra::NOTRANS, *soln_dxdp, ss_soln_dxdp.ptr(), 1.0, 0.0);
    }
    disc->writeSolutionMVToMeshDatabase(*ss_soln,ss_soln_dxdp,overlapped);
  }
}

void
STKDiscretization::writeMeshDatabaseToFile(
    const double        time,
    const bool          force_write_solution)
{
#ifdef ALBANY_DISABLE_OUTPUT_MESH
  *out << "[STKDiscretization::writeMeshDatabaseToFile] ALBANY_DISABLE_OUTPUT_MESH=TRUE. Skip.\n";
  (void) time;
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
      *out << "STKDiscretization::writeMeshDatabaseToFile: writing time " << time;
      if (time_label != time) *out << " with label " << time_label;
      *out << " to index " << out_step << " in file "
           << stkMeshStruct->exoOutFile << std::endl;
    }
  }
  outputInterval++;

  for (auto it : sideSetDiscretizations) {
    it.second->writeMeshDatabaseToFile(time, force_write_solution);
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

void
STKDiscretization::getField(Thyra_Vector& result, const std::string& name) const
{
  auto dof_mgr = getDOFManager(name);
  solutionFieldContainer->fillVector(result, name, dof_mgr, false);
}

void
STKDiscretization::getSolutionField(Thyra_Vector& result, const bool overlapped) const
{
  solutionFieldContainer->fillSolnVector(result, getDOFManager(), overlapped);
}

void
STKDiscretization::getSolutionMV(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  solutionFieldContainer->fillSolnMultiVector(result, getDOFManager(), overlapped);
}

void
STKDiscretization::getSolutionDxDp(
    Thyra_MultiVector& result,
    const bool         overlapped) const
{
  solutionFieldContainer->fillSolnSensitivity(result, getDOFManager(), overlapped);
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

  // Make sure we don't reuse old dof mgrs (if adapting)
  m_key_to_dof_mgr.clear();

  Teuchos::RCP<DOFManager> dof_mgr;

  // Solution dof mgr
  dof_mgr  = create_dof_mgr("",FE_Type::HGRAD,1,m_neq);
  m_dof_managers[solution_dof_name()][""] = dof_mgr;

  // Nodes dof mgr
  dof_mgr = create_dof_mgr("",FE_Type::HGRAD,1,1);
  m_dof_managers[nodes_dof_name()][""]    = dof_mgr;
  m_node_dof_managers[""]                 = dof_mgr;

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

    // NOTE: for now we hard code P1. In the future, we must be able to
    //       store this info somewhere and retrieve it here.
    dof_mgr = create_dof_mgr(sis->meshPart,FE_Type::HGRAD,1,dof_dim);
    m_dof_managers[sis->name][sis->meshPart] = dof_mgr;

    dof_mgr = create_dof_mgr(sis->meshPart,FE_Type::HGRAD,1,1);
    m_node_dof_managers[sis->meshPart] = dof_mgr;
  }

  const int meshDim = stkMeshStruct->numDim;
  coordinates.resize(meshDim * getLocalSubdim(getOverlapNodeVectorSpace()));
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
  for (int k=0; k < m_neq; ++k) {
    if (sideSetEquations.find(k) == sideSetEquations.end()) {
      volumeEqns.push_back(k);
    }
  }
  const int numVolumeEqns = volumeEqns.size();

  // The global solution dof manager
  const auto sol_dof_mgr = getDOFManager();
  const int num_elems = sol_dof_mgr->cell_indexer()->getNumLocalElements();

  // Handle the simple case, and return immediately
  if (numVolumeEqns==m_neq) {
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
        const auto& ss = m_sideSets[ws].at(ss_name);

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
            for (int eq=0; eq<m_neq; ++eq) {
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
            for (int col_eq=0; col_eq<m_neq; ++col_eq) {
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
          stkMeshStruct->meshSpecs[0]->ebNameToIndex[m_wsEBNames[i]];
    }
  }

  m_ws_elem_coords.resize(numBuckets);

  stkMeshStruct->get_field_accessor()->createStateArrays(m_workset_sizes);
  stkMeshStruct->get_field_accessor()->transferNodeStatesToElemStates();

  // Clear elem_LID->wsIdx index map if remeshing
  m_elem_ws_idx.clear();
  auto cell_indexer = getCellsGlobalLocalIndexer();
  int num_elems = cell_indexer->getNumLocalElements();
  m_elem_ws_idx.resize(num_elems);

  for (int b = 0; b < numBuckets; b++) {
    stk::mesh::Bucket& buck = *buckets[b];
    m_ws_elem_coords[b].resize(buck.size());

    stk::mesh::Entity element = buck[0];

    // i is the element index within bucket b
    for (std::size_t i = 0; i < buck.size(); i++) {
      // Traverse all the elements in this bucket
      element = buck[i];

      int elem_LID = cell_indexer->getLocalElement(stk_gid(element));

      // Now, save a map from element GID to workset on this PE
      m_elem_ws_idx[elem_LID].ws = b;
      m_elem_ws_idx[elem_LID].idx = i;

      // Set coords at nodes
      const auto* nodes = bulkData->begin_nodes(element);
      const int num_nodes = bulkData->num_nodes(element);
      m_ws_elem_coords[b][i].resize(num_nodes);
      for (int j=0; j<num_nodes; ++j) {
        m_ws_elem_coords[b][i][j] = stk::mesh::field_data(*coordinates_field, nodes[j]);
      }
    }
  }

  auto transformType = discParams->get<std::string>("Transform Type", "None");
  double alpha = discParams->get("LandIce alpha", 0.0);
  alpha *= M_PI / 180.;  // convert to radians
  auto& elemStateArrays = stkMeshStruct->get_field_accessor()->getElemStates();
  for (int d = 0; d < stkMeshStruct->numDim; d++) {
    if (stkMeshStruct->PBCStruct.periodic[d]) {
      for (int b = 0; b < numBuckets; b++) {
        auto has_sheight = elemStateArrays[b].count("surface_height")==1;
        DualDynRankView<double> sHeight;
        if (has_sheight) {
          sHeight = elemStateArrays[b]["surface_height"];
        }
        for (std::size_t i = 0; i < buckets[b]->size(); i++) {
          int  nodes_per_element = buckets[b]->num_nodes(i);
          for (int j = 0; j < nodes_per_element; j++) {
            if (m_ws_elem_coords[b][i][j][d] == 0.0) {
              double* xleak = new double[stkMeshStruct->numDim];
              for (int k = 0; k < stkMeshStruct->numDim; k++)
                if (k == d)
                  xleak[d] = stkMeshStruct->PBCStruct.scale[d];
                else
                  xleak[k] = m_ws_elem_coords[b][i][j][k];
              if ((transformType == "ISMIP-HOM Test A" ||
                   transformType == "ISMIP-HOM Test B" ||
                   transformType == "ISMIP-HOM Test C" ||
                   transformType == "ISMIP-HOM Test D") &&
                  d == 0) {
                xleak[2] -= stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                if (has_sheight) {
                  sHeight.host()(i, j) -= stkMeshStruct->PBCStruct.scale[d] * tan(alpha);
                }
              }
              m_ws_elem_coords[b][i][j] = xleak;  // replace ptr to coords
              toDelete.push_back(xleak);
            }
          }
        }
        if (has_sheight) {
          sHeight.sync_to_dev();
        }
      }
    }
  }
}

void
STKDiscretization::computeSideSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeSideSets");
  // Clean up existing sideset structure if remeshing
  for (auto& ss : m_sideSets) {
    ss.clear();  // empty the ith map
  }

  int numBuckets = m_wsEBNames.size();

  m_sideSets.resize(numBuckets);  // Need a sideset list per workset

  Teuchos::Array<GO> side_GIDs;
  auto cell_indexer = getCellsGlobalLocalIndexer();
  for (const auto& [ss_name,ss_part] : stkMeshStruct->ssPartVec) {
    // Make sure the sideset exist even if no sides are owned
    for (auto& ss : m_sideSets) {
      ss[ss_name].resize(0);
    }

    // Get all owned sides in this side set
    stk::mesh::Selector select_owned_in_sspart =
        stk::mesh::Selector(*ss_part) &
        stk::mesh::Selector(metaData->locally_owned_part());

    std::vector<stk::mesh::Entity> sides;
    stk::mesh::get_selected_entities(
        select_owned_in_sspart,  // sides local to this processor
        bulkData->buckets(metaData->side_rank()),
        sides);

    *out << "STKDisc: sideset " << ss_name << " has size " << sides.size()
         << "  on Proc 0." << std::endl;

    // loop over the sides to see what they are, then fill in the data holder
    // for side set options, look at
    // $TRILINOS_DIR/packages/stk/stk_usecases/mesh/UseCase_13.cpp

    side_GIDs.reserve(side_GIDs.size()+sides.size());
    for (const auto& side : sides) {
      TEUCHOS_TEST_FOR_EXCEPTION(
          bulkData->num_elements(side) != 1,
          std::logic_error,
          "STKDisc: cannot figure out side set topology for side set "
              << ss_name << std::endl);

      stk::mesh::Entity elem = bulkData->begin_elements(side)[0];

      // containing the side. Note that if the side is internal, it will show up
      // twice in the
      // element list, once for each element that contains it.

      SideStruct sStruct;

      // Save side stk GID.
      sStruct.side_GID = bulkData->identifier(side) - 1;
      side_GIDs.push_back(sStruct.side_GID);

      // Save elem GID and LID. Here, LID is the local id *within* the workset
      sStruct.elem_GID = bulkData->identifier(elem) - 1;
      int elem_LID = cell_indexer->getLocalElement(sStruct.elem_GID);
      sStruct.ws_elem_idx = m_elem_ws_idx[elem_LID].idx;

      // Get the ws that this element lives in
      int workset = m_elem_ws_idx[elem_LID].ws;

      // Save the position of the side within element (0-based).
      sStruct.side_pos = determine_entity_pos(elem, side);

      // Save the index of the element block that this elem lives in
      sStruct.elem_ebIndex =
          stkMeshStruct->meshSpecs[0]->ebNameToIndex[m_wsEBNames[workset]];

      // Get or create the vector of side structs for this side set on this workset
      auto& ss_vec = m_sideSets[workset][ss_name];
      ss_vec.push_back(sStruct);
    }
  }
  auto vs = createVectorSpace(comm,side_GIDs);
  m_sides_indexer = createGlobalLocalIndexer(vs);

  buildSideSetsViews();
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
      if (children[isub]==child)
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

Teuchos::RCP<ConnManager>
STKDiscretization::create_conn_mgr (const std::string& part_name)
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

  // Create conn manager
  return Teuchos::rcp(new STKConnManager(metaData,bulkData,elem_blocks));
}

void
STKDiscretization::computeNodeSets()
{
  TEUCHOS_FUNC_TIME_MONITOR("STKDiscretization: computeNodeSets");
  auto coordinates_field = stkMeshStruct->getCoordinatesField();

  const auto& sol_dof_mgr = getDOFManager();
  const auto& cell_indexer = sol_dof_mgr->cell_indexer();

  std::vector<std::vector<int>> offsets (sol_dof_mgr->getNumFields());
  for (int eq=0; eq<m_neq; ++eq) {
    offsets[eq] = sol_dof_mgr->getGIDFieldOffsets(eq);
  }

  // Loop over all node sets
  constexpr auto NODE_RANK = stk::topology::NODE_RANK;
  for (const auto& ns : stkMeshStruct->nsPartVec) {
    auto& ns_gids     = nodeSetGIDs[ns.first];
    auto& ns_elem_pos = nodeSets[ns.first];
    auto& ns_coords   = nodeSetCoords[ns.first];

    // Grab all nodes on this nodeset.
    // NOTE: we take the globally shared part, since the way Tpetra resolved
    //       sharing might be different from the way STK did. So we loop on ALL
    //       owned+shared nodes, and pick the ones that Tpetra marked as owned
    stk::mesh::Selector ns_selector = metaData->globally_shared_part();
    ns_selector |= metaData->locally_owned_part();
    ns_selector &= *ns.second;
    std::vector<stk::mesh::Entity> nodes;
    stk::mesh::get_selected_entities(ns_selector, bulkData->buckets(NODE_RANK), nodes);

    // Remove nodes that are not owned according to Tpetra
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
    const auto& node_numeration_map = m_side_nodes_to_ss_cell_nodes.at(it.first);
    const auto& side_cell_gid_map   = m_side_to_ss_cell.at(it.first);
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
      for (int eq=0; eq<m_neq; ++eq) {
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
  for (const auto& [ss_name, ss_disc] : sideSetDiscretizationsSTK) {
    ss_disc->updateMesh();

    stkMeshStruct->buildCellSideNodeNumerationMap(
        ss_name,
        m_side_to_ss_cell[ss_name],
        m_side_nodes_to_ss_cell_nodes[ss_name]);
  }

  if (sideSetDiscretizations.size()>0) {
    buildSideSetProjectors();
  }
}

void STKDiscretization::setFieldData()
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
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, m_neq, numDim, solution_vector, num_params));
  } else if (Teuchos::nonnull(oSTKFieldContainer)) {
    solutionFieldContainer = Teuchos::rcp(new OrdinarySTKFieldContainer(
      params, stkMeshStruct->metaData, stkMeshStruct->bulkData, m_neq, numDim, num_params));
  } else {
    ALBANY_ABORT ("Error! Failed to cast the AbstractSTKFieldContainer to a concrete type.\n");
  }

  // Proceed to set the solution field data in the side meshes as well (if any)
  for (auto& it : sideSetDiscretizations) {
    it.second->setFieldData();
  }
}

Teuchos::RCP<DOFManager>
STKDiscretization::
create_dof_mgr (const std::string& part_name,
                const FE_Type fe_type,
                const int order,
                const int dof_dim)
{
  auto& dof_mgr = get_dof_mgr(part_name,fe_type,order,dof_dim);
  if (Teuchos::nonnull(dof_mgr)) {
    // Not the first time we build a DOFManager for a field with these specs
    return dof_mgr;
  }

  // Create conn and dof managers
  auto conn_mgr = create_conn_mgr(part_name);
  dof_mgr  = Teuchos::rcp(new DOFManager(conn_mgr,comm,part_name));

  const auto& topo = stkMeshStruct->elementBlockCT_.at(conn_mgr->elem_block_name());
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
  //       and call the fields cmp_N, N=0,..,$dof_dim-1
  for (int i=0; i<dof_dim; ++i) {
    dof_mgr->addField("cmp_" + std::to_string(i),fp);
  }

  dof_mgr->build();

  return dof_mgr;
}

Teuchos::RCP<AdaptationData>
STKDiscretization::
checkForAdaptation (const Teuchos::RCP<const Thyra_Vector>& solution,
                    const Teuchos::RCP<const Thyra_Vector>& solution_dot,
                    const Teuchos::RCP<const Thyra_Vector>& solution_dotdot,
                    const Teuchos::RCP<const Thyra_MultiVector>& dxdp)
{
  auto adapt_data = Teuchos::rcp(new AdaptationData());

  auto& adapt_params = discParams->sublist("Mesh Adaptivity");
  auto adapt_type = adapt_params.get<std::string>("Type","None");
  if (adapt_type=="None") {
    return adapt_data;
  }
  TEUCHOS_TEST_FOR_EXCEPTION (adapt_type!="Minimally-Oscillatory", std::runtime_error,
      "Error! Adaptation type '" << adapt_type << "' not supported.\n"
      " - valid choices: None, Minimally-Oscillatory\n");

  // Only do adaptation for simple 1d problems
  auto mesh1d = Teuchos::rcp_dynamic_cast<TmplSTKMeshStruct<1>>(stkMeshStruct);
  TEUCHOS_TEST_FOR_EXCEPTION (mesh1d.is_null(), std::runtime_error,
      "Error! Adaptation for STK is only supported for a simple 1D problem, with STK1D discretization.\n");

  double tol = adapt_params.get<double>("Max Hessian");
  auto data = getLocalData(solution);
  // Simple check: refine if a proxy of the hessian of x is larger than a tolerance
  // TODO: replace with
  //  1. if |C_i| > threshold, mark for refinement the whole mesh
  //  2. Interpolate solution (and all elem/node fields if possible, but not necessary for adv-diff example)
  int num_nodes = data.size();
  getCoordinates();

  adapt_data->x = solution;
  adapt_data->x_dot = solution_dot;
  adapt_data->x_dotdot = solution_dotdot;
  adapt_data->dxdp = dxdp;
  for (int i=1; i<num_nodes-1; ++i) {
    auto h_prev = coordinates[i] - coordinates[i-1];
    auto h_next = coordinates[i+1] - coordinates[i];
    auto hess = (data[i-1] - 2*data[i] + data[i+1]) / (h_prev*h_next);
    auto grad_prev = (data[i]-data[i-1]) / h_prev;
    auto grad_next = (data[i+1]-data[i]) / h_next;
    if (std::fabs(hess)>tol and grad_prev*grad_next<0) {
      adapt_data->type = AdaptationType::Topology;
      break;
    }
  }

  return adapt_data;
}

void STKDiscretization::
adapt (const Teuchos::RCP<AdaptationData>& adaptData)
{
  // Not sure if we allow calling adapt in general, but just in case
  if (adaptData->type==AdaptationType::None) {
    return;
  }

  TEUCHOS_TEST_FOR_EXCEPTION (adaptData->type!=AdaptationType::Topology, std::runtime_error,
      "Error! Adaptation type not supported. Only 'None' and 'Topology' are currently supported.\n");

  // Solution oscillates. We need to half dx
  auto mesh1d = Teuchos::rcp_dynamic_cast<TmplSTKMeshStruct<1>>(stkMeshStruct);
  int num_params = mesh1d->getNumParams();
  int ne_x = discParams->get<int>("1D Elements");
  auto& adapt_params = discParams->sublist("Mesh Adaptivity");
  discParams->set("Workset Size", stkMeshStruct->meshSpecs()[0]->worksetSize);
  int factor = adapt_params.get("Refining Factor",2);
  discParams->set("1D Elements",factor*ne_x);
  auto sis = Teuchos::rcp(new StateInfoStruct(getMeshStruct()->get_field_accessor()->getAllSIS()));
  stkMeshStruct = Teuchos::rcp(new TmplSTKMeshStruct<1>(discParams,comm,num_params));
  stkMeshStruct->setFieldData(comm,sis,{});
  stkMeshStruct->getFieldContainer()->addStateStructs(sis);
  this->setFieldData();
  stkMeshStruct->setBulkData(comm);

  updateMesh();

  int num_time_deriv = discParams->get<int>("Number Of Time Derivatives");
  auto x_mv_new = Thyra::createMembers(getVectorSpace(),num_time_deriv);

  for (int ideriv=0; ideriv<num_time_deriv; ++ideriv) {
    auto data_new = getNonconstLocalData(x_mv_new->col(ideriv));
    auto x = ideriv==0 ? adaptData->x : (ideriv==1 ? adaptData->x_dot : adaptData->x_dotdot);
    auto data_old = getLocalData(x);
    int num_nodes_new = data_new.size();

    for (int inode=0; inode<num_nodes_new; ++inode) {
      int coarse = inode / factor;
      int rem    = inode % factor;
      if (rem == 0) {
        // Same node as coarse mesh
        data_new[inode] = data_old[coarse];
      } else {
        // Convex interpolation of two coarse points
        double alpha = static_cast<double>(rem) / factor;
        data_new[inode] = data_old[coarse]*(1-alpha) + data_old[coarse+1]*alpha;
      }
    }
  }

  writeSolutionMVToMeshDatabase(*x_mv_new, Teuchos::null, false);
}

}  // namespace Albany
