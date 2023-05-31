//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MESH_SPECS_HPP
#define ALBANY_MESH_SPECS_HPP

#include "Albany_DiscretizationUtils.hpp"

#include "Shards_CellTopology.hpp"
#include "Shards_CellTopologyData.h"
#include "Intrepid2_Polylib.hpp"

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <vector>
#include <string>
#include <map>

// The MeshSpecs holds information (loaded mostly from STK::metaData),
// which is needed to create an Albany::Problem.
// The idea is that in order to build the problem's evaluators, we do not
// need to have a mesh. All we need are general topological information about the mesh.
// This includes worksetSize, CellTopologyData, etc.

namespace Albany {

struct MeshSpecs
{
  // Empty initialization. This constructor initialize all entries to empty values,
  // or, if possible, invalid values. This will be used to create mesh specs for
  // sideSets that will have no mesh. Having no mesh allows to specify long
  // parameter lists in the input file. For instance, if all we need is to know
  // topological information on a sideset, there is no need to
  // create a (potentially heavy) side mesh/discretization. Hence, we instead
  // create mesh specs, with bare minimum information.
  // To facilitate this, we offer an empty constructor, and the routines that
  // build mesh specs for side sets will have to manually modify the specs.
  MeshSpecs();

  // This constructor initializes all the possible information in a mesh specs object.
  MeshSpecs(
      const CellTopologyData&  ctd_,
      int                      numDim_,
      std::vector<std::string> nsNames_,
      std::vector<std::string> ssNames_,
      int                      worksetSize_,
      const std::string        ebName_,
      std::map<std::string, int> ebNameToIndex_);

  // nonconst to allow replacement when the mesh adapts
  CellTopologyData ctd;
  int              numDim;
  // Node Sets Names
  std::vector<std::string> nsNames;
  // Side Sets Names
  std::vector<std::string> ssNames;
  int                      worksetSize;
  // Element block name for the EB that this struct corresponds to
  std::string ebName;

  // If there are multiple element blocks, store the name and index of all blocks
  std::map<std::string, int> ebNameToIndex;

  // We store the side meshes names so we have a way to index them with a number
  std::map<std::string, Teuchos::RCP<MeshSpecs>> sideSetMeshSpecs;
  std::vector<std::string> sideSetMeshNames;
};

} // namespace Albany

#endif // ALBANY_MESH_SPECS_HPP
