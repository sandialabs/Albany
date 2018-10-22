//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_MESH_SPECS_HPP
#define ALBANY_MESH_SPECS_HPP

#include "Shards_CellTopologyData.h"
#include "Intrepid2_Polylib.hpp"

#include "Teuchos_ArrayRCP.hpp"
#include "Teuchos_ParameterList.hpp"

#include <vector>
#include <string>
#include <map>

// The MeshSpecsStruct holds information (loaded mostly from STK::metaData),
// which is needed to create an Albany::Problem.
// The idea is that in order to build the problem's evaluators, we do not
// need to have a mesh. All we need are general topological information about the mesh.
// This includes worksetSize, CellTopologyData, etc.

namespace Albany {

struct MeshSpecsStruct
{
  // Empty initialization. This constructor initialize all entries to empty values,
  // or, if possible, invalid values. This will be used to create mesh specs for
  // sideSets that will have no mesh. Having no mesh allows to specify long
  // parameter lists in the input file. For instance, if all we need is to know
  // topological (and quadrature) information on a sideset, there is no need to
  // create a (potentially heavy) side mesh/discretization. Hence, we instead
  // create mesh specs, with bare minimum information.
  // To facilitate this, we offer an empty constructor, and the routines that
  // build mesh specs for side sets will have to manually modify the specs.
  MeshSpecsStruct();

  // This constructor initializes all the possible informations in a mesh specs object.
  MeshSpecsStruct(
      const CellTopologyData&  ctd_,
      int                      numDim_,
      int                      cubatureDegree_,
      std::vector<std::string> nsNames_,
      std::vector<std::string> ssNames_,
      int                      worksetSize_,
      const std::string        ebName_,
      std::map<std::string, int> ebNameToIndex_,
      bool                       interleavedOrdering_,
      const bool                 sepEvalsByEB_ = false,
      const Intrepid2::EPolyType cubatureRule_ = Intrepid2::POLYTYPE_GAUSS);

  // nonconst to allow replacement when the mesh adapts
  CellTopologyData ctd;
  int              numDim;
  int              cubatureDegree;
  // Node Sets Names
  std::vector<std::string> nsNames;
  // Side Sets Names
  std::vector<std::string> ssNames;
  int                      worksetSize;
  // Element block name for the EB that this struct corresponds to
  std::string ebName;

  // If there are multiple element block, store the name and index of all blocks
  std::map<std::string, int> ebNameToIndex;

  bool interleavedOrdering;
  // Records "Separate Evaluators by Element Block". This says whether there
  // are as many MeshSpecsStructs as there are element blocks. If there is
  // only one element block in the problem, then the value of this boolean
  // doesn't matter. It is intended that interface blocks (LCM) don't count,
  // but the user must enforce this intention.
  bool                       sepEvalsByEB;
  Intrepid2::EPolyType       cubatureRule;

  // We store the side meshes names so we have a way to index them with a number
  std::map<std::string, Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>> sideSetMeshSpecs;
  std::vector<std::string> sideSetMeshNames;
};

} // namespace Albany

#endif // ALBANY_MESH_SPECS_HPP
