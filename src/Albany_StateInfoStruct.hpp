//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

// IK, 9/12/14: no Epetra!

#ifndef ALBANY_STATEINFOSTRUCT
#define ALBANY_STATEINFOSTRUCT

// This file containts two structs that are containers for the
// Albany::Problem to interface to STK::Mesh.
// (1) The MeshSpecsStruct holds information that is loaded mostly
//     from STK::metaData, which is needed to create an Albany::Problem.
//     This includes worksetSize, CellTopologyData, etc.
// (2) The StateInfoStruct contains information from the Problem
//     (via the State Manager) that is used by STK to define Fields.
//     This includes name, number of quantitites (scalar,vector,tensor),
//     Element vs Node lcoation, etc.

#include <string>
#include <vector>
#include <map>

#include "Intrepid2_Polylib.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Shards_Array.hpp"
#include "Shards_CellTopologyData.h"

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Adapt_NodalDataBase.hpp"

//! Container for minimal mesh specification info needed to
//  construct an Albany Problem

namespace Albany {

//
// Using these most of the Albany code compiles, but there are some errors
// with converting from STK data structures.
// In any case, the operator= still does a shallow copym which was the
// motivation to try Kokkos::View
//
//using MDArray = Kokkos::View<double*, PHX::Device>;
//using IDArray = Kokkos::View<LO*, PHX::Device>;
//using StateArray = std::map<std::string, MDArray>;
//using StateArrayVec = std::vector<StateArray>;

using MDArray = shards::Array<double, shards::NaturalOrder>;
using IDArray = shards::Array<LO, shards::NaturalOrder>;
using StateArray = std::map<std::string, MDArray>;
using StateArrayVec = std::vector<StateArray>;

struct StateArrays
{
  StateArrayVec elemStateArrays;
  StateArrayVec nodeStateArrays;
};

struct MeshSpecsStruct
{
  MeshSpecsStruct(
      const CellTopologyData&  ctd_,
      int                      numDim_,
      int                      cubatureDegree_,
      std::vector<std::string> nsNames_,
      std::vector<std::string> ssNames_,
      int                      worksetSize_,
      const std::string        ebName_,
      std::map<std::string, int>& ebNameToIndex_,
      bool                       interleavedOrdering_,
      const bool                 sepEvalsByEB_ = false,
      const Intrepid2::EPolyType cubatureRule_ = Intrepid2::POLYTYPE_GAUSS)
      : ctd(ctd_),
        numDim(numDim_),
        cubatureDegree(cubatureDegree_),
        nsNames(nsNames_),
        ssNames(ssNames_),
        worksetSize(worksetSize_),
        ebName(ebName_),
        ebNameToIndex(ebNameToIndex_),
        interleavedOrdering(interleavedOrdering_),
        sepEvalsByEB(sepEvalsByEB_),
        cubatureRule(cubatureRule_)
  {
    TEUCHOS_TEST_FOR_EXCEPTION (cubatureDegree<0, Teuchos::Exceptions::InvalidArgument,
                                "Error! Invalid cubature degree on element block '" << ebName << "'.\n");
  }

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
  std::map<std::string, int>& ebNameToIndex;
  bool interleavedOrdering;
  // Records "Separate Evaluators by Element Block". This says whether there
  // are as many MeshSpecsStructs as there are element blocks. If there is
  // only one element block in the problem, then the value of this boolean
  // doesn't matter. It is intended that interface blocks (LCM) don't count,
  // but the user must enforce this intention.
  bool                       sepEvalsByEB;
  const Intrepid2::EPolyType cubatureRule;
  std::map<std::string, Teuchos::ArrayRCP<Teuchos::RCP<MeshSpecsStruct>>>
      sideSetMeshSpecs;

  // We store the side meshes names so we have a way to index them with a number
  std::vector<std::string> sideSetMeshNames;
};

//! Container to get state info from StateManager to STK. Made into a struct so
//  the information can continue to evolve without changing the interfaces.

struct StateStruct
{
  enum MeshFieldEntity
  {
    WorksetValue,
    NodalData,
    ElemNode,
    ElemData,
    NodalDataToElemNode,
    NodalDistParameter,
    QuadPoint
  };
  typedef std::vector<PHX::DataLayout::size_type> FieldDims;

  StateStruct(const std::string& name_, MeshFieldEntity ent)
      : name(name_),
        responseIDtoRequire(""),
        output(true),
        restartDataAvailable(false),
        saveOldState(false),
        meshPart(""),
        pParentStateStruct(NULL),
        entity(ent),
        layered(false)
  {
  }

  StateStruct(
      const std::string& name_,
      MeshFieldEntity    ent,
      const FieldDims&   dims,
      const std::string& type,
      const std::string& meshPart_ = "",
      const std::string& ebName_   = "")
      : name(name_),
        responseIDtoRequire(""),
        output(true),
        dim(dims),
        initType(type),
        restartDataAvailable(false),
        saveOldState(false),
        meshPart(meshPart_),
        ebName(ebName_),
        pParentStateStruct(NULL),
        entity(ent),
        layered(false)
  {
  }

  void
  setInitType(const std::string& type)
  {
    initType = type;
  }
  void
  setInitValue(const double val)
  {
    initValue = val;
  }
  void
  setFieldDims(const FieldDims& dims)
  {
    dim = dims;
  }
  void
  setMeshPart(const std::string& meshPart_)
  {
    meshPart = meshPart_;
  }
  void
  setEBName(const std::string& ebName_)
  {
    ebName = ebName_;
  }

  void
  print()
  {
    std::cout << "StateInfoStruct diagnostics for : " << name << std::endl;
    std::cout << "Dimensions : " << std::endl;
    for (unsigned int i = 0; i < dim.size(); i++)
      std::cout << "    " << i << " " << dim[i] << std::endl;
    std::cout << "Entity : " << entity << std::endl;
  }

  const std::string name;
  FieldDims         dim;
  MeshFieldEntity   entity;
  std::string       initType;
  double            initValue;
  std::map<std::string, std::string> nameMap;

  // For proper PHAL_SaveStateField functionality - maybe only needed
  // temporarily?
  // If nonzero length, the responseID for response
  // field manager to require (assume dummy data layout)
  std::string responseIDtoRequire;
  bool        output;
  bool        restartDataAvailable;
  // Bool that this state is to be copied into name+"_old"
  bool        saveOldState;
  bool        layered;
  std::string meshPart;
  std::string ebName;
  // If this is a copy (name = parentName+"_old"), ptr to parent struct
  StateStruct* pParentStateStruct;

  StateStruct();
};

// typedef std::vector<Teuchos::RCP<StateStruct> >  StateInfoStruct;
// New container class approach
class StateInfoStruct {
 public:
  typedef std::vector<Teuchos::RCP<StateStruct>>::const_iterator const_iterator;

  Teuchos::RCP<StateStruct>& operator[](int index) { return sis[index]; }
  const Teuchos::RCP<StateStruct> operator[](int index) const
  {
    return sis[index];
  }
  void
  push_back(const Teuchos::RCP<StateStruct>& ss)
  {
    sis.push_back(ss);
  }
  std::size_t
  size() const
  {
    return sis.size();
  }
  Teuchos::RCP<StateStruct>&
  back()
  {
    return sis.back();
  }
  const_iterator
  begin() const
  {
    return sis.begin();
  }
  const_iterator
  end() const
  {
    return sis.end();
  }

  // Create storage on access - only if used
  Teuchos::RCP<Adapt::NodalDataBase>
  createNodalDataBase()
  {
    if (Teuchos::is_null(nodal_data_base))
      nodal_data_base = Teuchos::rcp(new Adapt::NodalDataBase);
    return nodal_data_base;
  }
  const Teuchos::RCP<Adapt::NodalDataBase>&
  getNodalDataBase()
  {
    return nodal_data_base;
  }

 private:
  std::vector<Teuchos::RCP<StateStruct>> sis;
  Teuchos::RCP<Adapt::NodalDataBase>     nodal_data_base;
};
}
#endif
