//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STATE_INFO_STRUCT
#define ALBANY_STATE_INFO_STRUCT

// The StateInfoStruct contains information from the Problem
// (via the State Manager) that is used by STK to define Fields.
// This includes name, number of quantities (scalar,vector,tensor),
// Element vs Node location, etc.

#include "Albany_ScalarOrdinalTypes.hpp"
#include "Albany_DualDynRankView.hpp"

#include "Phalanx_DataLayout.hpp"
#include "Shards_Array.hpp"
#include "Shards_CellTopologyData.h"

#include <map>
#include <string>
#include <vector>

namespace Albany {

using StateView     = DualDynRankView<double>;
using StateArray    = std::map<std::string,StateView>;
using StateArrayVec = std::vector<StateArray>;

struct StateArrays
{
  StateArrayVec elemStateArrays;
  StateArrayVec nodeStateArrays;
};

//! Container to get state info from StateManager to STK. Made into a struct so
//  the information can continue to evolve without changing the interfaces.

struct StateStruct
{
  enum StateType
  {
    ElemState = 1,
    NodeState
  };
  
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
        entity(ent),
        responseIDtoRequire(""),
        output(true),
        restartDataAvailable(false),
        saveOldState(false),
        layered(false),
        meshPart(""),
        pParentStateStruct(NULL)
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
        dim(dims),
        entity(ent),
        initType(type),
        responseIDtoRequire(""),
        output(true),
        restartDataAvailable(false),
        saveOldState(false),
        layered(false),
        meshPart(meshPart_),
        ebName(ebName_),
        pParentStateStruct(NULL)
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
    for (unsigned i = 0; i < dim.size(); ++i) {
      std::cout << "    " << i << " " << dim[i] << std::endl;
    }
    std::cout << "Entity : " << entity << std::endl;
  }

  StateType stateType () const {
    switch (entity) {
      case StateStruct::WorksetValue:
      case StateStruct::ElemData:
      case StateStruct::QuadPoint:
      case StateStruct::ElemNode:
        return ElemState;
      case StateStruct::NodalData:
      case StateStruct::NodalDistParameter:
      case StateStruct::NodalDataToElemNode:
        return NodeState;
      default:
        TEUCHOS_TEST_FOR_EXCEPTION (true, std::runtime_error,
            "Error! Unhandled/unsupported state type.\n");
    }
  }

  const std::string                  name{""};
  FieldDims                          dim;
  MeshFieldEntity                    entity;
  std::string                        initType{""};
  double                             initValue{0.0};
  std::map<std::string, std::string> nameMap;

  // For proper PHAL_SaveStateField functionality - maybe only needed
  // temporarily?
  // If nonzero length, the responseID for response
  // field manager to require (assume dummy data layout)
  std::string responseIDtoRequire{""};
  bool        output{false};
  bool        restartDataAvailable{false};
  // Bool that this state is to be copied into name+"_old"
  bool        saveOldState{false};
  bool        layered{false};
  std::string meshPart{""};
  std::string ebName{""};
  // If this is a copy (name = parentName+"_old"), ptr to parent struct
  StateStruct* pParentStateStruct{nullptr};

  StateStruct();
};

// Alias to a vector of state struct pointers
using StateInfoStruct = std::vector<Teuchos::RCP<StateStruct>>;

}  // namespace Albany

#endif  // ALBANY_STATEINFOSTRUCT
