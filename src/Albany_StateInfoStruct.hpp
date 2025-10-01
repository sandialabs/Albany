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

//! Container to get state info from StateManager to STK. Made into a struct so
//  the information can continue to evolve without changing the interfaces.

struct StateStruct
{
  enum StateType
  {
    ElemState = 1,
    NodeState,
    GlobalState
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

  StateStruct() = delete;

  StateStruct(
      const std::string& name_,
      MeshFieldEntity    ent,
      const FieldDims&   dims = {},
      const std::string& type = "none",
      const std::string& meshPart_ = "",
      const std::string& ebName_   = "")
      : name(name_),
        dim(dims),
        entity(ent),
        initType(type),
        meshPart(meshPart_),
        ebName(ebName_)
  {
    // Nothing to do
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
        return GlobalState;
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

  const std::string                  name = "";
  FieldDims                          dim = {};
  MeshFieldEntity                    entity;
  std::string                        initType = "none";
  double                             initValue = 0;
  std::map<std::string, std::string> nameMap;

  // For proper PHAL_SaveStateField functionality - maybe only needed
  // temporarily?
  // If nonzero length, the responseID for response
  // field manager to require (assume dummy data layout)
  std::string responseIDtoRequire = "";
  bool        output = true;
  bool        restartDataAvailable = false;
  bool        layered  = false;
  std::string meshPart = "";
  std::string ebName   = "";

  // Flag for 3d states that are computed on the fly in extruded meshes from basal states
  bool        extruded = false;
  bool        interpolated = false;

  // If this is a copy (name = parentName+"_old"), ptr to parent struct
  StateStruct* pParentStateStruct = nullptr;
};

// Could just be an alias to a vector of state struct pointers,
// but inheriting allows to define some helper methods
class StateInfoStruct : public std::vector<Teuchos::RCP<StateStruct>>
{
public:
  StateInfoStruct () = default;

  Teuchos::RCP<StateStruct> find (const std::string& name) {
    for (const auto& entry : *this) {
      if (entry->name==name) return entry;
    }
    return Teuchos::null;
  }
};

}  // namespace Albany

#endif  // ALBANY_STATEINFOSTRUCT
