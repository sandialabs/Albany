//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef ALBANY_STATEINFOSTRUCT
#define ALBANY_STATEINFOSTRUCT

// THis file containts two structs that are containers for the
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
#include "Shards_CellTopologyData.h"
#include "Shards_Array.hpp"

  //! Container for minimal mesh specification info needed to 
  //  construct an Albany Problem

namespace Albany {

typedef shards::Array<double, shards::NaturalOrder> MDArray;
typedef std::map< std::string, MDArray > StateArray;
typedef std::vector<StateArray> StateArrayVec;

  struct StateArrays {
    StateArrayVec elemStateArrays;
    StateArrayVec nodeStateArrays;
  };

  struct MeshSpecsStruct {
    MeshSpecsStruct(const CellTopologyData& ctd_, int numDim_, 
                    int cubatureDegree_, std::vector<std::string> nsNames_,
                    std::vector<std::string> ssNames_,
                    int worsetSize_, const std::string ebName_,
                    const std::map<std::string, int>& ebNameToIndex_, bool interleavedOrdering_)
       :  ctd(ctd_), numDim(numDim_), cubatureDegree(cubatureDegree_),
          nsNames(nsNames_), ssNames(ssNames_), worksetSize(worsetSize_), 
          ebName(ebName_), ebNameToIndex(ebNameToIndex_),
          interleavedOrdering(interleavedOrdering_) {}
    CellTopologyData ctd;  // nonconst to allow replacement when the mesh adapts
    int numDim;
    int cubatureDegree;
    std::vector<std::string> nsNames;  //Node Sets Names
    std::vector<std::string> ssNames;  //Side Sets Names
    int worksetSize;
    const std::string ebName;  //Element block name for the EB that this struct corresponds to
    const std::map<std::string, int>& ebNameToIndex;
    bool interleavedOrdering;
  };
}

namespace Albany {

//! Container to get state info from StateManager to STK. Made into a struct so
//  the information can continue to evolve without changing the interfaces.


struct StateStruct {
  enum Entity {ScalarValue, NodePoint, QuadPoint, Vector, Tensor};
  enum ArrayClass {Element, Node, Dummy};
  //enum InitType {Zero, Identity, Restart, UndefinedInit};

  StateStruct (std::string name_): name(name_), responseIDtoRequire(""), output(true), 
	restartDataAvailable(false), saveOldState(false), pParentStateStruct(NULL) {};
   //StateStruct (std::string name_): name(name_), entity(UndefinedEntity), initType(UndefinedInit), output(true) {};
  ~StateStruct () {};

  const std::string name;
  std::vector<int> dim;
  //std::vector<MDArray> wsArray;
  //std::string entity; 
  Entity entity;
  ArrayClass aClass;
  std::string initType; //InitType initType;
  double initValue;
  std::map<std::string, std::string> nameMap;

  //For proper PHAL_SaveStateField functionality - maybe only needed temporarily?
  std::string responseIDtoRequire; //If nonzero length, the responseID for response 
                                   // field manager to require (assume dummy data layout)
  bool output;
  bool restartDataAvailable;
  bool saveOldState; // Bool that this state is to be copied into name+"_old"
  StateStruct *pParentStateStruct; // If this is a copy (name = parentName+"_old"), ptr to parent struct
  ArrayClass toClass(const std::string& dtype) const { if(dtype.compare("Cell") == 0) return Element;
                                       else if(dtype.compare("Node") == 0) return Node;
                                       else if(dtype.compare("Dummy") == 0) return Dummy;
                                       else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                            "StateStruct: ArrayClass - " << dtype << " - not supported" << std::endl); }
  Entity toEntity(const std::string& entity) const { if(entity.compare("ScalarValue") == 0) return ScalarValue;
                                       else if(entity.compare("Node") == 0) return NodePoint;
                                       else if(entity.compare("QuadPoint") == 0) return QuadPoint;
                                       else TEUCHOS_TEST_FOR_EXCEPTION(true, std::logic_error,
                                            "StateStruct: Entity type - " << entity << " - not supported" << std::endl); }
  void print(){

    std::cout << "StateInfoStruct diagnostics for : " << name << std::endl;
    std::cout << "Dimensions : " << std::endl;
    for(int i = 0; i < dim.size(); i++)
       std::cout << "    " << i << " " << dim[i] << std::endl;
    std::cout << "Class : " << aClass << " type : " << initType << " value : " << initValue << std::endl;
    std::cout << "Entity : " << entity << std::endl;
  }


  private:  
    StateStruct ();
};

typedef std::vector<Teuchos::RCP<StateStruct> >  StateInfoStruct;

}
#endif
