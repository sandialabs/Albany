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
#include "Intrepid_Polylib.hpp"

#include "Adapt_NodalDataBlock.hpp"

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
                    const std::map<std::string, int>& ebNameToIndex_, bool interleavedOrdering_,
                    const Intrepid::EIntrepidPLPoly cubatureRule_ = Intrepid::PL_GAUSS)
       :  ctd(ctd_), numDim(numDim_), cubatureDegree(cubatureDegree_),
          nsNames(nsNames_), ssNames(ssNames_), worksetSize(worsetSize_), 
          ebName(ebName_), ebNameToIndex(ebNameToIndex_),
          interleavedOrdering(interleavedOrdering_),
          cubatureRule(cubatureRule_) {}
    CellTopologyData ctd;  // nonconst to allow replacement when the mesh adapts
    int numDim;
    int cubatureDegree;
    std::vector<std::string> nsNames;  //Node Sets Names
    std::vector<std::string> ssNames;  //Side Sets Names
    int worksetSize;
    const std::string ebName;  //Element block name for the EB that this struct corresponds to
    const std::map<std::string, int>& ebNameToIndex;
    bool interleavedOrdering;
    const Intrepid::EIntrepidPLPoly cubatureRule;
  };

//! Container to get state info from StateManager to STK. Made into a struct so
//  the information can continue to evolve without changing the interfaces.

struct StateStruct {

  enum MeshFieldEntity {WorksetValue, NodalData, ElemNode, QuadPoint};
  typedef std::vector<int> FieldDims;

  StateStruct (const std::string& name_, MeshFieldEntity ent): 
        name(name_), responseIDtoRequire(""), output(true), 
	restartDataAvailable(false), saveOldState(false), pParentStateStruct(NULL), entity(ent)
  {};

  StateStruct (const std::string& name_, MeshFieldEntity ent, const FieldDims& dims, const std::string& type): 
        name(name_), responseIDtoRequire(""), output(true), dim(dims), initType(type),
	restartDataAvailable(false), saveOldState(false), pParentStateStruct(NULL), entity(ent)
  {};

  void setInitType(const std::string& type) { initType = type; }
  void setInitValue(const double val) { initValue = val; }
  void setFieldDims(const FieldDims& dims) { dim = dims; }

  void print(){

    std::cout << "StateInfoStruct diagnostics for : " << name << std::endl;
    std::cout << "Dimensions : " << std::endl;
    for(unsigned int i = 0; i < dim.size(); i++)
       std::cout << "    " << i << " " << dim[i] << std::endl;
    std::cout << "Entity : " << entity << std::endl;
  }

  const std::string name;
  FieldDims dim;
  MeshFieldEntity entity;
  std::string initType;
  double initValue;
  std::map<std::string, std::string> nameMap;

  //For proper PHAL_SaveStateField functionality - maybe only needed temporarily?
  std::string responseIDtoRequire; //If nonzero length, the responseID for response 
                                   // field manager to require (assume dummy data layout)
  bool output;
  bool restartDataAvailable;
  bool saveOldState; // Bool that this state is to be copied into name+"_old"
  StateStruct *pParentStateStruct; // If this is a copy (name = parentName+"_old"), ptr to parent struct

  StateStruct ();

};

//typedef std::vector<Teuchos::RCP<StateStruct> >  StateInfoStruct;
// New container class approach
class StateInfoStruct {
public:

   typedef std::vector<Teuchos::RCP<StateStruct> >::const_iterator const_iterator;

   Teuchos::RCP<StateStruct>& operator[](int index){ return sis[index]; }
   const Teuchos::RCP<StateStruct> operator[](int index) const { return sis[index]; }
   void push_back(const Teuchos::RCP<StateStruct>& ss){ sis.push_back(ss); }
   std::size_t size() const { return sis.size(); }
   Teuchos::RCP<StateStruct>& back(){ return sis.back(); }
   const_iterator begin() const { return sis.begin(); }
   const_iterator end() const { return sis.end(); }

// Create storage on access - only if used
   Teuchos::RCP<Adapt::NodalDataBlock> getNodalDataBlock(){ return nodal_data_block; }

   Teuchos::RCP<Adapt::NodalDataBlock> createNodalDataBlock(){ 
        if(Teuchos::is_null(nodal_data_block))
            nodal_data_block = Teuchos::rcp(new Adapt::NodalDataBlock);
        return nodal_data_block; 
   }

private:

   std::vector<Teuchos::RCP<StateStruct> > sis;
   Teuchos::RCP<Adapt::NodalDataBlock> nodal_data_block;
   
};

}
#endif
