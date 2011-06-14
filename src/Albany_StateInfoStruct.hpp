/********************************************************************\
*            Albany, Copyright (2010) Sandia Corporation             *
*                                                                    *
* Notice: This computer software was prepared by Sandia Corporation, *
* hereinafter the Contractor, under Contract DE-AC04-94AL85000 with  *
* the Department of Energy (DOE). All rights in the computer software*
* are reserved by DOE on behalf of the United States Government and  *
* the Contractor as provided in the Contract. You are authorized to  *
* use this computer software for Governmental purposes but it is not *
* to be released or distributed to the public. NEITHER THE GOVERNMENT*
* NOR THE CONTRACTOR MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR      *
* ASSUMES ANY LIABILITY FOR THE USE OF THIS SOFTWARE. This notice    *
* including this sentence must appear on any copies of this software.*
*    Questions to Andy Salinger, agsalin@sandia.gov                  *
\********************************************************************/


#ifndef ALBANY_STATEINFOSTRUCT
#define ALBANY_STATEINFOSTRUCT

// THis file containts two structs that are containers for the
// Albany::Problem to interface to STK::Mesh.
// (1) The MeshSpecsStruct holds information that is loaded mostly
//     from STK::metaData, which is needed to create an Albany::Problem.
//     THis includes worksetSize, CellTopologyData, etc.
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
  struct MeshSpecsStruct {
    MeshSpecsStruct(const CellTopologyData& ctd_, int numDim_, 
                    int cubatureDegree_, std::vector<std::string> nsNames_,
                    int worsetSize_)
       :  ctd(ctd_), numDim(numDim_), cubatureDegree(cubatureDegree_),
          nsNames(nsNames_), worksetSize(worsetSize_) {}
    const CellTopologyData ctd;
    int numDim;
    int cubatureDegree;
    std::vector<std::string> nsNames;  //Node Sets Names
    int worksetSize;
  };
}

namespace Albany {

//! Container to get state info from StateManager to STK. Made into a struct so
//  the information can continue to evolve without changing the interfaces.


struct StateStruct {
  //enum Entity {Node, Element, UndefinedEntity};
  //enum InitType {Zero, Identity, Restart, UndefinedInit};

   StateStruct (std::string name_): name(name_),output(true), saveOldState(false) {};
   //StateStruct (std::string name_): name(name_), entity(UndefinedEntity), initType(UndefinedInit), output(true) {};
  ~StateStruct () {};

  const std::string name;
  std::vector<int> dim;
  //std::vector<shards::Array* > wsArray;
  std::vector<shards::Array<double,shards::NaturalOrder>* > wsArray;
  std::string entity; //Entity entity;
  std::string initType; //InitType initType;
  bool output;
  bool saveOldState; // Bool that this state is to be copied into name+"_old"

  private:  
    StateStruct ();
};

typedef std::vector<Teuchos::RCP<StateStruct> >  StateInfoStruct;

typedef std::map< std::string, shards::Array<double,shards::NaturalOrder> > StateArray;
typedef std::vector<StateArray> StateArrays;

}
#endif
