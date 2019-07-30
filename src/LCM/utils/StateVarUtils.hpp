//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//
#if !defined(LCM_StateVarUtils_hpp)
#define LCM_StateVarUtils_hpp

#include <map>
#include <vector>
#include "Albany_DataTypes.hpp"
#include "Albany_StateInfoStruct.hpp"
#include "Albany_StateManager.hpp"
namespace LCM {

//
// These are to mirror Albany::StateArrays, which are shards:Arrays
// under the hood, which in turn use for storage a raw pointer that comes
// from the depths of STK. Thus, to make a copy of the states without
// touching that pointer, we create these so that the values can be
// passed back and forth between LCM::StateArrays and Albany::StateArrays
// whenever we need to reset states.
//
using StateArray    = std::map<std::string, std::vector<ST>>;
using StateArrayVec = std::vector<StateArray>;

struct StateArrays
{
  StateArrayVec element_state_arrays;
  StateArrayVec node_state_arrays;
};

void
fromTo(Albany::StateArrayVec const& src, LCM::StateArrayVec& dst);

void
fromTo(LCM::StateArrayVec const& src, Albany::StateArrayVec& dst);

void
fromTo(Albany::StateArrays const& src, LCM::StateArrays& dst);

void
fromTo(LCM::StateArrays const& src, Albany::StateArrays& dst);

void
printElementStates(Albany::StateManager const& state_mgr);

}  // namespace LCM

#endif  // LCM_StateVarUtils_hpp
