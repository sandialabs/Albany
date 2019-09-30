//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "StateVarUtils.hpp"

namespace LCM {

void
fromTo(Albany::StateArrayVec const& src, LCM::StateArrayVec& dst)
{
  auto const num_ws = src.size();
  dst.resize(num_ws);
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto&& src_map = src[ws];
    auto&& dst_map = dst[ws];
    for (auto&& kv : src_map) {
      auto&&     state_name = kv.first;
      auto&&     src_states = kv.second;
      auto&&     dst_states = dst_map[state_name];
      auto const num_states = src_states.size();
      dst_states.resize(num_states);
      for (auto s = 0; s < num_states; ++s) { dst_states[s] = src_states[s]; }
    }
  }
}

void
fromTo(LCM::StateArrayVec const& src, Albany::StateArrayVec& dst)
{
  auto const num_ws = src.size();
  assert(num_ws == dst.size());
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto&& src_map = src[ws];
    auto&& dst_map = dst[ws];
    for (auto&& kv : src_map) {
      auto&& state_name = kv.first;
      auto&& src_states = kv.second;
      assert(dst_map.find(state_name) != dst_map.end());
      auto&&    dst_states = dst_map[state_name];
      const int num_states = src_states.size();
      assert(num_states == dst_states.size());
      for (auto s = 0; s < num_states; ++s) { dst_states[s] = src_states[s]; }
    }
  }
}

void
fromTo(Albany::StateArrays const& src, LCM::StateArrays& dst)
{
  fromTo(src.elemStateArrays, dst.element_state_arrays);
  fromTo(src.nodeStateArrays, dst.node_state_arrays);
}

void
fromTo(LCM::StateArrays const& src, Albany::StateArrays& dst)
{
  fromTo(src.element_state_arrays, dst.elemStateArrays);
  fromTo(src.node_state_arrays, dst.nodeStateArrays);
}

}  // namespace LCM
