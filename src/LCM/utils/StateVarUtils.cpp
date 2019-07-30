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

void
printElementStates(Albany::StateManager const& state_mgr)
{
  auto&      sa     = state_mgr.getStateArrays();
  auto       sis    = state_mgr.getStateInfoStruct();
  auto&      fos    = *Teuchos::VerboseObjectBase::getDefaultOStream();
  auto&      esa    = sa.elemStateArrays;
  auto const num_ws = esa.size();
  fos << "**** BEGIN ELEMENT STATES ****\n";
  for (auto ws = 0; ws < num_ws; ++ws) {
    for (auto s = 0; s < sis->size(); ++s) {
      std::string const& state_name = (*sis)[s]->name;
      std::string const& init_type  = (*sis)[s]->initType;
      // AQUI
      if (state_name != "ACE Failure Indicator") continue;
      Albany::StateStruct::FieldDims dims;
      esa[ws][state_name].dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      if (init_type == "scalar") {
        switch (size) {
          case 1:
            for (auto cell = 0; cell < dims[0]; ++cell) {
              double& value = esa[ws][state_name](cell);
              fos << "**** # INDEX 1, " << state_name << "(" << cell << ")"
                  << " = " << value << '\n';
            }
            break;
          case 2:
            for (auto cell = 0; cell < dims[0]; ++cell) {
              for (auto qp = 0; qp < dims[1]; ++qp) {
                double& value = esa[ws][state_name](cell, qp);
                fos << "**** # INDEX 2, " << state_name << "(" << cell << ","
                    << qp << ")"
                    << " = " << value << '\n';
              }
            }
            break;
          case 3:
            for (auto cell = 0; cell < dims[0]; ++cell) {
              for (auto qp = 0; qp < dims[1]; ++qp) {
                for (auto i = 0; i < dims[2]; ++i) {
                  double& value = esa[ws][state_name](cell, qp, i);
                  fos << "**** # INDEX 3, " << state_name << "(" << cell << ","
                      << qp << "," << i << ")"
                      << " = " << value << '\n';
                }
              }
            }
            break;
          case 4:
            for (int cell = 0; cell < dims[0]; ++cell) {
              for (int qp = 0; qp < dims[1]; ++qp) {
                for (int i = 0; i < dims[2]; ++i) {
                  for (int j = 0; j < dims[3]; ++j) {
                    double& value = esa[ws][state_name](cell, qp, i, j);
                    fos << "**** # INDEX 4, " << state_name << "(" << cell
                        << "," << qp << "," << i << "," << j << ")"
                        << " = " << value << '\n';
                  }
                }
              }
            }
            break;
          case 5:
            for (int cell = 0; cell < dims[0]; ++cell) {
              for (int qp = 0; qp < dims[1]; ++qp) {
                for (int i = 0; i < dims[2]; ++i) {
                  for (int j = 0; j < dims[3]; ++j) {
                    for (int k = 0; k < dims[4]; ++k) {
                      double& value = esa[ws][state_name](cell, qp, i, j, k);
                      fos << "**** # INDEX 5, " << state_name << "(" << cell
                          << "," << qp << "," << i << "," << j << "," << k
                          << ")"
                          << " = " << value << '\n';
                    }
                  }
                }
              }
            }
            break;
          default: ALBANY_ASSERT(1 <= size && size <= 5, ""); break;
        }
      } else if (init_type == "identity") {
        for (int cell = 0; cell < dims[0]; ++cell) {
          for (int qp = 0; qp < dims[1]; ++qp) {
            for (int i = 0; i < dims[2]; ++i) {
              for (int j = 0; j < dims[3]; ++j) {
                double& value = esa[ws][state_name](cell, qp, i, j);
                fos << "**** # INDEX 4, " << state_name << "(" << cell << ","
                    << qp << "," << i << "," << j << ")"
                    << " = " << value << '\n';
              }
            }
          }
        }
      }
    }
  }
  fos << "**** END ELEMENT STATES ****\n";
}

void
printNodeStates(Albany::StateManager const& state_mgr)
{
  auto&      sa     = state_mgr.getStateArrays();
  auto       sis    = state_mgr.getStateInfoStruct();
  auto&      fos    = *Teuchos::VerboseObjectBase::getDefaultOStream();
  auto&      nsa    = sa.nodeStateArrays;
  auto const num_ws = nsa.size();
  fos << "**** BEGIN NODE STATES ****\n";
  for (auto ws = 0; ws < num_ws; ++ws) {
    for (auto s = 0; s < sis->size(); ++s) {
      std::string const& state_name = (*sis)[s]->name;
      std::string const& init_type  = (*sis)[s]->initType;
      Albany::StateStruct::FieldDims dims;
      nsa[ws][state_name].dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (auto node = 0; node < dims[0]; ++node) {
            double& value = nsa[ws][state_name](node);
            fos << "**** # SCALAR, " << state_name << "(" << node << ")"
                << " = " << value << '\n';
          }
          break;
        case 2:
          for (auto node = 0; node < dims[0]; ++node) {
            for (auto i = 0; i < dims[1]; ++i) {
              double& value = nsa[ws][state_name](node, i);
              fos << "**** # VECTOR, " << state_name << "(" << node << "," << i
                  << ")"
                  << " = " << value << '\n';
            }
          }
          break;
        case 3:
          for (int node = 0; node < dims[0]; ++node) {
            for (int i = 0; i < dims[1]; ++i) {
              for (int j = 0; j < dims[2]; ++j) {
                double& value = nsa[ws][state_name](node, i, j);
                fos << "**** # INDEX 4, " << state_name << "(" << node << ","
                    << i << "," << j << ")"
                    << " = " << value << '\n';
              }
            }
          }
          break;
        default: ALBANY_ASSERT(1 <= size && size <= 3, ""); break;
      }
    }
  }
  fos << "**** END NODE STATES ****\n";
}

}  // namespace LCM
