//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Albany_StateInfoStruct.hpp"
#include "Albany_Utils.hpp"

namespace Albany {

void
printStateArrays(StateArrays const& sa, std::string const& where)
{
  if (where != "") {
    auto& fos = *Teuchos::VerboseObjectBase::getDefaultOStream();
    fos << "**** PRINTING FROM : " << where << '\n';
  }
  printElementStates(sa);
  printNodeStates(sa);
}

void
printElementStates(StateArrays const& sa)
{
  auto&      esa    = sa.elemStateArrays;
  auto const num_ws = esa.size();
  auto&      fos    = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "**** BEGIN ELEMENT STATES ****\n";
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto const num_states = esa[ws].size();
    for (auto state_mda : esa[ws]) {
      auto const state_name = state_mda.first;
      // AQUI
      if (state_name != "ACE Failure Indicator") continue;
      auto                           mda = state_mda.second;
      Albany::StateStruct::FieldDims dims;
      mda.dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            double& value = mda(cell);
            fos << "**** # INDEX 1, " << state_name << "(" << cell << ")"
                << " = " << value << '\n';
          }
          break;
        case 2:
          for (auto cell = 0; cell < dims[0]; ++cell) {
            for (auto qp = 0; qp < dims[1]; ++qp) {
              double& value = mda(cell, qp);
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
                double& value = mda(cell, qp, i);
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
                  double& value = mda(cell, qp, i, j);
                  fos << "**** # INDEX 4, " << state_name << "(" << cell << ","
                      << qp << "," << i << "," << j << ")"
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
                    double& value = mda(cell, qp, i, j, k);
                    fos << "**** # INDEX 5, " << state_name << "(" << cell
                        << "," << qp << "," << i << "," << j << "," << k << ")"
                        << " = " << value << '\n';
                  }
                }
              }
            }
          }
          break;
        default: ALBANY_ASSERT(1 <= size && size <= 5, ""); break;
      }
    }
  }
  fos << "**** END ELEMENT STATES ****\n";
}

void
printNodeStates(StateArrays const& sa)
{
  auto&      nsa    = sa.nodeStateArrays;
  auto const num_ws = nsa.size();
  auto&      fos    = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "**** BEGIN NODE STATES ****\n";
  for (auto ws = 0; ws < num_ws; ++ws) {
    auto const num_states = nsa[ws].size();
    for (auto state_mda : nsa[ws]) {
      auto const                     state_name = state_mda.first;
      auto                           mda        = state_mda.second;
      Albany::StateStruct::FieldDims dims;
      mda.dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (auto node = 0; node < dims[0]; ++node) {
            double& value = mda(node);
            fos << "**** # SCALAR, " << state_name << "(" << node << ")"
                << " = " << value << '\n';
          }
          break;
        case 2:
          for (auto node = 0; node < dims[0]; ++node) {
            for (auto i = 0; i < dims[1]; ++i) {
              double& value = mda(node, i);
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
                double& value = mda(node, i, j);
                fos << "**** # TENSOR, " << state_name << "(" << node << ","
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

}  // namespace Albany
