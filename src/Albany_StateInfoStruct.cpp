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
  const size_t num_ws = esa.size();
  auto&      fos    = *Teuchos::VerboseObjectBase::getDefaultOStream();
  fos << "**** BEGIN ELEMENT STATES ****\n";
  for (size_t ws = 0; ws < num_ws; ++ws) {
    for (auto state_mda : esa[ws]) {
      auto const                     state_name = state_mda.first;
      auto                           mda        = state_mda.second;
      Albany::StateStruct::FieldDims dims;
      mda.dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (size_t cell = 0; cell < dims[0]; ++cell) {
            double& value = mda(cell);
            fos << "**** # INDEX 1, " << state_name << "(" << cell << ")"
                << " = " << value << '\n';
          }
          break;
        case 2:
          for (size_t cell = 0; cell < dims[0]; ++cell) {
            for (size_t qp = 0; qp < dims[1]; ++qp) {
              double& value = mda(cell, qp);
              fos << "**** # INDEX 2, " << state_name << "(" << cell << ","
                  << qp << ")"
                  << " = " << value << '\n';
            }
          }
          break;
        case 3:
          for (size_t cell = 0; cell < dims[0]; ++cell) {
            for (size_t qp = 0; qp < dims[1]; ++qp) {
              for (size_t i = 0; i < dims[2]; ++i) {
                double& value = mda(cell, qp, i);
                fos << "**** # INDEX 3, " << state_name << "(" << cell << ","
                    << qp << "," << i << ")"
                    << " = " << value << '\n';
              }
            }
          }
          break;
        case 4:
          for (size_t cell = 0; cell < dims[0]; ++cell) {
            for (size_t qp = 0; qp < dims[1]; ++qp) {
              for (size_t i = 0; i < dims[2]; ++i) {
                for (size_t j = 0; j < dims[3]; ++j) {
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
          for (size_t cell = 0; cell < dims[0]; ++cell) {
            for (size_t qp = 0; qp < dims[1]; ++qp) {
              for (size_t i = 0; i < dims[2]; ++i) {
                for (size_t j = 0; j < dims[3]; ++j) {
                  for (size_t k = 0; k < dims[4]; ++k) {
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
  for (size_t ws = 0; ws < num_ws; ++ws) {
    for (auto state_mda : nsa[ws]) {
      auto const                     state_name = state_mda.first;
      auto                           mda        = state_mda.second;
      Albany::StateStruct::FieldDims dims;
      mda.dimensions(dims);
      int size = dims.size();
      if (size == 0) return;
      switch (size) {
        case 1:
          for (size_t node = 0; node < dims[0]; ++node) {
            double& value = mda(node);
            fos << "**** # SCALAR, " << state_name << "(" << node << ")"
                << " = " << value << '\n';
          }
          break;
        case 2:
          for (size_t node = 0; node < dims[0]; ++node) {
            for (size_t i = 0; i < dims[1]; ++i) {
              double& value = mda(node, i);
              fos << "**** # VECTOR, " << state_name << "(" << node << "," << i
                  << ")"
                  << " = " << value << '\n';
            }
          }
          break;
        case 3:
          for (size_t node = 0; node < dims[0]; ++node) {
            for (size_t i = 0; i < dims[1]; ++i) {
              for (size_t j = 0; j < dims[2]; ++j) {
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
