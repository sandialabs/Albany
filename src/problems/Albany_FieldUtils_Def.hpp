//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef SRC_PROBLEMS_ALBANY_FIELDUTILS_DEF_HPP_
#define SRC_PROBLEMS_ALBANY_FIELDUTILS_DEF_HPP_

#include <stdexcept>

#include "Teuchos_TestForException.hpp"

#include "Albany_FieldUtils.hpp"

namespace Albany {

template <typename EvalT>
void
FieldUtils::setComputeBasisFunctionsFields()
{
  // allocateComputeBasisFunctionsFields() must be called first
  TEUCHOS_TEST_FOR_EXCEPTION(!BF.get_static_view().is_allocated(), std::logic_error,
      "The unmanaged field, BF, is not allocated!")

  fm.setUnmanagedField<EvalT>(BF);

  // If MeshScalarT=RealType, we can reuse MDFields
  // this is true if the mesh does not depend on the solution
  using MeshScalarT = typename EvalT::MeshScalarT;
  if (std::is_same<RealType, MeshScalarT>::value) {
    fm.setUnmanagedField<EvalT>(weighted_measure);
    fm.setUnmanagedField<EvalT>(jacobian_det);
    fm.setUnmanagedField<EvalT>(wBF);
    fm.setUnmanagedField<EvalT>(GradBF);
    fm.setUnmanagedField<EvalT>(wGradBF);
  }
}

} // namespace Albany

#endif /* SRC_PROBLEMS_ALBANY_FIELDUTILS_DEF_HPP_ */
