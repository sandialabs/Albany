//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#ifndef PROBLEMS_ALBANY_FIELDUTILS_HPP_
#define PROBLEMS_ALBANY_FIELDUTILS_HPP_

#include "Phalanx_FieldManager.hpp"
#include "Phalanx_MDField.hpp"
#include "Teuchos_RCP.hpp"

#include "Albany_Layouts.hpp"
#include "Albany_ScalarOrdinalTypes.hpp"
#include "PHAL_AlbanyTraits.hpp"
#include "PHAL_Dimension.hpp"

namespace Albany {

class FieldUtils
{
public:
  FieldUtils(
      PHX::FieldManager<PHAL::AlbanyTraits>& fm,
      Teuchos::RCP<Albany::Layouts> dl);

  //! Allocate memory for unmanaged fields in ComputeBasisFunctions
  void allocateComputeBasisFunctionsFields();

  //! Set ComputeBasisFunctions fields in field manager as unmanaged
  template <typename EvalT>
  void setComputeBasisFunctionsFields();

private:
  PHX::FieldManager<PHAL::AlbanyTraits>& fm;
  Teuchos::RCP<Albany::Layouts> dl;

  //! ComputeBasisFunctions fields
  mutable PHX::MDField<RealType,Cell,QuadPoint> weighted_measure;
  mutable PHX::MDField<RealType,Cell,QuadPoint> jacobian_det;
  mutable PHX::MDField<RealType,Cell,Node,QuadPoint> BF;
  mutable PHX::MDField<RealType,Cell,Node,QuadPoint> wBF;
  mutable PHX::MDField<RealType,Cell,Node,QuadPoint,Dim> GradBF;
  mutable PHX::MDField<RealType,Cell,Node,QuadPoint,Dim> wGradBF;
};

} // namespace Albany

#endif /* PROBLEMS_ALBANY_FIELDUTILS_HPP_ */
