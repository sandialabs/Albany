//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_MDField_UnmanagedAllocator.hpp"

#include "Albany_FieldUtils_Def.hpp"
#include "Albany_GeneralPurposeFieldsNames.hpp"

namespace Albany {

FieldUtils::FieldUtils(
    PHX::FieldManager<PHAL::AlbanyTraits>& fm,
    Teuchos::RCP<Albany::Layouts> dl) : 
    fm(fm),
    dl(dl)
{
}

void
FieldUtils::allocateComputeBasisFunctionsFields()
{
  // Allocate unmanaged fields once to save memory. These can then be reused in all evaluation types as
  // long as the mesh does not depend on the solution
  weighted_measure = PHX::allocateUnmanagedMDField<RealType,Cell,QuadPoint>(Albany::weights_name, dl->qp_scalar);
  jacobian_det = PHX::allocateUnmanagedMDField<RealType,Cell,QuadPoint>(Albany::jacobian_det_name, dl->qp_scalar);
  BF = PHX::allocateUnmanagedMDField<RealType,Cell,Node,QuadPoint>(Albany::bf_name, dl->node_qp_scalar);
  wBF = PHX::allocateUnmanagedMDField<RealType,Cell,Node,QuadPoint>(Albany::weighted_bf_name, dl->node_qp_scalar);
  GradBF = PHX::allocateUnmanagedMDField<RealType,Cell,Node,QuadPoint,Dim>(Albany::grad_bf_name, dl->node_qp_gradient);
  wGradBF = PHX::allocateUnmanagedMDField<RealType,Cell,Node,QuadPoint,Dim>(Albany::weighted_grad_bf_name, dl->node_qp_gradient);
}

} // namespace Albany

template void Albany::FieldUtils::setComputeBasisFunctionsFields<PHAL::AlbanyTraits::Residual>();
template void Albany::FieldUtils::setComputeBasisFunctionsFields<PHAL::AlbanyTraits::Jacobian>();
template void Albany::FieldUtils::setComputeBasisFunctionsFields<PHAL::AlbanyTraits::Tangent>();
template void Albany::FieldUtils::setComputeBasisFunctionsFields<PHAL::AlbanyTraits::DistParamDeriv>();
template void Albany::FieldUtils::setComputeBasisFunctionsFields<PHAL::AlbanyTraits::HessianVec>();
