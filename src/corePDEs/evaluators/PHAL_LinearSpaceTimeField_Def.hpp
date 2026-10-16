//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_Array.hpp"
#include "Phalanx_DataLayout.hpp"

#include "PHAL_Workset.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
LinearSpaceTimeField<EvalT, Traits>::
LinearSpaceTimeField (const Teuchos::ParameterList& p,
                      const Teuchos::RCP<Albany::Layouts>& dl) :
  coordVec (p.get<std::string>("QP Coordinate Vector Name"), dl->qp_vector),
  field    (p.get<std::string>("Field Name"), dl->qp_scalar)
{
  this->addDependentField(coordVec);
  this->addEvaluatedField(field);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_vector->dimensions(dims);
  numQPs  = dims[1];
  numDims = dims[2];

  TEUCHOS_TEST_FOR_EXCEPTION (numDims<1 || numDims>3, std::logic_error,
      "Error! LinearSpaceTimeField supports 1, 2 or 3 spatial dimensions, not " << numDims << ".\n");

  const auto coeffs = p.get<Teuchos::Array<double>>("Coefficients");
  TEUCHOS_TEST_FOR_EXCEPTION (coeffs.size()!=numDims+2, Teuchos::Exceptions::InvalidParameter,
      "Error! The field \"" << field.fieldTag().name() << "\" is linear in space and time, and needs "
      << numDims+2 << " \"Coefficients\" [c_0, c_1, ..., c_" << numDims << ", c_t], "
      << "for c_0 + c_1 x_1 + ... + c_" << numDims << " x_" << numDims << " + c_t t, "
      << "but " << coeffs.size() << " were given.\n");

  c_0 = coeffs[0];
  for (int d=0; d<3; ++d) {
    coeffs_x[d] = d<numDims ? coeffs[d+1] : 0.0;
  }
  c_t = coeffs[numDims+1];
  value_at_origin = c_0;

  this->setName("LinearSpaceTimeField(" + field.fieldTag().name() + ")" + PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinearSpaceTimeField<EvalT, Traits>::
postRegistrationSetup (typename Traits::SetupData d,
                       PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(coordVec,fm);
  this->utils.setFieldData(field,fm);

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void LinearSpaceTimeField<EvalT, Traits>::
operator() (const LinearSpaceTimeField_Tag&, const int& cell) const {
  for (int qp=0; qp<numQPs; ++qp) {
    field(cell,qp) = value_at_origin;
    for (int d=0; d<numDims; ++d) {
      field(cell,qp) += coeffs_x[d]*coordVec(cell,qp,d);
    }
  }
}

//**********************************************************************
template<typename EvalT, typename Traits>
void LinearSpaceTimeField<EvalT, Traits>::
evaluateFields (typename Traits::EvalData workset)
{
  // c_0 + c_t t, used by the functor
  value_at_origin = c_0 + c_t*workset.current_time;

  Kokkos::parallel_for(LinearSpaceTimeField_Policy(0,workset.numCells),*this);
}

} // namespace PHAL
