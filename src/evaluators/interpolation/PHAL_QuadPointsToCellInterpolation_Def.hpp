//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_Print.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::
QuadPointsToCellInterpolationBase (const Teuchos::ParameterList& p,
                                  const Teuchos::RCP<Albany::Layouts>& dl,
                                  const Teuchos::RCP<PHX::DataLayout>& qp_layout,
                                  const Teuchos::RCP<PHX::DataLayout>& cell_layout) :
  w_measure (p.get<std::string>("Weighted Measure Name"), dl->qp_scalar)
{
  qp_layout->dimensions(qp_dims);

  field_qp   = decltype(field_qp)(p.get<std::string> ("Field QP Name"), qp_layout);
  field_cell = decltype(field_cell)(p.get<std::string> ("Field Cell Name"), cell_layout);

  this->addDependentField (field_qp.fieldTag());
  this->addDependentField (w_measure.fieldTag());
  this->addEvaluatedField (field_cell);

  this->setName("QuadPointsToCellInterpolation"+PHX::print<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field_qp,fm);
  this->utils.setFieldData(w_measure,fm);

  this->utils.setFieldData(field_cell,fm);

  TEUCHOS_TEST_FOR_EXCEPTION (qp_dims.size() > 5, Teuchos::Exceptions::InvalidParameter, "Error! val_side has more dimensions than expected.\n");

  for (unsigned int i = 0; i < qp_dims.size(); ++i)
    dimsArray[i] = qp_dims[i];

  d.fill_field_dependencies(this->dependentFields(),this->evaluatedFields());
  if (d.memoizer_active()) memoizer.enable_memoizer();
}

// *********************************************************************
// Kokkos functor
template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Dim0_Tag& tag, const int& cell) const {
  
  MeshScalarT meas = 0.0;
      
  for (unsigned int qp(0); qp<dimsArray[1]; ++qp)
    meas += w_measure(cell,qp);

  field_cell(cell) = 0.0;
  for (unsigned int qp(0); qp<dimsArray[1]; ++qp) {
    field_cell(cell) += field_qp(cell,qp)*w_measure(cell,qp);
  }
  field_cell(cell) /= meas;

}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Dim1_Tag& tag, const int& cell) const {
  
  MeshScalarT meas = 0.0;
      
  for (size_t qp(0); qp<dimsArray[1]; ++qp)
    meas += w_measure(cell,qp);

  for (size_t i(0); i<dimsArray[2]; ++i)
  {
    field_cell(cell,i) = 0;
    for (size_t qp(0); qp<dimsArray[1]; ++qp)
      field_cell(cell,i) += field_qp(cell,qp,i)*w_measure(cell,qp);
    field_cell(cell,i) /= meas;
  }

}

template<typename EvalT, typename Traits, typename ScalarT>
KOKKOS_INLINE_FUNCTION
void QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::
operator() (const Dim2_Tag& tag, const int& cell) const {
  
  MeshScalarT meas = 0.0;
      
  for (size_t qp(0); qp<dimsArray[1]; ++qp)
    meas += w_measure(cell,qp);

  for (size_t i(0); i<dimsArray[2]; ++i)
  {
    for (size_t j(0); j<dimsArray[3]; ++j)
    {
      field_cell(cell,i,j) = 0;
      for (size_t qp(0); qp<dimsArray[1]; ++qp)
        field_cell(cell,i,j) += field_qp(cell,qp,i,j)*w_measure(cell,qp);
      field_cell(cell,i,j) /= meas;
    }
  }

}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  if (memoizer.have_saved_data(workset,this->evaluatedFields())) return;

  switch(qp_dims.size())
  {
    case 2: //scalar
      Kokkos::parallel_for(Dim0_Policy(0, workset.numCells), *this);
      break;
    case 3: //vector
      Kokkos::parallel_for(Dim1_Policy(0, workset.numCells), *this);
      break;
    case 4: //tensor
      Kokkos::parallel_for(Dim2_Policy(0, workset.numCells), *this);
      break;
    default:
      TEUCHOS_TEST_FOR_EXCEPTION (true, std::logic_error, "Error! Field dimension not supported (this error should have already appeared).\n");
  }

}

} // Namespace PHAL
