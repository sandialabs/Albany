//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

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

  this->addDependentField (field_qp);
  this->addDependentField (w_measure);
  this->addEvaluatedField (field_cell);

  this->setName("QuadPointsToCellInterpolation"+PHX::typeAsString<EvalT>());
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
}

//**********************************************************************
template<typename EvalT, typename Traits, typename ScalarT>
void QuadPointsToCellInterpolationBase<EvalT, Traits, ScalarT>::evaluateFields (typename Traits::EvalData workset)
{
  ScalarT meas;
  int numQPs = qp_dims[1];

  for (int cell=0; cell<workset.numCells; ++cell)
  {
    meas = 0.0;
    for (int qp(0); qp<numQPs; ++qp)
    {
      meas += w_measure(cell,qp);
    }

    if(qp_dims.size()==2)  //scalar
    {
      field_cell(cell) = 0;
      for (int qp(0); qp<numQPs; ++qp)
        field_cell(cell) += field_qp(cell,qp)*w_measure(cell,qp);
      field_cell(cell) /= meas;
    }
    else if(qp_dims.size()==3)  //vector
    {
      for (int dim(0); dim<qp_dims[2]; ++dim)
      {
        field_cell(cell,dim) = 0;
        for (int qp(0); qp<numQPs; ++qp)
          field_cell(cell,dim) += field_qp(cell,qp,dim)*w_measure(cell,qp);
        field_cell(cell,dim) /= meas;
      }
    }
    else if(qp_dims.size()==4)  //tensor
    {
      for (int dim0(0); dim0<qp_dims[2]; ++dim0)
        for (int dim1(0); dim1<qp_dims[3]; ++dim1)
      {
        field_cell(cell,dim0, dim1) = 0;
        for (int qp(0); qp<numQPs; ++qp)
          field_cell(cell,dim0, dim1) += field_qp(cell,qp,dim0, dim1)*w_measure(cell,qp);
        field_cell(cell,dim0, dim1) /= meas;
      }
    }
  }
}

} // Namespace PHAL
