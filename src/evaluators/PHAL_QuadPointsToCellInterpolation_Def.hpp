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
                                  const Teuchos::RCP<Albany::Layouts>& dl) :
  w_measure (p.get<std::string>("Weighted Measure Name"), dl->qp_scalar)
{
  isVectorField = p.get<bool>("Is Vector Field");

  if (isVectorField)
  {
    field_qp   = PHX::MDField<ScalarT> (p.get<std::string> ("Field QP Name"), dl->qp_vector);
    field_cell = PHX::MDField<ScalarT> (p.get<std::string> ("Field Cell Name"), dl->cell_vector);

    numQPs = dl->qp_vector->dimension(1);
    vecDim = dl->qp_vector->dimension(2);
  }
  else
  {
    field_qp   = PHX::MDField<ScalarT> (p.get<std::string> ("Field QP Name"), dl->qp_scalar);
    field_cell = PHX::MDField<ScalarT> (p.get<std::string> ("Field Cell Name"), dl->cell_scalar2);

    numQPs = dl->qp_scalar->dimension(1);
  }

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

  for (int cell=0; cell<workset.numCells; ++cell)
  {
    meas = 0.0;
    for (int qp(0); qp<numQPs; ++qp)
    {
      meas += w_measure(cell,qp);
    }

    if (isVectorField)
    {
      for (int dim(0); dim<vecDim; ++dim)
      {
        field_cell(cell,dim) = 0;
        for (int qp(0); qp<numQPs; ++qp)
          field_cell(cell,dim) += field_qp(cell,qp,dim)*w_measure(cell,qp);
        field_cell(cell,dim) /= meas;
      }
    }
    else
    {
      field_cell(cell) = 0;
      for (int qp(0); qp<numQPs; ++qp)
        field_cell(cell) += field_qp(cell,qp)*w_measure(cell,qp);
      field_cell(cell) /= meas;
    }
    field_cell(cell) /= numQPs;
  }
}

} // Namespace PHAL
