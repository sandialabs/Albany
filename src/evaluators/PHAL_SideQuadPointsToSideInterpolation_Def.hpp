//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
SideQuadPointsToSideInterpolation<EvalT, Traits>::
SideQuadPointsToSideInterpolation (const Teuchos::ParameterList& p,
                               const Teuchos::RCP<Albany::Layouts>& dl)
{
  isVectorField = p.get<bool>("Is Vector Field");

  sideSetName = p.get<std::string>("Side Set Name");
  w_measure    = PHX::MDField<ScalarT> (p.get<std::string>("Weighted Measure Name"), dl->side_qp_scalar);
  if (isVectorField)
  {
    field_qp   = PHX::MDField<ScalarT> (p.get<std::string> ("Field QP Name"), dl->side_qp_vector);
    field_side = PHX::MDField<ScalarT> (p.get<std::string> ("Field Side Name"), dl->side_vector);

    numSides = dl->side_qp_vector->dimension(1);
    numQPs   = dl->side_qp_vector->dimension(2);
    vecDim   = dl->side_qp_vector->dimension(3);
  }
  else
  {
    field_qp   = PHX::MDField<ScalarT> (p.get<std::string> ("Field QP Name"), dl->side_qp_scalar);
    field_side = PHX::MDField<ScalarT> (p.get<std::string> ("Field Side Name"), dl->side_scalar);

    numSides = dl->side_qp_scalar->dimension(1);
    numQPs   = dl->side_qp_scalar->dimension(2);
  }

  this->addDependentField (field_qp);
  this->addDependentField (w_measure);
  this->addEvaluatedField (field_side);

  this->setName("SideQuadPointsToSideInterpolation"+PHX::typeAsString<EvalT>());
}

template<typename EvalT, typename Traits>
void SideQuadPointsToSideInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData (field_qp,fm);
  this->utils.setFieldData (w_measure,fm);

  this->utils.setFieldData (field_side,fm);
}

template<typename EvalT, typename Traits>
void SideQuadPointsToSideInterpolation<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  if (workset.sideSets->find(sideSetName)==workset.sideSets->end())
    return;

  const std::vector<Albany::SideStruct>& sideSet = workset.sideSets->at(sideSetName);
  for (auto it_side : sideSet)
  {
    // Get the local data of side and cell
    const int cell = it_side.elem_LID;
    const int side = it_side.side_local_id;

    ScalarT meas = 0.0;
    for (int qp(0); qp<numQPs; ++qp)
    {
      meas += w_measure(cell,side,qp);
    }

    if (isVectorField)
    {
      for (int dim(0); dim<vecDim; ++dim)
      {
        field_side(cell,side,dim) = 0;
        for (int qp(0); qp<numQPs; ++qp)
          field_side(cell,side,dim) += field_qp(cell,side,qp,dim)*w_measure(cell,side,qp);
        field_side(cell,side,dim) /= meas;
      }
    }
    else
    {
      field_side(cell,side) = 0.0;
      for (int qp(0); qp<numQPs; ++qp)
        field_side(cell,side) += field_qp(cell,side,qp)*w_measure(cell,side,qp);
      field_side(cell,side) /= meas;
    }
  }
}

} // Namespace PHAL
