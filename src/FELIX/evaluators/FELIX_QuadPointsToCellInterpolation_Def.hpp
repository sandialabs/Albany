//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Phalanx_DataLayout.hpp"
#include "Phalanx_TypeStrings.hpp"

namespace FELIX {

//**********************************************************************
template<typename EvalT, typename Traits>
QuadPointsToCellInterpolation<EvalT, Traits>::QuadPointsToCellInterpolation (const Teuchos::ParameterList& p,
                                                                           const Teuchos::RCP<Albany::Layouts>& dl) :
  field_qp   (p.get<std::string> ("Field QP Name"), dl->qp_scalar),
  field_cell (p.get<std::string> ("Field Cell Name"), dl->cell_scalar2)
{
  this->addDependentField(field_qp);

  this->addEvaluatedField(field_cell);

  std::vector<PHX::DataLayout::size_type> dims;
  dl->qp_scalar->dimensions(dims);
  numQPt = dims[1];

  this->setName("QuadPointsToCellInterpolation"+PHX::typeAsString<EvalT>());
}

//**********************************************************************
template<typename EvalT, typename Traits>
void QuadPointsToCellInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(field_qp,fm);
  this->utils.setFieldData(field_cell,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void QuadPointsToCellInterpolation<EvalT, Traits>::evaluateFields (typename Traits::EvalData workset)
{
  for (int cell=0; cell<workset.numCells; ++cell)
  {
    field_cell(cell) = 0.0;
    for (int qp(0); qp<numQPt; ++qp)
    {
      field_cell(cell) += field_qp(cell,qp);
    }
    field_cell(cell) /= numQPt;
  }
}

} // Namespace FELIX
