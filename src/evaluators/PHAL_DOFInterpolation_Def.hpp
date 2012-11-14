//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"

namespace PHAL {

//**********************************************************************
template<typename EvalT, typename Traits>
DOFInterpolation<EvalT, Traits>::
DOFInterpolation(const Teuchos::ParameterList& p,
                              const Teuchos::RCP<Albany::Layouts>& dl) :
  val_node    (p.get<std::string>   ("Variable Name"), dl->node_scalar),
  BF          (p.get<std::string>   ("BF Name"), dl->node_qp_scalar),
  val_qp      (p.get<std::string>   ("Variable Name"), dl->qp_scalar )
{
  this->addDependentField(val_node);
  this->addDependentField(BF);
  this->addEvaluatedField(val_qp);

  this->setName("DOFInterpolation"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(val_node,fm);
  this->utils.setFieldData(BF,fm);
  this->utils.setFieldData(val_qp,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void DOFInterpolation<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  // This is needed, since evaluate currently sums into
  for (int i=0; i < val_qp.size() ; i++) val_qp[i] = 0.0;

  Intrepid::FunctionSpaceTools::
      evaluate<ScalarT>(val_qp, val_node, BF);
}

//**********************************************************************
}

