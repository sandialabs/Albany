//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_SPressureResid<EvalT, Traits>::
XZHydrostatic_SPressureResid(const Teuchos::ParameterList& p,
              const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF      (p.get<std::string> ("Weighted BF Name"), dl->node_qp_scalar),
  wGradBF  (p.get<std::string> ("Weighted Gradient BF Name"),dl->node_qp_gradient),
  sp       (p.get<std::string> ("QP Variable Name"), dl->qp_scalar),
  spDot    (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_scalar),
  Residual (p.get<std::string> ("Residual Name"), dl->node_scalar),
  numNodes ( dl->node_scalar             ->dimension(1)),
  numQPs   ( dl->node_qp_scalar          ->dimension(2)),
  numDims  ( dl->node_qp_gradient        ->dimension(3)),
  numLevels( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(spDot);
  this->addDependentField(wBF);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_SPressureResid"+PHX::TypeString<EvalT>::value);

  sp0 = 0.0;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SPressureResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(spDot,fm);
  this->utils.setFieldData(wBF,fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_SPressureResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node=0; node < numNodes; ++node) {
        Residual(cell,node) += spDot(cell,qp)*wBF(cell,node,qp);
      }
    }
  }
}

//**********************************************************************
template<typename EvalT,typename Traits>
typename XZHydrostatic_SPressureResid<EvalT,Traits>::ScalarT& 
XZHydrostatic_SPressureResid<EvalT,Traits>::getValue(const std::string &n)
{
  if (n=="SPressure") return sp0;
}

}
