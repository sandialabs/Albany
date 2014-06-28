//*****************************************************************//
//    Albany 2.0:  Copyright 2012 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"

#include "Intrepid_FunctionSpaceTools.hpp"
#include "Aeras_Layouts.hpp"

namespace Aeras {

//**********************************************************************
template<typename EvalT, typename Traits>
XZHydrostatic_TracerResid<EvalT, Traits>::
XZHydrostatic_TracerResid(Teuchos::ParameterList& p,
                      const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF        (p.get<std::string> ("Weighted BF Name"),                 dl->node_qp_scalar   ),
  XDot       (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_scalar_level  ),        
  UTracerGrad(p.get<std::string> ("Gradient QP UTracer"),              dl->qp_gradient_level),        
  TracerSrc  (p.get<std::string> ("Tracer Source Name"),               dl->qp_scalar_level  ),        
  etadotdTracer (p.get<std::string> ("Tracer EtaDotd Name"),           dl->qp_scalar_level  ),        
  Residual   (p.get<std::string> ("Residual Name"),                    dl->node_scalar_level),        
  numNodes   (dl->node_scalar             ->dimension(1)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(XDot);
  this->addDependentField(UTracerGrad);
  this->addDependentField(wBF);
  this->addDependentField(TracerSrc);
  this->addDependentField(etadotdTracer);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_TracerResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_TracerResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(XDot,           fm);
  this->utils.setFieldData(UTracerGrad,    fm);
  this->utils.setFieldData(wBF,            fm);
  this->utils.setFieldData(TracerSrc,      fm);
  this->utils.setFieldData(etadotdTracer,  fm);
  this->utils.setFieldData(Residual,       fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_TracerResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          Residual(cell,node,level) +=          XDot(cell,qp,level)*wBF(cell,node,qp);
          Residual(cell,node,level) +=     TracerSrc(cell,qp,level)*wBF(cell,node,qp);
          Residual(cell,node,level) += etadotdTracer(cell,qp,level)*wBF(cell,node,qp);
          for (int j=0; j < numDims; ++j) 
            Residual(cell,node,level) += UTracerGrad(cell,qp,level,j)*wBF(cell,node,qp);
        }
      }
    }
  }
}
}
