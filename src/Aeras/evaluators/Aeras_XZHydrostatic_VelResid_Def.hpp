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
XZHydrostatic_VelResid<EvalT, Traits>::
XZHydrostatic_VelResid(const Teuchos::ParameterList& p,
                       const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF         (p.get<std::string> ("Weighted BF Name"),                 dl->node_qp_scalar),
  wGradBF     (p.get<std::string> ("Weighted Gradient BF Name"),        dl->node_qp_gradient),
  keGrad      (p.get<std::string> ("Gradient QP Kinetic Energy"),       dl->qp_gradient_level),
  PhiGrad     (p.get<std::string> ("Gradient QP GeoPotential"),         dl->qp_gradient_level),
  uDot        (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_scalar_level),
  coordVec    (p.get<std::string> ("QP Coordinate Vector Name"),        dl->qp_gradient),
  density     (p.get<std::string> ("QP Density"),                       dl->qp_scalar_level),
  pGrad       (p.get<std::string> ("Gradient QP Pressure"),             dl->qp_gradient_level),
  etadotdVelx (p.get<std::string> ("EtaDotdVelx"),                      dl->qp_scalar_level),
  Residual    (p.get<std::string> ("Residual Name"),                    dl->node_scalar_level),
  numNodes    ( dl->node_scalar             ->dimension(1)),
  numQPs      ( dl->node_qp_scalar          ->dimension(2)),
  numDims     ( dl->node_qp_gradient        ->dimension(3)),
  numLevels   ( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(keGrad);
  this->addDependentField(PhiGrad);
  this->addDependentField(density);
  this->addDependentField(etadotdVelx);
  this->addDependentField(pGrad);
  this->addDependentField(uDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(coordVec);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_VelResid"+PHX::TypeString<EvalT>::value);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_VelResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(keGrad     , fm);
  this->utils.setFieldData(PhiGrad    , fm);
  this->utils.setFieldData(density    , fm);
  this->utils.setFieldData(etadotdVelx, fm);
  this->utils.setFieldData(pGrad      , fm);
  this->utils.setFieldData(uDot       , fm);
  this->utils.setFieldData(wBF        , fm);
  this->utils.setFieldData(wGradBF    , fm);
  this->utils.setFieldData(coordVec   , fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_VelResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          // Transient Term
          Residual(cell,node,level) += uDot(cell,qp,level)*wBF(cell,node,qp);
          // Advection Term
          for (int j=0; j < numDims; ++j) {
            Residual(cell,node,level) += ( keGrad(cell,qp,level,j) + PhiGrad(cell,qp,level,j) )*wBF(cell,node,qp);
            Residual(cell,node,level) += ( (1.0/density(cell,qp,level))*pGrad(cell,qp,level,j) + etadotdVelx(cell,qp,level) )*wBF(cell,node,qp);
          }
        }
      }
    }
  }
}
}
