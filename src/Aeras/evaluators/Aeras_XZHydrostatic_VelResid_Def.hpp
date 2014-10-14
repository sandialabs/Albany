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
  wGradGradBF (p.isParameter("Hydrostatic Problem") &&  
               p.get<Teuchos::ParameterList*>("Hydrostatic Problem")->isParameter("HyperViscosity") ?
               p.get<std::string> ("Weighted Gradient Gradient BF Name") : "None", dl->node_qp_tensor) ,
  keGrad      (p.get<std::string> ("Gradient QP Kinetic Energy"),       dl->qp_gradient_level),
  PhiGrad     (p.get<std::string> ("Gradient QP GeoPotential"),         dl->qp_gradient_level),
  etadotdVelx (p.get<std::string> ("EtaDotdVelx"),                      dl->qp_vector_level),
  pGrad       (p.get<std::string> ("Gradient QP Pressure"),             dl->qp_gradient_level),

  uDot        (p.get<std::string> ("QP Time Derivative Variable Name"), dl->qp_vector_level),
  DVelx       (p.get<std::string> ("D Vel Name"),                       dl->qp_vector_level),
  LaplaceVelx (p.isParameter("Hydrostatic Problem") &&
                p.get<Teuchos::ParameterList*>("Hydrostatic Problem")->isParameter("HyperViscosity") ?
                p.get<std::string> ("Laplace Vel Name") : "None",dl->qp_scalar_level),
  density     (p.get<std::string> ("QP Density"),                       dl->qp_scalar_level),
  Residual    (p.get<std::string> ("Residual Name"),                    dl->node_vector_level),

  viscosity   (p.isParameter("XZHydrostatic Problem") ? 
                p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Viscosity", 0.0):
                p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("Viscosity", 0.0)),
  hyperviscosity(p.isParameter("XZHydrostatic Problem") ? 
                p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("HyperViscosity", 0.0):
                p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("HyperViscosity", 0.0)),
  numNodes    ( dl->node_scalar             ->dimension(1)),
  numQPs      ( dl->node_qp_scalar          ->dimension(2)),
  numDims     ( dl->node_qp_gradient        ->dimension(3)),
  numLevels   ( dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(keGrad);
  this->addDependentField(PhiGrad);
  this->addDependentField(density);
  this->addDependentField(etadotdVelx);
  this->addDependentField(DVelx);
  this->addDependentField(pGrad);
  this->addDependentField(uDot);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  if (hyperviscosity) this->addDependentField(LaplaceVelx);
  if (hyperviscosity) this->addDependentField(wGradGradBF);

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
  this->utils.setFieldData(DVelx      , fm);
  this->utils.setFieldData(pGrad      , fm);
  this->utils.setFieldData(uDot       , fm);
  this->utils.setFieldData(wBF        , fm);
  this->utils.setFieldData(wGradBF    , fm);
  if (hyperviscosity) this->utils.setFieldData(LaplaceVelx, fm);
  if (hyperviscosity) this->utils.setFieldData(wGradGradBF, fm);

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_VelResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  for (int i=0; i < Residual.size(); ++i) Residual(i)=0.0;

  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int dim=0; dim < numDims; ++dim) {
            Residual(cell,node,level,dim) += ( keGrad(cell,qp,level,dim) + PhiGrad(cell,qp,level,dim) )*wBF(cell,node,qp);
            Residual(cell,node,level,dim) += ( pGrad (cell,qp,level,dim)/density(cell,qp,level) )      *wBF(cell,node,qp);
            Residual(cell,node,level,dim) += etadotdVelx(cell,qp,level,dim)                            *wBF(cell,node,qp);
            Residual(cell,node,level,dim) += uDot(cell,qp,level,dim)                                   *wBF(cell,node,qp);
            Residual(cell,node,level,dim) += viscosity * DVelx(cell,qp,level,dim) * wGradBF(cell,node,qp,dim);
            if (hyperviscosity) 
              Residual(cell,node,level,dim) -= hyperviscosity * LaplaceVelx(cell,qp,level) * wGradGradBF(cell,node,qp,dim,dim);
          }
        }
      }
    }
  }
}
}
