//*****************************************************************//
//    Albany 3.0:  Copyright 2016 Sandia Corporation               //
//    This Software is released under the BSD license detailed     //
//    in the file "license.txt" in the top-level Albany directory  //
//*****************************************************************//

#include "Teuchos_TestForException.hpp"
#include "Teuchos_VerboseObject.hpp"
#include "Teuchos_RCP.hpp"
#include "Phalanx_DataLayout.hpp"
#include "Sacado_ParameterRegistration.hpp"
#include "PHAL_Utilities.hpp"

#include "Intrepid2_FunctionSpaceTools.hpp"
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
  etadotdVelx (p.get<std::string> ("EtaDotdVelx"),                      dl->node_vector_level),
  pGrad       (p.get<std::string> ("Gradient QP Pressure"),             dl->qp_gradient_level),

  uDot        (p.get<std::string> ("QP Time Derivative Variable Name"), dl->node_vector_level),
  DVelx       (p.get<std::string> ("D Vel Name"),                       dl->qp_vector_level),
  density     (p.get<std::string> ("QP Density"),                       dl->node_scalar_level),
  Residual    (p.get<std::string> ("Residual Name"),                    dl->node_vector_level),

  viscosity      (p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Viscosity", 0.0)),
  hyperviscosity (p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("HyperViscosity", 0.0)),
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

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_VelResid" + PHX::typeAsString<EvalT>());

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

  this->utils.setFieldData(Residual,fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_VelResid<EvalT, Traits>::
operator() (const int cell, const int node, const int level) const{
  for (int dim=0; dim < numDims; ++dim) {
    int qp = node;
    ScalarT wbf=wBF(cell,node,qp); 
    Residual(cell,node,level,dim) = 
	( keGrad(cell,qp,level,dim) + PhiGrad(cell,qp,level,dim) )*wbf
        + ( pGrad (cell,qp,level,dim)/density(cell,qp,level) )*wbf
        + etadotdVelx(cell,qp,level,dim)*wbf
        + uDot(cell,qp,level,dim)*wbf;
     for (int qp=0; qp < numQPs; ++qp) {
          Residual(cell,node,level,dim) += viscosity * DVelx(cell,qp,level,dim) * wGradBF(cell,node,qp,dim);
     }
   }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_VelResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < numLevels; ++level) {
        for (int dim=0; dim < numDims; ++dim) {
          int qp = node; 
          Residual(cell,node,level,dim) = ( keGrad(cell,qp,level,dim) + PhiGrad(cell,qp,level,dim) )*wBF(cell,node,qp)
                                        + ( pGrad (cell,qp,level,dim)/density(cell,qp,level) )      *wBF(cell,node,qp)
                                        + etadotdVelx(cell,qp,level,dim)                            *wBF(cell,node,qp)
                                        + uDot(cell,qp,level,dim)                                   *wBF(cell,node,qp);

          for (int qp=0; qp < numQPs; ++qp) {
            Residual(cell,node,level,dim) += viscosity * DVelx(cell,qp,level,dim) * wGradBF(cell,node,qp,dim);
          }
        }
      }
    }
  }

#else
  XZHydrostatic_VelResid_Policy range(
             {0,0,0}, {(int)workset.numCells,(int)numNodes,(int)numLevels}, XZHydrostatic_VelResid_TileSize);
  Kokkos::Experimental::md_parallel_for(range,*this);

#endif
}
}
