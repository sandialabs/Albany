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
XZHydrostatic_TracerResid<EvalT, Traits>::
XZHydrostatic_TracerResid(Teuchos::ParameterList& p,
                      const Teuchos::RCP<Aeras::Layouts>& dl) :
  wBF        (p.get<std::string> ("Weighted BF Name"),                 dl->node_qp_scalar   ),
  wGradBF         (p.get<std::string> ("Weighted Gradient BF Name"),   dl->node_qp_gradient),
  piTracerGrad (p.get<std::string> ("Gradient QP PiTracer"),           dl->qp_gradient_level),
  TracerDot  (p.get<std::string> ("QP Time Derivative Variable Name"), dl->node_scalar_level  ),        
  TracerSrc  (p.get<std::string> ("Tracer Source Name"),               dl->qp_scalar_level  ),        
  UTracerDiv (p.get<std::string> ("Divergence QP UTracer"),            dl->qp_scalar_level),        
  //etadotdTracer (p.get<std::string> ("Tracer EtaDotd Name"),         dl->qp_scalar_level  ),        
  dedotpiTracerde (p.get<std::string> ("Tracer EtaDotd Name"),         dl->qp_scalar_level  ),        
  viscosity       (p.isParameter("XZHydrostatic Problem") ? 
                   p.get<Teuchos::ParameterList*>("XZHydrostatic Problem")->get<double>("Viscosity", 0.0):
                   p.get<Teuchos::ParameterList*>("Hydrostatic Problem")  ->get<double>("Viscosity", 0.0)),
  Residual   (p.get<std::string> ("Residual Name"),                    dl->node_scalar_level),        
  numNodes   (dl->node_scalar             ->dimension(1)),
  numQPs     (dl->node_qp_scalar          ->dimension(2)),
  numDims    (dl->node_qp_gradient        ->dimension(3)),
  numLevels  (dl->node_scalar_level       ->dimension(2))
{
  this->addDependentField(TracerDot);
  this->addDependentField(UTracerDiv);
  this->addDependentField(wBF);
  this->addDependentField(wGradBF);
  this->addDependentField(piTracerGrad);
  this->addDependentField(TracerSrc);
  //this->addDependentField(etadotdTracer);
  this->addDependentField(dedotpiTracerde);

  this->addEvaluatedField(Residual);

  this->setName("Aeras::XZHydrostatic_TracerResid" + PHX::typeAsString<EvalT>());

  Schmidt = 1.0;
}

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_TracerResid<EvalT, Traits>::
postRegistrationSetup(typename Traits::SetupData d,
                      PHX::FieldManager<Traits>& fm)
{
  this->utils.setFieldData(TracerDot,      fm);
  this->utils.setFieldData(UTracerDiv,     fm);
  this->utils.setFieldData(wBF,            fm);
  this->utils.setFieldData(wGradBF,        fm);
  this->utils.setFieldData(piTracerGrad,   fm);
  this->utils.setFieldData(TracerSrc,      fm);
  //this->utils.setFieldData(etadotdTracer,  fm);
  this->utils.setFieldData(dedotpiTracerde,  fm);
  this->utils.setFieldData(Residual,       fm);
}

//**********************************************************************
// Kokkos kernels
#ifdef ALBANY_KOKKOS_UNDER_DEVELOPMENT
template<typename EvalT, typename Traits>
KOKKOS_INLINE_FUNCTION
void XZHydrostatic_TracerResid<EvalT, Traits>::
operator() (const XZHydrostatic_TracerResid_Tag& tag, const int& cell) const{
  for (int qp=0; qp < numQPs; ++qp) {
    int node = qp; 
    for (int level=0; level < numLevels; ++level) {
      Residual(cell,node,level) +=     TracerDot(cell,qp,level) * wBF(cell,node,qp)
                                +      TracerSrc(cell,qp,level) * wBF(cell,node,qp)
                                +     UTracerDiv(cell,qp,level) * wBF(cell,node,qp)
                                +  dedotpiTracerde(cell,qp,level) * wBF(cell,node,qp);
    }
  }

  for (int node=0; node < numNodes; ++node) {
    for (int level=0; level < 2; ++level) {
      for (int qp=0; qp < numQPs; ++qp) {
        for (int dim=0; dim < numDims; ++dim) {
          Residual(cell,node,level) += (viscosity/Schmidt)*piTracerGrad(cell,qp,level,dim)*wGradBF(cell,node,qp,dim);
        }
      }
    }
  }
}

#endif

//**********************************************************************
template<typename EvalT, typename Traits>
void XZHydrostatic_TracerResid<EvalT, Traits>::
evaluateFields(typename Traits::EvalData workset)
{
  PHAL::set(Residual, 0.0);

#ifndef ALBANY_KOKKOS_UNDER_DEVELOPMENT
  for (int cell=0; cell < workset.numCells; ++cell) {
    for (int qp=0; qp < numQPs; ++qp) {
      int node = qp; 
      //for (int node=0; node < numNodes; ++node) {
        for (int level=0; level < numLevels; ++level) {
          Residual(cell,node,level) +=     TracerDot(cell,qp,level) * wBF(cell,node,qp)
                                    +      TracerSrc(cell,qp,level) * wBF(cell,node,qp)
                                    +     UTracerDiv(cell,qp,level) * wBF(cell,node,qp)
                                    +  dedotpiTracerde(cell,qp,level) * wBF(cell,node,qp);
          //Residual(cell,node,level) += etadotdTracer(cell,qp,level) * wBF(cell,node,qp);

          //std::cout <<"IN TRACERS: TracerSrc" << TracerSrc(cell,qp,level) <<" UTracerDiv "<< UTracerDiv(cell,qp,level) << "\n";

        //}
      }
    }

    for (int node=0; node < numNodes; ++node) {
      for (int level=0; level < 2; ++level) {
        for (int qp=0; qp < numQPs; ++qp) {
          for (int dim=0; dim < numDims; ++dim) {
            Residual(cell,node,level) += (viscosity/Schmidt)*piTracerGrad(cell,qp,level,dim)*wGradBF(cell,node,qp,dim);
          }
        }
      }
    }
  }

#else
  Kokkos::parallel_for(XZHydrostatic_TracerResid_Policy(0,workset.numCells),*this);

#endif
}
}
